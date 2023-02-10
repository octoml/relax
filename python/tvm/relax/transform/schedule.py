# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, unused-argument
"""Relax passes related to scheduling functions for target hardware."""
import tempfile
from typing import Union, List

from tvm import relax
from tvm.ir import transform, IRModule
from tvm.target import Target
from tvm import meta_schedule as ms
from .tuning_api import Trace


@transform.module_pass(opt_level=2, name="schedule_for_target")
class ScheduleForTarget:
    """Apply a minimal set of transformations to enable running on a specific target."""

    def __init__(self, target: Union[Target, str], trials_per_task: int = 4):
        """
        This function returns a pass which applies basic schedule transformations to each
        primitive function in the input module for the specified target. This is useful
        when a hardware target requires certain intrinsics for kernels to be valid. For
        example, on GPUs, each kernel must have loops bound to a thread and block index.
        By default, primitive functions do not contain this binding. Applying a single
        step of Metaschedule's transform rules inserts bindings that enable the functions
        to be run.

        Thus, this pass is a convenience wrapper around the simplist possible invocation
        of Metaschedule tuning. It performs only a few schedules per task, skips benchmarking,
        and verifies that they can be built.

        Parameters
        ----------
        target : Union[Target, str]
            The tvm target that fucntions should be scheduled for.
        trials_per_task : int
            The number of transformations to try per task. The higher this number is,
            the longer the pass will take, but the less likely it is for an invalid
            schedule to be picked.
        """
        if isinstance(target, str):
            target = Target(target)
        self.target = target
        self.trials_per_task = trials_per_task
        # Create a fake runner function that does not perform benchmarking. This
        # allows us to save time when transforming primitive functions in the module.
        @ms.derived_object
        class FakeRunner(ms.runner.PyRunner):
            def run(
                self, runner_inputs: List[ms.runner.RunnerInput]
            ) -> List[ms.runner.RunnerFuture]:
                return [ms.runner.LocalRunnerFuture([0.0], None)]

        self.runner = FakeRunner()

    def transform_module(self, mod: IRModule, ctx: transform.PassContext) -> IRModule:
        """Apply a minimal set of tuning to transform the input module.

        Parameters
        ----------
        mod : IRModule
            The input module to schedule.
        ctx : transform.PassContext
            Information about the current pass, not currently used.

        Returns
        -------
        scheduled_mod : IRModule
            The input module with hardware specific transformations applied.
        """
        # Extract the number of tasks in the input module so that we can
        # determine the minimal number of transformations to try.
        num_tasks = len(ms.relax_integration.extract_tasks(mod, self.target))

        # Perform a minimal set of metaschedule tuning on the input module.
        with tempfile.TemporaryDirectory() as work_dir:
            with self.target, transform.PassContext(trace=Trace(mod), opt_level=0):
                # Create a pass that performs a few trials per task in the module.
                tuning_pass = relax.transform.MetaScheduleTuneIRMod(
                    params={},
                    work_dir=work_dir,
                    max_trials_global=self.trials_per_task * num_tasks,
                    max_trials_per_task=1,
                    runner=self.runner,
                )

                # Apply the pass on our module.
                mod = tuning_pass(mod)

                # Use the results of tuning to schedule the module.
                application_pass = relax.transform.MetaScheduleApplyDatabase(work_dir)
                mod = application_pass(mod)

        return mod
