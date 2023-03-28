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

from pathlib import Path
from typing import *
from types import ModuleType

import numpy as np

from benchmarking_utils import (
    Timer,
    eprint,
    ImportError,
    BaseRunner,
)

import tvm
import logging
from tvm import octo
from tvm.relax.frontend.onnx.onnx_frontend import ONNXGraphImporter


class RelaxBase(BaseRunner):
    def __init__(self, target, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target = target

    def metadata(self) -> Dict[str, Any]:
        return {
            "relax-version": "unknown",
        }

    def load_model(self) -> "onnx.ModelProto":
        return self._model

    def run(
        self,
        inputs,
        gold_results: List[np.ndarray],
    ):
        runtimes_ms = []
        output_deltas = []
        run_config = self.benchmark_config

        tvm_version = tvm.support.libinfo()["GIT_COMMIT_HASH"]
        id = run_config.config.id()
        eprint(
            f"[{id}] Running octo.compile (from {tvm_version}) with shapes={self.benchmark_config.config.shapes}, target={self.target}, and tuning_steps={self.benchmark_config.config.tuning_steps}"
        )
        eprint(
            f"[{id}]    Running for {run_config.warmup_runs} warmups and {run_config.test_runs} tests"
        )

        # Disable tuning logs
        ms_logger = logging.getLogger("tvm.meta_schedule")
        ms_logger.setLevel(logging.CRITICAL)
        for name in logging.root.manager.loggerDict:
            if "tvm" in name:
                logger = logging.getLogger(name)
                logger.setLevel(logging.CRITICAL)

        with Timer() as compile_timer:
            try:
                tvm_model = octo.compile(
                    self._model,
                    shape_dict=self.benchmark_config.config.shapes,
                    target=tvm.target.Target(self.target),
                    tuning_steps=self.benchmark_config.config.tuning_steps,
                )
            except Exception as e:
                return e, ImportError.FAILED_OCTO_COMPILE, 0, [], []

        compile_time_ms = compile_timer.ms_duration
        breakpoint()

        # NOTE: ONNX frontend sanitizes input names. This hack is brittle and presumes Python dict ordering is the same
        # between invocations. The real fix should be that OctoModel carries a mapping of framework names to Relax names.
        importer = ONNXGraphImporter({}, {})
        for k in list(inputs):
            sanitized_k = importer._sanitize_name(k)
            if sanitized_k != k:
                inputs[sanitized_k] = inputs[k]
                del inputs[k]

        for i in range(run_config.warmup_runs):
            eprint(
                f"[{id}][{i + 1} / {run_config.warmup_runs}][{self.name}] Warmup {run_config.config.id()}"
            )
            tvm_model.run(inputs)

        # Run the model a few times and record the end to end execution time
        for i in range(run_config.test_runs):
            eprint(
                f"[{id}][{i + 1} / {run_config.test_runs}][{self.name}] Running {run_config.config.id()}"
            )
            try:
                with Timer() as timer:
                    output = tvm_model.run(inputs)

                # Stash the runtime
                runtimes_ms.append(timer.ms_duration)

            except Exception as e:
                return e, ImportError.FAILED_EXECUTION, 0, [], []

            # Check accuracy
            output_deltas.append([gold_results[i] - output[i] for i in range(len(output))])

        return None, None, compile_time_ms, runtimes_ms, output_deltas
