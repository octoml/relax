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

from benchmarking_utils import BenchmarkConfig, Timer, eprint, BaseRunner


class OnnxBase(BaseRunner):
    def __init__(
        self,
        benchmark_config: BenchmarkConfig,
        providers: List[str],
        ort: ModuleType,
    ):
        self.ort = ort
        self.providers = providers

        super().__init__(benchmark_config=benchmark_config)

    def metadata(self) -> Dict[str, Any]:
        return {
            "ort-version": self.ort.__version__,
        }

    def run(
        self,
        inputs,
        gold_results: List[np.ndarray],
    ):
        """
        Run an onnx `model` with onnxruntime's TensorRT EP
        """
        runtimes_ms = []
        output_deltas = []
        sess_opt = self.ort.SessionOptions()

        run_config = self.benchmark_config

        id = run_config.config.id()
        eprint(f"[{id}] Running onnx trt")
        eprint(
            f"[{id}]    Running for {run_config.warmup_runs} warmups and {run_config.test_runs} tests"
        )

        # Set up an onnx inference on the specified providers
        sess = self.ort.InferenceSession(
            self._model.SerializeToString(),
            sess_options=sess_opt,
            providers=self.providers,
        )

        # Unwrap input if necessary
        if isinstance(inputs, list) and len(inputs) == 1:
            inputs = inputs[0]
        else:
            inputs = inputs

        output_names = []
        compile_time_ms = 0
        for i in range(run_config.warmup_runs):
            eprint(
                f"[{id}][{i + 1} / {run_config.warmup_runs}][onnx] Warmup {run_config.config.id()}"
            )
            sess.run(output_names, inputs)

        # Run the model a few times and record the end to end execution time
        for i in range(run_config.test_runs):
            eprint(
                f"[{id}][{i + 1} / {run_config.test_runs}][onnx] Running {run_config.config.id()}"
            )
            with Timer() as timer:
                output = sess.run(output_names, inputs)

            # Stash the runtime
            runtimes_ms.append(timer.ms_duration)

            # Check accuracy
            output_deltas.append([gold_results[i] - output[i] for i in range(len(output))])

        return None, None, compile_time_ms, runtimes_ms, output_deltas, {}
