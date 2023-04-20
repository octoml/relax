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

import datetime
import json

from pathlib import Path
from typing import List, Optional, Dict, Any

import numpy as np
from benchmarking_utils import BenchmarkConfig, FROM_HUB, sha256sum


def flush_result(
    result_path: Optional[Path],
    compile_time_ms: float,
    import_error: Optional[str],
    runtimes_ms: List[float],
    shapes: Dict[str, List[int]],
    output_deltas: List[List[np.ndarray]],
    run_config: BenchmarkConfig,
    framework_ops: List[str],
    runtime_metadata: Dict[str, Any],
    relax_ops: List[Dict[str, Any]],
    extra_info: Dict[str, Any],
) -> None:
    """
    Print the results of a run to stdout if 'result_path' is None otherwise
    write it to that file
    """
    model_config = run_config.config

    outputs_match_onnx = True
    for deltas in output_deltas:
        for delta in deltas:
            if not np.allclose(
                np.zeros_like(delta), delta, atol=run_config.atol, rtol=run_config.rtol
            ):
                outputs_match_onnx = False

    end_to_end_runtimes_ms = np.array(runtimes_ms)
    # branch, sha = git_info()
    branch = "tbd"
    sha = "tbd"

    if model_config.file() == FROM_HUB:
        # TODO: implement this for hub models
        model_sha = "unknown"
    else:
        model_sha = sha256sum(model_config.file())

    if import_error is not None or len(runtimes_ms) == 0:
        mean_sec = 0
        p95_sec = 0
        std_dev_sec = 0
        variance_sec2 = 0
        cov = 0
    else:
        runtimes_s = np.array(end_to_end_runtimes_ms) / 1000.0
        mean_s = np.mean(runtimes_s)
        std_dev_s = np.std(runtimes_s)
        mean_sec = mean_s
        p95_sec = np.percentile(runtimes_s, 95)
        std_dev_sec = std_dev_s
        variance_sec2 = np.var(runtimes_s)
        cov = std_dev_s / mean_s

    data = {
        #  identifying fields
        "test_run_id": "to be filled in",
        "run_at": "to be filled in",
        "test_suite_id": "to be filled in",
        "model_set_id": model_config.set,
        "model_name": model_config.name,
        "config_name": model_config.flow_config,
        # info to reproduce results
        "model_hash": model_sha,
        "runtime_metadata": runtime_metadata,
        "import_error": import_error,
        "warmup_runs": run_config.warmup_runs,
        "test_runs": run_config.test_runs,
        "input_shapes": shapes,
        # coarse grained timings
        "inference_stats": {
            "mean_sec": mean_sec,
            "p95_sec": p95_sec,
            "std_dev_sec": std_dev_sec,
            "variance_sec2": variance_sec2,
            "cov": cov,
        },
        "compile_time_ms": compile_time_ms,
        "raw_stats_ms": runtimes_ms,
        "outputs_match_onnx": outputs_match_onnx,
        # model details
        "framework_ops": framework_ops,
        "relax_ops": relax_ops,
        # coverage results
        **extra_info,
    }

    if result_path is None:
        print(json.dumps(data, indent=2), flush=True)
    else:
        with open(result_path, "w") as f:
            json.dump(data, f, indent=2)
