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
import tvm
import tvm.testing
from tvm import relax as rx
from tvm.script import relax as R
from tvm.script import tir as T
from pathlib import Path
import sys
import json
import onnx
from tvm import relay
import pytest
import datetime
import scipy
import numpy as np
from tvm.tir import Any as Any
from typing import *
from utils import *
import onnxruntime.backend
from onnx import hub
import configparser
import onnxruntime as ort
from pathlib import Path
from tvm.contrib.cutlass import (
    has_cutlass,
    num_cutlass_partitions,
    finalize_modules,
    finalize_modules_vm,
)
from onnxruntime.tools import symbolic_shape_infer
from tvm.relay.op.contrib.cutlass import partition_for_cutlass

# Directory to store output JSON files
OUTPUT_DIR = Path("coverage_results")
FROM_HUB = object()
ONNX_REPO = "onnx/models"
ONNX_REPO_SHA = "8e893eb39b131f6d3970be6ebd525327d3df34ea"
REPO_ROOT = Path(__file__).parent.parent.parent.parent

# Load additional tests from disk
MODELS_JSON = REPO_ROOT / "models.ini"
ON_DISK_MODELS = []
if MODELS_JSON.exists():
    config = configparser.ConfigParser()
    config.read("models.ini")
    for key in config.keys():
        if key == "DEFAULT":
            continue

        group, name = key.split(".")
        path = config.get(key, "path")
        for flow_config in config.get(key, "flow_configs").split(","):
            ON_DISK_MODELS.append([group, name, path, flow_config])


np.random.seed(0)


def flush_result(model_name, model_file, model_set, runtimes_ms, target_flow, test_runs):
    """
    Output a model result to disk
    """
    output_dir = OUTPUT_DIR / model_set / model_name
    output_dir.mkdir(exist_ok=True, parents=True)
    i = 0
    while True:
        output_path = output_dir / f"{target_flow}_{i}.json"
        if not output_path.exists():
            break
        i += 1

    id = f'{datetime.datetime.now().strftime("%Y-%m-%d-%H:%M:%S")}-{i}'
    end_to_end_runtimes_ms = np.array(runtimes_ms)
    unqiue_onnx_op_names: List[str] = []
    unique_relay_ops: List[Dict[str, Any]] = []
    relay_fusion_groups: List[List[int]] = []
    branch, sha = git_info()

    if model_file == FROM_HUB:
        # TODO: implement this for hub models
        model_sha = "unknown"
    else:
        model_sha = sha256sum(model_file)

    data = {
        #  identifying fields
        "test_run_id": id,
        "model_set_id": model_set,
        "model_name": model_name,
        "config_name": target_flow,
        # info to reproduce results
        "model_hash": model_sha,
        "repo": {
            "owner": "octoml",
            "repo": "relax",
            "sha": sha,
            "branch": branch,
        },
        "test_runs": test_runs,
        # coarse grained timings
        "inference_stats": {
            "mean_sec": np.mean(end_to_end_runtimes_ms) / 1000.0,
            "p95_sec": np.percentile(end_to_end_runtimes_ms, 95) / 1000.0,
        },
        # model details
        "framework_ops": unqiue_onnx_op_names,
        "relay_ops": unique_relay_ops,
        "relay_fusion_groups": relay_fusion_groups,
        # coverage results
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote results to {output_path}")


def run_onnx(providers):
    def fn(*args, **kwargs):
        return run_onnx_impl(*args, **kwargs, providers=providers)

    return fn


def run_onnx_impl(id, model, inputs, warmup_runs, test_runs, shapes, dtypes, providers):
    """
    Run an onnx `model` with onnxruntime's TensorRT EP
    """
    runtimes_ms = []
    sess_opt = ort.SessionOptions()

    # Set up an onnx inference on GPU
    sess = ort.InferenceSession(
        model.SerializeToString(), sess_options=sess_opt, providers=providers
    )

    # Unwrap input if necessary
    if isinstance(inputs, list) and len(inputs) == 1:
        inp = inputs[0]
    else:
        inp = inputs
    output_names = []

    # Run the model a few times and record the end to end execution time
    for i in range(test_runs):
        print(f"[{i} / {test_runs}] Running {id}")
        with Timer() as timer:
            output = sess.run(output_names, inp)
        print(output)

        runtimes_ms.append(timer.ms_duration)
    return runtimes_ms


def profile_and_build(
    mod,
    params,
    sm,
    split_k_slices=[1],
    tmp_dir="./tmp",
    use_fast_math=False,
    use_3xtf32=True,
):
    print(
        f"before partitioning:\n {mod}",
    )
    mod = partition_for_cutlass(mod)
    print(
        f"after partitioning:\n {mod}",
    )

    num_cutlass_partition = num_cutlass_partitions(mod)
    host = tvm.target.Target("llvm")
    cuda = tvm.target.Target("cuda", host=host)
    cutlass = tvm.target.Target(
        {
            "kind": "cutlass",
            "sm": sm,
            "use_3xtf32": use_3xtf32,
            "split_k_slices": split_k_slices,
            "profile_all_alignments": False,
            "find_first_valid": True,
            "use_multiprocessing": True,
            "use_fast_math": use_fast_math,
            "tmp_dir": tmp_dir,
        },
        host=host,
    )
    with tvm.transform.PassContext(opt_level=3):
        lib = relay.build(mod, target=[cuda, cutlass], params=params)
    lib = finalize_modules(lib, "compile.so", tmp_dir)
    dev = tvm.device("cuda", 0)
    rt_mod = tvm.contrib.graph_executor.GraphModule(lib["default"](dev))
    return rt_mod, dev, num_cutlass_partition


def byoc_cutlass(id, model, inputs, warmup_runs, test_runs, shapes, dtypes):
    raise NotImplementedError()
    assert has_cutlass()
    mod, params = relay.frontend.from_onnx(model, shapes, opset=None, convert_config=None)
    params = inputs
    vm, dev, num_cutlass = profile_and_build(mod=mod, sm=80, params=params)
    return [1, 2, 3, 4]


def tvm_relay_msat(id, model, inputs, warmup_runs, test_runs, shapes, dtypes):
    raise NotImplementedError()
    return [1, 2, 3, 4]


def tvm_llvm(id, model, inputs, warmup_runs, test_runs, shapes, dtypes):
    """
    Run an onnx `model` with the vm executor
    """
    mod, params = relay.frontend.from_onnx(model, shapes, opset=None, convert_config=None)
    params = inputs
    dev = tvm.cpu()

    target = "llvm"
    print(mod, dev, target)
    executor = relay.create_executor("vm", mod=mod, device=dev, target=target)
    runner = executor.evaluate()
    runtimes_ms = []
    print(list(inputs.values()))
    input_data = list(inputs.values())
    for i in range(test_runs):
        print(f"[{i} / {test_runs}] tvm llvm running {id}")
        with Timer() as timer:
            result = runner(*input_data, **params)

        runtimes_ms.append(timer.ms_duration)
    return runtimes_ms


@pytest.mark.parametrize(
    "model_set,model_name,model_file,flow_config",
    [
        # ["onnx-opensource", "ResNet50", FROM_HUB, "tvm_llvm"],
        # ["onnx-opensource", "ResNet50", FROM_HUB, "onnx_trt"],
        # # TODO: Re-enable these when the relevant configs are implemented
        # # ["onnx-opensource", "ResNet50", FROM_HUB, "tvm_relay_msat"],
        # # ["onnx-opensource", "ResNet50", FROM_HUB, "byoc_cutlass"],
    ]
    + ON_DISK_MODELS,  # Also run models not defined in this file
)
def test_mean_runtime(model_set, model_name, model_file, flow_config):
    """
    Tests end to end mean/p95 runtime for models on available backends
    """
    target_flow = flow_config

    # Load the model from a local file or download it from ONNX hub
    if model_file == FROM_HUB:
        repo = f"{ONNX_REPO}:{ONNX_REPO_SHA}"
        model = hub.load(model_name, repo=repo, silent=True)
    else:
        model = onnx.load(model_file)

    # Determine the model inputs
    shapes, dtypes = infer_inputs(model)
    input_names = list(shapes.keys())
    # inputs = [np.random.uniform(size=shapes[name]).astype(dtypes[name]) for name in input_names]
    inputs = {}
    for name in input_names:
        # if "int" in dtypes[name]:
        #     inputs[name] = (np.random.uniform(size=shapes[name]) * 100).astype(dtypes[name])
        # else:
        inputs[name] = np.random.uniform(size=shapes[name]).astype(dtypes[name])

    # Select the runnner
    configs = {
        # - TVM with Relay + BYOC CUTLASS
        "byoc_cutlass": byoc_cutlass,
        # - TVM with Relay + MetaSchedule + AutoTensorization
        "tvm_relay_msat": tvm_relay_msat,
        # - ONNX-RT + TensorRT EP
        "onnx_trt": run_onnx(providers=["CUDAExecutionProvider"]),
        # "onnx_trt": run_onnx(providers="TensorrtExecutionProvider", "CUDAExecutionProvider"]),
        # - ONNX CPU
        # "onnx_cpu": run_onnx(providers=["CPUExecutionProvider"]),
        # TVM LLVM graph executor
        "tvm_llvm": tvm_llvm,
    }

    runner = configs[flow_config]
    name = f"{model_set}-{model_name}"
    test_runs = 3
    warmup_runs = 0

    # Run the model a few times and extract timings
    runtimes_ms = runner(
        name,
        model,
        inputs,
        warmup_runs=warmup_runs,
        test_runs=test_runs,
        shapes=shapes,
        dtypes=dtypes,
    )

    # Send the output results to a JSON file on disk
    flush_result(
        model_name=model_name,
        model_set=model_set,
        model_file=model_file,
        runtimes_ms=runtimes_ms,
        target_flow=target_flow,
        test_runs=test_runs,
    )


if __name__ == "__main__":
    tvm.testing.main()
