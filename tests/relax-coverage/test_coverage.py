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

import configparser
import datetime
import json
import os
import shutil

from pathlib import Path
from typing import *

# unused but needed to get CUDA working in onnx, too lazy to actually fix the
# issue
# https://stackoverflow.com/questions/75267445/why-does-onnxruntime-fail-to-create-cudaexecutionprovider-in-linuxubuntu-20/75267493#75267493
import torch
import numpy as np
import onnxruntime.backend
import pytest
import tvm
import tvm.testing
import tabulate
import yaml
import onnxruntime

from tvm.tir import Any as Any
from utils import *
from runners import *

# Directory to store output JSON files
JSON_SCHEMA = REPO_ROOT / "schema" / "schema.jsonschema"
OUTPUT_DIR = REPO_ROOT / ".coverage_results"
UPLOAD_GCP = os.getenv("UPLOAD_GCP", "0") == "1"
UPLOAD_PG = os.getenv("UPLOAD_PG", "0") == "1"
TEST_RUNS = int(os.getenv("TEST_RUNS", "1"))
WARMUP_RUNS = int(os.getenv("WARMUP_RUNS", "0"))
ONNX_NIGHTLY_PATH = os.environ["ONNX_NIGHTLY_PATH"]

# Load additional tests from disk
MODELS_YAML = REPO_ROOT / "models.yaml"
ON_DISK_MODELS: List[ModelConfig] = []
if MODELS_YAML.exists():
    with open(MODELS_YAML) as f:
        data = yaml.safe_load(f)

    for item in data:
        set = item["set"]
        name = item["name"]
        sha256 = item["sha256"]

        requires_toposort = bool(item.get("requires_toposort"))
        atol = float(item.get("atol", 0.0001))
        input_scale = float(item.get("input_scale", 1))
        rtol = float(item.get("rtol", 0.0001))

        shapes = None
        dtypes = None

        shapes_entry = item.get("shapes")
        if shapes_entry is not None:
            shapes = {}
            dtypes = {}

            for entry in shapes_entry:
                shapes[entry["name"]] = entry["shape"]
                dtypes[entry["name"]] = entry["dtype"]

        for flow_config in item["configs"]:
            ON_DISK_MODELS.append(
                BenchmarkConfig(
                    config=ModelConfig(
                        set=set,
                        name=name,
                        sha256=sha256,
                        flow_config=flow_config,
                        requires_toposort=requires_toposort,
                        input_scale=input_scale,
                        shapes=shapes,
                        dtypes=dtypes,
                    ),
                    warmup_runs=WARMUP_RUNS,
                    test_runs=TEST_RUNS,
                    check_accuracy=True,
                    atol=atol,
                    rtol=rtol,
                )
            )


RUNNERS = {
    # - TVM with Relay + BYOC CUTLASS
    "byoc_cutlass": byoc_cutlass,
    # - TVM with Relay + MetaSchedule + AutoTensorization
    "tvm_relay_msat": tvm_relay_msat,
    # - ONNX-RT + TensorRT EP
    "onnx_trt": run_onnx(
        use_nightly=False,
        nightly_path=ONNX_NIGHTLY_PATH,
        providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"],
    ),
    "onnx_trt_nightly": run_onnx(
        use_nightly=True,
        nightly_path=ONNX_NIGHTLY_PATH,
        providers=["TensorrtExecutionProvider", "CUDAExecutionProvider"],
    ),
    "onnx_cpu_nightly": run_onnx(
        use_nightly=True,
        nightly_path=ONNX_NIGHTLY_PATH,
        providers=["CPUExecutionProvider"],
    ),
    # "onnx_trt": run_onnx(providers="TensorrtExecutionProvider", "CUDAExecutionProvider"]),
    # - ONNX CPU
    # "onnx_cpu": run_onnx(providers=["CPUExecutionProvider"]),
    # TVM LLVM graph executor
    "tvm_llvm": tvm_llvm,
    "relax_native": relax_native,
    "relax_cuda": relax_cuda,
}


def flush_result(
    result_directory: Path,
    compile_time_ms: float,
    import_error: Optional[str],
    runtimes_ms: List[float],
    shapes: Dict[str, List[int]],
    output_deltas: List[List[np.ndarray]],
    run_config: BenchmarkConfig,
    framework_ops: List[str],
    relay_ops: List[Dict[str, Any]],
) -> Path:
    """
    Output a model result to disk
    """
    model_config = run_config.config
    result_directory.mkdir(exist_ok=True, parents=True)
    i = 0
    while True:
        output_path = result_directory / f"{model_config.flow_config}_{i}.json"
        if not output_path.exists():
            break
        i += 1

    DATE_FORMAT = "%Y-%m-%d-%H:%M:%S"
    date = datetime.datetime.now().strftime(DATE_FORMAT)
    id = f"{date}-{i}"

    outputs_match_onnx = True
    for deltas in output_deltas:
        for delta in deltas:
            if not np.allclose(
                np.zeros_like(delta), delta, atol=run_config.atol, rtol=run_config.rtol
            ):
                outputs_match_onnx = False

    end_to_end_runtimes_ms = np.array(runtimes_ms)
    relay_fusion_groups: List[List[int]] = []
    # branch, sha = git_info()
    branch = "tbd"
    sha = "tbd"

    if model_config.file() == FROM_HUB:
        # TODO: implement this for hub models
        model_sha = "unknown"
    else:
        model_sha = sha256sum(model_config.file())

    print("runtimes")
    for rt in runtimes_ms:
        print(rt)

    test_suite_id = result_directory.name

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
        "test_run_id": id,
        "run_at": date,
        "test_suite_id": test_suite_id,
        "model_set_id": model_config.set,
        "model_name": model_config.name,
        "config_name": model_config.flow_config,
        # info to reproduce results
        "model_hash": model_sha,
        "repo": {
            "owner": "octoml",
            "repo": "relax",
            "sha": sha,
            "branch": branch,
        },
        "onnx_version": onnxruntime.__version__,
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
        "relay_ops": relay_ops,
        "relay_fusion_groups": relay_fusion_groups,
        # coverage results
    }

    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Wrote results to {output_path}")

    return output_path


@pytest.fixture(scope="session")
def result_directory(request):
    output_dir = gen_test_output_dir(base=OUTPUT_DIR)

    def finalizer():
        catted_results_path = output_dir / "concatted_results.jsonl"

        print(f"Preparing files in {output_dir} for upload")
        outputs = list(output_dir.glob("*.json"))
        if len(outputs) == 0:
            print(f"Skipping result upload for {output_dir} there were no result files")
            return

        with open(catted_results_path, "w") as f:
            concat_test_results(
                schema_file=JSON_SCHEMA,
                test_results=list(output_dir.glob("*.json")),
                output=f,
            )

        if UPLOAD_GCP:
            print(f"[GCP] Uploading records from {catted_results_path}")
            dataset_id = "contrived_test_results_1"
            # prod table
            table_id = "contrived-string"

            # testing table
            # table_id = "gitlab_ci_data"
            job = bigquery_upload(
                jsonl_file=catted_results_path, dataset_id=dataset_id, table_id=table_id
            )
            print(
                f"[GCP] Done uploading {job.output_rows} records to {dataset_id}:{table_id}"
            )
        else:
            print(
                f"[GCP] Skipping result upload for {output_dir} since UPLOAD_GCP was not 1 ({os.getenv('UPLOAD_GCP', 'unset')})"
            )

        if UPLOAD_PG:
            database = "scorecard"
            table_name = "gitlab_data"
            rows = postgres_upload(
                jsonl_file=catted_results_path,
                database=database,
                table_name=table_name,
            )
            print(f"[PG] Done uploading {rows} records to {database}:{table_name}")
        else:
            print(
                f"[PG] Skipping result upload for {output_dir} since UPLOAD_PG was not 1 ({os.getenv('UPLOAD_PG', 'unset')})"
            )

        # Plain text report
        data = load_jsonl(jsonl_file=catted_results_path)
        rows = []
        for item in data:
            item = json.loads(item["r"])
            name = (
                f"{item['model_set_id']} / {item['model_name']} / {item['config_name']}"
            )
            if len(item["raw_stats_ms"]) == 0:
                rows.append([name, "no data", "no data", "no data", "no data"])
                continue

            runtimes_s = [x / 1000.0 for x in item["raw_stats_ms"]]
            if item["import_error"] is None:
                rows.append(
                    [
                        name,
                        item["inference_stats"]["mean_sec"],
                        np.min(runtimes_s),
                        np.var(runtimes_s),
                        item["inference_stats"]["cov"],
                    ]
                )
            else:
                rows.append(
                    [
                        name,
                        "err",
                        "err",
                        "err",
                        "err",
                    ]
                )
        rows = sorted(rows, key=lambda row: row[0])

        print(f"benchmark over {TEST_RUNS} runs and {WARMUP_RUNS} warmup runs")
        print(
            tabulate.tabulate(
                rows, headers=["model", "mean (s)", "min (s)", "var (s^2)"]
            )
        )

    request.addfinalizer(finalizer)

    yield output_dir


_gold_cache = {}


def get_gold_results(
    model: onnx.ModelProto, inputs, model_config: ModelConfig
) -> List[np.ndarray]:
    key = f"{model_config.set}.{model_config.name}"
    if key not in _gold_cache:
        sess_opt = ort.SessionOptions()

        # Set up an onnx inference on GPU
        sess = ort.InferenceSession(
            model.SerializeToString(),
            sess_options=sess_opt,
            providers=["CPUExecutionProvider"],
        )
        output_names = []
        output = sess.run(output_names, inputs)
        _gold_cache[key] = output

    return _gold_cache[key]


# @pytest.mark.parametrize(
#     "run_config",
#     [
#         # ["onnx-opensource", "ResNet50", FROM_HUB, "tvm_llvm"],
#         # ["onnx-opensource", "ResNet50", FROM_HUB, "onnx_trt"],
#         # # TODO: Re-enable these when the relevant configs are implemented
#         # # ["onnx-opensource", "ResNet50", FROM_HUB, "tvm_relay_msat"],
#         # # ["onnx-opensource", "ResNet50", FROM_HUB, "byoc_cutlass"],
#     ]
#     + ON_DISK_MODELS,  # Also run models not defined in this file
# )
# TODO: Initialize with models from ONNX hub
HUB_MODELS = []


@parameterize_configs(ON_DISK_MODELS + HUB_MODELS)
def test_mean_runtime(result_directory, run_config: BenchmarkConfig):
    """
    Tests end to end mean/p95 runtime for models on available backends
    """
    np.random.seed(0)
    # Load the model from a local file or download it from ONNX hub
    model_config = run_config.config
    model = model_config.load_model()

    # Determine the model inputs
    if model_config.shapes is None:
        shapes, dtypes = infer_inputs(model)
    else:
        shapes = model_config.shapes
        dtypes = model_config.dtypes

    if model_config.name == "turing_vortex_fp16":
        for shape in shapes.values():
            shape[1] = 512

    input_names = list(shapes.keys())
    # inputs = [np.random.uniform(size=shapes[name]).astype(dtypes[name]) for name in input_names]
    inputs = {}
    for name in input_names:
        # if "int" in dtypes[name]:
        #     inputs[name] = (np.random.uniform(size=shapes[name]) * 100).astype(dtypes[name])
        # else:
        inputs[name] = (
            np.random.uniform(size=shapes[name]) * model_config.input_scale
        ).astype(dtypes[name])

    # Select the runnner
    runner = RUNNERS[model_config.flow_config]

    gold_results = []
    if run_config.test_runs > 0:
        gold_results = get_gold_results(
            model=model, inputs=inputs, model_config=model_config
        )

    # Run the model a few times and extract timings
    error, import_error, compile_time_ms, runtimes_ms, output_deltas = runner(
        model=model,
        inputs=inputs,
        run_config=run_config,
        shapes=shapes,
        dtypes=dtypes,
        gold_results=gold_results,
    )

    try:
        framework_ops = extract_framework_ops(model)
    except Exception as e:
        framework_ops = []
        error = e

    if "onnx" in model_config.flow_config:
        relay_ops = []
    else:
        try:
            relay_ops = extract_relay_ops(model, framework_ops, shapes)
        except Exception as e:
            relay_ops = []
            error = e

    # Send the output results to a JSON file on disk
    flush_result(
        result_directory=result_directory,
        run_config=run_config,
        runtimes_ms=runtimes_ms,
        shapes=shapes,
        import_error=import_error,
        compile_time_ms=compile_time_ms,
        output_deltas=output_deltas,
        relay_ops=relay_ops,
        framework_ops=framework_ops,
    )

    # Re-raise any failures
    if error is not None:
        raise error


if __name__ == "__main__":
    # test_result_dir = OUTPUT_DIR / "testing"
    # if test_result_dir.exists():
    #     shutil.rmtree(test_result_dir)
    # test_result_dir.mkdir()
    # test_mean_runtime(
    #     result_directory=test_result_dir,
    #     run_config=BenchmarkConfig(
    #         config=ModelConfig(
    #             set="ms-models",
    #             # name="antisemitism_detection_fp16",  # GOOD
    #             # name="stable_diffusion",  # ERROR Cannot load
    #             # name="turing_text_large_fp16",  # GOOD
    #             # name="turing_vision_large_fp32",  # GOOD
    #             name="turing_vortex_fp16",  # GOOD
    #             input_scale=1,
    #             flow_config="relax_onnx",
    #             requires_toposort=False,
    #         ),
    #         test_runs=1,
    #         warmup_runs=0,
    #         atol=0.0001,
    #         rtol=0.0001,
    #         check_accuracy=True,
    #     ),
    # )

    model = ModelConfig(
        set="ms-models",
        name="turing_vortex_fp16",
        input_scale=1,
        flow_config="",
        sha256="",
        shapes=None,
        dtypes=None,
        requires_toposort=False,
    ).load_model(verify_sha256=False)

    fops = extract_framework_ops(model)
    print(fops)
    print("")
    print("")
    shapes = {
        "q_title_token_ids": [1, 512],
        "q_title_token_types": [1, 512],
        "q_title_token_masks": [1, 512],
    }
    ops = extract_relay_ops(model, fops, shapes)
    print(ops)
    # tvm.testing.main()

else:
    for config in ON_DISK_MODELS:
        print(
            f"Running {config.config.id()} on {config.config.flow_config} ({config.warmup_runs} warmups, {config.test_runs} runs)"
        )
