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

import json
import os
import shlex
import tempfile
import shutil
import re
import sys
import datetime

from pathlib import Path
from typing import *


import pytest
import tabulate
import yaml
import numpy as np

from onnx import hub
from utils import *
from runners.cloud_utils import *

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
HUB_MODELS_YAML = REPO_ROOT / "hub_models.yaml"
MODELS: List[Dict[str, Any]] = []
CLI = Path(__file__).resolve().parent / "runners" / "cli.py"


def generate_configs(item: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [{**item, "executor": config} for config in item["configs"]]


def get_configs_from_file(path: Path) -> List[Dict[str, Any]]:
    configs = []
    if path.exists():
        with open(path) as f:
            data = yaml.safe_load(f)

        for item in data:
            configs.extend(generate_configs(item))
    return configs


# MODELS = get_configs_from_file(HUB_MODELS_YAML) + get_configs_from_file(MODELS_YAML)
MODELS = get_configs_from_file(MODELS_YAML)


def flush_print(*args):
    print(*args, flush=True)


@pytest.fixture(scope="function", autouse=True)
def show_test_name(request):
    flush_print(f"Test '{request.node.nodeid}' STARTED")

    def fin():
        flush_print(f"Test '{request.node.nodeid}' COMPLETED")

    request.addfinalizer(fin)


@pytest.fixture(scope="session")
def upload_coverage(request):
    output_dir = gen_test_output_dir(base=OUTPUT_DIR)
    output_dir = output_dir / "coverage"

    def finalizer():
        pass

        if UPLOAD_GCP:
            flush_print(f"[GCP] Uploading records from {catted_results_path}")
            dataset_id = "contrived_test_results_1"
            # prod table
            table_id = "sampledate_coverage_name"

            # testing table
            # table_id = "gitlab_ci_data"
            job = bigquery_upload(
                jsonl_file=catted_results_path, dataset_id=dataset_id, table_id=table_id
            )
            flush_print(
                f"[GCP] Done uploading {job.output_rows} records to {dataset_id}:{table_id}"
            )
        else:
            flush_print(
                f"[GCP] Skipping result upload for {output_dir} since UPLOAD_GCP was not 1 ({os.getenv('UPLOAD_GCP', 'unset')})"
            )

    request.addfinalizer(finalizer)

    yield output_dir


@pytest.fixture(scope="session")
def result_directory(request):
    output_dir = gen_test_output_dir(base=OUTPUT_DIR)

    def finalizer():
        catted_results_path = output_dir / "concatted_results.jsonl"

        flush_print(f"Preparing files in {output_dir} for upload")
        outputs = list(output_dir.glob("*.json"))
        if len(outputs) == 0:
            flush_print(f"Skipping result upload for {output_dir} there were no result files")
            return

        with open(catted_results_path, "w") as f:
            concat_test_results(
                schema_file=JSON_SCHEMA,
                test_results=list(output_dir.glob("*.json")),
                output=f,
            )

        if UPLOAD_GCP:
            flush_print(f"[GCP] Uploading records from {catted_results_path}")
            dataset_id = "contrived_test_results_1"
            # prod table
            table_id = "contrived-string"

            # testing table
            # table_id = "gitlab_ci_data"
            job = bigquery_upload(
                jsonl_file=catted_results_path, dataset_id=dataset_id, table_id=table_id
            )
            flush_print(
                f"[GCP] Done uploading {job.output_rows} records to {dataset_id}:{table_id}"
            )
        else:
            flush_print(
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
            flush_print(f"[PG] Done uploading {rows} records to {database}:{table_name}")
        else:
            flush_print(
                f"[PG] Skipping result upload for {output_dir} since UPLOAD_PG was not 1 ({os.getenv('UPLOAD_PG', 'unset')})"
            )

        # Plain text report
        data = load_jsonl(jsonl_file=catted_results_path)
        rows = []
        for item in data:
            item = json.loads(item["r"])
            name = f"{item['model_set_id']} / {item['model_name']} / {item['config_name']}"
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

        flush_print(f"benchmark over {TEST_RUNS} runs and {WARMUP_RUNS} warmup runs")
        flush_print(
            tabulate.tabulate(rows, headers=["model", "mean (s)", "min (s)", "var (s^2)", "cov"])
        )

    request.addfinalizer(finalizer)

    yield output_dir


BAD_WARNINGS = [
    "UserWarning: Specified provider 'TensorrtExecutionProvider' is not in available provider names",
    "UserWarning: Specified provider 'CUDAExecutionProvider' is not in available provider names",
]


def _test_impl(request, slug, result_directory, run_config: Dict[str, Any]):
    DATE_FORMAT = "%Y-%m-%d-%H:%M:%S"
    run_at = datetime.datetime.now().strftime(DATE_FORMAT)
    executor = run_config["executor"]
    cmd = [
        sys.executable,
        CLI,
        "run",
        "--sha",
        run_config["sha256"],
        "--model",
        slug,
        "--executor",
        executor,
        "--random-inputs",
        "--runs",
        TEST_RUNS,
        "--warmup-runs",
        WARMUP_RUNS,
    ]
    if "shapes" in run_config and run_config["shapes"] is not None:
        cmd.append("--shapes")
        cmd.append(json.dumps(run_config["shapes"]))

    if run_config.get("requires_toposort", False):
        cmd.append("--toposort")

    if "cuda-sm" in run_config:
        cmd.append("--cuda-sm")
        cmd.append(run_config["cuda-sm"])

    if "tuning-steps" in run_config:
        cmd.append("--tuning-steps")
        cmd.append(run_config["tuning-steps"])

    cmd = [shlex.quote(str(c)) for c in cmd]
    cmd = " ".join(cmd)
    env = os.environ.copy()
    env["CUDA_PATH"] = "/usr/local/cuda-11.8"
    env["CUDA_MODULE_LOADING"] = "LAZY"
    flush_print(f"+ {cmd}")
    with tempfile.NamedTemporaryFile() as stderr_file:
        full_cmd = cmd + f" 2> >(tee -a {stderr_file.name} >&2)"
        proc = subprocess.run(
            full_cmd,
            check=False,
            stdout=subprocess.PIPE,
            encoding="utf-8",
            shell=True,
            env=env,
            executable=shutil.which("bash"),
        )

        with open(stderr_file.name) as f:
            stderr = f.read().strip()

    stdout = proc.stdout.strip()

    if stdout == "":
        raise RuntimeError(f"No stdout found from process. stderr: {stderr}")

    try:
        data = json.loads(stdout)
    except json.decoder.JSONDecodeError as e:
        raise RuntimeError(f"Could not decode JSON: {e}\n{stdout}")

    result_directory.mkdir(exist_ok=True, parents=True)
    data["test_run_id"] = f"{result_directory.name}-{run_at}-{slug}"
    data["run_at"] = run_at
    data["test_suite_id"] = result_directory.name
    i = 0
    while True:
        output_path = result_directory / f"{executor}_{i}.json"
        if not output_path.exists():
            break
        i += 1

    flush_print(f"Writing to {output_path}")
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

    if proc.returncode != 0 and not stderr.endswith("free(): invalid pointer"):
        raise RuntimeError(f"Process failed: stdout:\n{proc.stdout}\nstderr:{stderr}")

    # Prints from C++ don't get captured by Python at all, so check them after the
    # fact to see if cli.py ran any native code that printed warnings we don't
    # want to see
    for warning in BAD_WARNINGS:
        if warning in stdout:
            raise RuntimeError(f"Found {warning} in stdout:\n{stdout}")
        if warning in stderr:
            raise RuntimeError(f"Found {warning} in stderr:\n{stderr}")


def regex_for_unsupported_ops(ops: List[str]):
    """
    The list of unsupported ops can be in any order
    """
    match_any_op = "|".join(ops)
    match_any_op = f"({match_any_op})"
    op_str = ", ".join([match_any_op for i in range(len(ops))])
    return re.compile(
        f"tvm.error.OpNotImplemented: The following operators are not supported for frontend ONNX: {op_str}"
    )


encoder_ops = regex_for_unsupported_ops(ops=["Range", "Log", "Abs", "Greater", "Less", "Min"])
decoder_ops = regex_for_unsupported_ops(
    ops=["Log", "Max", "LessOrEqual", "Range", "Less", "Min", "Neg"]
)
pt_decoder_ops = regex_for_unsupported_ops(
    ops=["Less", "Min", "Range", "Neg", "Log", "Identity", "LessOrEqual"]
)
gptj_ops = regex_for_unsupported_ops(ops=["Einsum", "Cos", "Sin", "Range", "Neg"])
dynamic_shape = "AttributeError: <class 'tvm.tir.expr.Var'> has no attribute value"
missing_weights = re.compile(r"No such file or directory: '.*/weights.pb'")
missing_trt = "'TensorrtExecutionProvider' is not in available provider"
cuda_initialization = "CUDA initialization failure with error: 35"
missing_shapes_in_models_yaml = "Unspecified dynamic shapes detected"
cutlass_offload_failure = "KeyError: <MathOperation.multiply_add"

EXPECTED_FAILURES = {
    # Missing features in relax
    "oss-onnx.t5-decoder-with-lm-head-12v1-relax-cuda": dynamic_shape,
    "oss-onnx.t5-decoder-with-lm-head-12v1-relax-native": dynamic_shape,
    "oss-onnx.t5-encoder-12v1-relax-cuda": dynamic_shape,
    "oss-onnx.t5-encoder-12v1-relax-native": dynamic_shape,
}


def matches(pattern: Union[str, re.Pattern], text: str) -> bool:
    if isinstance(pattern, str):
        return pattern in text

    return pattern.search(text) is not None


@parameterize_configs(MODELS)
def test_offload_coverage(request, show_test_name, result_directory, run_config: Dict[str, Any]):
    """ """


@parameterize_configs(MODELS)
def test_mean_runtime(request, show_test_name, result_directory, run_config: Dict[str, Any]):
    """
    Tests end to end mean/p95 runtime for models on available backends
    """
    if run_config["executor"] == "relax-native":
        pytest.skip("relax-native results are slow and not needed")

    cuda_arg = request.config.getoption("--cuda-sm")
    if cuda_arg:
        run_config["cuda-sm"] = cuda_arg

    slug = f"{run_config['set']}.{run_config['name']}@{run_config['version']}"
    pyslug = pytest_slug(run_config)
    xfail_regex = EXPECTED_FAILURES.get(pyslug)

    # The entire test runner is wrapped in this try..except to implement some
    # custom behavior around xfailing, namely that the xfail happens with a
    # specific message in the error output.
    try:
        _test_impl(request, slug=slug, result_directory=result_directory, run_config=run_config)
    except Exception as e:
        if xfail_regex is not None:
            # This test should xfail, check if the error matches
            if not matches(xfail_regex, str(e)):
                # The error does not match, don't xfail and raise a normal exceptino
                raise RuntimeError(
                    f"Test {pyslug} is in EXPECTED_FAILURES but the expected error regex {xfail_regex} was not found in {str(e)}"
                )
            else:
                # The test failed and the message matches, xfail
                pytest.xfail(reason=f"{pyslug} is in EXPECTED_FAILURES")

        raise e

    # The test passed, but if the slug is in the xfail list this should be an
    # error (to mimic pytest.mark.xfail(strict=True))
    if xfail_regex is not None:
        raise RuntimeError(f"Expected test {pyslug} to fail but it passed")


if __name__ != "__main__":
    # Running under pytest
    for config in MODELS:
        pass
        # print(
        #     f"Running {config.config.id()} on {config.config.flow_config} ({config.warmup_runs} warmups, {config.test_runs} runs)"
        # )
