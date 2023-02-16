import json
import random
import string
import subprocess
import sys
import time
import os

from typing import *
from pathlib import Path
from dataclasses import dataclass

import psycopg2
import commentjson
import jsonschema
import onnx
import pytest

from google.cloud import bigquery
from google.oauth2 import service_account
from onnx import hub
from tvm import relax


REPO_ROOT = Path(__file__).parent.parent
FROM_HUB = object()
ONNX_REPO = "onnx/models"
ONNX_REPO_SHA = "8e893eb39b131f6d3970be6ebd525327d3df34ea"
IS_IN_CI = False
MODELS_DIR = REPO_ROOT / "model-data"


class ImportError:
    FAILED_ONNX_IMPORT = "failed_onnx_import"
    FAILED_RELAX_BUILD = "failed_relax_build"
    FAILED_EXECUTION = "failed_execution"


@dataclass
class ModelConfig:
    set: str
    name: str
    sha256: str
    flow_config: str
    requires_toposort: bool
    input_scale: float
    shapes: Optional[Dict[str, List[int]]]
    dtypes: Optional[Dict[str, str]]

    def id(self) -> str:
        return f"{self.set}-{self.name}"

    def file(self) -> Path:
        if self.set == "onnx-opensource":
            return FROM_HUB
        return MODELS_DIR / self.set / f"{self.name}.onnx"

    def load_model(self, verify_sha256: bool = True) -> onnx.ModelProto:
        path = self.file()
        if path == FROM_HUB:
            repo = f"{ONNX_REPO}:{ONNX_REPO_SHA}"
            model = hub.load(self.name, repo=repo, silent=True)
            if verify_sha256:
                print(
                    f"Skipping verification for {path} since it was loaded from ONNX Hub"
                )
        else:
            if not path.exists():
                print(
                    f"Model file at {path} not found, trying to download from Azure storage..."
                )
                (MODELS_DIR / self.set).mkdir(exist_ok=True, parents=True)
                azure_download(blob_name=path.name, out_path=MODELS_DIR / self.set)
                raise RuntimeError()
                # raise RuntimeError(
                #     f"Model file at {path} not found, has it been downloaded?"
                # )
            if verify_sha256:
                actual_sha256 = sha256sum(path)
                if actual_sha256 != self.sha256:
                    raise RuntimeError(
                        f"Model's sha256 ({actual_sha256}) did not match expected sha256 ({self.sha256})"
                    )

        if IS_IN_CI:
            raise NotImplementedError(
                f"Cannot load {self.set}: {self.name} from Azure Storage yet"
            )
        else:
            model = onnx.load(path)

        if self.requires_toposort:
            import onnx_graphsurgeon as gs

            sorted = gs.import_onnx(model)
            sorted.toposort()
            model = gs.export_onnx(sorted)

        return model


@dataclass
class BenchmarkConfig:
    config: ModelConfig
    warmup_runs: int
    test_runs: int
    check_accuracy: bool
    atol: float
    rtol: float

    def __str__(self):
        return f"{self.config.set}.{self.config.name}.{self.config.flow_config}"


def parameterize_configs(configs: List[BenchmarkConfig]):
    names = [str(c) for c in configs]
    return pytest.mark.parametrize("run_config", configs, ids=names)


def sha256sum(model_file_name: str):
    proc = subprocess.run(
        ["sha256sum", model_file_name],
        stdout=subprocess.PIPE,
        check=True,
        encoding="utf-8",
    )
    return proc.stdout.strip().split()[0]


def git_info() -> Tuple[str, str]:
    """
    Determine the git branch and sha
    """
    proc = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        stdout=subprocess.PIPE,
        check=True,
        encoding="utf-8",
    )
    branch = proc.stdout.strip()
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        stdout=subprocess.PIPE,
        check=True,
        encoding="utf-8",
    )
    sha = proc.stdout.strip()
    return branch, sha


_manifest = None


def manifest():
    global _manifest
    if _manifest is None:
        with open(Path(__file__).resolve().parent / "ONNX_HUB_MANIFEST.json") as f:
            _manifest = json.load(f)

    return _manifest


def find_model(name):
    for model in manifest():
        if model["model_path"] == name:
            return model
    raise ValueError(f"{name} not found")


def infer_inputs(model):
    """
    Infers this model's input shapes and input dtypes.

    :return: input shapes and input dtypes of this Hulk ONNX Model.
    :raises: ONNXInferInputsUnknownDataType when dtype is unknown.
    """
    # N.B. Defer the import so as not to unconditionally require other runtimes.
    from tvm import relay
    from tvm.tir import Any as Any

    input_shapes = {}
    input_dtypes = {}
    initializer_names = [n.name for n in model.graph.initializer]
    # The inputs contains both the inputs and parameters. We are just interested in the
    # inputs so skip all parameters listed in graph.initializer
    for input_info in model.graph.input:
        if input_info.name not in initializer_names:
            name, shape, dtype, axis_names = relay.frontend.onnx.get_info(input_info)
            if dtype is None:
                raise RuntimeError(
                    f"Unknown dtype on input '{input_info.name}' is not supported. inputs: '{input_info.name}'",
                )

            # Normalize the shape dimensions to integers
            assert isinstance(input_shapes, dict)
            new_shape = []
            for value, name in zip(shape, axis_names):
                # print("Unspecified shape, assuming 1", value)
                value = int(value) if not isinstance(value, Any) else 1
                # new_shape[name] = value
                new_shape.append(value)
            input_shapes.update({input_info.name: new_shape})
            input_dtypes.update({input_info.name: dtype})

    return input_shapes, input_dtypes


class Timer(object):
    def __enter__(self):
        self.start = time.perf_counter_ns()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter_ns()
        self.ms_duration = (self.end - self.start) / 1000 / 1000


_bigquery_client_and_config = None


def bigquery_client_and_config(
    key_path: str = "gcp_auth.json", schema: Optional[List[bigquery.SchemaField]] = None
):
    if not Path(key_path).exists():
        raise RuntimeError(f"{key_path} was not found, did you forget to mount it?")

    global _bigquery_client_and_config
    if _bigquery_client_and_config is None:
        credentials = service_account.Credentials.from_service_account_file(
            key_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"],
        )
        client = bigquery.Client(
            credentials=credentials,
            project=credentials.project_id,
        )

        if schema is None:
            schema = [
                bigquery.SchemaField("r", "STRING", mode="REQUIRED"),
            ]

        job_config = bigquery.LoadJobConfig(
            schema=schema,
        )
        job_config.source_format = bigquery.SourceFormat.NEWLINE_DELIMITED_JSON
        job_config.autodetect = True
        _bigquery_client_and_config = (client, job_config)

    return _bigquery_client_and_config


def bigquery_upload(jsonl_file: Path, dataset_id: str, table_id: str) -> int:
    client, job_config = bigquery_client_and_config()
    dataset_ref = client.dataset(dataset_id)
    table_ref = dataset_ref.table(table_id)
    with open(jsonl_file, "rb") as source_file:
        job = client.load_table_from_file(
            source_file,
            table_ref,
            location="us-west1",  # Must match the destination dataset location.
            job_config=job_config,
        )  # API request

    job.result()

    return job


def load_jsonl(jsonl_file: Path) -> List[Dict[str, Any]]:
    with open(jsonl_file) as f:
        data = [json.loads(line) for line in f.readlines()]

    return data


def postgres_upload(jsonl_file: Path, database: str, table_name: str) -> int:
    """
    Uploads records in jsonl_file (one JSON document per line) to postgres
    """
    rows = [(json.dumps(d["r"]),) for d in load_jsonl(jsonl_file=jsonl_file)]
    sql = f"INSERT INTO {table_name} (r) VALUES (%s)"
    conn = None
    password = os.environ["POSTGRES_PASSWORD"]
    ip = os.environ["POSTGRES_IP"]
    user = os.getenv("POSTGRES_USER", "ci")
    try:
        conn = psycopg2.connect(
            host=ip,
            database=database,
            user=user,
            password=password,
        )
        cur = conn.cursor()
        cur.executemany(sql, rows)
        conn.commit()
        cur.close()
    finally:
        if conn is not None:
            conn.close()

    return len(rows)


def gen_test_output_dir(base: Path) -> str:
    """
    Creates a 5 character id for the test used to store the result JSONs
    """
    for _ in range(1000):
        test_run_id = "".join([random.choice(string.ascii_lowercase) for _ in range(5)])
        test_output_dir = base / test_run_id
        if not test_output_dir.exists():
            return test_output_dir

    raise RuntimeError("Unable to generate a unique ID for this test run")


def _load_and_strip_comments(f):
    return commentjson.loads(f.read())


def concat_test_results(schema_file: Path, test_results: List[Path], output: TextIO):
    with open(schema_file) as f:
        schema = _load_and_strip_comments(f)

    for path in test_results:
        with open(path, "r") as f:
            try:
                data = _load_and_strip_comments(f)
            except Exception as e:
                print(f"while loading {path}:", file=sys.stderr)
                raise e

        try:
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.SchemaError as e:
            print(f"while validating {path}:", file=sys.stderr)
            raise e
        output.write(json.dumps({"r": json.dumps(data)}))
        output.write("\n")


def azure_download(
    blob_name: str,
    out_path: Path,
    container_name: str = "models",
    account_name: str = "scorecardmodels",
    account_key: Optional[str] = None,
):
    if out_path.is_dir():
        out_path = out_path / blob_name
    if account_key is None:
        account_key = os.getenv("AZURE_ACCOUNT_KEY")
        if account_key is None:
            raise RuntimeError(
                "Environment variable 'AZURE_ACCOUNT_KEY' must be set to download from Azure. You can get one on Azure at Home > Storage accounts > scorecardmodels > Access keys"
            )

    command = [
        "az",
        "storage",
        "blob",
        "download",
        "--container-name",
        container_name,
        "--file",
        out_path,
        "--name",
        blob_name,
        "--account-name",
        account_name,
        "--account-key",
        account_key,
    ]
    command = [str(c) for c in command]
    print(f"+ {' '.join(command)}")
    subprocess.run(command, check=True)
    return out_path


def extract_framework_ops(model: onnx.ModelProto) -> List[Dict[str, str]]:
    return [{"name": node.name, "op_type": node.op_type} for node in model.graph.node]


def extract_relay_ops(
    model: onnx.ModelProto,
    framework_ops: List[Dict[str, str]],
    shapes: Dict[str, List[int]],
) -> List[str]:
    tvm_model = relax.from_onnx(model, shape=shapes)

    ops = []
    for item in tvm_model.functions.keys():
        ops.append(
            {
                "framework_op_index": -1,
                "name": item.name_hint,
                "schedule_method": "unknown",
            }
        )

    return ops


if __name__ == "__main__":
    azure_download(
        blob_name="antisemitism_detection_fp16.onnx", out_path=Path("/opt/models")
    )
    azure_download(
        blob_name="spacev5_norm_op13_fp16_optimized.randomized.onnx",
        out_path=Path("/opt/models"),
    )
    azure_download(blob_name="stable_diffusion.onnx", out_path=Path("/opt/models"))
    azure_download(
        blob_name="turing_text_large_fp16.onnx", out_path=Path("/opt/models")
    )
    azure_download(
        blob_name="turing_vision_large_fp32.onnx", out_path=Path("/opt/models")
    )
    azure_download(blob_name="turing_vortex_fp16.onnx", out_path=Path("/opt/models"))
    azure_download(blob_name="weights.pb", out_path=Path("/opt/models"))
