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
import random
import string
import subprocess
import sys
import time
import os
import argparse
import collections
import functools

from typing import *
from pathlib import Path
from dataclasses import dataclass

import psycopg2
import commentjson
import jsonschema
import onnx
import pytest

from cloud_utils import aws_download, IS_IN_CI

import numpy as np
from google.cloud import bigquery
from google.oauth2 import service_account
from onnx import hub
from tvm import relax


REPO_ROOT = Path(__file__).parent.parent.parent
FROM_HUB = object()
ONNX_REPO = "onnx/models"
ONNX_REPO_SHA = "8e893eb39b131f6d3970be6ebd525327d3df34ea"
MODELS_DIR = REPO_ROOT / "model-data"


class ImportError:
    FAILED_ONNX_IMPORT = "failed_onnx_import"
    FAILED_RELAX_BUILD = "failed_relax_build"
    FAILED_EXECUTION = "failed_execution"
    FAILED_OCTO_COMPILE = "failed_octo_compile"


def eprint(*args):
    print(*args, file=sys.stderr, flush=True)


@dataclass
class ModelConfig:
    set: str
    name: str
    sha256: str
    version: str
    flow_config: str
    requires_toposort: bool
    input_scale: float
    tuning_steps: Optional[int]
    shapes: Optional[Dict[str, List[int]]]
    dtypes: Optional[Dict[str, str]]
    files: Optional[List[str]]

    @staticmethod
    def from_json(raw: str) -> "ModelConfig":
        data = json.loads(raw)
        return ModelConfig(
            set=data["set"],
            name=data["name"],
            sha256=data["sha256"],
            version=data["version"],
            flow_config=data["flow_config"],
            requires_toposort=data["requires_toposort"],
            input_scale=data["input_scale"],
            shapes=data["shapes"],
            dtypes=data["dtypes"],
        )

    def id(self) -> str:
        return f"{self.set}-{self.name}"

    def model_dir(self) -> Path:
        if self.set == "onnx-hub":
            return FROM_HUB
        return MODELS_DIR / self.set / f"{self.name}@{self.version}"

    def file(self) -> Path:
        if self.set == "onnx-hub":
            return FROM_HUB
        return self.model_dir() / "model.onnx"

    def load_model(self, verify_sha256: bool = True) -> onnx.ModelProto:
        path = self.file()
        s3_prefix = f"{self.set}/{self.name}@{self.version}"
        if path == FROM_HUB:
            repo = f"{ONNX_REPO}:{ONNX_REPO_SHA}"
            model = hub.load(self.name, repo=repo, silent=True)
            if verify_sha256:
                eprint(f"Skipping verification for {path} since it was loaded from ONNX Hub")
            return model
        else:
            if not path.exists():
                eprint(f"Model file at {path} not found, trying to download from S3 storage...")
                (MODELS_DIR / self.set).mkdir(exist_ok=True, parents=True)
                out_path = aws_download(
                    blob_name=f"{s3_prefix}/model.onnx",
                    out_path=path,
                )
                if not out_path.exists():
                    raise RuntimeError(f"Model file at {path} not found, has it been downloaded?")
            if verify_sha256:
                actual_sha256 = sha256sum(path)
                if actual_sha256 != self.sha256:
                    raise RuntimeError(
                        f"Model's sha256 ({actual_sha256}) did not match expected sha256 ({self.sha256})"
                    )

        eprint(f"Loading model at {path}")
        model = onnx.load(path, load_external_data=False)
        for external_file in self.files:
            external_file_path = self.model_dir() / external_file
            if not external_file_path.exists():
                eprint(f"{external_file_path} does not exist, downloading from S3...")
                aws_download(
                    blob_name=f"{s3_prefix}/{external_file}",
                    out_path=external_file_path,
                )
        onnx.load_external_data_for_model(model, self.model_dir())

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
    cuda_sm: int

    @staticmethod
    def from_json(raw: str) -> "BenchmarkConfig":
        data = json.loads(raw)
        return BenchmarkConfig(
            config=ModelConfig.from_json(data["config"]),
            warmup_runs=data["warmup_runs"],
            test_runs=data["test_runs"],
            check_accuracy=data["check_accuracy"],
            atol=data["atol"],
            rtol=data["rtol"],
            cuda_sm=data["cuda_sm"],
        )

    def __str__(self):
        return f"{self.config.set}.{self.config.name}.{self.config.flow_config}"


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


def infer_shapes(
    model: "onnx.ModelProto", axes: Optional[Dict[str, int]] = None
) -> Dict[str, List[int]]:
    # N.B. Defer the import so as not to unconditionally require other runtimes.
    from tvm import relay
    from tvm.tir import Any as Any

    input_shapes = {}
    if axes is None:
        axes = {}
    initializer_names = [n.name for n in model.graph.initializer]
    # The inputs contains both the inputs and parameters. We are just interested in the
    # inputs so skip all parameters listed in graph.initializer
    unspecified_dynamic_axes = []
    for input_info in model.graph.input:
        if input_info.name not in initializer_names:
            name, shape, dtype, axis_names = relay.frontend.onnx.get_info(input_info)

            # Normalize the shape dimensions to integers
            assert isinstance(input_shapes, dict)
            new_shape = []
            for value, axis_name in zip(shape, axis_names):
                if isinstance(value, Any):
                    lookup_value = axes.get(axis_name)
                    if lookup_value is None:
                        unspecified_dynamic_axes.append((axis_name, name))
                        value = -1
                    else:
                        value = lookup_value
                else:
                    value = int(value)

                new_shape.append(value)
            input_shapes.update({input_info.name: new_shape})

    if len(unspecified_dynamic_axes) > 0:
        axes_to_inputs = collections.defaultdict(list)
        for axis_name, input_name in unspecified_dynamic_axes:
            axes_to_inputs[axis_name].append(input_name)

        msg = "\n".join(
            [
                f"    {axis_name} on {', '.join(input_names)}"
                for axis_name, input_names in axes_to_inputs.items()
            ]
        )
        raise RuntimeError(
            f"Unspecified dynamic shapes detected, shapes must be manually specified or an $axis entry provided:\n{msg}"
        )
    return input_shapes


def infer_dtypes(model: "onnx.ModelProto") -> Dict[str, str]:
    # N.B. Defer the import so as not to unconditionally require other runtimes.
    from tvm import relay
    from tvm.tir import Any as Any

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

            input_dtypes.update({input_info.name: dtype})

    return input_dtypes


class Timer(object):
    def __enter__(self):
        self.start = time.perf_counter_ns()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter_ns()
        self.ms_duration = (self.end - self.start) / 1000 / 1000


def extract_framework_ops(model: onnx.ModelProto) -> List[Dict[str, str]]:
    return []
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


class BaseRunner:
    benchmark_config: BenchmarkConfig

    def __init__(self, benchmark_config: BenchmarkConfig):
        self.benchmark_config = benchmark_config

        self._model = self.benchmark_config.config.load_model(
            verify_sha256=benchmark_config.config.sha256 is not None
        )

    def metadata(self):
        raise NotImplementedError

    def run(self, *args, **kwargs):
        raise NotImplementedError

    def load_model(self) -> "onnx.ModelProto":
        return self._model

    def run_onnx_cpu_inference(self, inputs: Dict[str, "np.ndarray"]) -> List["np.ndarray"]:
        import onnxruntime as ort

        sess_opt = ort.SessionOptions()

        # Set up an onnx inference on GPU
        sess = ort.InferenceSession(
            self._model.SerializeToString(),
            sess_options=sess_opt,
            providers=["CPUExecutionProvider"],
        )
        output_names = []
        output = sess.run(output_names, inputs)
        return output

    def generate_inputs(self, n: int) -> List[Dict[str, np.ndarray]]:
        all_inputs = []

        inferred_dtypes = None

        if self.benchmark_config.config.shapes is None:
            shapes = infer_shapes(self._model)
        else:
            axes = self.benchmark_config.config.shapes.get("$axes")
            if len(self.benchmark_config.config.shapes) == 1 and axes is not None:
                shapes = infer_shapes(self._model, axes=axes)
            else:
                shapes = self.benchmark_config.config.shapes

        if self.benchmark_config.config.dtypes is None:
            if inferred_dtypes is not None:
                dtypes = inferred_dtypes
            else:
                dtypes = infer_dtypes(self._model)
        else:
            dtypes = self.benchmark_config.config.dtypes

        for _ in range(n):
            input_names = list(shapes.keys())
            inputs = {}
            for name in input_names:
                inputs[name] = (
                    np.random.uniform(size=shapes[name]) * self.benchmark_config.config.input_scale
                ).astype(dtypes[name])

            all_inputs.append(inputs)

        return all_inputs
