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
import sys
import os

from typing import TextIO, List, Dict, Any
from pathlib import Path

import commentjson
import jsonschema
import pytest


REPO_ROOT = Path(__file__).parent.parent
FROM_HUB = object()
ONNX_REPO = "onnx/models"
ONNX_REPO_SHA = "8e893eb39b131f6d3970be6ebd525327d3df34ea"
IS_IN_CI = os.getenv("IS_IN_CI", "0") == "1"


def pytest_slug(item: Dict[str, Any]):
    return f"{item['set']}.{item['name']}v{item['version']}-{item['executor']}"


def parameterize_configs(configs: List[Dict[str, Any]]):
    names = [f"{pytest_slug(c)}" for i, c in enumerate(configs)]
    return pytest.mark.parametrize("run_config", configs, ids=names)


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


def load_jsonl(jsonl_file: Path) -> List[Dict[str, Any]]:
    with open(jsonl_file) as f:
        data = [json.loads(line) for line in f.readlines()]

    return data


def gen_test_output_dir(base: Path) -> str:
    """
    Creates a 5 character id for the test used to store the result JSONs
    """

    if "TEST_SUITE_ID" in os.environ:
        # CI sets this in the docker build
        test_suite_id = os.environ["TEST_SUITE_ID"]
        print(f"Using TEST_SUITE_ID from env: {test_suite_id}")
        return base / test_suite_id

    for _ in range(1000):
        test_suite_id = "".join([random.choice(string.ascii_lowercase) for _ in range(5)])
        test_output_dir = base / test_suite_id
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
                print(f"while loading {path}:", file=sys.stderr, flush=True)
                raise e

        try:
            jsonschema.validate(instance=data, schema=schema)
        except jsonschema.SchemaError as e:
            print(f"while validating {path}:", file=sys.stderr, flush=True)
            raise e
        output.write(json.dumps({"r": json.dumps(data)}))
        output.write("\n")
