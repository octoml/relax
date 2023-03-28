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

import os
import sys

# unused but needed to get CUDA working in onnx, too lazy to actually fix the
# issue
# https://stackoverflow.com/questions/75267445/why-does-onnxruntime-fail-to-create-cudaexecutionprovider-in-linuxubuntu-20/75267493#75267493
import torch

# Load the nightly ONNX version from its install directory
sys.path.insert(0, os.environ["ONNX_NIGHTLY_PATH"])
import onnxruntime as ort

from benchmarking_utils import BenchmarkConfig
from onnx_base import OnnxBase


class OnnxTrt(OnnxBase):
    def __init__(self, benchmark_config: BenchmarkConfig):
        super().__init__(
            benchmark_config=benchmark_config,
            ort=ort,
            providers=["CPUExecutionProvider"],
        )


Runner = OnnxTrt
