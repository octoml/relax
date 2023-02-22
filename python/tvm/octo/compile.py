# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, wrong-import-position
"""Simplified interface for TVM Unity Flow."""

import tvm
from tvm import relax
from typing import Union, Optional, Dict, List
from pathlib import Path
import onnx


def load_onnx_model(
    model_file: Union[str, Path, onnx.ModelProto], shape_dict: Optional[Dict[str, List]] = None
) -> tvm.IRModule:
    """Convert an input onnx model into a relax module.

    Parameters
    ----------
    model_file : Union[str, Path, onnx.ModelProto]
        An input onnx model to convert. Can either be a path to a model or an already
        loaded onnx protobuf.

    shape_dict : Optional[Dict[str, List]]
        An optional dictionary that maps inputs to specific shapes. If not provided,
        the default values in the onnx graph will be used.

    Returns
    -------
    relax_mod : tvm.IRModule
        A Relax module implementing the input onnx graph.
    """
    # Check input format and load if needed.
    if not isinstance(model_file, onnx.ModelProto):
        model_file = onnx.load(model_file)

    # Convert the graph into a relax implementation.
    relax_mod = relax.from_onnx(model_file)

    return relax_mod


def offload_cutlass(mod: tvm.IRModule, target: tvm.target.Target) -> tvm.IRModule:
    """Converts appropriate subgraphs to CUTLASS

    Parameters
    ----------
    mod : tvm.IRModule
        The input module that should have subgraphs rewritten to CUTLASS.
    target : tvm.target.Target
        The target used for compilation. Needed to parameterize CUTLASS.

    Returns
    -------
    cutlass_mod : tvm.IRModule
        The input graph with all possible subgraphs rewritten.
    """
    # Extract the sm version of the current target.
    assert target.arch is not "", "Target architecture must be specified."
    sm = int(target.arch.split("_")[1])

    # Construct CUTLASS codegen pass.
    cutlass_codegen_pass = relax.transform.RunCodegen(
        {"cutlass": {"sm": sm, "find_first_valid": True}}
    )

    # Construct pattern identification pass.
    # TODO(jwfromm) rebase on cutlass pattern language

    # Run passes on input module.
    seq = tvm.transform.Sequential(
        [
            # relax.transform.FuseOpPattern(patterns, annotate_codegen=True),
            cutlass_codegen_pass
        ]
    )

    return seq(mod)
