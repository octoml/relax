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
# pylint: disable=invalid-name, wrong-import-position, redefined-builtin, not-callable
"""Simplified interface for TVM Unity Flow."""
from pathlib import Path
from typing import Union, Optional, Dict, List
import onnx
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass
from .utils import get_cuda_target, get_llvm_target
from .octo_model import OctoModel


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
    if isinstance(model_file, (Path, str)):
        model_file = onnx.load(model_file)
    else:
        assert isinstance(
            model_file, onnx.ModelProto
        ), f"model_file must be one of (str, Path, onnx.ModelProto) but got {type(model_file)})"

    # Convert the graph into a relax implementation.
    relax_mod = from_onnx(model_file, shape_dict=shape_dict)

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
        The input module after the partition_for_cutlass and RunCodegen passes
        are applied. In the first step, subgraphs that cutlass supports are
        found and annotated. Next, those subgraphs are compiled using nvcc.
        The result is a graph containing a mixture of relax operators
        and external calls to the compiled cutlass kernels.
    """
    # Extract the sm version of the current target.
    assert target.arch, "Target architecture must be specified."
    sm = int(target.arch.split("_")[1])
    # Cutlass only has support up to sm80, future sms will work with
    # earlier kernels though.
    if sm > 80:
        sm = 80

    # Apply partitioning to offload patterns to cutlass.
    mod = partition_for_cutlass(mod)

    # Construct CUTLASS codegen pass.
    cutlass_codegen_pass = relax.transform.RunCodegen(
        {"cutlass": {"sm": sm, "find_first_valid": True}}
    )

    # Generate code for matched cutlass kernels.
    mod = cutlass_codegen_pass(mod)
    return mod


def compile(
    model: Union[str, Path, onnx.ModelProto],
    target: Optional[tvm.target.Target] = None,
    shape_dict: Optional[Dict[str, List]] = None,
):
    """Entrypoint to compiling a model using the Unity Flow.

    Parameters
    ----------
    model : Union[str, Path, onnx.ModelProto]
        An input onnx model to convert. Can either be a path to a model or an already
        loaded onnx protobuf.

    target : Optional[tvm.target.Target]
        A description of the hardware to compile to. If not provided, one will be extracted for
        the current host machine.

    shape_dict : Optional[Dict[str, List]]
        An optional dictionary that maps inputs to specific shapes. If not provided,
        the default values in the onnx graph will be used.

    Returns
    -------
    octo_model: OctoModel
        A convenience wrapper around the compiled model that provides utility functions.
    """
    # Determine current target.
    if target is None:
        # Check if this is gpu enabled.
        if tvm.cuda(0).exist:
            target = get_cuda_target()
        else:
            target = get_llvm_target()
        print(f"Auto-selected target {target}")

    # Convert model into a relax module.
    relax_mod = load_onnx_model(model, shape_dict)

    # Perform constant folding where possible.
    c_mod = relax.transform.FoldConstant()(relax_mod)
    breakpoint()

    # Extract information about input shapes and types so we can
    # randomly generate them later if needed.
    input_info = {}
    for inp in relax_mod["main"].params:
        input_shape = [i.value for i in inp.struct_info.shape]
        input_dtype = inp.struct_info.dtype
        input_info[inp.name_hint] = (input_shape, input_dtype)

    # If target is gpu and compiled with Cutlass, offload where possible.
    if target.kind.name == "cuda":
        if tvm.get_global_func("relax.ext.cutlass", True):
            # Match subgraphs that can be offloaded to cutlass and offload them.
            relax_mod = offload_cutlass(relax_mod, target)
        else:
            print("Cutlass backend not detected. Consider enabling it for better performance.")

    # Perform legalization to lower Relax operators.
    relax_mod = relax.transform.LegalizeOps()(relax_mod)

    # Schedule all remaining functions to be compatible with gpu if needed.
    if target.kind.name == "cuda":
        with target, tvm.transform.PassContext(opt_level=3):
            relax_mod = tvm.tir.transform.DefaultGPUSchedule()(relax_mod)

    # Compile the module.
    exe = relax.build(relax_mod, target)

    # Create an OctoModel from the compiled artifact.
    return OctoModel(exe, input_info, target=target)
