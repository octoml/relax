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
import tempfile
from pathlib import Path
from typing import Union, Optional, Dict, List
import onnx
import onnx_graphsurgeon as gs
import tvm
from tvm import relax, meta_schedule as ms
from tvm.relax.frontend.onnx import from_onnx
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass
from tvm.relax.transform.tuning_api import Trace
from .utils import get_cuda_target, get_llvm_target
from .octo_model import OctoModel
from .schedule_cumsum import ScheduleCumsum
from .inject_op_pattern import InjectOpPattern


def load_onnx_model(
    model_file: Union[str, Path, onnx.ModelProto],
    shape_dict: Optional[Dict[str, List]] = None,
    dtype_dict: Optional[Union[str, Dict[str, str]]] = "float32",
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

    dtype_str: Optional[Union[str, Dict[str, str]]]
        An optional string or dictionary that maps inputs to its specific data type.
        If not provided, the default type of "float32" will be used.

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

    # Make sure nodes are topologically sorted.
    sorted_graph = gs.import_onnx(model_file)
    sorted_graph.toposort()
    model_file = gs.export_onnx(sorted_graph)

    # Convert the graph into a relax implementation.
    relax_mod = from_onnx(model_file, shape_dict=shape_dict, dtype_dict=dtype_dict)

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
        {"cutlass": {"sm": sm, "find_first_valid": False}}
    )

    # Generate code for matched cutlass kernels.
    mod = cutlass_codegen_pass(mod)
    return mod


def default_schedule_func(
    primfunc: tvm.tir.PrimFunc, target: tvm.target.Target
) -> Union[tvm.tir.PrimFunc, None]:
    """Apply some basic optimizations on the PrimFunc.

    For example, for GPUs we apply AutoBind, AutoInline, CrossThreadReduction,
    InlineConstantScalars, ParallelizeVectorizeUnroll

    Parameters
    ----------
    primfunc : tvm.tir.PrimFunc
        The input primfunc.
    target : tvm.target.Target
        The target used for compilation. Needed to generate design space.

    Returns
    -------
    optimized_primfunc : Union[tvm.tir.PrimFunc, None]
        The primfunc is applying some basic optimizations by picking a
        random candidate from the design space. Returns None if the picked
        candidate fails compilation.

    """
    mod = tvm.IRModule({"main": primfunc})
    sch_rules = ms.ScheduleRule.create(target.kind.name)
    spaces = ms.TuneContext(
        mod=mod,
        target=target,
        space_generator=ms.space_generator.PostOrderApply(
            sch_rules=sch_rules,
            postprocs=[],
            mutator_probs={},
        ),
        task_name="main",
    ).generate_design_space()
    if len(spaces) == 0:
        print("No valid candidate for PrimFunc: ", primfunc.script())
        return None
    # randomly pick up the last candidate from design space
    # anecdotally the last candidate puts intermediate buffers in shared memory
    space = spaces[-1]
    mod = space.mod
    # verify that the valid candidate can be successfully compiled
    # should we run verify_gpu_code pass?
    # This only fails for some already scheduled cumsum operations in space_v5 but
    # leaving it here as a safety net.
    try:
        _ = relax.build(mod, target)
    except:
        print("Unable to build func ", primfunc.script())
        return None

    return mod["main"]


def compile(
    model: Union[str, Path, onnx.ModelProto],
    target: Optional[tvm.target.Target] = None,
    shape_dict: Optional[Dict[str, List]] = None,
    dtype_dict: Optional[Union[str, Dict[str, str]]] = "float32",
    tuning_steps: Optional[int] = None,
    work_dir: Optional[str] = None,
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

    type_dict : Optional[Union[str, Dict[str, str]]]
        An optional string or dictionary that maps inputs to its specific data type.
        If not provided, the default type of "float32" will be used.

    tuning_steps : Optional[int]
        The number of tuning trials to perform on the model. By default, no tuning will be done
        and kernels will intead either be offloaded or use a default gpu binding. Doing a small
        amount of tuning, however, can help accelerate certain models by quite a bit.

    work_dir : Optional[str]
        An optional directory where tuning logs will be saved. If not provided, a temporary
        directory will be used. This argument can be helpful for saving time when doing repeated
        runs with tuning.

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
    if isinstance(target, str):
        target = tvm.target.Target(target)

    # Convert model into a relax module.
    relax_mod = load_onnx_model(model, shape_dict=shape_dict, dtype_dict=dtype_dict)

    # Extract information about input shapes and types so we can
    # randomly generate them later if needed.
    input_info = {}
    for inp in relax_mod["main"].params:
        input_shape = [i.value for i in inp.struct_info.shape]
        input_dtype = inp.struct_info.dtype
        input_info[inp.name_hint] = (input_shape, input_dtype)

    # If target is gpu and compiled with Cutlass, offload where possible.
    if target.kind.name == "cuda":
        # Apply layout rewriting so that convolution can be properly offloaded.
        relax_mod = relax.transform.ConvertLayout({"relax.nn.conv2d": ["NHWC", "OHWI"]})(relax_mod)
        # Schedule any cumsum operators if needed. We need to do this explicitly
        # to make it work with thrust.
        with target:
            relax_mod = ScheduleCumsum()(relax_mod)
        if tvm.get_global_func("relax.ext.cutlass", True):
            # Match subgraphs that can be offloaded to cutlass and offload them.
            relax_mod = offload_cutlass(relax_mod, target)
        else:
            print("Cutlass backend not detected. Consider enabling it for better performance.")

    # Perform legalization to lower Relax operators.
    relax_mod = relax.transform.LegalizeOps()(relax_mod)
    relax_mod = relax.transform.FoldConstant()(relax_mod)
    relax_mod = relax.transform.AnnotateTIROpPattern()(relax_mod)
    # Manually note certain ops as fusable.
    relax_mod = InjectOpPattern()(relax_mod)
    relax_mod = relax.transform.FuseOps()(relax_mod)
    relax_mod = relax.transform.FuseTIR()(relax_mod)

    # If specified, perform tuning to optimize remaining workloads.
    if work_dir is None:
        work_dir = tempfile.mkdtemp()
    with target:
        if tuning_steps:
            with tvm.transform.PassContext(trace=Trace(relax_mod), opt_level=0):
                tuning_pass = relax.transform.MetaScheduleTuneIRMod(
                    params={},
                    work_dir=work_dir,
                    max_trials_global=tuning_steps,
                )
                relax_mod = tuning_pass(relax_mod)

        application_pass = relax.transform.MetaScheduleApplyDatabase(work_dir)
        relax_mod = application_pass(relax_mod)

    # Apply some basic scheduling optimizations to untuned kernels
    # This is a temporary workaround to get basic scheduling optimizations
    # (such as AutoInline, Unroll) without spending a lot of time tuning
    # computationally light kernels like softmax, reductions. Ideally MS API
    # should be updated to perform these optimizations without spending extensive
    # amounts of tuning for each kernel.
    for gv in relax_mod.get_global_vars():
        func = relax_mod[gv]
        if isinstance(func, tvm.tir.PrimFunc):
            if func.get_attr("tir.is_scheduled") == True:
                continue
            updated_func = default_schedule_func(func, target)
            if updated_func:
                relax_mod[gv] = updated_func.with_attrs({"tir.is_scheduled": True})

    # Finally, add thread binding to remaining kernels to allow them to run on gpu.
    if target.kind.name == "cuda":
        with target, tvm.transform.PassContext(opt_level=3):
            relax_mod = tvm.tir.transform.DefaultGPUSchedule()(relax_mod)

    # Compile the module.
    exe = relax.build(relax_mod, target)

    # Create an OctoModel from the compiled artifact.
    return OctoModel(exe, input_info, target=target)
