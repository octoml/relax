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
from typing import Union, Optional, Dict, List, Tuple
import onnx
import onnx_graphsurgeon as gs
import tvm
from tvm import relax
from tvm.relax.frontend.onnx import from_onnx
from tvm.relax.backend.contrib.cutlass import partition_for_cutlass
from .utils import get_cuda_target, get_llvm_target
from .octo_model import OctoModel
from .schedule_cumsum import ScheduleCumsum


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

    # Make sure nodes are topologically sorted.
    sorted_graph = gs.import_onnx(model_file)
    sorted_graph.toposort()
    model_file = gs.export_onnx(sorted_graph)

    # Convert the graph into a relax implementation.
    relax_mod = from_onnx(model_file, shape_dict=shape_dict)

    return relax_mod


def offload_cutlass(mod: tvm.IRModule, target: tvm.target.Target) -> Tuple[tvm.IRModule, Dict]:
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
    schedule_map : Dict[str, List[Tuple[int, str]]]
        A dictionary mapping framework op names to a list of tuples. Each tuple
        contains a relax op id and the schedule that was applied to it.
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

    # Construct a mapping framework op -> (relax op id, schedule_method)
    schedule_map = construct_schedule_map(mod)

    # Construct CUTLASS codegen pass.
    cutlass_codegen_pass = relax.transform.RunCodegen(
        {"cutlass": {"sm": sm, "find_first_valid": True}}
    )

    # Generate code for matched cutlass kernels.
    mod = cutlass_codegen_pass(mod)

    return mod, schedule_map


def construct_schedule_map(mod: tvm.IRModule) -> Dict[str, List[Tuple[int, str]]]:
    """Constructs a mapping from framework op names to a corresponding list of
    relax ops and how they were scheduled.

    Parameters
    ----------
    mod : tvm.IRModule
        The input IRModule

    Returns
    -------
    schedule_map : Dict[str, List[Tuple[int, str]]]
        A dictionary mapping framework op names to a list of tuples. Each tuple
        contains a relax op id and the schedule that was applied to it.
    """

    def maybe_decode_multiple_spans(span: tvm.ir.Span) -> List[tvm.ir.Span]:
        """Checks if multiple spans are encoded into one and (maybe) decodes them.
        Multiple spans can be merged and encoded during op fusion.
        Must be in sync with fuse_ops::CreateSpanForCallSite which does the encoding.
        """

        if span is None:
            return []
        source = str(span.source_name.name)
        # If the source does not start with 0xMULSPAN! then this is a single span
        if not source.startswith("0xMULSPAN!"):
            return [(span, False)]
        # Skip the 0xMULSPAN! prefix
        source = source[10:]
        spans = []
        # Split source by ;
        for encoded_span in source.split(";"):
            if encoded_span == "":
                continue
            # Extract converter name and index. The substring before the first [
            converter_name_index = encoded_span[: encoded_span.index("[")]
            # The substring until the first ( is the converter name
            converter_name = converter_name_index[: converter_name_index.index("(")]
            converter_index = int(converter_name_index[converter_name_index.index("(") + 1 : -1])

            ops = encoded_span[encoded_span.index("[") + 1 : encoded_span.index("]")]
            # Split ops by ,
            for op in ops.split(","):
                # Extract the op name.
                op_name = op[: op.index("(")]
                # Extract the op index.
                op_index = int(op[op.index("(") + 1 : op.index(")")])

                # Create a new span and add it to the list.
                source_string = converter_name + ";" + op_name
                spans.append(
                    (
                        tvm.ir.Span(
                            tvm.ir.SourceName(source_string), converter_index, 0, op_index, 0
                        ),
                        True,
                    )
                )

        return spans

    def get_offload_type(mod: tvm.IRModule, op: relax.Call) -> str:
        """Checks if the given relax op is offloaded to a framework.
        Returns "native" if not.
        """
        call_tir_op = tvm.relay.op.get("relax.call_tir")
        assert op.op != call_tir_op, "call_tir should not be in the IRModule at this point."
        assert isinstance(
            op.op, (tvm.ir.GlobalVar, tvm.ir.Op)
        ), "Expecting op to be an Op or GlobalVar."

        target_name = get_op_name(op)
        function_names = [var.name_hint for var in mod.get_global_vars()]

        if target_name in function_names:
            func = mod[target_name]
            if isinstance(func, relax.Function) and "Codegen" in func.attrs.keys():
                return func.attrs["Codegen"]

        return "native"

    def get_op_name(call: relax.Call) -> str:
        """Returns the name of target of this call."""
        if isinstance(call.op, tvm.ir.Op):
            return call.op.name
        elif isinstance(call.op, tvm.ir.GlobalVar):
            return call.op.name_hint
        else:
            raise ValueError("Expecting call.op to be an Op or GlobalVar.")

    framework_op_to_relax_op = {}

    @relax.expr_functor.visitor
    class Visitor(tvm.relax.PyExprVisitor):
        """Visitor that populates the framework_op_to_relax_op dict."""

        def visit_call_(self, call: relax.Call):  # pylint: disable=arguments-differ
            if call.span is None:
                print("Warning: expecting a span to be attached to call node {}.".format(call))

            offload_type = get_offload_type(mod, call)
            spans = maybe_decode_multiple_spans(call.span)

            for span, from_multi_span in spans:
                if not from_multi_span:
                    # This is not a fused span. Get the op name from the call node.
                    op_name = get_op_name(call)
                    converter_name = str(span.source_name.name)
                else:
                    converter_name, op_name = str(span.source_name.name).split(";")

                converter_name_and_index = converter_name + "(" + str(span.line) + ")"
                if converter_name_and_index not in framework_op_to_relax_op.keys():
                    framework_op_to_relax_op[converter_name_and_index] = []

                op_and_index = op_name + "(" + str(span.column) + ")"
                # Column is the index of the relax op in the converter.
                framework_op_to_relax_op[converter_name_and_index].append(
                    (op_and_index, offload_type)
                )

            super().visit_call_(call)

    visitor = Visitor()
    main_func = mod["main"]
    if isinstance(main_func, relax.Function):
        visitor.visit_expr(main_func)

    return framework_op_to_relax_op


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
    if isinstance(target, str):
        target = tvm.target.Target(target)

    # Convert model into a relax module.
    relax_mod = load_onnx_model(model, shape_dict)

    # Extract information about input shapes and types so we can
    # randomly generate them later if needed.
    input_info = {}
    for inp in relax_mod["main"].params:
        input_shape = [i.value for i in inp.struct_info.shape]
        input_dtype = inp.struct_info.dtype
        input_info[inp.name_hint] = (input_shape, input_dtype)

    schedule_map = {}
    # If target is gpu and compiled with Cutlass, offload where possible.
    if target.kind.name == "cuda":
        # Schedule any cumsum operators if needed. We need to do this explicitly
        # to make it work with thrust.
        with target:
            relax_mod = ScheduleCumsum()(relax_mod)
        if tvm.get_global_func("relax.ext.cutlass", True):
            # Match subgraphs that can be offloaded to cutlass and offload them.
            relax_mod, schedule_map = offload_cutlass(relax_mod, target)
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
    return OctoModel(exe, input_info, target=target, schedule_map=schedule_map)
