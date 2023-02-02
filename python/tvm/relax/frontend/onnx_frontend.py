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
# pylint: disable=invalid-name, import-self, len-as-condition, unused-argument, too-many-lines
# pylint: disable=import-outside-toplevel
"""ONNX: Open Neural Network Exchange frontend for Relax."""
import math
import warnings
from typing import Union, Optional

import numpy as _np

import tvm
from tvm import relax, topi, relay
from tvm.target import Target
from tvm.ir import IRModule
from tvm.relax import testing, PyExprMutator
from tvm._ffi import base as _base
from tvm.runtime import ndarray as _nd
from tvm.relay.expr import TupleWrapper, Var, GlobalVar
from tvm.relay.frontend.onnx import OnnxOpConverter as RelayOnnxOpConverter


def new_var(var_name, shape, dtype="float32"):
    return testing.nn.Parameter(shape=shape, dtype=dtype, name=var_name)


def get_type(elem_type):
    """Converts onnx integer datatype to numpy datatype"""
    # If a string was passed instead of a tensor type, it does not need
    # conversion and can be returned.
    if isinstance(elem_type, str):
        return elem_type

    try:
        from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE
    except ImportError as e:
        raise ImportError("Unable to import onnx which is required {}".format(e))

    try:
        from onnx import TensorProto
    except ImportError as e:
        raise ImportError("Unable to import TensorProto from onnx {}".format(e))

    return str(TENSOR_TYPE_TO_NP_TYPE[elem_type])


def get_info(info_proto):
    """Extract the shape from a ValueInfoProto."""
    shape = []
    shape_name = []
    for dim in info_proto.type.tensor_type.shape.dim:
        name = dim.dim_param
        value = dim.dim_value
        if value is None or value == 0:
            value = tvm.tir.Var("d", "int64")
            shape_name.append(name)
        else:
            shape_name.append(value)
        shape.append(value)

    name = info_proto.name
    if info_proto.type.tensor_type.elem_type:
        dtype = get_type(info_proto.type.tensor_type.elem_type)
    else:
        dtype = None
    return name, shape, dtype, shape_name


def get_numpy(tensor_proto):
    """Grab data in TensorProto and convert to numpy array."""
    try:
        from onnx.numpy_helper import to_array
    except ImportError as e:
        raise ImportError("Unable to import onnx which is required {}".format(e))
    return to_array(tensor_proto)


class onnx_input(list):
    """A helper extension to list that returns None for out of bound indices."""

    def __getitem__(self, item):
        if isinstance(item, slice):
            if item.stop is None:
                stop = len(self)
            else:
                stop = item.stop
            indices = list(range(stop)[item])
            return [self[i] for i in indices]
        if isinstance(item, int):
            return list(self)[item] if item < len(self) else None
        raise TypeError("list indices must be integers or slices, not %s" % type(item).__name__)


class OnnxOpConverter(object):
    """A helper class for holding onnx op converters."""

    @classmethod
    def get_converter(cls, opset):
        """Get converter matches given opset.

        Parameters
        ----------
        opset: int
            opset from model.

        Returns
        -------
        converter, which should be `_impl_vx`. Number x is the biggest
            number smaller than or equal to opset belongs to all support versions.
        """
        versions = [int(d.replace("_impl_v", "")) for d in dir(cls) if "_impl_v" in d]
        versions = sorted(versions + [opset])
        version = versions[max([i for i, v in enumerate(versions) if v == opset]) - 1]
        if hasattr(cls, "_impl_v{}".format(version)):
            return getattr(cls, "_impl_v{}".format(version))
        raise NotImplementedError(
            "opset version {} of {} not implemented".format(version, cls.__name__)
        )


class MatMul(OnnxOpConverter):
    """Converts an onnx MatMul node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.emit_te(topi.matmul, inputs[0], inputs[1])


class Div(OnnxOpConverter):
    """Converts an onnx Div node into an equivalent Relax expression."""

    @classmethod
    def _impl_v14(cls, bb, inputs, attr):
        return bb.emit_te(topi.divide, inputs[0], inputs[1])


class Sigmoid(OnnxOpConverter):
    """Converts an onnx Sigmoid node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.emit_te(topi.sigmoid, inputs[0])


class Softmax(OnnxOpConverter):
    """Converts an onnx Softmax node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        axis = attr.get("axis", -1)
        return bb.emit_te(topi.nn.softmax, inputs[0], axis=axis)


class Transpose(OnnxOpConverter):
    """Converts an onnx Transpose node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        perm = attr.get("perm", None)
        return bb.emit_te(topi.transpose, inputs[0], axes=perm)


class Unsqueeze(OnnxOpConverter):
    """Converts an onnx Unsqueeze node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        input = inputs[0]
        axes = inputs[1]

        if isinstance(axes, relax.Constant):
            constant_axes = list(axes.data.numpy())
            constant_axes = list(map(int, constant_axes))
            constant_axes = sorted(constant_axes)
            for axis in constant_axes:
                input = bb.emit_te(topi.expand_dims, input, axis=axis, num_newaxis=1)
            return input

        raise NotImplementedError("Unsqueeze with dynamic axes is not supported.")


class Concat(OnnxOpConverter):
    """Convert an onnx Concat node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        axis = attr.get("axis", 0)
        return bb.emit_te(topi.concatenate, inputs, axis)


class Add(OnnxOpConverter):
    """Convert an onnx Add node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.emit_te(topi.add, inputs[0], inputs[1])


class Mul(OnnxOpConverter):
    """Convert an onnx Mul node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.emit_te(topi.multiply, inputs[0], inputs[1])


class Cast(OnnxOpConverter):
    """Convert an onnx Cast node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        to_type = get_type(attr["to"])
        return bb.emit_te(topi.cast, inputs[0], to_type)


class Gather(OnnxOpConverter):
    """Convert an onnx Gather node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        # TODO This assumes positive only indices.
        axis = attr.get("axis", 0)
        return bb.emit_te(topi.take, inputs[0], inputs[1], axis)


class Gemm(OnnxOpConverter):
    """Convert an onnx Gemm node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        alpha = attr.get("alpha", None)
        beta = attr.get("beta", None)
        transA = attr.get("transA", False)
        transB = attr.get("transB", False)
        A = inputs[0]
        B = inputs[1]
        C = inputs[2]
        dtype = A.checked_type.dtype

        # Compute Y = alpha * A X B + beta * C

        if alpha is not None:
            A = bb.emit_te(topi.multiply, A, relax.const(alpha, dtype=dtype))

        Y = bb.emit_te(topi.matmul, A, B, transA, transB)

        if C is not None:
            if beta is not None:
                C = bb.emit_te(topi.multiply, C, relax.const(beta, dtype=dtype))
            Y = bb.emit_te(topi.add, Y, C)

        return Y


class Reshape(OnnxOpConverter):
    """Convert an onnx Reshape node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        from tvm.script import relax as R

        data = inputs[0]
        # TODO We assume new_shape is a constant, need to enable tensor input to reshape
        # for full support.
        new_shape = inputs[1].data.numpy()

        # Convert -1 dims in new_shape into positive equivalent.
        if -1 in new_shape:
            breakpoint()
            data_shape = [dim.value for dim in data.shape.values]
            total_elements = _np.prod(data_shape)
            new_product = 1
            for dim in new_shape:
                if dim > 0:
                    new_product *= dim

            # Replace -1 with positive equivalent
            for i, dim in enumerate(new_shape):
                if dim == -1:
                    new_shape[i] = int(total_elements / new_product)

        return bb.emit_te(topi.reshape, data, new_shape)


class Gelu(OnnxOpConverter):
    """Operator converter for Gelu from Microsoft onnxruntime contrib opset.

    gelu(x) = 0.5x(1 + erf(x/sqrt(2)))
    """

    @classmethod
    def _impl_v1(cls, bb, inputs, attr):
        x = inputs[0]

        # Declare constants
        const_dtype = x.checked_type.dtype
        half = relax.const(0.5, dtype=const_dtype)
        one = relax.const(1.0, dtype=const_dtype)
        sqrt2 = relax.const(math.sqrt(2.0), dtype=const_dtype)

        # Compute gelu
        term1 = bb.emit_te(topi.multiply, half, x)
        erf = bb.emit_te(topi.erf, bb.emit_te(topi.divide, x, sqrt2))
        term2 = bb.emit_te(topi.add, one, erf)
        return bb.emit_te(topi.multiply, term1, term2)


class BiasGelu(OnnxOpConverter):
    """Operator converter for BiasGelu from Microsoft onnxruntime contrib opset.

    bias_gelu(x, b) = 0.5(x + b)(1 + erf((x + b)/sqrt(2)))
    """

    @classmethod
    def _impl_v1(cls, bb, inputs, attr):
        x = inputs[0]
        b = inputs[1]

        b_dims = b.checked_type.ndim
        assert b_dims == 1, "BiasGelu bias term must be a 1D tensor."

        inp = bb.emit_te(topi.add, x, b)
        return Gelu._impl_v1(bb, [inp], attr)


class Where(OnnxOpConverter):
    """Convert an onnx Where node into an equivalent Relax expression."""

    @classmethod
    def _impl_v16(cls, bb, inputs, attr):
        return bb.emit_te(topi.where, *inputs)


class Clip(OnnxOpConverter):
    """Converts an onnx Clip node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        results = inputs[0]
        if len(inputs) >= 2:
            results = bb.emit_te(topi.maximum, results, inputs[1])
        if len(inputs) >= 3:
            results = bb.emit_te(topi.minimum, results, inputs[2])
        return results


class Equal(OnnxOpConverter):
    """Converts an onnx Equal node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.emit_te(topi.equal, inputs[0], inputs[1])


class Shape(OnnxOpConverter):
    """Converts an onnx Equal node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.emit_te(topi.shape, inputs[0], "int64")


class Not(OnnxOpConverter):
    """Converts an onnx Not node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.emit_te(topi.bitwise_not, inputs[0])


class Tanh(OnnxOpConverter):
    """Converts an onnx Tanh node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.emit_te(topi.tanh, inputs[0])


class Sqrt(OnnxOpConverter):
    """Converts an onnx Sqrt node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.emit_te(topi.sqrt, inputs[0])


class Relu(OnnxOpConverter):
    """Converts an onnx Relu node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.emit_te(topi.nn.relu, inputs[0])


class Pow(OnnxOpConverter):
    """Converts an onnx Pow node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.emit_te(topi.power, inputs[0], inputs[1])


class Conv(OnnxOpConverter):
    """Convert an onnx Conv node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        # not supported yet
        assert "auto_pad" not in attr
        assert "group" not in attr
        # supported conv2d
        return bb.emit_te(
            topi.add,
            bb.emit_te(
                topi.nn.conv2d,
                inputs[0],
                inputs[1],
                strides=attr.get("strides", 1),
                padding=attr.get("pads", 0),
                dilation=attr.get("dilations", 1),
            ),
            bb.emit_te(topi.expand_dims, inputs[2], axis=1, num_newaxis=2),
        )


class Erf(OnnxOpConverter):
    """Converts an onnx Erf node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.emit_te(topi.erf, inputs[0])


class CumSum(OnnxOpConverter):
    """Converts an onnx CumSum node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        data = inputs[0]
        if len(inputs) > 1:
            axis = int(inputs[1].data.numpy())
        else:
            axis = None
        if getattr(attr, "reverse", 0) != 0:
            data = bb.emit_te(topi.flip, data, axis=axis if axis else 0)
        data = bb.emit_te(
            topi.cumsum,
            data=data,
            axis=axis,
            exclusive=attr.get("exclusive", None),
        )
        if getattr(attr, "reverse", 0) != 0:
            data = bb.emit_te(topi.flip, data, axis=axis if axis else 0)
        return data


class Squeeze(OnnxOpConverter):
    """Converts an onnx Squeeze node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        if len(inputs) > 1:
            axis = [int(x) for x in inputs[1].data.numpy()]
        else:
            axis = None
        return bb.emit_te(topi.squeeze, inputs[0], axis=axis)


class Constant(OnnxOpConverter):
    """Converts an onnx Constant node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        def const(
            value: Union[bool, int, float, _np.ndarray, tvm.nd.NDArray],
            dtype: Optional[str] = None,
            span: Optional[relax.Span] = None,
        ):
            """Create a constant value.

            Parameters
            ----------
            value: Union[bool, int, float, numpy.ndarray, tvm.nd.NDArray]
                The constant value.

            dtype: str, optional
                The data type of the resulting constant.

            span: Optional[relax.Span]
                Span that points to original source code.

            Note
            ----
            When dtype is None, we use the following rule:

            - int maps to "int32"
            - float maps to "float32"
            - bool maps to "bool"
            - other using the same default rule as numpy.
            """
            if isinstance(value, (_base.numeric_types, (bool, list))):
                value = _np.array(value, dtype=dtype)

            if not dtype:
                # when dtype is None: int maps to "int32", float maps to "float32"
                dtype = {_np.dtype("int64"): _np.int32, _np.dtype("float64"): _np.float32}.get(
                    value.dtype, None
                )

            if isinstance(value, (_np.ndarray, _np.generic)):
                if dtype is not None:
                    value = value.astype(dtype)
                value = _nd.array(value)

            if not isinstance(value, _nd.NDArray):
                raise ValueError("value has to be scalar or NDArray")

            return relax.Constant(value, span)

        if "value" not in attr:
            raise ValueError("no value in Constant")
        value = attr.pop("value")
        # Constants may rarely have string types. These are likely exported
        # from other frameworks and not actually used in TVM. We'll just use
        # a zero valued constant for compatibility.
        if isinstance(value, bytes):
            np_value = _np.asarray([0]).astype("int64")
        else:
            np_value = get_numpy(value)
        dtype = np_value.dtype.name
        value = const(np_value, dtype)
        return value


class Sub(OnnxOpConverter):
    """Converts an onnx Sub node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.emit_te(topi.subtract, inputs[0], inputs[1])

class Split(OnnxOpConverter):
    """Converts an onnx Split node into an equivalent Relax expression."""

    @classmethod
    def _impl_v1(cls, bb, inputs, attr):
        splits = attr.get("split", None)
        if splits is not None and len(splits) > 1:
            indices = []
            index = 0
            for i in splits[:-1]:
                index += i
                indices.append(index)
        # When splits isnt specified divide evenly over axis.
        else:
            indices = attr["tvm_custom"]["num_outputs"]
        output = bb.emit_te(topi.split, inputs[0], indices, attr.get("axis", 0))
        return output

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        splits = inputs[1]
        splits_rank = None
        if splits is not None:
            splits_rank = splits.checked_type.ndim
        if splits is not None and splits_rank > 0:
            if isinstance(splits, relax.Constant):
                splits = splits.data.asnumpy()
                indices = []
                index = 0
                for i in splits[:-1]:
                    index += i
                    indices.append(index)
            else:
                raise ValueError("Dynamic Split not yet supported")
        # When splits isnt specified divide evenly over axis.
        else:
            indices = attr["tvm_custom"]["num_outputs"]
        output = bb.emit_te(topi.split, inputs[0], indices, axis=attr.get("axis", 0))
        return output


def _get_convert_map(opset):
    return {
        "MatMul": MatMul,
        "Concat": Concat,
        "Add": Add,
        "Mul": Mul,
        "Cast": Cast,
        "Gather": Gather,
        "Gemm": Gemm,
        "Reshape": relay.frontend.onnx.Reshape,
        "Div": Div,
        "Sigmoid": Sigmoid,
        "Softmax": Softmax,
        "Transpose": Transpose,
        "Unsqueeze": Unsqueeze,
        "Gelu": Gelu,
        "BiasGelu": BiasGelu,
        "Where": Where,
        "Clip": Clip,
        "Equal": Equal,
        "Shape": Shape,
        "Not": Not,
        "Tanh": Tanh,
        "Sqrt": Sqrt,
        "Relu": Relu,
        "Conv": Conv,
        "Pow": Pow,
        "Erf": Erf,
        "CumSum": CumSum,
        "Squeeze": Squeeze,
        "Constant": Constant,
        "Sub": Sub,
        "LayerNormalization": relay.frontend.onnx.LayerNormalization,
        "SkipLayerNormalization": relay.frontend.onnx.SkipLayerNormalization,
        "EmbedLayerNormalization": relay.frontend.onnx.EmbedLayerNormalization,
        # defs/reduction
        "ReduceMax": relay.frontend.onnx.ReduceMax,
        "ReduceMin": relay.frontend.onnx.ReduceMin,
        "ReduceSum": relay.frontend.onnx.ReduceSum,
        "ReduceMean": relay.frontend.onnx.ReduceMean,
        "ReduceProd": relay.frontend.onnx.ReduceProd,
        "ReduceLogSumExp": relay.frontend.onnx.ReduceLogSumExp,
        "ReduceLogSum": relay.frontend.onnx.ReduceLogSum,
        "ReduceSumSquare": relay.frontend.onnx.ReduceSumSquare,
        "ReduceL1": relay.frontend.onnx.ReduceL1,
        "ReduceL2": relay.frontend.onnx.ReduceL2,
        "Expand": relay.frontend.onnx.Expand,
        "ConstantOfShape": relay.frontend.onnx.ConstantOfShape,
        "Slice": relay.frontend.onnx.Slice,
        "Attention": relay.frontend.onnx.Attention,
        "Pad": relay.frontend.onnx.Pad,
        "Split": Split,
        "Tile": relay.frontend.onnx.Tile,
    }


class GraphProto:
    """A helper class for handling Relax expression copying from pb2.GraphProto.
    Definition: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto
        Parameters
    ----------
    shape : dict of str to tuple, optional
        The input shape to the graph
    dtype : str or dict of str to str
        The input types to the graph
    """

    current = None

    def __init__(self, shape, dtype, target):
        self._nodes = {}
        self._inputs = {}
        self._num_input = 0
        self._shape = shape.copy() if shape else {}
        self._input_names = []
        self._dtype = dtype
        self.opset = None
        self._target = target
        self.bb = relax.BlockBuilder()

    def from_onnx(self, graph, opset) -> IRModule:
        """Construct Relax expression from ONNX graph.
        Onnx graph is a python protobuf object.
        The companion parameters will be handled automatically.
        However, the input names from onnx graph is vague, mixing inputs and
        network weights/bias such as "1", "2"...
        For convenience, we rename the `real` input names to "input_0",
        "input_1"... And renaming parameters to "param_0", "param_1"...
        Parameters
        ----------
        graph : onnx protobuf object
            The loaded onnx graph
        opset : opset version
        Returns
        -------
        mod : tvm.IRModule
            The returned relax module
        params : dict
            A dict of name: tvm.nd.array pairs, used as pretrained weights
        """
        with self.bb.function("main"):
            with self.bb.dataflow() as df:
                self.opset = opset
                self._parse_graph_initializers(graph)
                self._parse_graph_input(graph)
                self._check_for_unsupported_ops(graph)
                self._construct_nodes(graph)

                # now return the outputs
                outputs = [self._nodes[self._parse_value_proto(i)] for i in graph.output]
                outputs = outputs[0] if len(outputs) == 1 else relax.Tuple(outputs)

                ## Maintain the order of inputs and parameters from the ONNX graph, but only include
                ## those parameters that are needed to execute the relax graph
                nodes = {v: k for k, v in self._nodes.items()}
                # Create a function from our output expression and all input variables.
                param_list = [v for k, v in self._inputs.items()]
                output_var = self.bb.emit_output(outputs)
            self.bb.emit_func_output(output_var, params=param_list)
        return self.bb.get()

    def _parse_graph_initializers(self, graph):
        """Parse network inputs to relax, aka parameters."""
        for init_tensor in graph.initializer:
            if not init_tensor.name.strip():
                raise ValueError("Tensor's name is required.")
            array = self._parse_array(init_tensor)
            self._nodes[init_tensor.name] = relax.const(array)

    def _parse_graph_input(self, graph):
        for i in graph.input:
            # from onnx v0.2, GraphProto.input has type ValueInfoProto,
            #  and the name is 'i.name'
            i_name, i_shape, d_type, i_shape_name = get_info(i)
            if i_name in self._nodes:
                continue
            else:
                self._num_input += 1
                self._input_names.append(i_name)
                if i_name in self._shape:
                    i_shape = self._shape[i_name]
                else:
                    if "?" in str(i_shape):
                        warning_msg = (
                            "Input %s has unknown dimension shapes: %s. "
                            "Specifying static values may improve performance"
                            % (i_name, str(i_shape_name))
                        )
                        warnings.warn(warning_msg)
                if isinstance(self._dtype, dict):
                    dtype = self._dtype[i_name] if i_name in self._dtype else d_type
                else:
                    dtype = d_type
                self._nodes[i_name] = new_var(i_name, shape=i_shape, dtype=dtype)
            self._inputs[i_name] = self._nodes[i_name]

    def _check_for_unsupported_ops(self, graph):
        convert_map = _get_convert_map(self.opset)
        unsupported_ops = set()
        for node in graph.node:
            op_name = node.op_type
            if (
                op_name not in convert_map
                and op_name != "Constant"
                # and op_name not in _identity_list
            ):
                unsupported_ops.add(op_name)
        if unsupported_ops:
            msg = "The following operators are not supported for frontend ONNX: "
            msg += ", ".join(unsupported_ops)
            raise tvm.error.OpNotImplemented(msg)

    def _construct_nodes(self, graph):
        """Nodes are stored as directed acyclic graph."""
        for node in graph.node:
            op_name = node.op_type
            attr = self._parse_attr(node.attribute)
            # Create and populate input list.
            inputs = onnx_input()
            for i in node.input:
                if i != "":
                    inputs.append(self._nodes[i])
                else:
                    inputs.append(None)
            i_name = self._parse_value_proto(node)
            outputs = node.output
            attr["tvm_custom"] = {}
            attr["tvm_custom"]["name"] = i_name
            attr["tvm_custom"]["num_outputs"] = len(outputs)

            op = self._convert_operator(op_name, inputs, attr, self.opset)

            if not isinstance(op, relax.Tuple):
                if isinstance(op.checked_type, tvm.ir.type.TupleType):
                    # This is a var bound to a tuple. We need to unpack it and create
                    # a new tuple.
                    tuple_items = []
                    for i in range(len(op.checked_type.fields)):
                        tuple_items.append(self.bb.emit(relax.TupleGetItem(op, i)))
                    op = relax.Tuple(tuple_items)
                    outputs_num = len(tuple_items)
                else:
                    outputs_num = 1
            else:
                outputs_num = len(op)

            assert len(outputs) == outputs_num, "Number of output mismatch {} vs {} in {}.".format(
                len(outputs), outputs_num, op_name
            )

            if outputs_num == 1:
                self._nodes[outputs[0]] = op
            else:
                for k, i in zip(list(outputs), range(len(outputs))):
                    self._nodes[k] = op[i]

    def _parse_value_proto(self, value_proto):
        """Parse ValueProto or raw str."""
        try:
            name = value_proto.name
        except AttributeError:
            name = value_proto
        return name

    def _parse_array(self, tensor_proto):
        np_array = get_numpy(tensor_proto).reshape(tuple(tensor_proto.dims))
        return tvm.nd.array(np_array)

    def _parse_attr(self, attr_proto):
        """Convert a list of AttributeProto to a dict, with names as keys."""
        attrs = {}
        for a in attr_proto:
            for f in ["f", "i", "s", "g"]:
                if a.HasField(f):
                    attrs[a.name] = getattr(a, f)
            for f in ["floats", "ints", "strings"]:
                if list(getattr(a, f)):
                    assert a.name not in attrs, "Only one type of attr is allowed"
                    attrs[a.name] = tuple(getattr(a, f))
            for f in ["t"]:
                if a.HasField(f):
                    attrs[a.name] = getattr(a, f)
            for f in ["tensors"]:
                if list(getattr(a, f)):
                    assert a.name not in attrs, "Only one type of attr is allowed"
                    attrs[a.name] = tuple(getattr(a, f))
            for f in ["graphs"]:
                if list(getattr(a, f)):
                    raise NotImplementedError("Field {} is not supported in relax.".format(f))
            if a.name not in attrs:
                raise ValueError("Cannot parse attribute: \n{}\n.".format(a))
        return attrs

    def _relay_input_adapter(self, inputs):
        """Creates equivalent input Relay vars from the input Relax vars"""
        relay_vars = onnx_input()
        for relax_var in inputs:
            shape_values = []
            # Some inputs may be None to indicate that input isnt used.
            if relax_var is None:
                relay_vars.append(None)
            # Otherwise construct a new relay variable mirroring the relax one.
            else:
                for shape_value in relax_var.struct_info.shape.values:
                    shape_values.append(shape_value)
                if isinstance(relax_var, relax.Constant):
                    relay_vars.append(relay.const(relax_var.data, dtype=relax_var.checked_type.dtype))
                else:
                    relay_vars.append(
                        relay.var(
                            relax_var.name_hint, shape=shape_values, dtype=relax_var.checked_type.dtype
                        )
                    )
        return relay_vars

    def _relay_output_adapter(self, relax_inputs, relay_inputs, relay_output):
        """Given the output of a relay op from the Onnx relay frontend,
        calls into the relay to relax translator to obtain the equivalent Relax.
        Then unpacks the IRModule obtained and adds the TIR funcs and the
        associated call_tirs to the block builder in use.

        Parameters
        ----------
        relax_inputs : list(relax.Var, relay.Constant)
                The list of relax vars that are inputs to the relax op.
        relay_inputs : list(relay.Var, relay.Constant)
                The list of relay vars that are inputs to the relay op. This is
                obtianed from the _relay_input_adapter function.
        relay_output : relay.Expr
                The output of the relay op from the Onnx relay frontend.
        Returns
        -------
        output : relax.Expr
                The output of the equivalent relax op.
        """
        if isinstance(relay_output, TupleWrapper):
            relay_output = relay_output.tuple_value

        # Create a Relay function with the body returned by the Relay op.
        relay_var_inputs = [input for input in relay_inputs if isinstance(input, relay.Var)]
        function = relay.Function(relay_var_inputs, relay_output)
        # Save the current in-use block builder. The translator uses its own block builder.
        prev_bb = relax.BlockBuilder._current
        relax.BlockBuilder._current = None
        relax_mod = testing.relay_translator.from_relay(function, self._target)
        # Restore the block builder used by the frontend.
        relax.BlockBuilder._current = prev_bb

        # This dict is used by the Mapper mutator to replace the globar vars
        # in the relax_mod with global_vars registered with the in-use block builder.
        global_var_dict = {}
        for gv, func in relax_mod.functions.items():
            if gv.name_hint != "main":
                global_var_dict[gv] = self.bb.add_func(func, gv.name_hint)

        # This dict is used by the Mapper mutator to replace the relax vars
        # with the inputs.
        relax_input_dict = {}
        for relax_var in relax_inputs:
            if isinstance(relax_var, relax.Var):
                relax_input_dict[relax_var.name_hint] = relax_var

        @relax.expr_functor.mutator
        class Mapper(PyExprMutator):
            def visit_var_(self, var_node: Var):
                if var_node.name_hint in relax_input_dict:
                    return relax_input_dict[var_node.name_hint]
                return var_node

            def visit_global_var_(self, gv_node: GlobalVar):
                if gv_node in global_var_dict:
                    return global_var_dict[gv_node]
                return gv_node

        updated_func = Mapper().visit_expr(relax_mod["main"])

        var_bindings = updated_func.body.blocks[0].bindings
        if isinstance(updated_func.ret_struct_info, relax.TupleStructInfo):
            # Returning a tuple.
            final_binding = var_bindings[-2]
            for binding in var_bindings[:-2]:
                self.bb.emit_normalized(binding)
        else:
            final_binding = var_bindings[-1]
            for binding in var_bindings[:-1]:
                self.bb.emit_normalized(binding)

        return final_binding.value

    def _convert_operator(self, op_name, inputs, attrs, opset):
        """Convert ONNX operator into a Relax operator.
        The converter must specify conversions explicitly for incompatible name, and
        apply handlers to operator attributes.
        Parameters
        ----------
        op_name : str
            Operator name, such as Convolution, FullyConnected
        inputs : list of tvm.relax.function.Function
            List of inputs.
        attrs : dict
            Dict of operator attributes
        opset : int
            Opset version
        Returns
        -------
        sym : tvm.relax.function.Function
            Converted relax function
        """
        convert_map = _get_convert_map(opset)
        if op_name in convert_map:
            convert_class = convert_map[op_name]
            op_function = convert_class.get_converter(opset)
            # If the op_function is a subclass of Relay OnnxOpConverter then it is a relay op.
            if issubclass(convert_class, RelayOnnxOpConverter):
                relay_inputs = self._relay_input_adapter(inputs)
                # The op_function might change the inputs to the relay op. Use a copy of the inputs.
                relay_inputs_copy = onnx_input()
                for relay_input in relay_inputs:
                    relay_inputs_copy.append(relay_input)
                # TODO handle params passing
                relay_output = op_function(relay_inputs_copy, attrs, params=[])
                sym = self._relay_output_adapter(inputs, relay_inputs, relay_output)
            else:
                sym = op_function(self.bb, inputs, attrs)
        else:
            raise NotImplementedError("Operator {} not implemented.".format(op_name))
        return sym


def from_onnx(model, shape=None, dtype="float32", opset=None, target: Union[str, Target] = "llvm"):
    """Convert a ONNX model into an equivalent Relax Function.
    ONNX graphs are represented as Python Protobuf objects.
    The companion parameters will be handled automatically.
    However, the input names from onnx graph is vague, mixing inputs and
    network weights/bias such as "1", "2"...
    For convenience, we rename the `real` input names to "input_0",
    "input_1"... And renaming parameters to "param_0", "param_1"...
    By default, ONNX defines models in terms of dynamic shapes. The ONNX importer
    retains that dynamism upon import, and the compiler attempts to convert the
    model into a static shapes at compile time. If this fails, there may still
    be dynamic operations in the model. Not all TVM kernels currently support
    dynamic shapes, please file an issue on discuss.tvm.apache.org
    if you hit an error with dynamic kernels.
    Parameters
    ----------
    model : protobuf object
        ONNX ModelProto after ONNX v1.1.0
    shape : dict of str to tuple, optional
        The input shape to the graph
    dtype : str or dict of str to str
        The input types to the graph
    opset : int, optional
        Override to autodetected opset.
        This can be helpful for some testing.
    target : str or Target, optional
        The compilation target used by the Relay to Relax translator.

    Returns
    -------
    mod : tvm.IRModule
        The relax module for compilation
    params : dict of str to tvm.nd.NDArray
        The parameter dict to be used by relax
    """
    try:
        import onnx

        if hasattr(onnx.checker, "check_model"):
            # try use onnx's own model checker before converting any model
            try:
                onnx.checker.check_model(model)
            except Exception as e:  # pylint: disable=c-extension-no-member, broad-except
                # the checker is a bit violent about errors, so simply print warnings here
                warnings.warn(str(e))
    except ImportError:
        pass

    if isinstance(target, str):
        target = Target(target)
    g = GraphProto(shape, dtype, target)
    graph = model.graph

    try:
        opset_in_model = 1
        if model.opset_import:
            # TODO: for now we only really support ai.onnx op set
            # TODO: handle other namespaces well see https://github.com/apache/tvm/issues/10950
            for opset_identifier in model.opset_import:
                # As per https://github.com/onnx/onnx/blob/main/docs/IR.md
                # All operator sets except the default one must specify the operator version
                if str(opset_identifier.domain) in ["ai.onnx", ""]:
                    opset_in_model = opset_identifier.version
                    break
    except AttributeError:
        opset_in_model = 1

    if opset is None:
        opset = opset_in_model
    elif opset < opset_in_model:
        warnings.warn(
            ""
            f"You are overwritting original opset ver = {opset_in_model} by lower ver = {opset}. "
            f"That might cause model conversion errors."
        )

    # Use the graph proto as a scope so that ops can access other nodes if needed.
    return g.from_onnx(graph, opset)
