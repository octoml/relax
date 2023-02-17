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
# pylint: disable=consider-using-f-string
"""ONNX: Open Neural Network Exchange importer for Relax.

This module implemnets the required functionality to read ONNX models
and convert them into equivalent Relax functions. The entry point that encapsulates
this functionality is the function from_onnx.

In order to extend the functionality of the importer, you can add new
operators to the operator registry. The operator registry is a dictionary
that maps operator names to operator converters. The registry is defined
in the _get_converter_map function. To add a new operator, you can define
a new class that inherits from the OnnxOpConverter class and implement
the _impl method.

By default, ONNX defines models in terms of dynamic shapes. The ONNX importer
retains dynamic shapes upon import, and when possible, the compiler attempts to
convert the model to use static shapes at compile time.
If this fails, there may still be dynamic operations in the model.
Not all TVM kernels currently support dynamic shapes, please file an issue on
github.com/apache/tvm/issues if you hit an error with dynamic kernels.
"""
import warnings
from typing import Union, List, Dict, Tuple, Any
import numpy as _np

import tvm
from tvm import relax, relay
from tvm.target import Target
from tvm.ir import IRModule
from tvm.relax import testing, PyExprMutator
from tvm.relay.expr import TupleWrapper, Var, GlobalVar
from tvm.relay.frontend.onnx import OnnxOpConverter as RelayOnnxOpConverter
from tvm.script import relax as R

import onnx.onnx_ml_pb2


def new_var(var_name: str, shape: Tuple, dtype: str = "float32"):
    return testing.nn.Parameter(shape=shape, dtype=dtype, name=var_name)


def get_type(elem_type: Union[str, int]) -> str:
    """Converts onnx integer datatype to numpy datatype"""
    # If a string was passed instead of a tensor type, it does not need
    # conversion and can be returned.
    if isinstance(elem_type, str):
        return elem_type

    try:
        from onnx.mapping import TENSOR_TYPE_TO_NP_TYPE  # pylint: disable=import-outside-toplevel
    except ImportError as exception:
        raise ImportError(
            "Unable to import onnx which is required {}".format(exception)
        ) from exception

    return str(TENSOR_TYPE_TO_NP_TYPE[elem_type])


def get_info(info_proto: onnx.onnx_ml_pb2.ValueInfoProto) -> Tuple[str, List, str, List]:
    """Extract the shape from a ValueInfoProto.

    Parameters
    ----------
    info_proto: onnx.onnx_ml_pb2.ValueInfoProto
        The ValueInfoProto to extract the info from.

    Returns
    -------
    Tuple[str, List, str, List]
        The name, shape, type, and shape name of the ValueInfoProto.
    """
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


def get_numpy(tensor_proto: onnx.onnx_ml_pb2.TensorProto) -> _np.ndarray:
    """Grab data in TensorProto and convert to numpy array."""
    try:
        from onnx.numpy_helper import to_array  # pylint: disable=import-outside-toplevel
    except ImportError as exception:
        raise ImportError(
            "Unable to import onnx which is required {}".format(exception)
        ) from exception
    return to_array(tensor_proto)


class onnx_input(list):  # pylint: disable=invalid-name
    """A list that returns None when out-of-bounds indices are accessed."""

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


# pylint: disable=invalid-name, len-as-condition, unused-argument, too-many-lines, redefined-builtin
class OnnxOpConverter(object):
    """A helper class for holding the common logic for ONNX op converters.
    Each converter maps to a single ONNX op and defines the equivalent
    functionality using Relax expressions. The converter can define multiple versions
    of the op and the version is selected based on the opset version of the model.
    """

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


class Concat(OnnxOpConverter):
    """Convert an onnx Concat node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        axis = attr.get("axis", 0)
        return bb.normalize(R.concat(inputs, axis))


class Matmul(OnnxOpConverter):
    """Convert an onnx Matmul node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.normalize(relax.op.linear_algebra.matmul(inputs[0], inputs[1]))


class Add(OnnxOpConverter):
    """Convert an onnx Add node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.normalize(relax.op.add(inputs[0], inputs[1]))


class Cast(OnnxOpConverter):
    """Convert an onnx Cast node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        to_type = get_type(attr["to"])
        return bb.normalize(relax.op.astype(inputs[0], to_type))


class Gather(OnnxOpConverter):
    """Convert an onnx Gather node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        axis = attr.get("axis", 0)
        # This assumes that the indices are 1-d and positive
        return bb.normalize(relax.op.take(inputs[0], inputs[1], axis))


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
            if transA:
                A = relax.op.permute_dims(A)
            if transB:
                B = relax.op.permute_dims(B)

            Y = relax.op.multiply(
                relax.op.linear_algebra.matmul(A, B),
                relax.const(alpha, dtype=dtype),
            )

        if C is not None:
            if beta is not None:
                C = relax.op.multiply(C, relax.const(beta, dtype=dtype))
            if alpha is not None:
                Y = relax.op.add(Y, C)
            else:
                Y = C
        else:
            if alpha is None:
                Y = relax.const(0, dtype=dtype)

        return bb.normalize(Y)


class Mul(OnnxOpConverter):
    """Convert an onnx Mul node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.normalize(relax.op.multiply(inputs[0], inputs[1]))


class Reshape(OnnxOpConverter):
    """Convert an onnx Reshape node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        assert isinstance(inputs[1], relax.Constant), """Currently only supports constant shape."""
        shape = inputs[1].data.numpy().tolist()
        assert shape.count(-1) <= 1, """Reshape only supports at most one -1 in shape."""
        return bb.normalize(relax.op.reshape(inputs[0], shape))


class Shape(OnnxOpConverter):
    """Convert an onnx Shape node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.normalize(relax.op.shape_of(inputs[0]))


class Clip(OnnxOpConverter):
    """Convert an onnx Clip node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        data = inputs[0]
        if len(inputs) >= 2:
            min_value = inputs[1]
            assert isinstance(
                min_value, relax.Constant
            ), """Currently only supports constant min."""
            min_value = min_value.data.numpy().tolist()
            assert isinstance(min_value, (int, float)), """Clip only supports scalar min."""
        else:
            min_value = None

        if len(inputs) >= 3:
            max_value = inputs[2]
            assert isinstance(
                max_value, relax.Constant
            ), """Currently only supports constant max."""
            max_value = max_value.data.numpy().tolist()
            assert isinstance(max_value, (int, float)), """Clip only supports scalar max."""
        else:
            max_value = None
        return bb.normalize(relax.op.clip(data, min_value, max_value))


class Div(OnnxOpConverter):
    """Convert an onnx Div node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.normalize(relax.op.divide(inputs[0], inputs[1]))


class Expand(OnnxOpConverter):
    """Convert an onnx Expand node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        data = inputs[0]
        shape = inputs[1]

        shape_ndim = [dim.value for dim in shape.struct_info.shape.values][0]
        shape_dataflow_var = bb.emit(
            relax.Call(
                relax.ExternFunc("vm.builtin.tensor_to_shape"),
                [shape],
                sinfo_args=[relax.ShapeStructInfo(ndim=shape_ndim)],
            )
        )

        shape_vars = []
        for i in range(shape_ndim):
            shape_vars.append(tvm.tir.Var("shape_var_%d" % i, "int64"))
        # TODO: confirm this is the correct way to do this
        bb.match_cast(shape_dataflow_var, R.Shape(shape_vars))
        return bb.normalize(relax.op.broadcast_to(data, relax.ShapeExpr(shape_vars)))


class ReduceMean(OnnxOpConverter):
    """Convert an onnx ReduceMean node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        axes = attr.get("axes", None)
        keepdims = attr.get("keepdims", 1)
        return bb.normalize(relax.op.mean(inputs[0], axes, keepdims))


class Sigmoid(OnnxOpConverter):
    """Convert an onnx Sigmoid node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.normalize(relax.op.sigmoid(inputs[0]))


class Slice(OnnxOpConverter):
    """Convert an onnx Slice node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        data = inputs[0]
        starts = inputs[1]
        ends = inputs[2]
        axes = inputs[3] if len(inputs) >= 4 else None
        steps = inputs[4] if len(inputs) >= 5 else None

        # TODO (jwfromm) currently only supports constant parameters.
        if not all(
            [
                (isinstance(param, relax.Constant) or param is None)
                for param in [starts, ends, axes, steps]
            ]
        ):
            raise ValueError("Only constant Slice parameters are currently supported.")

        # Convert parameters to constant lists.
        starts = starts.data.numpy().tolist()
        ends = ends.data.numpy().tolist()
        if axes is not None:
            axes = axes.data.numpy().tolist()
        else:
            axes = list(range(len(starts)))
        if steps is not None:
            steps = steps.data.numpy().tolist()
        else:
            steps = [1] * len(axes)
        return bb.normalize(relax.op.strided_slice(data, axes, starts, ends, steps))


class Softmax(OnnxOpConverter):
    """Convert an onnx Softmax node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        axis = attr.get("axis", -1)
        return bb.normalize(relax.op.nn.softmax(inputs[0], axis))


class Transpose(OnnxOpConverter):
    """Convert an onnx Transpose node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        axes = attr.get("perm", None)
        return bb.normalize(relax.op.permute_dims(inputs[0], axes))


class Unsqueeze(OnnxOpConverter):
    """Convert an onnx Unsqueeze node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        data = inputs[0]
        axes = inputs[1]

        if isinstance(axes, relax.Constant):
            constant_axes = list(axes.data.numpy())  # convert to list
            constant_axes = list(map(int, constant_axes))  # dedupulicate
            constant_axes = sorted(constant_axes)  # sort
            return bb.normalize(relax.op.expand_dims(data, constant_axes))
        raise NotImplementedError("Unsqueeze with dynamic axes is not supported.")


class BiasGelu(OnnxOpConverter):
    """Convert an onnx BiasGelu node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        a = inputs[0]
        b = inputs[1]

        b_dims = b.checked_type.ndim
        assert b_dims == 1, "BiasGelu bias term must be a 1D tensor."

        return bb.normalize(relax.op.nn.gelu(relax.op.add(a, b)))


class Conv(OnnxOpConverter):
    """Convert an onnx Conv node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        assert (
            inputs[0].checked_type.ndim == 4 and inputs[1].checked_type.ndim == 4
        ), """Only 2D convolutions are supported for now."""

        strides = attr.get("strides", [1, 1])
        if "auto_pad" not in attr or attr["auto_pad"] == b"NOTSET":
            padding = attr.get("pads", None)
        elif attr["auto_pad"] == b"VALID":
            padding = [0, 0, 0, 0]
        elif attr["auto_pad"] == b"SAME_UPPER" or attr["auto_pad"] == b"SAME_LOWER":
            padding_front = []
            padding_back = []
            for i in range(2):
                pad = max(strides[i] - (inputs[0].struct_info.shape.values[i + 2] % strides[i]), 0)
                pad_front = pad // 2
                if pad % 2 == 1 and attr["auto_pad"] == "SAME_LOWER":
                    pad_front += 1
                pad_back = pad - pad_front
                padding_front.append(pad_front)
                padding_back.append(pad_back)
                padding = padding_front + padding_back
        else:
            raise ValueError("Unsupported auto_pad value: {}".format(attr["auto_pad"]))
        # default NCHW / OIHW layout
        return bb.normalize(
            relax.op.nn.conv2d(
                inputs[0],
                inputs[1],
                padding=padding,
                strides=strides,
                dilation=attr.get("dilations", [1, 1]),
                groups=attr.get("group", 1),
            )
        )


class Equal(OnnxOpConverter):
    """Convert an onnx Equal node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.normalize(relax.op.equal(inputs[0], inputs[1]))


class Erf(OnnxOpConverter):
    """Convert an onnx Erf node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        x = inputs[0]
        sqrt2 = relax.op.sqrt(relax.const(2, dtype="float32"))
        # TODO: replace with erf operator once it is implemented
        return bb.normalize(
            relax.op.add(
                relax.op.divide(
                    relax.op.multiply(relax.op.nn.gelu(relax.op.multiply(x, sqrt2)), sqrt2), x
                ),
                relax.const(-1, dtype="float32"),
            )
        )


class Not(OnnxOpConverter):
    """Convert an onnx Not node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.normalize(relax.op.equal(inputs[0], relax.const(0, dtype="bool")))


class Pow(OnnxOpConverter):
    """Convert an onnx Pow node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        x = inputs[0]
        y = inputs[1]
        return bb.normalize(relax.op.exp(relax.op.multiply(relax.op.log(x), y)))


class SkipLayerNormalization(OnnxOpConverter):
    """Convert an onnx SkipLayerNormalization node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        eps = attr.get("epsilon", 1e-12)
        data = inputs[0]
        assert data.checked_type.ndim == 3, "SkipLayerNormalization input must be 3D tensor."
        skip = inputs[1]
        assert skip.checked_type.ndim == 3, "SkipLayerNormalization skip must be 3D tensor."
        gamma = inputs[2]
        beta = inputs[3] if len(inputs) > 3 else None
        bias = inputs[4] if len(inputs) > 4 else None

        x = relax.op.add(data, skip)
        if bias is not None:
            x = relax.op.add(x, bias)
        output = relax.op.nn.layer_norm(
            x,
            gamma,
            beta,
            axes=2,  # feature dimension
            epsilon=eps,
        )

        placeholder = relax.const(0, dtype="float32")

        # TODO: according to relay documentation, onnxruntime doesn't compute
        # the other outputs, despite the documentation. Need to confirm and
        # make sure this is the correct way to handle relax tuple return.
        return relax.Tuple([output, placeholder, placeholder, placeholder])


class Sqrt(OnnxOpConverter):
    """Convert an onnx Sqrt node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.normalize(relax.op.sqrt(inputs[0]))


class Squeeze(OnnxOpConverter):
    """Convert an onnx Squeeze node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        data = inputs[0]
        if len(inputs) > 1:
            axes = inputs[1]
            assert isinstance(axes, relax.Constant), "Currently only support constant axes."
            axes = axes.data.numpy().tolist()
            for axis in axes:
                assert data.struct_info.shape[axis] == 1, "Squeeze axis must be 1."
        else:
            axes = None
        return bb.normalize(relax.op.squeeze(data, axes))


class Sub(OnnxOpConverter):
    """Convert an onnx Sub node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.normalize(relax.op.subtract(inputs[0], inputs[1]))


class Tanh(OnnxOpConverter):
    """Convert an onnx Tanh node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.normalize(relax.op.tanh(inputs[0]))


class Where(OnnxOpConverter):
    """Convert an onnx Where node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.normalize(relax.op.where(inputs[0], inputs[1], inputs[2]))


class Constant(OnnxOpConverter):
    """Convert an onnx Constant node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
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
        value = relax.const(np_value, dtype)
        return value


class Relu(OnnxOpConverter):
    """Convert an onnx Relu node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        return bb.normalize(relax.op.nn.relu(inputs[0]))


class Split(OnnxOpConverter):
    """Convert an onnx Split node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        axis = attr.get("axis", 0)
        input = inputs[0]
        split = inputs[1] if len(inputs) > 1 else None
        if split is None:
            raise ValueError("split is required for Split")
        assert isinstance(split, relax.Constant), "Currently only support constant split."
        split = split.data.numpy().tolist()
        assert sum(split) == input.struct_info.shape[axis], "Split sum must equal input size."
        return bb.normalize(relax.op.split(input, split, axis))


class EmbedLayerNormalization(OnnxOpConverter):
    """Convert an onnx EmbedLayerNormalization node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        eps = attr.get("epsilon", 1e-12)
        input_ids = inputs[0]
        assert (
            input_ids.checked_type.ndim == 2
        ), "EmbedLayerNormalization input ids must be 2D tensor."

        segment_ids = inputs[1]
        assert (
            segment_ids.checked_type.ndim == 2
        ), "EmbedLayerNormalization segment ids must be 2D tensor."
        if (
            segment_ids.struct_info.shape[0] != input_ids.struct_info.shape[0]
            or segment_ids.struct_info.shape[1] != input_ids.struct_info.shape[1]
        ):
            # segment_ids is optional
            segment_ids = None
            next_input_id = 1
        else:
            next_input_id = 2

        word_embedding = inputs[next_input_id]
        assert (
            word_embedding.checked_type.ndim == 2
        ), "EmbedLayerNormalization word embedding must be 2D tensor."
        position_embedding = inputs[next_input_id + 1]
        assert (
            position_embedding.checked_type.ndim == 2
        ), "EmbedLayerNormalization position embedding must be 2D tensor."
        next_input_id += 2
        segment_embedding = inputs[next_input_id]
        if segment_embedding.checked_type.ndim != 2:
            # segment embedding is optional
            segment_embedding = None
        else:
            next_input_id += 1
        gamma = inputs[next_input_id]
        beta = inputs[next_input_id + 1]
        mask = inputs[next_input_id + 2] if len(inputs) > next_input_id + 2 else None
        position_ids = inputs[next_input_id + 3] if len(inputs) > next_input_id + 3 else None

        (batch_size, seq_len) = input_ids.struct_info.shape
        batch_size = int(batch_size)
        seq_len = int(seq_len)

        if segment_ids:
            assert segment_embedding is not None

        if position_ids is None:
            position_ids = relax.const([list(range(int(seq_len)))] * int(batch_size), dtype="int64")

        input_ids_1d = bb.normalize(relax.op.reshape(input_ids, [-1]))
        word_vec_2d = bb.normalize(relax.op.take(word_embedding, input_ids_1d, axis=0))
        word_vec = bb.normalize(relax.op.reshape(word_vec_2d, [batch_size, seq_len, -1]))

        segment_ids_1d = bb.normalize(relax.op.reshape(segment_ids, [-1]))
        segment_vec_2d = bb.normalize(relax.op.take(segment_embedding, segment_ids_1d, axis=0))
        segment_vec = bb.normalize(relax.op.reshape(segment_vec_2d, [batch_size, seq_len, -1]))

        pos_ids_1d = bb.normalize(relax.op.reshape(position_ids, [-1]))
        pos_vec_2d = bb.normalize(relax.op.take(position_embedding, pos_ids_1d, axis=0))
        pos_vec = bb.normalize(relax.op.reshape(pos_vec_2d, [batch_size, seq_len, -1]))

        vec_sum = bb.normalize(relax.op.add(word_vec, pos_vec))
        if segment_ids:
            vec_sum = bb.normalize(relax.op.add(vec_sum, segment_vec))

        vec_sum_2d = bb.normalize(relax.op.reshape(vec_sum, [batch_size * seq_len, -1]))
        ln = bb.normalize(relax.op.nn.layer_norm(vec_sum_2d, gamma, beta, axes=1, epsilon=eps))
        ln = bb.normalize(relax.op.reshape(ln, [batch_size, seq_len, -1]))

        mask_index = relax.const(
            _np.zeros(
                batch_size,
            ),
            dtype="int64",
        )
        if mask:
            # calculate number of words per sentence
            mask_index = bb.normalize(relax.op.sum(mask, axis=1))
        return relax.Tuple([ln, mask_index, vec_sum])


class Attention(OnnxOpConverter):
    """Transform an onnx Attention node into an equivalent Relax expression."""

    @classmethod
    def _impl_v13(cls, bb, inputs, attr):
        # mask_filter_value = attr.get("mask_filter_value", -1e4)
        num_heads = attr.pop("num_heads")
        # past_present_share_buffer = attr.pop("past_present_share_buffer")
        assert (
            "qkv_hidden_sizes" not in attr
        ), "different hidden sizes for Q, K, V are not currently supported"
        # scale = attr.pop("scale")
        assert "unidirectional" not in attr, "unidirectional attention not current supported"

        # (batch, seq, in_hidden)
        input_emb = inputs[0]

        # (in_hidden, 3 * out_hidden), where out_hidden = num_heads * head_size
        weight = inputs[1]

        # (3 * out_hidden,)
        bias = inputs[2]

        # 1. (    batch,              1,        max_seq, max_seq)
        # 2. (    batch, past_seq + seq,)
        # 3. (    batch,            seq, past_seq + seq,)
        # 4. (    batch,)
        # 5. (2 * batch,)
        # For now, we only support case 2.
        mask_index = inputs[3]

        # (2, batch, num_heads, past_seq, head_size)
        past = inputs[4]

        # (batch, num_heads, seq, seq)
        extra_add = inputs[5]

        (batch_size, seq_len, _) = input_emb.struct_info.shape
        (out_hidden_x3,) = bias.struct_info.shape
        assert out_hidden_x3 % 3 == 0, "bias shape should be divisible by 3"
        out_hidden = out_hidden_x3 // 3
        assert (
            out_hidden % num_heads == 0
        ), "output hidden size should be divisible by number of attention heads"
        head_size = out_hidden // num_heads

        assert (
            mask_index is not None
        ), "Attention import currently only supports required mask_index"
        mask_index_shape = mask_index.struct_info.shape
        assert (
            len(mask_index_shape) == 2
            and mask_index_shape[0] == batch_size
            and mask_index_shape[1] == seq_len
        ), "currently only support (batch_size, sequence_length) mask index"

        assert past is None, "past K, V state is not currently supported"
        assert extra_add is None, "extra add to QxK not currently supported"

        # split weight and biases and do the matmuls
        w_Q, w_K, w_V = relax.op.split(weight, 3, axis=1)
        b_Q, b_K, b_V = relax.op.split(bias, 3, axis=0)
        # need to merge batch dimensions since TVM matmul is 2D
        input_emb = relax.op.reshape(input_emb, (-1, 0))
        Q = relax.op.add(relax.op.linear_algebra.matmul(input_emb, w_Q), b_Q)
        K = relax.op.add(relax.op.linear_algebra.matmul(input_emb, w_K), b_K)
        V = relax.op.add(relax.op.linear_algebra.matmul(input_emb, w_V), b_V)

        # massage tensors in preparation for batched matmul
        def massage(tensor):
            tensor = relax.op.reshape(tensor, (batch_size, seq_len, num_heads, head_size))

            # (batch_size, num_heads, seq_len, head_size)
            tensor = relax.op.permute_dims(tensor, axes=[0, 2, 1, 3])

            # (batch_size * num_heads, seq_len, head_size)
            return relax.op.reshape(tensor, (-1, 0, 0))

        Q = massage(Q)
        K = massage(K)
        V = massage(V)

        K_present = relax.op.reshape(K, (batch_size, num_heads, seq_len, head_size))
        V_present = relax.op.reshape(V, (batch_size, num_heads, seq_len, head_size))
        present = relax.op.concat([K_present, V_present], axis=0)

        K_T = relax.op.permute_dims(K)
        att_scores = relax.op.linear_algebra.matmul(Q, K_T)
        score_dtype = att_scores.struct_info.dtype
        att_scores = relax.op.divide(
            att_scores, relax.const(_np.sqrt(head_size), dtype=att_scores.struct_info.dtype)
        )
        att_scores = relax.op.reshape(att_scores, (batch_size, num_heads, seq_len, seq_len))

        # build the attention mask
        att_mask = relax.op.astype(mask_index, score_dtype)
        att_mask = relax.op.expand_dims(att_mask, 1)
        att_mask = relax.op.subtract(relax.const(1, dtype=score_dtype), att_mask)
        att_mask = relax.op.multiply(att_mask, relax.const(-10000, dtype=score_dtype))

        # apply the mask
        att_scores = relax.op.add(att_scores, att_mask)
        att_scores = relax.op.reshape(att_scores, (batch_size * num_heads, seq_len, seq_len))

        att_probs = relax.op.nn.softmax(att_scores, axis=-1)

        output = relax.op.matmul(att_probs, V)
        output = relax.op.reshape(output, (-1, num_heads, 0, 0))
        output = relax.op.permute_dims(output, axes=[0, 2, 1, 3])
        output = relax.op.reshape(output, (0, 0, out_hidden))

        return relax.Tuple(bb.emit(output), present)


# pylint: enable=invalid-name, import-self, len-as-condition, unused-argument, too-many-lines


def _get_convert_map():
    return {
        "Concat": Concat,
        "MatMul": Matmul,
        "Add": Add,
        "Cast": Cast,
        "Gather": Gather,
        "Gemm": Gemm,
        "Mul": Mul,
        "Reshape": Reshape,
        "Shape": Shape,
        "Clip": Clip,
        "Div": Div,
        "Expand": Expand,
        "ReduceMean": ReduceMean,
        "Sigmoid": Sigmoid,
        "Slice": Slice,
        "Softmax": Softmax,
        "Transpose": Transpose,
        "Unsqueeze": Unsqueeze,
        "BiasGelu": BiasGelu,
        "Conv": Conv,
        "Equal": Equal,
        "Erf": Erf,
        "Not": Not,
        "Pow": Pow,
        "SkipLayerNormalization": SkipLayerNormalization,
        "Sqrt": Sqrt,
        "Squeeze": Squeeze,
        "Sub": Sub,
        "Tanh": Tanh,
        "Where": Where,
        "Constant": Constant,
        "Relu": Relu,
        "Split": Split,
        "EmbedLayerNormalization": EmbedLayerNormalization,
        "Attention": Attention,
    }


class ONNXGraphImporter:
    """A helper class for handling Relax expression copying from pb2.GraphProto.
    Definition: https://github.com/onnx/onnx/blob/main/onnx/onnx.proto

    Parameters
    ----------
    shape : dict of str to tuple, optional
        The input shape to the graph
    dtype : str or dict of str to str
        The input types to the graph
    target : tvm.target.Target
        The target device of the compiled functions when using the translator.
    """

    current = None

    def __init__(self, shape: Dict[str, Tuple], dtype: Union[str, Dict[str, str]], target: Target):
        self._nodes = {}
        self._inputs = {}
        self._num_input = 0
        self._shape = shape.copy() if shape else {}
        self._input_names = []
        self._dtype = dtype
        self.opset = None
        self._target = target
        self.bb = relax.BlockBuilder()  # pylint: disable=invalid-name

    def from_onnx(
        self, graph: onnx.onnx_ml_pb2.ModelProto, opset: int
    ) -> Tuple[IRModule, Dict[str, tvm.nd.array]]:
        """Construct Relax expressions from the ONNX graph.
        Onnx graph is a python protobuf object.

        #TODO (gigiblender): Handle model input name sanitization. This has been a problem
        in the Relay importer in the past and we should be careful to avoid it here.

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
            with self.bb.dataflow() as df:  # pylint: disable=invalid-name, unused-variable
                self.opset = opset
                self._parse_graph_initializers(graph)
                self._parse_graph_input(graph)
                self._check_for_unsupported_ops(graph)
                self._construct_nodes(graph)

                # now return the outputs
                outputs = [self._nodes[self._parse_value_proto(i)] for i in graph.output]
                outputs = outputs[0] if len(outputs) == 1 else relax.Tuple(outputs)

                # Create a function from our output expression and all input variables.
                param_list = [v for k, v in self._inputs.items()]
                output_var = self.bb.emit_output(outputs)
            self.bb.emit_func_output(output_var, params=param_list)
        return self.bb.get()

    def _parse_graph_initializers(self, graph: onnx.onnx_ml_pb2.GraphProto):
        """Parse network inputs to relax, aka parameters."""
        for init_tensor in graph.initializer:
            if not init_tensor.name.strip():
                raise ValueError("Tensor's name is required.")
            array = self._parse_array(init_tensor)
            self._nodes[init_tensor.name] = relax.const(array)

    def _parse_graph_input(self, graph: onnx.onnx_ml_pb2.GraphProto):
        """Parse model inputs to Relax parameters."""
        for i in graph.input:
            # from onnx v0.2, GraphProto.input has type ValueInfoProto,
            #  and the name is 'i.name'
            i_name, i_shape, d_type, i_shape_name = get_info(i)
            if i_name not in self._nodes:
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

    def _check_for_unsupported_ops(self, graph: onnx.onnx_ml_pb2.GraphProto):
        convert_map = _get_convert_map()
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

    def _construct_nodes(self, graph: onnx.onnx_ml_pb2.GraphProto):
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
            assert (
                len(outputs) <= outputs_num
            ), "Missing outputs during conversion. Expected {} but Got {} in {}.".format(
                len(outputs), outputs_num, op_name
            )

            if outputs_num == 1:
                self._nodes[outputs[0]] = op
            else:
                for k, i in zip(list(outputs), range(len(outputs))):
                    self._nodes[k] = op[i]

    def _parse_value_proto(self, value_proto: onnx.onnx_ml_pb2.GraphProto):
        """Parse ValueProto or raw str."""
        try:
            name = value_proto.name
        except AttributeError:
            name = value_proto
        return name

    def _parse_array(self, tensor_proto: onnx.onnx_ml_pb2.TensorProto) -> tvm.nd.array:
        np_array = get_numpy(tensor_proto).reshape(tuple(tensor_proto.dims))
        return tvm.nd.array(np_array)

    def _parse_attr(self, attr_proto: onnx.onnx_ml_pb2.AttributeProto) -> Dict[str, Any]:
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

    def _relay_input_adapter(self, inputs: List[relax.Var]) -> List[relay.Var]:
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
                    relay_vars.append(
                        relay.const(relax_var.data, dtype=relax_var.checked_type.dtype)
                    )
                else:
                    relay_vars.append(
                        relay.var(
                            relax_var.name_hint,
                            shape=shape_values,
                            dtype=relax_var.checked_type.dtype,
                        )
                    )
        return relay_vars

    def _relay_output_adapter(
        self,
        relax_inputs: List[Union[relax.Var, relax.Constant]],
        relay_inputs: List[Union[relay.Var, relay.Constant]],
        relay_output: relay.Expr,
    ) -> relax.Expr:
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
        prev_bb = relax.BlockBuilder._current  # pylint: disable=protected-access
        relax.BlockBuilder._current = None  # pylint: disable=protected-access
        relax_mod = testing.relay_translator.from_relay(function, self._target)
        # Restore the block builder used by the frontend.
        relax.BlockBuilder._current = prev_bb  # pylint: disable=protected-access

        # This dict is used by the Mapper mutator to replace the globar vars
        # in the relax_mod with global_vars registered with the in-use block builder.
        global_var_dict = {}
        for global_var, func in relax_mod.functions.items():
            if global_var.name_hint != "main":
                global_var_dict[global_var] = self.bb.add_func(func, global_var.name_hint)

        # This dict is used by the Mapper mutator to replace the relax vars
        # with the inputs.
        relax_input_dict = {}
        for relax_var in relax_inputs:
            if isinstance(relax_var, relax.Var):
                relax_input_dict[relax_var.name_hint] = relax_var

        @relax.expr_functor.mutator
        class Mapper(PyExprMutator):
            """Mutator to replace the global vars and relax vars in the relax_mod
            with the global vars registered with the in-use block builder and the
            relax vars with the inputs.
            """

            def visit_span(self, span: relax.Span):
                return span

            def visit_var_(
                self, var_node: Var
            ):  # pylint: disable=arguments-differ,arguments-renamed
                if var_node.name_hint in relax_input_dict:
                    return relax_input_dict[var_node.name_hint]
                return var_node

            def visit_global_var_(
                self, gv_node: GlobalVar
            ):  # pylint: disable=arguments-differ,arguments-renamed
                if gv_node in global_var_dict:
                    return global_var_dict[gv_node]
                return gv_node

        assert (
            len([f for f in relax_mod.functions.values() if isinstance(f, relax.Function)]) == 1
        ), "Expected only one Relax function in the module."
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

    def _convert_operator(
        self, op_name: str, inputs: List[relax.Function], attrs: Dict, opset: int
    ) -> relax.Function:
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
        convert_map = _get_convert_map()
        if op_name in convert_map:
            convert_class = convert_map[op_name]
            op_function = convert_class.get_converter(opset)
            # If the op_function is a subclass of Relay OnnxOpConverter then it is a relay op.
            if issubclass(convert_class, RelayOnnxOpConverter):
                relay_inputs = self._relay_input_adapter(inputs)
                # The op_function might change relay_inputs array. Use a copy of the inputs.
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


def from_onnx(
    model: onnx.onnx_ml_pb2.GraphProto,
    shape: Dict[str, Tuple] = None,
    dtype: str = "float32",
    opset: int = None,
    target: Union[str, Target] = "llvm",
) -> Tuple[IRModule, Dict]:
    """Convert a ONNX model into an equivalent Relax Function.
    ONNX graphs are represented as Python Protobuf objects.

    The current implementation assumes that the input model is after ONNX v1.1.0.

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
    # Error if the model version is below 1.1.0
    if model.ir_version < 3:
        raise ValueError(
            "Model IR version {} not supported. Must be at least after 1.1.0.".format(
                model.ir_version
            )
        )

    try:
        import onnx  # pylint: disable=import-outside-toplevel, redefined-outer-name

        if hasattr(onnx.checker, "check_model"):
            # try use onnx's own model checker before converting any model
            try:
                onnx.checker.check_model(model)
            except Exception as exception:  # pylint: disable=c-extension-no-member, broad-except
                # the checker is a bit violent about errors, so simply print warnings here
                warnings.warn(str(exception))
    except ImportError as exception:
        raise ImportError(
            "Unable to import onnx which is required {}".format(exception)
        ) from exception

    if isinstance(target, str):
        target = Target(target)
    g = ONNXGraphImporter(shape, dtype, target)
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
