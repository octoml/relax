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
# pylint: disable=unused-argument
"""
ONNX testcases
================
This file is a test script to test Relax ONNX frontend coverage.
"""

from typing import Optional, Dict

import numpy as np
import pytest

import tvm
import tvm.testing
from tvm import relax

import onnx
from onnx import helper
from onnx import TensorProto, ModelProto, ValueInfoProto
import onnxruntime


def generate_random_inputs(
    model: ModelProto, inputs: Optional[Dict[str, np.array]] = None
) -> Dict[str, np.array]:
    input_values = {}
    # Iterate through model inputs and extract their shape.
    for i in model.graph.input:
        if inputs is not None and i.name in inputs:
            input_values[i.name] = inputs[i.name]
            continue
        shape = []
        for dim in i.type.tensor_type.shape.dim:
            shape.append(dim.dim_value)

        # Extract datatype for the input.
        if i.type.tensor_type.elem_type:
            dtype = str(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[i.type.tensor_type.elem_type])
        else:
            dtype = "float32"

        # Generate random inputs for each input.
        if dtype == "bool":
            random_value = np.random.choice(a=[False, True], size=shape)
        else:
            random_value = np.random.normal(size=shape).astype(dtype)
        input_values[i.name] = random_value

    return input_values


def check_correctness(model: ModelProto, inputs: Optional[Dict[str, np.array]] = None) -> None:
    """Run an onnx model in both onnxruntime and TVM through our importer
       confirm that the results match. Otherwise, an exception will be raised.

    Parameters
    ----------
    model: ModelProto
        The input onnx model that should be tested.
    inputs: Optional[Dict[str, np.array]]
        An optional dictionary containing values for each input in the onnx model.
    """
    # If inputs are not provided, extract them from the onnx graph and produce random
    # values that we'll use for testing.
    inputs = generate_random_inputs(model, inputs)

    # Run the model through onnx to get the expected result.
    ort_session = onnxruntime.InferenceSession(model.SerializeToString())
    ort_output = ort_session.run([], inputs)

    # Convert the onnx model into relax through the onnx importer.
    tvm_model = relax.from_onnx(model)
    # Compile the relax graph into a VM then run.
    with tvm.transform.PassContext(opt_level=3):
        # TODO add target configuration.
        ex = relax.vm.build(tvm_model, target="llvm")
        vm = relax.VirtualMachine(ex, tvm.cpu())
    vm.set_input("main", **inputs)
    vm.invoke_stateful("main")
    tvm_output = vm.get_outputs("main")
    # Wrap as a list if there is only one output.
    if isinstance(tvm_output, tvm.nd.NDArray):
        tvm_output = [tvm_output]

    assert len(tvm_output) == len(ort_output), "Unequal number of outputs"

    for (tvm_out, ort_out) in zip(tvm_output, ort_output):
        # TODO Allow configurable tolerance.
        tvm.testing.assert_allclose(tvm_out.numpy(), ort_out, atol=1e-5)


def test_matmul():
    matmul_node = helper.make_node("MatMul", ["a", "b"], ["c"])

    graph = helper.make_graph(
        [matmul_node],
        "matmul_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [32, 32]),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, [32, 32])],
    )

    model = helper.make_model(graph, producer_name="matmul_test")
    check_correctness(model)


def test_concat():
    concat_node = helper.make_node("Concat", ["a", "b"], ["ab"], axis=0)

    graph = helper.make_graph(
        [concat_node],
        "concat_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 32]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 32]),
        ],
        outputs=[helper.make_tensor_value_info("ab", TensorProto.FLOAT, [2, 32])],
    )

    model = helper.make_model(graph, producer_name="concat_test")
    check_correctness(model)


def test_add():
    add_node = helper.make_node("Add", ["a", "b"], ["ab"])

    graph = helper.make_graph(
        [add_node],
        "add_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 32]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 32]),
        ],
        outputs=[helper.make_tensor_value_info("ab", TensorProto.FLOAT, [1, 32])],
    )

    model = helper.make_model(graph, producer_name="add_test")
    check_correctness(model)


def test_mul():
    mul_node = helper.make_node("Mul", ["a", "b"], ["ab"])

    graph = helper.make_graph(
        [mul_node],
        "mul_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 32]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 32]),
        ],
        outputs=[helper.make_tensor_value_info("ab", TensorProto.FLOAT, [1, 32])],
    )

    model = helper.make_model(graph, producer_name="mul_test")
    check_correctness(model)
    check_correctness(model)


def test_cast():
    cast_node = helper.make_node("Cast", ["a"], ["a_float"], to=TensorProto.FLOAT)

    graph = helper.make_graph(
        [cast_node],
        "cast_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.INT32, [1, 32]),
        ],
        outputs=[helper.make_tensor_value_info("a_float", TensorProto.FLOAT, [1, 32])],
    )

    model = helper.make_model(graph, producer_name="cast_test")
    check_correctness(model)


def test_gather():
    gather_node = helper.make_node("Gather", ["data", "indices"], ["y"], axis=0)

    graph = helper.make_graph(
        [gather_node],
        "gather_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [5, 4, 3, 2]),
            helper.make_tensor_value_info("indices", TensorProto.INT32, [3]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 4, 3, 2])],
    )

    model = helper.make_model(graph, producer_name="gather_test")
    input_values = {
        "data": np.random.randn(5, 4, 3, 2).astype("float32"),
        "indices": np.array([0, 1, 3]).astype("int32"),
    }
    check_correctness(model, inputs=input_values)


def test_gemm():
    gemm_node = helper.make_node(
        "Gemm", ["a", "b", "c"], ["y"], alpha=0.25, beta=0.35, transA=1, transB=1
    )

    graph = helper.make_graph(
        [gemm_node],
        "gemm_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [4, 3]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [5, 4]),
            helper.make_tensor_value_info("c", TensorProto.FLOAT, [1, 5]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 5])],
    )

    model = helper.make_model(graph, producer_name="gemm_test")
    check_correctness(model)


@pytest.mark.skip
def test_reshape():
    reshape_node = helper.make_node("Reshape", ["data", "shape"], ["reshaped"])

    graph = helper.make_graph(
        [reshape_node],
        "reshape_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [1, 32]),
            helper.make_tensor_value_info("shape", TensorProto.INT64, [1]),
        ],
        initializer=[helper.make_tensor("shape", TensorProto.INT64, [1], [-1])],
        outputs=[helper.make_tensor_value_info("reshaped", TensorProto.FLOAT, [32])],
    )
    input_values = {
        "data": np.random.randn(1, 32).astype("float32"),
    }
    model = helper.make_model(graph, producer_name="reshape_test")
    check_correctness(model, inputs=input_values)


def test_div():
    div_node = helper.make_node("Div", ["a", "b"], ["c"])

    graph = helper.make_graph(
        [div_node],
        "div_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [32, 32]),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, [32, 32])],
    )

    model = helper.make_model(graph, producer_name="div_test")
    check_correctness(model)


def test_sigmoid():
    sigmoid_node = helper.make_node("Sigmoid", ["a"], ["b"])

    graph = helper.make_graph(
        [sigmoid_node],
        "sigmoid_test",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32])],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [32, 32])],
    )

    model = helper.make_model(graph, producer_name="sigmoid_test")
    check_correctness(model)


def test_softmax():
    softmax_node = helper.make_node("Softmax", ["a"], ["b"])

    graph = helper.make_graph(
        [softmax_node],
        "softmax_test",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32, 32])],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [32, 32, 32])],
    )

    model = helper.make_model(graph, producer_name="softmax_test")
    check_correctness(model)


def test_transpose():
    transpose_node = helper.make_node("Transpose", ["a"], ["b"], perm=[1, 2, 0])

    graph = helper.make_graph(
        [transpose_node],
        "transpose_test",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32, 32])],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [32, 32, 32])],
    )

    model = helper.make_model(graph, producer_name="transpose_test")
    check_correctness(model)


def test_unsqueeze():
    unsqueeze_node = helper.make_node("Unsqueeze", ["a", "axes"], ["b"])

    graph = helper.make_graph(
        [unsqueeze_node],
        "unsqueeze",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32])],
        initializer=[helper.make_tensor("axes", TensorProto.INT64, [3], vals=[0, 2, 3])],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 32, 1, 1, 32])],
    )

    model = helper.make_model(graph, producer_name="unsqueeze_test")
    check_correctness(model)


def test_gelu():
    gelu_node = helper.make_node("Gelu", ["a"], ["b"], domain="com.microsoft")

    graph = helper.make_graph(
        [gelu_node],
        "gelu_test",
        inputs=[helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32])],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [32, 32])],
    )

    model = helper.make_model(graph, producer_name="gelu_test")
    check_correctness(model)


def test_bias_gelu():
    bias_gelu_node = helper.make_node("BiasGelu", ["a", "b"], ["c"], domain="com.microsoft")

    graph = helper.make_graph(
        [bias_gelu_node],
        "bias_gelu_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [32]),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, [32, 32])],
    )

    model = helper.make_model(graph, producer_name="bias_gelu_test")
    check_correctness(model)


def test_where():
    where_node = helper.make_node("Where", ["a", "b", "c"], ["d"])

    graph = helper.make_graph(
        [where_node],
        "where_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.BOOL, [32, 32]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("c", TensorProto.FLOAT, [32, 32]),
        ],
        outputs=[helper.make_tensor_value_info("d", TensorProto.FLOAT, [32, 32])],
    )

    model = helper.make_model(graph, producer_name="where_test")
    check_correctness(model)


def test_clip():
    clip_node = helper.make_node("Clip", ["input", "min", "max"], ["output"])

    graph = helper.make_graph(
        [clip_node],
        "clip_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [32, 64]),
            helper.make_tensor_value_info("min", TensorProto.FLOAT, ()),
            helper.make_tensor_value_info("max", TensorProto.FLOAT, ()),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, [32, 64])],
    )

    model = helper.make_model(graph, producer_name="clip_test")
    check_correctness(model)


def test_equal():
    equal_node = helper.make_node("Equal", ["a", "b"], ["output"])

    graph = helper.make_graph(
        [equal_node],
        "equal_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [32, 32]),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.BOOL, [32, 32])],
    )

    model = helper.make_model(graph, producer_name="equal_test")
    check_correctness(
        model, {"a": np.zeros([32, 32], dtype="float32"), "b": np.zeros([32, 32], dtype="float32")}
    )
    check_correctness(
        model, {"a": np.ones([32, 32], dtype="float32"), "b": np.zeros([32, 32], dtype="float32")}
    )
    check_correctness(model)


def test_shape():
    shape_node = helper.make_node("Shape", ["data"], ["output"])

    graph = helper.make_graph(
        [shape_node],
        "shape_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [3, 4, 5, 6]),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.INT64, [4])],
    )

    model = helper.make_model(graph, producer_name="shape_test")
    check_correctness(model)


def test_not():
    not_node = helper.make_node("Not", ["x"], ["y"])
    shape = [3, 4, 5, 6]
    graph = helper.make_graph(
        [not_node],
        "not_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.BOOL, shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.BOOL, shape)],
    )

    model = helper.make_model(graph, producer_name="not_test")
    check_correctness(model, {"x": np.zeros(shape, dtype="bool")})
    check_correctness(model, {"x": np.ones(shape, dtype="bool")})


def test_tanh():
    tanh_node = helper.make_node("Tanh", ["x"], ["y"])
    shape = [9, 8, 7, 6]
    graph = helper.make_graph(
        [tanh_node],
        "tanh_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )

    model = helper.make_model(graph, producer_name="tanh_test")
    check_correctness(model)


def test_sqrt():
    sqrt_node = helper.make_node("Sqrt", ["x"], ["y"])
    shape = [32, 32]
    graph = helper.make_graph(
        [sqrt_node],
        "sqrt_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )

    model = helper.make_model(graph, producer_name="sqrt_test")
    check_correctness(model)


def test_relu():
    relu_node = helper.make_node("Relu", ["x"], ["y"])
    shape = [32, 32]
    graph = helper.make_graph(
        [relu_node],
        "relu_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )

    model = helper.make_model(graph, producer_name="relu_test")
    check_correctness(model)


def test_conv():
    conv_node = helper.make_node("Conv", ["x", "w", "b"], ["y"])
    nchw_shape = [3, 12, 32, 32]
    graph = helper.make_graph(
        [conv_node],
        "conv_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, nchw_shape),
            helper.make_tensor_value_info("w", TensorProto.FLOAT, [4, 12, 3, 3]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [4]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 4, 30, 30])],
    )

    model = helper.make_model(graph, producer_name="conv_test")
    check_correctness(model)


def test_pow():
    pow_node = helper.make_node("Pow", ["x", "y"], ["z"])
    shape = [32, 32]
    graph = helper.make_graph(
        [pow_node],
        "pow_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("y", TensorProto.FLOAT, shape),
        ],
        outputs=[helper.make_tensor_value_info("z", TensorProto.FLOAT, shape)],
    )

    model = helper.make_model(graph, producer_name="pow_test")
    check_correctness(model)


def test_erf():
    erf_node = helper.make_node("Erf", ["x"], ["y"])
    shape = [32, 32]
    graph = helper.make_graph(
        [erf_node],
        "erf_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )

    model = helper.make_model(graph, producer_name="erf_test")
    check_correctness(model)


def test_cumsum():
    cumsum_node = helper.make_node("CumSum", ["x", "axis"], ["y"])
    shape = [32, 32]
    type_proto = onnx.TypeProto()
    tensor_type_proto = type_proto.tensor_type
    tensor_type_proto.elem_type = TensorProto.INT64
    graph = helper.make_graph(
        [cumsum_node],
        "cumsum_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        initializer=[helper.make_tensor("axis", TensorProto.INT64, (), [1])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )

    model = helper.make_model(graph, producer_name="cumsum_test")
    check_correctness(model)


if __name__ == "__main__":
    test_matmul()
    test_concat()
    test_add()
    test_mul()
    test_cast()
    test_gather()
    test_gemm()
    test_equal()
    test_not()
    test_tanh()
    test_sqrt()
    test_relu()
    test_clip()
    test_conv()
    test_pow()
    test_erf()
    test_cumsum()

    # TODO, still has issues
    # test_reshape()
    test_div()
    test_sigmoid()
    test_softmax()
    test_transpose()
    test_unsqueeze()
    # test_shape()
