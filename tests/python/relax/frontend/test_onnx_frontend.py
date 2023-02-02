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
from onnx import helper, TensorProto, ModelProto, ValueInfoProto, mapping
import onnxruntime


def generate_random_inputs(
    model: ModelProto, inputs: Optional[Dict[str, np.array]] = None
) -> Dict[str, np.array]:
    input_values = {}
    # Iterate through model inputs and extract their shape.
    for i in model.graph.input:
        if inputs is not None and i.name in inputs and inputs[i.name] is not None:
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


def check_correctness(model: ModelProto, inputs: Optional[Dict[str, np.array]] = None, opset: int = None) -> None:
    """Run an onnx model in both onnxruntime and TVM through our importer
       confirm that the results match. Otherwise, an exception will be raised.

    Parameters
    ----------
    model: ModelProto
        The input onnx model that should be tested.
    inputs: Optional[Dict[str, np.array]]
        An optional dictionary containing values for each input in the onnx model.
    opset: int
        The opset version to use for the onnx importer.
    """
    if opset is not None:
        model.opset_import[0].version = opset

    # If inputs are not provided, extract them from the onnx graph and produce random
    # values that we'll use for testing.
    inputs = generate_random_inputs(model, inputs)

    # Run the model through onnx to get the expected result.
    ort_session = onnxruntime.InferenceSession(model.SerializeToString())
    ort_output = ort_session.run([], inputs)

    # Convert the onnx model into relax through the onnx importer.
    tvm_model = relax.from_onnx(model, opset=opset)
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
        # Sometimes None is used to indicate an unused output.
        if ort_out is not None:
            tvm.testing.assert_allclose(tvm_out.numpy(), ort_out, atol=1e-5)


@pytest.mark.parametrize("dynamic", [True, False])
def test_matmul(dynamic):
    matmul_node = helper.make_node("MatMul", ["a", "b"], ["c"])

    tensor_size = [32, 32]
    if dynamic:
        tensor_size = ["?", "?"]

    graph = helper.make_graph(
        [matmul_node],
        "matmul_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, tensor_size),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, tensor_size),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, tensor_size)],
    )

    model = helper.make_model(graph, producer_name="matmul_test")
    inputs = None
    if dynamic:
        inputs = {
            "a": np.random.normal(size=(32, 48)).astype("float32"),
            "b": np.random.normal(size=(48, 64)).astype("float32"),
        }
    check_correctness(model, inputs)


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


@pytest.mark.parametrize("dynamic", [True, False])
def test_reshape(dynamic):
    reshape_node = helper.make_node("Reshape", ["data", "shape"], ["reshaped"])

    data_shape = ["?", 32, 32, 8] if dynamic else [7, 32, 32, 8]
    output_shape = ["?", "?"] if dynamic else [7, 8192]

    graph = helper.make_graph(
        [reshape_node],
        "reshape_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
        ],
        initializer=[helper.make_tensor("shape", TensorProto.INT64, [2], [-1, 8192])],
        outputs=[helper.make_tensor_value_info("reshaped", TensorProto.FLOAT, output_shape)],
    )
    input_values = {
        "data": np.random.randn(7, 32, 32, 8).astype("float32"),
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


def test_squeeze():
    squeeze_node = helper.make_node("Squeeze", ["x", "axis"], ["y"])
    shape = [1, 32, 1, 32]
    graph = helper.make_graph(
        [squeeze_node],
        "squeeze_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
        ],
        initializer=[helper.make_tensor("axis", TensorProto.INT64, [2], [0, 2])],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [32, 32])],
    )

    model = helper.make_model(graph, producer_name="squeeze_test")
    check_correctness(model)


def test_const():
    shape = [32, 32]
    const_node = helper.make_node(
        "Constant",
        [],
        ["y"],
        value=helper.make_tensor(
            "value", TensorProto.FLOAT, shape, np.random.rand(*shape).astype(np.float32).flatten()
        ),
    )
    graph = helper.make_graph(
        [const_node],
        "const_test",
        inputs=[],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, shape)],
    )

    model = helper.make_model(graph, producer_name="const_test")
    check_correctness(model)


def test_sub():
    sub_node = helper.make_node("Sub", ["x", "y"], ["z"])
    shape = [32, 16]
    graph = helper.make_graph(
        [sub_node],
        "sub_test",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, shape),
            helper.make_tensor_value_info("y", TensorProto.FLOAT, shape),
        ],
        outputs=[helper.make_tensor_value_info("z", TensorProto.FLOAT, shape)],
    )

    model = helper.make_model(graph, producer_name="sub_test")
    check_correctness(model)


def test_layer_norm():
    layer_norm_node = helper.make_node(
        "LayerNormalization", ["a", "b", "c"], ["d", "mean", "std_dev"], epsilon=1e-12
    )

    graph = helper.make_graph(
        [layer_norm_node],
        "layer_norm_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [32]),
            helper.make_tensor_value_info("c", TensorProto.FLOAT, [32]),
        ],
        outputs=[
            helper.make_tensor_value_info("d", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("mean", TensorProto.FLOAT, [32]),
            helper.make_tensor_value_info("std_dev", TensorProto.FLOAT, [32]),
        ],
    )

    model = helper.make_model(graph, producer_name="layer_norm_test")
    check_correctness(model)


# TODO Enable dynamism
@pytest.mark.parametrize("dynamic", [False])
def test_skiplayernormalization(dynamic):
    """test_skiplayernormalization"""

    def verify_skiplayernormalization(input_, skip, gamma, beta, bias):
        node = onnx.helper.make_node(
            "SkipLayerNormalization",
            inputs=["input", "skip", "gamma", "beta", "bias"],
            outputs=["output", "mean", "std_dev"],
            domain="com.microsoft",
        )

        node.attribute.append(onnx.helper.make_attribute("epsilon", 1e-4))

        input_shape = list(input_.shape)
        skip_shape = list(skip.shape)
        gamma_shape = list(gamma.shape)
        beta_shape = list(beta.shape)
        bias_shape = list(bias.shape)
        output_shape = list(input_.shape)
        mean_shape = list([1])
        std_dev_shape = list([1])
        if dynamic:
            input_shape = ["?" for _ in range(len(input_.shape))]
            skip_shape = ["?" for _ in range(len(skip.shape))]
            gamma_shape = ["?" for _ in range(len(gamma.shape))]
            beta_shape = ["?" for _ in range(len(beta.shape))]
            bias_shape = ["?" for _ in range(len(bias.shape))]
            output_shape = ["?" for _ in range(len(input_.shape))]

        graph = helper.make_graph(
            [node],
            "skiplayernormalization_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
                helper.make_tensor_value_info("skip", TensorProto.FLOAT, skip_shape),
                helper.make_tensor_value_info("gamma", TensorProto.FLOAT, gamma_shape),
                helper.make_tensor_value_info("beta", TensorProto.FLOAT, beta_shape),
                helper.make_tensor_value_info("bias", TensorProto.FLOAT, bias_shape),
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape),
                helper.make_tensor_value_info("mean", TensorProto.FLOAT, mean_shape),
                helper.make_tensor_value_info("std_dev", TensorProto.FLOAT, std_dev_shape)
            ],
        )

        model = helper.make_model(graph, producer_name="skiplayernormalization_test")
        check_correctness(model, inputs={"input": input_, "skip": skip, "gamma": gamma, "beta": beta, "bias": bias})

    hidden_size = 384
    batch_size = 4
    sequence_length = 4

    dtype = "float32"
    input_array = np.random.random((batch_size, sequence_length, hidden_size)).astype(dtype)
    skip = np.random.random((batch_size, sequence_length, hidden_size)).astype(dtype)
    gamma = np.random.uniform(0.5, 0.7, hidden_size).astype(dtype)
    beta = np.random.randn(hidden_size).astype(dtype) * 0.1
    bias = np.random.randn(hidden_size).astype(dtype)

    verify_skiplayernormalization(input_array, skip, gamma, beta, bias)


def test_embedlayernormalization():
    """test_embedlayernormalization"""

    def verify_embedlayernormalization(
            input_ids,
            segment_ids,
            word_embedding,
            position_embedding,
            segment_embedding,
            gamma,
            beta,
    ):
        node = onnx.helper.make_node(
            "EmbedLayerNormalization",
            inputs=[
                "input_ids",
                "" if segment_ids is None else "segment_ids",
                "word_embedding",
                "position_embedding",
                "" if segment_embedding is None else "segment_embedding",
                "gamma",
                "beta",
            ],
            outputs=["output", "mask_index"],
            domain="com.microsoft",
        )

        node.attribute.append(onnx.helper.make_attribute("epsilon", 1e-4))

        segment_ids_shape = [] if segment_ids is None else segment_ids.shape
        segment_embedding_shape = [] if segment_embedding is None else segment_embedding.shape

        graph = helper.make_graph(
            [node],
            "embedlayernormalization_test",
            inputs=[
                helper.make_tensor_value_info(
                    "input_ids", TensorProto.INT32, list(input_ids.shape)
                ),
                helper.make_tensor_value_info("segment_ids", TensorProto.INT32, segment_ids_shape),
                helper.make_tensor_value_info(
                    "word_embedding", TensorProto.FLOAT, list(word_embedding.shape)
                ),
                helper.make_tensor_value_info(
                    "position_embedding", TensorProto.FLOAT, list(position_embedding.shape)
                ),
                helper.make_tensor_value_info(
                    "segment_embedding", TensorProto.FLOAT, segment_embedding_shape
                ),
                helper.make_tensor_value_info("gamma", TensorProto.FLOAT, list(gamma.shape)),
                helper.make_tensor_value_info("beta", TensorProto.FLOAT, list(beta.shape)),
            ],
            outputs=[
                helper.make_tensor_value_info(
                    "output", TensorProto.FLOAT, list((batch_size, sequence_length, hidden_size))
                ),
                helper.make_tensor_value_info("mask_index", TensorProto.INT32, [batch_size]),
            ],
        )

        model = helper.make_model(graph, producer_name="embedlayernormalization_test")

        inputs = {
            "input_ids": input_ids,
            "segment_ids": segment_ids,
            "word_embedding": word_embedding,
            "position_embedding": position_embedding,
            "segment_embedding": segment_embedding,
            "gamma": gamma,
            "beta": beta
        }
        check_correctness(model, inputs=inputs)

        # TODO(@anwang2009): onnxruntime v1.9.0 requires empty list for optional argument,
        # but v1.10.0+ requires None instead.
        # verify_with_ort_with_inputs(
        #     model,
        #     [
        #         input_ids,
        #         np.empty(0, dtype="int32") if segment_ids is None else segment_ids,
        #         word_embedding,
        #         position_embedding,
        #         np.empty(0, dtype="float32") if segment_embedding is None else segment_embedding,
        #         gamma,
        #         beta,
        #     ],
        #     [
        #         (batch_size, sequence_length, hidden_size),
        #         batch_size,
        #     ],
        #     target=target,
        #     dev=dev,
        #     rtol=1e-4,
        #     atol=1e-4,
        # )

    hidden_size = 384
    batch_size = 4
    sequence_length = 3
    vocab_size = 5

    input_ids = np.full((batch_size, sequence_length), 3).astype("int32")
    segment_ids = np.zeros((batch_size, sequence_length)).astype("int32")
    word_embedding = np.full((vocab_size, hidden_size), 1).astype("float32")
    position_embedding = np.full((sequence_length, hidden_size), 2).astype("float32")
    segment_embedding = np.full((vocab_size, hidden_size), 3).astype("float32")

    gamma = np.random.uniform(0.5, 0.7, hidden_size).astype("float32")
    beta = np.random.randn(hidden_size).astype("float32") * 0.1

    verify_embedlayernormalization(
        input_ids, segment_ids, word_embedding, position_embedding, segment_embedding, gamma, beta
    )

    # Test with undefined segment embedding
    verify_embedlayernormalization(
        input_ids, None, word_embedding, position_embedding, None, gamma, beta
    )


def create_reduce_test_parameters():
    output = []
    for value in [True, False]:
        output.append(("ReduceMax", value))
        output.append(("ReduceMean", value))
        output.append(("ReduceMin", value))
        output.append(("ReduceProd", value))
        output.append(("ReduceSum", value))
        output.append(("ReduceSumSquare", value))
        output.append(("ReduceLogSum", value))
        output.append(("ReduceLogSumExp", value))
        output.append(("ReduceL1", value))
        output.append(("ReduceL2", value))
    return output


@pytest.mark.parametrize("func, dynamic", create_reduce_test_parameters())
def test_all_reduce_funcs(func, dynamic):
    """test_all_reduce_funcs"""

    def verify_reduce_func(func, data, axis, keepdims):
        inshape = data.shape
        outshape = np.sum(data, axis=axis, keepdims=keepdims == 1).shape

        if axis:
            node = onnx.helper.make_node(
                func, inputs=["x"], outputs=["y"], axes=axis, keepdims=keepdims
            )
        else:
            node = onnx.helper.make_node(func, inputs=["x"], outputs=["y"], keepdims=keepdims)

        if dynamic:
            in_list = ["?" for _ in range(len(inshape))]
            out_list = ["?" for _ in range(len(outshape))]
        else:
            in_list = list(inshape)
            out_list = list(outshape)
        graph = helper.make_graph(
            [node],
            "reduce_test",
            inputs=[helper.make_tensor_value_info("x", TensorProto.FLOAT, in_list)],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, out_list)]
        )

        model = helper.make_model(graph, producer_name="reduce_test")

        inputs_dict = {"x": data}
        check_correctness(model, inputs_dict, opset=11)

    verify_reduce_func(func, np.array(1.0).astype(np.float32), axis=None, keepdims=False)

    for keepdims in [True, False]:
        verify_reduce_func(
            func, np.random.randn(3, 2, 2).astype(np.float32), axis=None, keepdims=keepdims
        )

        verify_reduce_func(
            func, np.random.randn(3, 2, 3).astype(np.float32), axis=None, keepdims=keepdims
        )

        verify_reduce_func(
            func, np.random.randn(3, 3, 3).astype(np.float32), axis=(1,), keepdims=keepdims
        )

        verify_reduce_func(
            func, np.random.randn(3, 3, 3, 1).astype(np.float32), axis=(1, 2), keepdims=keepdims
        )

        verify_reduce_func(
            func, np.random.randn(3, 3, 3, 1).astype(np.float32), axis=(1,), keepdims=keepdims
        )

        verify_reduce_func(
            func, np.random.randn(1, 3, 4, 1).astype(np.float32), axis=(1,), keepdims=keepdims
        )


@pytest.mark.parametrize("dynamic", [False, True])
def test_expand(dynamic):
    """test_expand"""
    if dynamic:
        # TODO: Support dynamic shape for Expand
        pytest.skip("Dynamic expand is not supported yet")

    def _test_expand(name, data, shape, ref_data):
        shape_array = np.array(shape)
        shape_node = onnx.helper.make_node(
            "Constant",
            inputs=[],
            outputs=["shape"],
            value=onnx.helper.make_tensor(
                name="const_tensor",
                data_type=onnx.TensorProto.INT64,
                dims=shape_array.shape,
                vals=shape_array.flatten().astype("int64"),
            ),
        )
        expand_node = helper.make_node("Expand", ["in", "shape"], ["out"])

        in_shape = list(data.shape)
        out_shape = list(ref_data.shape)
        if dynamic:
            in_shape = ["?" for _ in range(len(in_shape))]
            out_shape = ["?" for _ in range(len(out_shape))]
        graph = helper.make_graph(
            [shape_node, expand_node],
            "expand_teint64st",
            inputs=[helper.make_tensor_value_info("in", TensorProto.FLOAT, in_shape)],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, out_shape)],
        )

        model = helper.make_model(graph, producer_name=name)
        check_correctness(model, inputs={"in": data})

    in_shape = (3, 1)
    shape = (3, 4)
    data = np.random.uniform(size=in_shape).astype(np.float32)
    ref_data = np.tile(data, 4)
    _test_expand("expand_with_dim_unchanged_test", data, shape, ref_data)

    in_shape = (3, 1)
    shape = (2, 1, 6)
    data = np.random.uniform(size=in_shape).astype(np.float32)
    ref_data = data * np.ones(shape, dtype=np.float32)
    _test_expand("expand_larger_target_shape_test", data, shape, ref_data)

    in_shape = (1, 1)
    shape = (3,)
    data = np.random.uniform(size=in_shape).astype(np.float32)
    ref_data = data * np.ones(shape, dtype=np.float32)
    _test_expand("expand_smaller_target_shape_test", data, shape, ref_data)


def test_constantofshape():
    """test_constantofshape"""

    def verify_constantofshape(input_dim, value, dtype):
        fill_node = helper.make_node(
            "ConstantOfShape",
            ["input"],
            ["output"],
            value=helper.make_tensor(
                "value", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], (1,), (value,)
            ),
        )

        inputs = [helper.make_tensor_value_info("input", TensorProto.INT64, [len(input_dim)])]

        graph = helper.make_graph(
            [fill_node],
            "fill_test",
            inputs,
            initializer=[helper.make_tensor("input", TensorProto.INT64, [len(input_dim)], np.asarray(input_dim).astype("int64"))],
            outputs=[
                helper.make_tensor_value_info(
                    "output", mapping.NP_TYPE_TO_TENSOR_TYPE[np.dtype(dtype)], input_dim
                )
            ],
        )

        model = helper.make_model(graph, producer_name="fill_test")
        input_np = np.array(input_dim).astype("int64")
        check_correctness(model, inputs={"input": input_np})

    verify_constantofshape((2, 3, 4, 5), 10, "float32")
    verify_constantofshape((3, 3), 0, "int32")
    verify_constantofshape((1, 2, 3), -1, "float32")


def test_slice():
    def verify_slice(data_shape, output_shape, starts, ends, axes=None, steps=None):
        if isinstance(starts, list):
            starts = np.array(starts, "int64")
        if isinstance(ends, list):
            ends = np.array(ends, "int64")
        if isinstance(axes, list):
            axes = np.array(axes, "int64")
        if isinstance(steps, list):
            steps = np.array(steps, "int64")

        
        slice_inputs=["x", "starts", "ends"]
        initializer=[
            helper.make_tensor("starts", TensorProto.INT64, starts.shape, starts),
            helper.make_tensor("ends", TensorProto.INT64, ends.shape, ends),
        ]

        if axes is not None:
            initializer.append(helper.make_tensor("axes", TensorProto.INT64, axes.shape, axes))
            slice_inputs.append("axes")
        if steps is not None:
            initializer.append(helper.make_tensor("steps", TensorProto.INT64, steps.shape, steps))
            slice_inputs.append("steps")

        slice_node = helper.make_node(
            "Slice",
            inputs=slice_inputs,
            outputs=["y"]
        )

        graph = helper.make_graph(
            [slice_node],
            "slice_test",
            inputs=[
                helper.make_tensor_value_info("x", TensorProto.FLOAT, data_shape),
            ],
            outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, output_shape)],
            initializer=initializer,
        )

        model = helper.make_model(graph, producer_name="slice_test")
        check_correctness(model)

    # Test with all parameters set.
    verify_slice([20, 10, 5], [3, 10, 5], starts=[0, 0], ends=[3, 10], axes=[0, 1], steps=[1, 1])
    # Test with default axes and steps.
    verify_slice([20, 10, 5], [3, 10, 5], starts=[0, 0], ends=[3, 10])
    # Test with negative steps.
    verify_slice([20, 10, 5], [19, 3, 2], starts=[20, 10, 4], ends=[0, 0, 1], steps=[-1, -3, -2], axes=[0, 1, 2])


# TODO Enable dynamism
@pytest.mark.parametrize("dynamic", [True, False])
def test_attention(dynamic):
    """test_attention"""

    def verify_attention(input_, weight, bias, mask_index, num_heads):
        node = onnx.helper.make_node(
            "Attention",
            inputs=["input", "weight", "bias", "mask_index"],
            outputs=["output", "present"],
            domain="com.microsoft",
            num_heads=num_heads,
        )

        present_output_shape = (2, batch_size, num_heads, sequence_length, head_size)

        input_shape = list(input_.shape)
        weight_shape = list(weight.shape)
        bias_shape = list(bias.shape)
        mask_shape = list(mask_index.shape)
        output_shape = list(input_.shape)
        present_shape = list(present_output_shape)
        if dynamic:
            input_shape = ["?" for _ in range(len(input_.shape))]
            weight_shape = ["?" for _ in range(len(weight.shape))]
            bias_shape = ["?" for _ in range(len(bias.shape))]
            mask_shape = ["?" for _ in range(len(mask_index.shape))]
            output_shape = ["?" for _ in range(len(input_.shape))]
            present_shape = ["?" for _ in range(len(present_output_shape))]

        graph = helper.make_graph(
            [node],
            "attention_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, input_shape),
                helper.make_tensor_value_info("weight", TensorProto.FLOAT, weight_shape),
                helper.make_tensor_value_info("bias", TensorProto.FLOAT, bias_shape),
                helper.make_tensor_value_info(
                    "mask_index", TensorProto.INT32, mask_shape
                ),
            ],
            outputs=[
                helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape),
                helper.make_tensor_value_info(
                    "present", TensorProto.FLOAT, present_shape
                ),
            ],
        )

        model = helper.make_model(graph, producer_name="attention_test")

        check_correctness(model, inputs={"input": input_, "weight": weight, "bias": bias, "mask_index": mask_index})
        # "present" output should be nullptr when the "past" input isn't included,
        # but ort requires an output shape to be specified?
        # verify_with_ort_with_inputs(
        #     model,
        #     [input_, weight, bias, mask_index],
        #     [input_.shape, present_output_shape],
        #     target=target,
        #     dev=dev,
        #     rtol=1e-4,
        #     atol=1e-4,
        # )

    hidden_size = 384
    batch_size = 4
    sequence_length = 4
    num_heads = 12
    head_size = 32

    dtype = "float32"
    input_array = np.random.random((batch_size, sequence_length, hidden_size)).astype(dtype)
    weight = np.random.normal(size=(hidden_size, 3 * hidden_size)).astype(dtype) * 0.1
    bias = np.random.randn(3 * hidden_size).astype(dtype)
    mask_index = np.full((batch_size, sequence_length), 1).astype("int32")

    verify_attention(input_array, weight, bias, mask_index, num_heads)


@pytest.mark.parametrize("dynamic", [True, False])
def test_pad(dynamic):
    """test_pad"""

    if dynamic:
        pytest.skip("Dynamic pad not supported")

    def verify_pad(input_shape, pads, mode="constant", value=0.0):
        indata = np.random.normal(size=input_shape).astype(np.float32)
        #  numpy expect result
        len_dim = len(pads) // 2
        np_pads = [(pads[i], pads[i + len_dim]) for i in range(len_dim)]
        pads = np.array(pads)
        #  onnx graph
        if mode in ["edge", "reflect"]:
            outdata = np.pad(indata, pad_width=np_pads, mode=mode)
            node = helper.make_node("Pad", inputs=["input", "pads"], outputs=["output"], mode=mode)
            graph = helper.make_graph(
                [node],
                "pad_test",
                inputs=[
                    helper.make_tensor_value_info("input", TensorProto.FLOAT, list(indata.shape))
                ],
                initializer=[helper.make_tensor("pads", TensorProto.INT64, (len(pads),), pads)],
                outputs=[
                    helper.make_tensor_value_info("output", TensorProto.FLOAT, list(outdata.shape))
                ],
            )
        else:
            outdata = np.pad(indata, pad_width=np_pads, mode="constant", constant_values=value)
            node = helper.make_node(
                "Pad",
                inputs=["input", "pads", "constant_value"],
                outputs=["output"],
                mode="constant",
            )
            graph = helper.make_graph(
                [node],
                "pad_test",
                inputs=[
                    helper.make_tensor_value_info("input", TensorProto.FLOAT, list(indata.shape))
                ],
                initializer=[
                    helper.make_tensor("pads", TensorProto.INT64, (len(pads),), pads),
                    helper.make_tensor("constant_value", TensorProto.FLOAT, (1,), [value]),
                ],
                outputs=[
                    helper.make_tensor_value_info("output", TensorProto.FLOAT, list(outdata.shape))
                ],
            )
        model = helper.make_model(graph, producer_name="pad_test")
        check_correctness(model)

    verify_pad((2, 2), [0, 1, 0, 0], "constant", 0.0)
    verify_pad((2, 3), [1, 0, 0, 1], "constant", 0.0)
    verify_pad((3, 2), [0, 0, 1, 0], "constant", 5.0)
    verify_pad((1, 3, 4, 5), [0, 1, 1, 1, 0, 0, 1, 1], "reflect")


@pytest.mark.parametrize("dynamic", [True, False])
def test_pad_constant_value(dynamic):
    """test_pad_constant_value"""
    if dynamic:
        pytest.skip("Dynamic shape is not supported yet")

    def verify_pad_constant_value(constant_value):
        tensor_shape = [1, 2, 257, 126]
        output_shape = [1, 2, 258, 128]
        pad_values = [0, 0, 0, 2, 0, 0, 1, 0]
        graph_inputs = [
            helper.make_tensor_value_info("input", TensorProto.FLOAT, tensor_shape),
        ]
        initializer = [
            helper.make_tensor("pads", TensorProto.INT64, dims=[len(pad_values)], vals=pad_values),
            helper.make_tensor("constant_value", TensorProto.FLOAT, dims=[], vals=[constant_value])]
        graph_outputs = [helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)]

        pad_node = helper.make_node(
            "Pad", ["input", "pads", "constant_value"], ["output"], mode="constant"
        )
        graph_nodes = [pad_node]
        graph = helper.make_graph(
            graph_nodes,
            "test_pad_constant_value",
            inputs=graph_inputs,
            outputs=graph_outputs,
            initializer=initializer
        )
        model = helper.make_model(
            graph,
            producer_name="test_pad_constant_value",
        )
        check_correctness(model)

    verify_pad_constant_value(0)


@pytest.mark.parametrize("fp_arith", [np.float16, np.float32])
@pytest.mark.parametrize("dynamic", [True, False])
def test_split(fp_arith, dynamic):
    """test_split"""

    def verify_split(indata_shape, outdata_shapes, split, axis=0, pass_split=True, opset=11):
        indata = np.random.normal(size=indata_shape).astype(fp_arith)
        input_names = ["input"]
        initializer = []

        if split:
            split_index = range(len(split))
        else:
            split_index = range(len(outdata_shapes))

        indata_shape = list(indata.shape)
        if dynamic:
            indata_shape = ["?" for _ in range(len(indata.shape))]
            outdata_shapes = [["?" for _ in range(len(o))] for o in outdata_shapes]

        inputs = [
            helper.make_tensor_value_info(
                "input", mapping.NP_TYPE_TO_TENSOR_TYPE[indata.dtype], indata_shape
            )
        ]

        if pass_split:
            if opset >= 13:
                np_split = np.array(split).astype(np.int64)
                initializer.append(
                    helper.make_tensor("split", TensorProto.INT64, list(np_split.shape), np_split)
                )
        node = helper.make_node(
            "Split",
            inputs=input_names,
            outputs=[f"output_{i}" for i in range(len(split_index))],
            axis=axis,
        )

        if pass_split and opset < 13:
            split_attr = helper.make_attribute("split", split)
            node.attribute.append(split_attr)

        graph = helper.make_graph(
            [node],
            "split_test",
            inputs=inputs,
            initializer=initializer,
            outputs=[
                helper.make_tensor_value_info(
                    f"output_{i}", mapping.NP_TYPE_TO_TENSOR_TYPE[indata.dtype], list(outdata_shapes[i])
                )
                for i in range(len(split_index))
            ],
        )
        model = helper.make_model(graph, producer_name="split_test")
        check_correctness(model, inputs={"input": indata}, opset=opset)

    # 1D
    verify_split(6, [[2], [2], [2]], [2, 2, 2], 0)
    verify_split(
        6, [[2], [2], [2]], [2, 2, 2], 0, False
    )
    verify_split(6, [[2], [1], [3]], [2, 1, 3], 0)
    verify_split(
        6, [[2], [1], [3]], [2, 1, 3], 0, opset=13
    )
    # 2D
    verify_split(
        (4, 4),
        [[2, 2], [2, 2]],
        [2, 2],
        1,
    )
    verify_split(
        (4, 4),
        [[2, 2], [2, 2]],
        [2, 2],
        1,
        opset=13,
    )
    # Split evenly (unstack)
    verify_split(3, [[1], [1], [1]], False, 0, False)
    # Split a single value to a single value
    verify_split(1, [[1]], [1], pass_split=True)
    # Test that the default case modifies nothing when split list has length one
    verify_split((1, 2), [[2]], [2], 1)
    verify_split((1, 2), [[2]], [1], 0)


@pytest.mark.skip
@pytest.mark.parametrize("dynamic", [True, False])
def test_tile(dynamic):
    """test_tile"""

    def verify_tile_v6(indata, repeats, outdata):
        node = helper.make_node("Tile", inputs=["input", "repeats"], outputs=["out"])

        indata_shape = list(indata.shape)
        repeats_shape = list(repeats.shape)
        outdata_shape = list(outdata.shape)
        if dynamic:
            indata_shape = ["?" for _ in range(len(indata_shape))]
            repeats_shape = ["?" for _ in range(len(repeats_shape))]
            outdata_shape = ["?" for _ in range(len(outdata_shape))]

        graph = helper.make_graph(
            [node],
            "tile_test",
            inputs=[
                helper.make_tensor_value_info("input", TensorProto.FLOAT, indata_shape),
                helper.make_tensor_value_info("repeats", TensorProto.INT64, repeats_shape)
            ],
            outputs=[helper.make_tensor_value_info("out", TensorProto.FLOAT, outdata_shape)],
        )

        model = helper.make_model(graph, producer_name="tile_test")
        check_correctness(model, inputs={"input": indata, "repeats": repeats}, opset=6)
        # verify_with_ort_with_inputs(
        #     model, [indata, repeats], use_vm=True, opset=6, target=target, dev=dev
        # )

    x = np.random.rand(2, 3, 4, 5).astype(np.float32)
    repeats = np.random.randint(low=1, high=10, size=(np.ndim(x),)).astype(np.int64)
    z_array = np.tile(x, repeats)
    verify_tile_v6(x, repeats, z_array)



if __name__ == "__main__":
    tvm.testing.main()