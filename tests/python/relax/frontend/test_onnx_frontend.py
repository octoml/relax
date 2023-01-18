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
from onnx import TensorProto, ModelProto
import onnxruntime


def generate_random_inputs(model: ModelProto) -> Dict[str, np.array]:
    input_values = {}
    # Iterate through model inputs and extract their shape.
    for i in model.graph.input:
        shape = []
        for dim in i.type.tensor_type.shape.dim:
            shape.append(dim.dim_value)

        # Extract datatype for the input.
        if i.type.tensor_type.elem_type:
            dtype = str(onnx.mapping.TENSOR_TYPE_TO_NP_TYPE[i.type.tensor_type.elem_type])
        else:
            dtype = "float32"

        # Generate random inputs for each input.
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
    if inputs is None:
        inputs = generate_random_inputs(model)
    
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
        outputs = [helper.make_tensor_value_info("c", TensorProto.FLOAT, [32, 32])]
    )

    model = helper.make_model(graph, producer_name="matmul_test")
    check_correctness(model)


if __name__ == "__main__":
    test_matmul()