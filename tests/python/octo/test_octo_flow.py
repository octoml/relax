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
This file tests the functionality of the octoml flow API.
"""
from onnx import helper, TensorProto
import tvm.octo
import tvm.testing
from tvm.contrib import utils


def get_simple_onnx_model():
    # Create a single onnx matmul model that can be used for testing.
    matmul_node = helper.make_node("MatMul", ["a", "b"], ["c"])
    graph = helper.make_graph(
        [matmul_node],
        "minimal_matmul",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [32, 32]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [32, 32]),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, [32, 32])],
    )
    model = helper.make_model(graph, producer_name="minimal_matmul")
    return model


@tvm.testing.requires_gpu
def test_e2e_flow():
    # Try a full end to end flow and confirm functionality of features.
    test_model = get_simple_onnx_model()
    # Apply the simplified octoml API.
    octo_model = tvm.octo.compile(test_model, target=tvm.target.Target("cuda"))
    # Check that the produced model has properly formed shape info.
    assert octo_model.input_info["a"] == ([32, 32], "float32")

    # Test that the OctoModel can be saved and loaded.
    temp = utils.tempdir()
    model_path = temp.relpath("model.tar")
    octo_model.save(model_path)
    loaded_model = tvm.octo.OctoModel(model_path=model_path)
    # Confirm that the loaded model is equivalent to the saved one.
    tvm.ir.assert_structural_equal(octo_model.exe.as_text(), loaded_model.exe.as_text())
    # Confirm targets were saved and loaded correctly.
    assert str(octo_model.target) == str(loaded_model.target)

    # Test the running and benchmarking helpers.
    outputs = octo_model.run()
    assert list(outputs[0].shape) == [32, 32]
    report = octo_model.profile()
    # Confirm report has expected cutlass offload of matmul.
    assert "matmul_cutlass" in str(report)
