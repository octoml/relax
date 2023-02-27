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
import os
import onnx
from onnx import helper, TensorProto
import tvm.octo
from tvm.contrib import utils


def get_simple_onnx_model():
    # Create a single onnx convolution model that can be used for testing.
    conv_node = helper.make_node("Conv", ["x", "w", "b"], ["y"])
    graph = helper.make_graph(
        [conv_node],
        "minimal_conv",
        inputs=[
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [1, 3, 32, 32]),
            helper.make_tensor_value_info("w", TensorProto.FLOAT, [16, 3, 3, 3]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [16]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [1, 16, 30, 30])],
    )
    model = helper.make_model(graph, producer_name="minimal_conv")
    return model


def test_e2e_flow():
    # Try a full end to end flow and confirm functionality of features.
    test_model = get_simple_onnx_model()
    octo_model = tvm.octo.compile(test_model)
    # Check that the produced model has properly formed shape info.
    assert octo_model.input_info["x"] == ([1, 3, 32, 32], "float32")

    # Test that the OctoModel can be saved and loaded.
    temp = utils.tempdir()
    model_path = temp.relpath("model.tar")
    octo_model.save(model_path)
    loaded_model = tvm.octo.OctoModel(model_path=model_path)
    # Confirm that the loaded model is equivalent to the saved one.
    tvm.ir.assert_structural_equal(octo_model.exe.as_text(), loaded_model.exe.as_text())
    # Confirm targets were saved and loaded correctly.
    assert str(octo_model.target) == str(loaded_model.target)
