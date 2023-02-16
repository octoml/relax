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
import tvm
import tvm.testing
from tvm import relax as rx
from tvm.script import relax as R
from tvm.script import tir as T
from pathlib import Path
import sys
import json
from tvm import relay
import pytest
import datetime
import scipy
import numpy as np
import onnxruntime.backend
from tvm.tir import Any as Any
from typing import *
from tvm.contrib import graph_executor, utils
import time

from utils import Timer


def get_onnxruntime_output(model, inputs):
    """Generic function to generate onnxruntime output"""
    rep = onnxruntime.backend.prepare(model.SerializeToString(), "GPU")
    if isinstance(inputs, list) and len(inputs) == 1:
        inp = inputs[0]
    else:
        inp = inputs

    with Timer() as onnx_timer:
        output = rep.run(inp)
    # Unpack output if there's only a single value.
    if len(output) == 1:
        output = output[0]
    return output, onnx_timer.ms_duration


def get_tvm_output(
    graph_def,
    input_data,
    target,
    dev,
    output_shape=None,
    output_dtype="float32",
    opset=None,
    opt_level=1,
    convert_config=None,
    steps_record=None,
):
    """Generic function to execute and get tvm output"""
    # TODO: Resolve the issues and remove the following lines
    input_names, shape_dict = get_input_data_shape_dict(graph_def, input_data)

    mod, params = relay.frontend.from_onnx(
        graph_def, shape_dict, opset=opset, convert_config=convert_config
    )
    if steps_record is not None:
        steps_record.append("IMPORTED")

    with tvm.transform.PassContext(opt_level=opt_level):
        graph, lib, params = relay.build(mod, target, params=params)

    if steps_record is not None:
        steps_record.append("COMPILED")

    m = graph_executor.create(graph, lib, dev)
    # set inputs
    if isinstance(input_data, list):
        for i, _ in enumerate(input_names):
            # Its possible for some onnx inputs to not be needed in the tvm
            # module, confirm its present before setting.
            # pylint: disable=unnecessary-list-index-lookup
            m.set_input(
                input_names[i], tvm.nd.array(input_data[i].astype(input_data[i].dtype))
            )
    else:
        m.set_input(input_names, tvm.nd.array(input_data.astype(input_data.dtype)))

    m.set_input(**params)
    # execute
    m.run()

    if steps_record is not None:
        steps_record.append("RAN")
    # get outputs
    if isinstance(output_shape, list):
        tvm_output_list = []
        for i, _ in enumerate(output_shape):
            tvm_output = m.get_output(i)
            tvm_output_list.append(tvm_output.numpy())
        return tvm_output_list
    else:
        tvm_output = m.get_output(0)
        return tvm_output.numpy()


def get_tvm_output_with_vm(
    graph_def,
    input_data,
    target,
    dev,
    opset=None,
    freeze_params=False,
    convert_config=None,
    validate_structural_equal=True,
):
    """Generic function to execute and get tvm output with vm executor"""
    if not isinstance(input_data, list):
        input_data = [input_data]
    _, shape_dict = get_input_data_shape_dict(graph_def, input_data)

    with tvm.testing.disable_span_filling():
        mod, params = relay.frontend.from_onnx(
            graph_def,
            shape_dict,
            opset=opset,
            freeze_params=freeze_params,
            convert_config=convert_config,
        )
    if validate_structural_equal:
        with tvm.testing.enable_span_filling():
            mod_with_span, _ = relay.frontend.from_onnx(
                graph_def,
                shape_dict,
                opset=opset,
                freeze_params=freeze_params,
                convert_config=convert_config,
            )
        assert tvm.ir.structural_equal(mod, mod_with_span)

    result = relay.create_executor("vm", mod=mod, device=dev, target=target).evaluate()(
        *input_data, **params
    )
    if isinstance(result, tvm.runtime.NDArray):
        return result.numpy()
    return [r.numpy() for r in result]


def get_input_data_shape_dict(graph_def, input_data):
    """Get input data shape"""
    if isinstance(input_data, list):
        input_names = {}
        shape_dict = {}
        for i, _ in enumerate(input_data):
            input_names[i] = graph_def.graph.input[i].name
            input_ = input_data[i]

            if input_ is None or not hasattr(input_, "shape") or input_.shape == ():
                # Skip adding input shape data when the input data is None;
                # This is to enable optional arguments for onnx operators.
                continue

            elif isinstance(input_, list):
                shape_dict[input_names[i]] = (len(input_),)

            else:
                shape_dict[input_names[i]] = input_.shape

    else:
        input_names = graph_def.graph.input[0].name
        shape_dict = {input_names: input_data.shape}

    return input_names, shape_dict


def verify_with_ort_with_inputs(
    model,
    inputs,
    out_shape=None,
    target=None,
    dev=None,
    use_vm=False,
    opset=None,
    freeze_params=False,
    dtype="float32",
    rtol=1e-5,
    atol=1e-5,
    apply_softmax=False,
    opt_level=1,
    convert_config=None,
    steps_record=None,
):
    """verify_with_ort_with_inputs"""
    if opset is not None:
        model.opset_import[0].version = opset

    ort_out, ort_ms = get_onnxruntime_output(model, inputs)

    tvm_out, tvm_ms = get_tvm_output(
        model,
        inputs,
        target,
        dev,
        out_shape,
        dtype,
        opset=opset,
        opt_level=opt_level,
        convert_config=convert_config,
        steps_record=steps_record,
    )

    if steps_record is not None:
        steps_record.append("IMPORTED")

    if not isinstance(tvm_out, list):
        tvm_out = [tvm_out]
    if not isinstance(ort_out, list):
        ort_out = [ort_out]
    for tvm_val, ort_val in zip(tvm_out, ort_out):
        if apply_softmax:
            ort_val = scipy.special.softmax(ort_val)
            tvm_val = scipy.special.softmax(tvm_val)
        tvm.testing.assert_allclose(ort_val, tvm_val, rtol=rtol, atol=atol)
        assert ort_val.dtype == tvm_val.dtype

    if steps_record is not None:
        steps_record.append("VERIFIED_RESULTS")
