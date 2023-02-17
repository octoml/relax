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
from typing import TYPE_CHECKING, List, Optional, Tuple, Union
import numpy as np

import tvm
import tvm.testing
from tvm import relax
from tvm.script import relax as R
from tvm.script import tir as T

import pytest
from onnx import helper

if TYPE_CHECKING:

    class TensorProto:
        """ONNX TensorProto values for type checking."""

        UNDEFINED = 0
        FLOAT = 1
        UINT8 = 2
        INT8 = 3
        UINT16 = 4
        INT16 = 5
        INT32 = 6
        INT64 = 7
        BOOL = 9
        FLOAT16 = 10
        DOUBLE = 11
        UINT32 = 12
        COMPLEX64 = 14
        COMPLEX128 = 15
        BFLOAT16 = 16

else:
    from onnx import TensorProto


bg = np.random.MT19937(0)
rg = np.random.Generator(bg)


# pylint: disable=no-self-argument,missing-class-docstring,missing-function-docstring,invalid-name
@tvm.script.ir_module
class ConcatModule:
    @R.function
    def main(
        a: R.Tensor((1, 32), dtype="float32"), b: R.Tensor((1, 32), dtype="float32")
    ) -> R.Tensor((2, 32), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((2, 32), dtype="float32") = R.concat((a, b), axis=0)
            R.output(gv)
        return gv


@tvm.script.ir_module
class MatMulModule:
    @R.function
    def main(
        a: R.Tensor((24, 32), dtype="float32"), b: R.Tensor((32, 16), dtype="float32")
    ) -> R.Tensor((24, 16), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((24, 16), dtype="float32") = R.matmul(a, b, out_dtype="")
            R.output(gv)
        return gv


@tvm.script.ir_module
class AddModule:
    @R.function
    def main(
        a: R.Tensor((24, 32), dtype="float32"), b: R.Tensor((24, 32), dtype="float32")
    ) -> R.Tensor((24, 32), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((24, 32), dtype="float32") = R.add(a, b)
            R.output(gv)
        return gv


@tvm.script.ir_module
class CastModule_6_6:
    @R.function
    def main(a: R.Tensor((24, 32), dtype="int32")) -> R.Tensor((24, 32), dtype="int32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((24, 32), dtype="int32") = R.astype(a, dtype="int32")
            R.output(gv)
        return gv


@tvm.script.ir_module
class CastModule_1_6:
    @R.function
    def main(a: R.Tensor((24, 32), dtype="float32")) -> R.Tensor((24, 32), dtype="int32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((24, 32), dtype="int32") = R.astype(a, dtype="int32")
            R.output(gv)
        return gv


@tvm.script.ir_module
class CastModule_10_6:
    @R.function
    def main(a: R.Tensor((24, 32), dtype="float16")) -> R.Tensor((24, 32), dtype="int32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((24, 32), dtype="int32") = R.astype(a, dtype="int32")
            R.output(gv)
        return gv


@tvm.script.ir_module
class CastModule_11_6:
    @R.function
    def main(a: R.Tensor((24, 32), dtype="float64")) -> R.Tensor((24, 32), dtype="int32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((24, 32), dtype="int32") = R.astype(a, dtype="int32")
            R.output(gv)
        return gv


@tvm.script.ir_module
class CastModule_6_1:
    @R.function
    def main(a: R.Tensor((24, 32), dtype="int32")) -> R.Tensor((24, 32), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((24, 32), dtype="float32") = R.astype(a, dtype="float32")
            R.output(gv)
        return gv


@tvm.script.ir_module
class CastModule_1_1:
    @R.function
    def main(a: R.Tensor((24, 32), dtype="float32")) -> R.Tensor((24, 32), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((24, 32), dtype="float32") = R.astype(a, dtype="float32")
            R.output(gv)
        return gv


@tvm.script.ir_module
class CastModule_10_1:
    @R.function
    def main(a: R.Tensor((24, 32), dtype="float16")) -> R.Tensor((24, 32), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((24, 32), dtype="float32") = R.astype(a, dtype="float32")
            R.output(gv)
        return gv


@tvm.script.ir_module
class CastModule_11_1:
    @R.function
    def main(a: R.Tensor((24, 32), dtype="float64")) -> R.Tensor((24, 32), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((24, 32), dtype="float32") = R.astype(a, dtype="float32")
            R.output(gv)
        return gv


@tvm.script.ir_module
class CastModule_6_10:
    @R.function
    def main(a: R.Tensor((24, 32), dtype="int32")) -> R.Tensor((24, 32), dtype="float16"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((24, 32), dtype="float16") = R.astype(a, dtype="float16")
            R.output(gv)
        return gv


@tvm.script.ir_module
class CastModule_1_10:
    @R.function
    def main(a: R.Tensor((24, 32), dtype="float32")) -> R.Tensor((24, 32), dtype="float16"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((24, 32), dtype="float16") = R.astype(a, dtype="float16")
            R.output(gv)
        return gv


@tvm.script.ir_module
class CastModule_10_10:
    @R.function
    def main(a: R.Tensor((24, 32), dtype="float16")) -> R.Tensor((24, 32), dtype="float16"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((24, 32), dtype="float16") = R.astype(a, dtype="float16")
            R.output(gv)
        return gv


@tvm.script.ir_module
class CastModule_11_10:
    @R.function
    def main(a: R.Tensor((24, 32), dtype="float64")) -> R.Tensor((24, 32), dtype="float16"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((24, 32), dtype="float16") = R.astype(a, dtype="float16")
            R.output(gv)
        return gv


@tvm.script.ir_module
class CastModule_6_11:
    @R.function
    def main(a: R.Tensor((24, 32), dtype="int32")) -> R.Tensor((24, 32), dtype="float64"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((24, 32), dtype="float64") = R.astype(a, dtype="float64")
            R.output(gv)
        return gv


@tvm.script.ir_module
class CastModule_1_11:
    @R.function
    def main(a: R.Tensor((24, 32), dtype="float32")) -> R.Tensor((24, 32), dtype="float64"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((24, 32), dtype="float64") = R.astype(a, dtype="float64")
            R.output(gv)
        return gv


@tvm.script.ir_module
class CastModule_10_11:
    @R.function
    def main(a: R.Tensor((24, 32), dtype="float16")) -> R.Tensor((24, 32), dtype="float64"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((24, 32), dtype="float64") = R.astype(a, dtype="float64")
            R.output(gv)
        return gv


@tvm.script.ir_module
class CastModule_11_11:
    @R.function
    def main(a: R.Tensor((24, 32), dtype="float64")) -> R.Tensor((24, 32), dtype="float64"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((24, 32), dtype="float64") = R.astype(a, dtype="float64")
            R.output(gv)
        return gv


@tvm.script.ir_module
class GatherModule_0:
    @R.function
    def main(
        data: R.Tensor((2, 4, 3, 2), dtype="float32"), indices: R.Tensor((2,), dtype="int32")
    ) -> R.Tensor((2, 4, 3, 2), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((2, 4, 3, 2), dtype="float32") = R.take(data, indices, axis=0)
            R.output(gv)
        return gv


@tvm.script.ir_module
class GatherModule_1:
    @R.function
    def main(
        data: R.Tensor((5, 2, 3, 2), dtype="float32"), indices: R.Tensor((2,), dtype="int32")
    ) -> R.Tensor((5, 2, 3, 2), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((5, 2, 3, 2), dtype="float32") = R.take(data, indices, axis=1)
            R.output(gv)
        return gv


@tvm.script.ir_module
class GatherModule_2:
    @R.function
    def main(
        data: R.Tensor((5, 4, 2, 2), dtype="float32"), indices: R.Tensor((2,), dtype="int32")
    ) -> R.Tensor((5, 4, 2, 2), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((5, 4, 2, 2), dtype="float32") = R.take(data, indices, axis=2)
            R.output(gv)
        return gv


@tvm.script.ir_module
class GatherModule_3:
    @R.function
    def main(
        data: R.Tensor((5, 4, 3, 2), dtype="float32"), indices: R.Tensor((2,), dtype="int32")
    ) -> R.Tensor((5, 4, 3, 2), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((5, 4, 3, 2), dtype="float32") = R.take(data, indices, axis=3)
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_None_None_False_False_False:
    @R.function
    def main(
        a: R.Tensor((3, 4), dtype="float32"), b: R.Tensor((4, 5), dtype="float32")
    ) -> R.Tensor((), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((), dtype="float32") = R.const(0, "float32")
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_0_25_None_False_False_False:
    @R.function
    def main(
        a: R.Tensor((3, 4), dtype="float32"), b: R.Tensor((4, 5), dtype="float32")
    ) -> R.Tensor((3, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 5), dtype="float32") = R.matmul(a, b, out_dtype="")
            gv: R.Tensor((3, 5), dtype="float32") = R.multiply(lv, R.const(0.25, "float32"))
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_None_0_35_False_False_False:
    @R.function
    def main(
        a: R.Tensor((3, 4), dtype="float32"), b: R.Tensor((4, 5), dtype="float32")
    ) -> R.Tensor((), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((), dtype="float32") = R.const(0, "float32")
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_0_25_0_35_False_False_False:
    @R.function
    def main(
        a: R.Tensor((3, 4), dtype="float32"), b: R.Tensor((4, 5), dtype="float32")
    ) -> R.Tensor((3, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 5), dtype="float32") = R.matmul(a, b, out_dtype="")
            gv: R.Tensor((3, 5), dtype="float32") = R.multiply(lv, R.const(0.25, "float32"))
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_None_None_True_False_False:
    @R.function
    def main(
        a: R.Tensor((4, 3), dtype="float32"), b: R.Tensor((4, 5), dtype="float32")
    ) -> R.Tensor((), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((), dtype="float32") = R.const(0, "float32")
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_0_25_None_True_False_False:
    @R.function
    def main(
        a: R.Tensor((4, 3), dtype="float32"), b: R.Tensor((4, 5), dtype="float32")
    ) -> R.Tensor((3, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 4), dtype="float32") = R.permute_dims(a, axes=None)
            lv1: R.Tensor((3, 5), dtype="float32") = R.matmul(lv, b, out_dtype="")
            gv: R.Tensor((3, 5), dtype="float32") = R.multiply(lv1, R.const(0.25, "float32"))
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_None_0_35_True_False_False:
    @R.function
    def main(
        a: R.Tensor((4, 3), dtype="float32"), b: R.Tensor((4, 5), dtype="float32")
    ) -> R.Tensor((), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((), dtype="float32") = R.const(0, "float32")
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_0_25_0_35_True_False_False:
    @R.function
    def main(
        a: R.Tensor((4, 3), dtype="float32"), b: R.Tensor((4, 5), dtype="float32")
    ) -> R.Tensor((3, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 4), dtype="float32") = R.permute_dims(a, axes=None)
            lv1: R.Tensor((3, 5), dtype="float32") = R.matmul(lv, b, out_dtype="")
            gv: R.Tensor((3, 5), dtype="float32") = R.multiply(lv1, R.const(0.25, "float32"))
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_None_None_False_True_False:
    @R.function
    def main(
        a: R.Tensor((3, 4), dtype="float32"), b: R.Tensor((5, 4), dtype="float32")
    ) -> R.Tensor((), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((), dtype="float32") = R.const(0, "float32")
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_0_25_None_False_True_False:
    @R.function
    def main(
        a: R.Tensor((3, 4), dtype="float32"), b: R.Tensor((5, 4), dtype="float32")
    ) -> R.Tensor((3, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((4, 5), dtype="float32") = R.permute_dims(b, axes=None)
            lv1: R.Tensor((3, 5), dtype="float32") = R.matmul(a, lv, out_dtype="")
            gv: R.Tensor((3, 5), dtype="float32") = R.multiply(lv1, R.const(0.25, "float32"))
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_None_0_35_False_True_False:
    @R.function
    def main(
        a: R.Tensor((3, 4), dtype="float32"), b: R.Tensor((5, 4), dtype="float32")
    ) -> R.Tensor((), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((), dtype="float32") = R.const(0, "float32")
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_0_25_0_35_False_True_False:
    @R.function
    def main(
        a: R.Tensor((3, 4), dtype="float32"), b: R.Tensor((5, 4), dtype="float32")
    ) -> R.Tensor((3, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((4, 5), dtype="float32") = R.permute_dims(b, axes=None)
            lv1: R.Tensor((3, 5), dtype="float32") = R.matmul(a, lv, out_dtype="")
            gv: R.Tensor((3, 5), dtype="float32") = R.multiply(lv1, R.const(0.25, "float32"))
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_None_None_True_True_False:
    @R.function
    def main(
        a: R.Tensor((4, 3), dtype="float32"), b: R.Tensor((5, 4), dtype="float32")
    ) -> R.Tensor((), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((), dtype="float32") = R.const(0, "float32")
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_0_25_None_True_True_False:
    @R.function
    def main(
        a: R.Tensor((4, 3), dtype="float32"), b: R.Tensor((5, 4), dtype="float32")
    ) -> R.Tensor((3, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 4), dtype="float32") = R.permute_dims(a, axes=None)
            lv1: R.Tensor((4, 5), dtype="float32") = R.permute_dims(b, axes=None)
            lv2: R.Tensor((3, 5), dtype="float32") = R.matmul(lv, lv1, out_dtype="")
            gv: R.Tensor((3, 5), dtype="float32") = R.multiply(lv2, R.const(0.25, "float32"))
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_None_0_35_True_True_False:
    @R.function
    def main(
        a: R.Tensor((4, 3), dtype="float32"), b: R.Tensor((5, 4), dtype="float32")
    ) -> R.Tensor((), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((), dtype="float32") = R.const(0, "float32")
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_0_25_0_35_True_True_False:
    @R.function
    def main(
        a: R.Tensor((4, 3), dtype="float32"), b: R.Tensor((5, 4), dtype="float32")
    ) -> R.Tensor((3, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 4), dtype="float32") = R.permute_dims(a, axes=None)
            lv1: R.Tensor((4, 5), dtype="float32") = R.permute_dims(b, axes=None)
            lv2: R.Tensor((3, 5), dtype="float32") = R.matmul(lv, lv1, out_dtype="")
            gv: R.Tensor((3, 5), dtype="float32") = R.multiply(lv2, R.const(0.25, "float32"))
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_None_None_False_False_True:
    @R.function
    def main(
        a: R.Tensor((3, 4), dtype="float32"),
        b: R.Tensor((4, 5), dtype="float32"),
        c: R.Tensor((1, 5), dtype="float32"),
    ) -> R.Tensor((1, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 5), dtype="float32") = c
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_0_25_None_False_False_True:
    @R.function
    def main(
        a: R.Tensor((3, 4), dtype="float32"),
        b: R.Tensor((4, 5), dtype="float32"),
        c: R.Tensor((1, 5), dtype="float32"),
    ) -> R.Tensor((3, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 5), dtype="float32") = R.matmul(a, b, out_dtype="")
            lv1: R.Tensor((3, 5), dtype="float32") = R.multiply(lv, R.const(0.25, "float32"))
            gv: R.Tensor((3, 5), dtype="float32") = R.add(lv1, c)
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_None_0_35_False_False_True:
    @R.function
    def main(
        a: R.Tensor((3, 4), dtype="float32"),
        b: R.Tensor((4, 5), dtype="float32"),
        c: R.Tensor((1, 5), dtype="float32"),
    ) -> R.Tensor((1, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 5), dtype="float32") = R.multiply(c, R.const(0.35, "float32"))
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_0_25_0_35_False_False_True:
    @R.function
    def main(
        a: R.Tensor((3, 4), dtype="float32"),
        b: R.Tensor((4, 5), dtype="float32"),
        c: R.Tensor((1, 5), dtype="float32"),
    ) -> R.Tensor((3, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 5), dtype="float32") = R.matmul(a, b, out_dtype="")
            lv1: R.Tensor((3, 5), dtype="float32") = R.multiply(lv, R.const(0.25, "float32"))
            lv2: R.Tensor((1, 5), dtype="float32") = R.multiply(c, R.const(0.35, "float32"))
            gv: R.Tensor((3, 5), dtype="float32") = R.add(lv1, lv2)
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_None_None_True_False_True:
    @R.function
    def main(
        a: R.Tensor((4, 3), dtype="float32"),
        b: R.Tensor((4, 5), dtype="float32"),
        c: R.Tensor((1, 5), dtype="float32"),
    ) -> R.Tensor((1, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 5), dtype="float32") = c
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_0_25_None_True_False_True:
    @R.function
    def main(
        a: R.Tensor((4, 3), dtype="float32"),
        b: R.Tensor((4, 5), dtype="float32"),
        c: R.Tensor((1, 5), dtype="float32"),
    ) -> R.Tensor((3, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 4), dtype="float32") = R.permute_dims(a, axes=None)
            lv1: R.Tensor((3, 5), dtype="float32") = R.matmul(lv, b, out_dtype="")
            lv2: R.Tensor((3, 5), dtype="float32") = R.multiply(lv1, R.const(0.25, "float32"))
            gv: R.Tensor((3, 5), dtype="float32") = R.add(lv2, c)
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_None_0_35_True_False_True:
    @R.function
    def main(
        a: R.Tensor((4, 3), dtype="float32"),
        b: R.Tensor((4, 5), dtype="float32"),
        c: R.Tensor((1, 5), dtype="float32"),
    ) -> R.Tensor((1, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 5), dtype="float32") = R.multiply(c, R.const(0.35, "float32"))
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_0_25_0_35_True_False_True:
    @R.function
    def main(
        a: R.Tensor((4, 3), dtype="float32"),
        b: R.Tensor((4, 5), dtype="float32"),
        c: R.Tensor((1, 5), dtype="float32"),
    ) -> R.Tensor((3, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 4), dtype="float32") = R.permute_dims(a, axes=None)
            lv1: R.Tensor((3, 5), dtype="float32") = R.matmul(lv, b, out_dtype="")
            lv2: R.Tensor((3, 5), dtype="float32") = R.multiply(lv1, R.const(0.25, "float32"))
            lv3: R.Tensor((1, 5), dtype="float32") = R.multiply(c, R.const(0.35, "float32"))
            gv: R.Tensor((3, 5), dtype="float32") = R.add(lv2, lv3)
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_None_None_False_True_True:
    @R.function
    def main(
        a: R.Tensor((3, 4), dtype="float32"),
        b: R.Tensor((5, 4), dtype="float32"),
        c: R.Tensor((1, 5), dtype="float32"),
    ) -> R.Tensor((1, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 5), dtype="float32") = c
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_0_25_None_False_True_True:
    @R.function
    def main(
        a: R.Tensor((3, 4), dtype="float32"),
        b: R.Tensor((5, 4), dtype="float32"),
        c: R.Tensor((1, 5), dtype="float32"),
    ) -> R.Tensor((3, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((4, 5), dtype="float32") = R.permute_dims(b, axes=None)
            lv1: R.Tensor((3, 5), dtype="float32") = R.matmul(a, lv, out_dtype="")
            lv2: R.Tensor((3, 5), dtype="float32") = R.multiply(lv1, R.const(0.25, "float32"))
            gv: R.Tensor((3, 5), dtype="float32") = R.add(lv2, c)
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_None_0_35_False_True_True:
    @R.function
    def main(
        a: R.Tensor((3, 4), dtype="float32"),
        b: R.Tensor((5, 4), dtype="float32"),
        c: R.Tensor((1, 5), dtype="float32"),
    ) -> R.Tensor((1, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 5), dtype="float32") = R.multiply(c, R.const(0.35, "float32"))
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_0_25_0_35_False_True_True:
    @R.function
    def main(
        a: R.Tensor((3, 4), dtype="float32"),
        b: R.Tensor((5, 4), dtype="float32"),
        c: R.Tensor((1, 5), dtype="float32"),
    ) -> R.Tensor((3, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((4, 5), dtype="float32") = R.permute_dims(b, axes=None)
            lv1: R.Tensor((3, 5), dtype="float32") = R.matmul(a, lv, out_dtype="")
            lv2: R.Tensor((3, 5), dtype="float32") = R.multiply(lv1, R.const(0.25, "float32"))
            lv3: R.Tensor((1, 5), dtype="float32") = R.multiply(c, R.const(0.35, "float32"))
            gv: R.Tensor((3, 5), dtype="float32") = R.add(lv2, lv3)
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_None_None_True_True_True:
    @R.function
    def main(
        a: R.Tensor((4, 3), dtype="float32"),
        b: R.Tensor((5, 4), dtype="float32"),
        c: R.Tensor((1, 5), dtype="float32"),
    ) -> R.Tensor((1, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 5), dtype="float32") = c
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_0_25_None_True_True_True:
    @R.function
    def main(
        a: R.Tensor((4, 3), dtype="float32"),
        b: R.Tensor((5, 4), dtype="float32"),
        c: R.Tensor((1, 5), dtype="float32"),
    ) -> R.Tensor((3, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 4), dtype="float32") = R.permute_dims(a, axes=None)
            lv1: R.Tensor((4, 5), dtype="float32") = R.permute_dims(b, axes=None)
            lv2: R.Tensor((3, 5), dtype="float32") = R.matmul(lv, lv1, out_dtype="")
            lv3: R.Tensor((3, 5), dtype="float32") = R.multiply(lv2, R.const(0.25, "float32"))
            gv: R.Tensor((3, 5), dtype="float32") = R.add(lv3, c)
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_None_0_35_True_True_True:
    @R.function
    def main(
        a: R.Tensor((4, 3), dtype="float32"),
        b: R.Tensor((5, 4), dtype="float32"),
        c: R.Tensor((1, 5), dtype="float32"),
    ) -> R.Tensor((1, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 5), dtype="float32") = R.multiply(c, R.const(0.35, "float32"))
            R.output(gv)
        return gv


@tvm.script.ir_module
class GemmModule_0_25_0_35_True_True_True:
    @R.function
    def main(
        a: R.Tensor((4, 3), dtype="float32"),
        b: R.Tensor((5, 4), dtype="float32"),
        c: R.Tensor((1, 5), dtype="float32"),
    ) -> R.Tensor((3, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 4), dtype="float32") = R.permute_dims(a, axes=None)
            lv1: R.Tensor((4, 5), dtype="float32") = R.permute_dims(b, axes=None)
            lv2: R.Tensor((3, 5), dtype="float32") = R.matmul(lv, lv1, out_dtype="")
            lv3: R.Tensor((3, 5), dtype="float32") = R.multiply(lv2, R.const(0.25, "float32"))
            lv4: R.Tensor((1, 5), dtype="float32") = R.multiply(c, R.const(0.35, "float32"))
            gv: R.Tensor((3, 5), dtype="float32") = R.add(lv3, lv4)
            R.output(gv)
        return gv


@tvm.script.ir_module
class MulModule:
    @R.function
    def main(
        a: R.Tensor((24, 32), dtype="float32"), b: R.Tensor((24, 32), dtype="float32")
    ) -> R.Tensor((24, 32), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((24, 32), dtype="float32") = R.multiply(a, b)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReshapeModuleConstant_0:
    @R.function
    def main(
        data: R.Tensor((7, 32, 32, 8), dtype="float32")
    ) -> R.Tensor((224, 256), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((224, 256), dtype="float32") = R.reshape(data, (224, 256))
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReshapeModuleConstant_1:
    @R.function
    def main(
        data: R.Tensor((7, 32, 32, 8), dtype="float32")
    ) -> R.Tensor((7, 8192), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((7, 8192), dtype="float32") = R.reshape(data, (7, 8192))
            R.output(gv)
        return gv


@tvm.script.ir_module
class ShapeModule:
    @R.function
    def main(data: R.Tensor((7, 32, 32, 8), dtype="float32")) -> R.Shape(ndim=-1):
        # block 0
        with R.dataflow():
            gv: R.Shape(ndim=-1) = R.shape_of(data)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ClipModule:
    @R.function
    def main(
        data: R.Tensor((7, 32, 32, 8), dtype="float32")
    ) -> R.Tensor((7, 32, 32, 8), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((7, 32, 32, 8), dtype="float32") = R.clip(
                data, R.prim_value(2), R.prim_value(3)
            )
            R.output(gv)
        return gv


@tvm.script.ir_module
class DivModule:
    @R.function
    def main(
        a: R.Tensor((24, 32), dtype="float32"), b: R.Tensor((24, 32), dtype="float32")
    ) -> R.Tensor((24, 32), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((24, 32), dtype="float32") = R.divide(a, b)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ExpandModule:
    @R.function
    def main(
        data: R.Tensor((3, 1), dtype="float32"), shape: R.Tensor((2,), dtype="int64")
    ) -> R.Tensor(dtype="float32", ndim=2):
        shape_var_0 = T.var("int64")
        shape_var_1 = T.var("int64")
        # block 0
        with R.dataflow():
            lv: R.Shape(ndim=2) = R.call_packed(
                "vm.builtin.tensor_to_shape", shape, sinfo_args=[R.Shape(ndim=2)]
            )
            lv1: R.Shape([shape_var_0, shape_var_1]) = R.match_cast(
                lv, R.Shape([shape_var_0, shape_var_1])
            )
            gv: R.Tensor((shape_var_0, shape_var_1), dtype="float32") = R.broadcast_to(
                data, (shape_var_0, shape_var_1)
            )
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__0__0:
    @R.function
    def main(data: R.Tensor((8, 7, 6, 5), dtype="float32")) -> R.Tensor((7, 6, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((7, 6, 5), dtype="float32") = R.mean(data, axis=[0], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__0__0:
    @R.function
    def main(data: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor((2, 3, 4), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((2, 3, 4), dtype="float32") = R.mean(data, axis=[0], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__1__0:
    @R.function
    def main(data: R.Tensor((8, 7, 6, 5), dtype="float32")) -> R.Tensor((8, 6, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((8, 6, 5), dtype="float32") = R.mean(data, axis=[1], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__1__0:
    @R.function
    def main(data: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor((1, 3, 4), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 3, 4), dtype="float32") = R.mean(data, axis=[1], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__2__0:
    @R.function
    def main(data: R.Tensor((8, 7, 6, 5), dtype="float32")) -> R.Tensor((8, 7, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((8, 7, 5), dtype="float32") = R.mean(data, axis=[2], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__2__0:
    @R.function
    def main(data: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor((1, 2, 4), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 2, 4), dtype="float32") = R.mean(data, axis=[2], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__3__0:
    @R.function
    def main(data: R.Tensor((8, 7, 6, 5), dtype="float32")) -> R.Tensor((8, 7, 6), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((8, 7, 6), dtype="float32") = R.mean(data, axis=[3], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__3__0:
    @R.function
    def main(data: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor((1, 2, 3), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 2, 3), dtype="float32") = R.mean(data, axis=[3], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__0_1__0:
    @R.function
    def main(data: R.Tensor((8, 7, 6, 5), dtype="float32")) -> R.Tensor((6, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((6, 5), dtype="float32") = R.mean(data, axis=[0, 1], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__0_1__0:
    @R.function
    def main(data: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor((3, 4), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((3, 4), dtype="float32") = R.mean(data, axis=[0, 1], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__0_2__0:
    @R.function
    def main(data: R.Tensor((8, 7, 6, 5), dtype="float32")) -> R.Tensor((7, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((7, 5), dtype="float32") = R.mean(data, axis=[0, 2], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__0_2__0:
    @R.function
    def main(data: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor((2, 4), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((2, 4), dtype="float32") = R.mean(data, axis=[0, 2], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__0_3__0:
    @R.function
    def main(data: R.Tensor((8, 7, 6, 5), dtype="float32")) -> R.Tensor((7, 6), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((7, 6), dtype="float32") = R.mean(data, axis=[0, 3], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__0_3__0:
    @R.function
    def main(data: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor((2, 3), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((2, 3), dtype="float32") = R.mean(data, axis=[0, 3], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__1_2__0:
    @R.function
    def main(data: R.Tensor((8, 7, 6, 5), dtype="float32")) -> R.Tensor((8, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((8, 5), dtype="float32") = R.mean(data, axis=[1, 2], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__1_2__0:
    @R.function
    def main(data: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor((1, 4), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 4), dtype="float32") = R.mean(data, axis=[1, 2], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__1_3__0:
    @R.function
    def main(data: R.Tensor((8, 7, 6, 5), dtype="float32")) -> R.Tensor((8, 6), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((8, 6), dtype="float32") = R.mean(data, axis=[1, 3], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__1_3__0:
    @R.function
    def main(data: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor((1, 3), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 3), dtype="float32") = R.mean(data, axis=[1, 3], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__2_3__0:
    @R.function
    def main(data: R.Tensor((8, 7, 6, 5), dtype="float32")) -> R.Tensor((8, 7), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((8, 7), dtype="float32") = R.mean(data, axis=[2, 3], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__2_3__0:
    @R.function
    def main(data: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor((1, 2), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 2), dtype="float32") = R.mean(data, axis=[2, 3], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__0_1_2__0:
    @R.function
    def main(data: R.Tensor((8, 7, 6, 5), dtype="float32")) -> R.Tensor((5,), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((5,), dtype="float32") = R.mean(data, axis=[0, 1, 2], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__0_1_2__0:
    @R.function
    def main(data: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor((4,), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((4,), dtype="float32") = R.mean(data, axis=[0, 1, 2], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__0_1_3__0:
    @R.function
    def main(data: R.Tensor((8, 7, 6, 5), dtype="float32")) -> R.Tensor((6,), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((6,), dtype="float32") = R.mean(data, axis=[0, 1, 3], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__0_1_3__0:
    @R.function
    def main(data: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor((3,), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((3,), dtype="float32") = R.mean(data, axis=[0, 1, 3], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__0_2_3__0:
    @R.function
    def main(data: R.Tensor((8, 7, 6, 5), dtype="float32")) -> R.Tensor((7,), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((7,), dtype="float32") = R.mean(data, axis=[0, 2, 3], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__0_2_3__0:
    @R.function
    def main(data: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor((2,), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((2,), dtype="float32") = R.mean(data, axis=[0, 2, 3], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__1_2_3__0:
    @R.function
    def main(data: R.Tensor((8, 7, 6, 5), dtype="float32")) -> R.Tensor((8,), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((8,), dtype="float32") = R.mean(data, axis=[1, 2, 3], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__1_2_3__0:
    @R.function
    def main(data: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor((1,), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1,), dtype="float32") = R.mean(data, axis=[1, 2, 3], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__0_1_2_3__0:
    @R.function
    def main(data: R.Tensor((8, 7, 6, 5), dtype="float32")) -> R.Tensor((), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((), dtype="float32") = R.mean(data, axis=[0, 1, 2, 3], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__0_1_2_3__0:
    @R.function
    def main(data: R.Tensor((1, 2, 3, 4), dtype="float32")) -> R.Tensor((), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((), dtype="float32") = R.mean(data, axis=[0, 1, 2, 3], keepdims=False)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__0__1:
    @R.function
    def main(
        data: R.Tensor((8, 7, 6, 5), dtype="float32")
    ) -> R.Tensor((1, 7, 6, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 7, 6, 5), dtype="float32") = R.mean(data, axis=[0], keepdims=True)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__0__1:
    @R.function
    def main(
        data: R.Tensor((1, 2, 3, 4), dtype="float32")
    ) -> R.Tensor((1, 2, 3, 4), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 2, 3, 4), dtype="float32") = R.mean(data, axis=[0], keepdims=True)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__1__1:
    @R.function
    def main(
        data: R.Tensor((8, 7, 6, 5), dtype="float32")
    ) -> R.Tensor((8, 1, 6, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((8, 1, 6, 5), dtype="float32") = R.mean(data, axis=[1], keepdims=True)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__1__1:
    @R.function
    def main(
        data: R.Tensor((1, 2, 3, 4), dtype="float32")
    ) -> R.Tensor((1, 1, 3, 4), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 1, 3, 4), dtype="float32") = R.mean(data, axis=[1], keepdims=True)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__2__1:
    @R.function
    def main(
        data: R.Tensor((8, 7, 6, 5), dtype="float32")
    ) -> R.Tensor((8, 7, 1, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((8, 7, 1, 5), dtype="float32") = R.mean(data, axis=[2], keepdims=True)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__2__1:
    @R.function
    def main(
        data: R.Tensor((1, 2, 3, 4), dtype="float32")
    ) -> R.Tensor((1, 2, 1, 4), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 2, 1, 4), dtype="float32") = R.mean(data, axis=[2], keepdims=True)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__3__1:
    @R.function
    def main(
        data: R.Tensor((8, 7, 6, 5), dtype="float32")
    ) -> R.Tensor((8, 7, 6, 1), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((8, 7, 6, 1), dtype="float32") = R.mean(data, axis=[3], keepdims=True)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__3__1:
    @R.function
    def main(
        data: R.Tensor((1, 2, 3, 4), dtype="float32")
    ) -> R.Tensor((1, 2, 3, 1), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 2, 3, 1), dtype="float32") = R.mean(data, axis=[3], keepdims=True)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__0_1__1:
    @R.function
    def main(
        data: R.Tensor((8, 7, 6, 5), dtype="float32")
    ) -> R.Tensor((1, 1, 6, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 1, 6, 5), dtype="float32") = R.mean(data, axis=[0, 1], keepdims=True)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__0_1__1:
    @R.function
    def main(
        data: R.Tensor((1, 2, 3, 4), dtype="float32")
    ) -> R.Tensor((1, 1, 3, 4), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 1, 3, 4), dtype="float32") = R.mean(data, axis=[0, 1], keepdims=True)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__0_2__1:
    @R.function
    def main(
        data: R.Tensor((8, 7, 6, 5), dtype="float32")
    ) -> R.Tensor((1, 7, 1, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 7, 1, 5), dtype="float32") = R.mean(data, axis=[0, 2], keepdims=True)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__0_2__1:
    @R.function
    def main(
        data: R.Tensor((1, 2, 3, 4), dtype="float32")
    ) -> R.Tensor((1, 2, 1, 4), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 2, 1, 4), dtype="float32") = R.mean(data, axis=[0, 2], keepdims=True)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__0_3__1:
    @R.function
    def main(
        data: R.Tensor((8, 7, 6, 5), dtype="float32")
    ) -> R.Tensor((1, 7, 6, 1), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 7, 6, 1), dtype="float32") = R.mean(data, axis=[0, 3], keepdims=True)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__0_3__1:
    @R.function
    def main(
        data: R.Tensor((1, 2, 3, 4), dtype="float32")
    ) -> R.Tensor((1, 2, 3, 1), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 2, 3, 1), dtype="float32") = R.mean(data, axis=[0, 3], keepdims=True)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__1_2__1:
    @R.function
    def main(
        data: R.Tensor((8, 7, 6, 5), dtype="float32")
    ) -> R.Tensor((8, 1, 1, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((8, 1, 1, 5), dtype="float32") = R.mean(data, axis=[1, 2], keepdims=True)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__1_2__1:
    @R.function
    def main(
        data: R.Tensor((1, 2, 3, 4), dtype="float32")
    ) -> R.Tensor((1, 1, 1, 4), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 1, 1, 4), dtype="float32") = R.mean(data, axis=[1, 2], keepdims=True)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__1_3__1:
    @R.function
    def main(
        data: R.Tensor((8, 7, 6, 5), dtype="float32")
    ) -> R.Tensor((8, 1, 6, 1), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((8, 1, 6, 1), dtype="float32") = R.mean(data, axis=[1, 3], keepdims=True)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__1_3__1:
    @R.function
    def main(
        data: R.Tensor((1, 2, 3, 4), dtype="float32")
    ) -> R.Tensor((1, 1, 3, 1), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 1, 3, 1), dtype="float32") = R.mean(data, axis=[1, 3], keepdims=True)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__2_3__1:
    @R.function
    def main(
        data: R.Tensor((8, 7, 6, 5), dtype="float32")
    ) -> R.Tensor((8, 7, 1, 1), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((8, 7, 1, 1), dtype="float32") = R.mean(data, axis=[2, 3], keepdims=True)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__2_3__1:
    @R.function
    def main(
        data: R.Tensor((1, 2, 3, 4), dtype="float32")
    ) -> R.Tensor((1, 2, 1, 1), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 2, 1, 1), dtype="float32") = R.mean(data, axis=[2, 3], keepdims=True)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__0_1_2__1:
    @R.function
    def main(
        data: R.Tensor((8, 7, 6, 5), dtype="float32")
    ) -> R.Tensor((1, 1, 1, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 1, 1, 5), dtype="float32") = R.mean(
                data, axis=[0, 1, 2], keepdims=True
            )
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__0_1_2__1:
    @R.function
    def main(
        data: R.Tensor((1, 2, 3, 4), dtype="float32")
    ) -> R.Tensor((1, 1, 1, 4), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 1, 1, 4), dtype="float32") = R.mean(
                data, axis=[0, 1, 2], keepdims=True
            )
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__0_1_3__1:
    @R.function
    def main(
        data: R.Tensor((8, 7, 6, 5), dtype="float32")
    ) -> R.Tensor((1, 1, 6, 1), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 1, 6, 1), dtype="float32") = R.mean(
                data, axis=[0, 1, 3], keepdims=True
            )
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__0_1_3__1:
    @R.function
    def main(
        data: R.Tensor((1, 2, 3, 4), dtype="float32")
    ) -> R.Tensor((1, 1, 3, 1), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 1, 3, 1), dtype="float32") = R.mean(
                data, axis=[0, 1, 3], keepdims=True
            )
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__0_2_3__1:
    @R.function
    def main(
        data: R.Tensor((8, 7, 6, 5), dtype="float32")
    ) -> R.Tensor((1, 7, 1, 1), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 7, 1, 1), dtype="float32") = R.mean(
                data, axis=[0, 2, 3], keepdims=True
            )
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__0_2_3__1:
    @R.function
    def main(
        data: R.Tensor((1, 2, 3, 4), dtype="float32")
    ) -> R.Tensor((1, 2, 1, 1), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 2, 1, 1), dtype="float32") = R.mean(
                data, axis=[0, 2, 3], keepdims=True
            )
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__1_2_3__1:
    @R.function
    def main(
        data: R.Tensor((8, 7, 6, 5), dtype="float32")
    ) -> R.Tensor((8, 1, 1, 1), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((8, 1, 1, 1), dtype="float32") = R.mean(
                data, axis=[1, 2, 3], keepdims=True
            )
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__1_2_3__1:
    @R.function
    def main(
        data: R.Tensor((1, 2, 3, 4), dtype="float32")
    ) -> R.Tensor((1, 1, 1, 1), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 1, 1, 1), dtype="float32") = R.mean(
                data, axis=[1, 2, 3], keepdims=True
            )
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__8_7_6_5__0_1_2_3__1:
    @R.function
    def main(
        data: R.Tensor((8, 7, 6, 5), dtype="float32")
    ) -> R.Tensor((1, 1, 1, 1), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 1, 1, 1), dtype="float32") = R.mean(
                data, axis=[0, 1, 2, 3], keepdims=True
            )
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReduceMeanModule__1_2_3_4__0_1_2_3__1:
    @R.function
    def main(
        data: R.Tensor((1, 2, 3, 4), dtype="float32")
    ) -> R.Tensor((1, 1, 1, 1), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 1, 1, 1), dtype="float32") = R.mean(
                data, axis=[0, 1, 2, 3], keepdims=True
            )
            R.output(gv)
        return gv


@tvm.script.ir_module
class SigmoidModule:
    @R.function
    def main(
        data: R.Tensor((7, 32, 32, 8), dtype="float32")
    ) -> R.Tensor((7, 32, 32, 8), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((7, 32, 32, 8), dtype="float32") = R.sigmoid(data)
            R.output(gv)
        return gv


@tvm.script.ir_module
class SliceModule_0:
    @R.function
    def main(x: R.Tensor((20, 10, 5), dtype="float32")) -> R.Tensor((3, 10, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((3, 10, 5), dtype="float32") = R.strided_slice(
                x, axes=[0, 1], begin=[0, 0], end=[3, 10], strides=[1, 1]
            )
            R.output(gv)
        return gv


@tvm.script.ir_module
class SliceModule_1:
    @R.function
    def main(x: R.Tensor((20, 10, 5), dtype="float32")) -> R.Tensor((3, 10, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((3, 10, 5), dtype="float32") = R.strided_slice(
                x, axes=[0, 1], begin=[0, 0], end=[3, 10], strides=[1, 1]
            )
            R.output(gv)
        return gv


@tvm.script.ir_module
class SliceModule_2:
    @R.function
    def main(x: R.Tensor((20, 10, 5), dtype="float32")) -> R.Tensor((20, 4, 2), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((20, 4, 2), dtype="float32") = R.strided_slice(
                x, axes=[0, 1, 2], begin=[20, 10, 4], end=[0, 0, 1], strides=[-1, -3, -2]
            )
            R.output(gv)
        return gv


@tvm.script.ir_module
class SliceModule_3:
    @R.function
    def main(x: R.Tensor((20, 10, 5), dtype="float32")) -> R.Tensor((20, 3, 10), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((20, 3, 10), dtype="float32") = R.strided_slice(
                x, axes=[1, 2], begin=[0, 0], end=[3, 10], strides=[1, 1]
            )
            R.output(gv)
        return gv


@tvm.script.ir_module
class SliceModule_4:
    @R.function
    def main(
        x: R.Tensor((20, 10, 5), dtype="float32")
    ) -> R.Tensor((-10, -3, -20), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((-10, -3, -20), dtype="float32") = R.strided_slice(
                x, axes=[-1, -3, -2], begin=[20, 10, 4], end=[0, 0, 1], strides=[1, 1, 1]
            )
            R.output(gv)
        return gv


@tvm.script.ir_module
class SoftmaxModule:
    @R.function
    def main(
        a: R.Tensor((7, 32, 32, 8), dtype="float32")
    ) -> R.Tensor((7, 32, 32, 8), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((7, 32, 32, 8), dtype="float32") = R.nn.softmax(a, axis=-1)
            R.output(gv)
        return gv


@tvm.script.ir_module
class TransposeModule:
    @R.function
    def main(a: R.Tensor((7, 32, 8), dtype="float32")) -> R.Tensor((32, 7, 8), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((32, 7, 8), dtype="float32") = R.permute_dims(a, axes=[1, 0, 2])
            R.output(gv)
        return gv


@tvm.script.ir_module
class UnsqueezeModule:
    @R.function
    def main(
        a: R.Tensor((7, 32, 8), dtype="float32")
    ) -> R.Tensor((1, 7, 32, 8, 1), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((1, 7, 32, 8, 1), dtype="float32") = R.expand_dims(a, axis=[0, 4])
            R.output(gv)
        return gv


@tvm.script.ir_module
class BiasGeluModule:
    @R.function
    def main(
        a: R.Tensor((7, 32, 8), dtype="float32"), b: R.Tensor((8,), dtype="float32")
    ) -> R.Tensor((7, 32, 8), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((7, 32, 8), dtype="float32") = R.add(a, b)
            gv: R.Tensor((7, 32, 8), dtype="float32") = R.nn.gelu(lv)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ConvModule:
    @R.function
    def main(
        x: R.Tensor((3, 12, 32, 32), dtype="float32"),
        w: R.Tensor((4, 12, 3, 3), dtype="float32"),
        b: R.Tensor((4,), dtype="float32"),
    ) -> R.Tensor((3, 4, 31, 31), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((3, 4, 31, 31), dtype="float32") = R.nn.conv2d(
                x,
                w,
                strides=[1, 1],
                padding=[0, 0, 1, 1],
                dilation=[1, 1],
                groups=1,
                data_layout="NCHW",
                kernel_layout="OIHW",
                out_layout="NCHW",
                out_dtype="",
            )
            R.output(gv)
        return gv


@tvm.script.ir_module
class EqualModule:
    @R.function
    def main(
        a: R.Tensor((7, 32, 8), dtype="float32"),
        b: R.Tensor((7, 32, 8), dtype="float32"),
    ) -> R.Tensor((7, 32, 8), dtype="bool"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((7, 32, 8), dtype="bool") = R.equal(a, b)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ErfModule:
    @R.function
    def main(a: R.Tensor((7, 32, 8), dtype="float32")) -> R.Tensor((7, 32, 8), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((), dtype="float32") = R.sqrt(R.const(2, "float32"))
            lv1: R.Tensor((7, 32, 8), dtype="float32") = R.multiply(a, lv)
            lv2: R.Tensor((7, 32, 8), dtype="float32") = R.nn.gelu(lv1)
            lv3: R.Tensor((7, 32, 8), dtype="float32") = R.multiply(lv2, lv)
            lv4: R.Tensor((7, 32, 8), dtype="float32") = R.divide(lv3, a)
            gv: R.Tensor((7, 32, 8), dtype="float32") = R.add(lv4, R.const(-1, "float32"))
            R.output(gv)
        return gv


@tvm.script.ir_module
class NotModule:
    @R.function
    def main(a: R.Tensor((7, 32, 8), dtype="bool")) -> R.Tensor((7, 32, 8), dtype="bool"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((7, 32, 8), dtype="bool") = R.equal(a, R.const(False, "bool"))
            R.output(gv)
        return gv


@tvm.script.ir_module
class PowModule:
    @R.function
    def main(
        a: R.Tensor((7, 32, 8), dtype="float32"),
        b: R.Tensor((7, 32, 8), dtype="float32"),
    ) -> R.Tensor((7, 32, 8), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((7, 32, 8), dtype="float32") = R.log(a)
            lv1: R.Tensor((7, 32, 8), dtype="float32") = R.multiply(lv, b)
            gv: R.Tensor((7, 32, 8), dtype="float32") = R.exp(lv1)
            R.output(gv)
        return gv


@tvm.script.ir_module
class SqrtModule:
    @R.function
    def main(a: R.Tensor((7, 32, 8), dtype="float32")) -> R.Tensor((7, 32, 8), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((7, 32, 8), dtype="float32") = R.sqrt(a)
            R.output(gv)
        return gv


@tvm.script.ir_module
class SkipLayerNormModule:
    @R.function
    def main(
        input: R.Tensor((3, 4, 5), dtype="float32"),
        skip: R.Tensor((3, 4, 5), dtype="float32"),
        gamma: R.Tensor((5,), dtype="float32"),
        beta: R.Tensor((5,), dtype="float32"),
        bias: R.Tensor((5,), dtype="float32"),
    ) -> R.Tensor((3, 4, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            lv: R.Tensor((3, 4, 5), dtype="float32") = R.add(input, skip)
            lv1: R.Tensor((3, 4, 5), dtype="float32") = R.add(lv, bias)
            gv: R.Tensor((3, 4, 5), dtype="float32") = R.nn.layer_norm(
                lv1, gamma, beta, axes=[2], epsilon=1e-05, center=True, scale=True
            )
            R.output(gv)
        return gv


@tvm.script.ir_module
class SqueezeModule:
    @R.function
    def main(a: R.Tensor((1, 3, 1, 5), dtype="float32")) -> R.Tensor((3, 5), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((3, 5), dtype="float32") = R.squeeze(a, axis=[0, 2])
            R.output(gv)
        return gv


@tvm.script.ir_module
class SubModule:
    @R.function
    def main(
        a: R.Tensor((7, 32, 8), dtype="float32"),
        b: R.Tensor((7, 32, 8), dtype="float32"),
    ) -> R.Tensor((7, 32, 8), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((7, 32, 8), dtype="float32") = R.subtract(a, b)
            R.output(gv)
        return gv


@tvm.script.ir_module
class TanhModule:
    @R.function
    def main(a: R.Tensor((7, 32, 8), dtype="float32")) -> R.Tensor((7, 32, 8), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((7, 32, 8), dtype="float32") = R.tanh(a)
            R.output(gv)
        return gv


@tvm.script.ir_module
class WhereModule:
    @R.function
    def main(
        condition: R.Tensor((7, 32, 8), dtype="bool"),
        x: R.Tensor((7, 32, 8), dtype="float32"),
        y: R.Tensor((7, 32, 8), dtype="float32"),
    ) -> R.Tensor((7, 32, 8), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((7, 32, 8), dtype="float32") = R.where(condition, x, y)
            R.output(gv)
        return gv


@tvm.script.ir_module
class ReluModule:
    @R.function
    def main(a: R.Tensor((7, 32, 8), dtype="float32")) -> R.Tensor((7, 32, 8), dtype="float32"):
        # block 0
        with R.dataflow():
            gv: R.Tensor((7, 32, 8), dtype="float32") = R.nn.relu(a)
            R.output(gv)
        return gv


@tvm.script.ir_module
class SplitModule:
    @R.function
    def main(
        data: R.Tensor((7, 32, 8), dtype="float32")
    ) -> R.Tuple(R.Tensor((2, 32, 8), dtype="float32"), R.Tensor((3, 32, 8), dtype="float32")):
        # block 0
        with R.dataflow():
            lv: R.Tuple(
                R.Tensor((2, 32, 8), dtype="float32"),
                R.Tensor((3, 32, 8), dtype="float32"),
                R.Tensor((2, 32, 8), dtype="float32"),
            ) = R.split(data, indices_or_sections=[2, 5], axis=0)
            lv1: R.Tensor((2, 32, 8), dtype="float32") = lv[0]
            lv2: R.Tensor((3, 32, 8), dtype="float32") = lv[1]
            lv3: R.Tensor((2, 32, 8), dtype="float32") = lv[2]
            gv: R.Tuple(
                R.Tensor((2, 32, 8), dtype="float32"),
                R.Tensor((3, 32, 8), dtype="float32"),
            ) = (lv1, lv2)
            R.output(gv)
        return gv


@tvm.script.ir_module
class EmbedLayerNormModule:
    @R.function
    def main(
        input_ids: R.Tensor((8, 64), dtype="int64"),
        segment_ids: R.Tensor((8, 64), dtype="int64"),
        word_embedding: R.Tensor((30522, 128), dtype="float32"),
        position_embedding: R.Tensor((512, 128), dtype="float32"),
        segment_embedding: R.Tensor((2, 128), dtype="float32"),
        gamma: R.Tensor((128,), dtype="float32"),
        beta: R.Tensor((128,), dtype="float32"),
        mask: R.Tensor((8, 64), dtype="bool"),
        position_ids: R.Tensor((8, 64), dtype="int64"),
    ) -> R.Tuple(
        R.Tensor((8, 64, 128), dtype="float32"),
        R.Tensor((8,), dtype="bool"),
        R.Tensor((8, 64, 128), dtype="float32"),
    ):
        # block 0
        with R.dataflow():
            lv: R.Tensor((512,), dtype="int64") = R.reshape(input_ids, (512,))
            lv1: R.Tensor((512, 128), dtype="float32") = R.take(word_embedding, lv, axis=0)
            lv2: R.Tensor((512,), dtype="int64") = R.reshape(segment_ids, (512,))
            lv3: R.Tensor((512, 128), dtype="float32") = R.take(segment_embedding, lv2, axis=0)
            lv4: R.Tensor((512,), dtype="int64") = R.reshape(position_ids, (512,))
            lv5: R.Tensor((512, 128), dtype="float32") = R.take(position_embedding, lv4, axis=0)
            lv6: R.Tensor((8, 64, 128), dtype="float32") = R.reshape(lv1, (8, 64, 128))
            lv7: R.Tensor((8, 64, 128), dtype="float32") = R.reshape(lv5, (8, 64, 128))
            lv8: R.Tensor((8, 64, 128), dtype="float32") = R.add(lv6, lv7)
            lv9: R.Tensor((8, 64, 128), dtype="float32") = R.reshape(lv3, (8, 64, 128))
            lv10: R.Tensor((8, 64, 128), dtype="float32") = R.add(lv8, lv9)
            lv11: R.Tensor((512, 128), dtype="float32") = R.reshape(lv10, (512, 128))
            lv12: R.Tensor((512, 128), dtype="float32") = R.nn.layer_norm(
                lv11, gamma, beta, axes=[1], epsilon=1e-12, center=True, scale=True
            )
            lv13: R.Tensor((8, 64, 128), dtype="float32") = R.reshape(lv12, (8, 64, 128))
            lv14: R.Tensor((8,), dtype="bool") = R.sum(mask, axis=[1], keepdims=False)
            gv: R.Tuple(
                R.Tensor((8, 64, 128), dtype="float32"),
                R.Tensor((8,), dtype="bool"),
                R.Tensor((8, 64, 128), dtype="float32"),
            ) = (lv13, lv14, lv10)
            R.output(gv)
        return gv


# pylint: enable=no-self-argument,missing-class-docstring,missing-function-docstring,invalid-name


def test_concat():
    """Test case for concat op."""
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
    assert tvm.ir.structural_equal(relax.from_onnx(model), ConcatModule)


def test_matmul():
    """Test case for matmul op."""
    matmul_node = helper.make_node("MatMul", ["a", "b"], ["c"])

    graph = helper.make_graph(
        [matmul_node],
        "matmul_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [24, 32]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [32, 16]),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, [24, 16])],
    )

    model = helper.make_model(graph, producer_name="matmul_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), MatMulModule)


def test_add():
    """Test case for add op."""
    add_node = helper.make_node("Add", ["a", "b"], ["c"])

    graph = helper.make_graph(
        [add_node],
        "add_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [24, 32]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [24, 32]),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, [24, 32])],
    )

    model = helper.make_model(graph, producer_name="add_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), AddModule)


@pytest.mark.parametrize(
    "from_type", [TensorProto.INT32, TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE]
)
@pytest.mark.parametrize(
    "to_type", [TensorProto.INT32, TensorProto.FLOAT, TensorProto.FLOAT16, TensorProto.DOUBLE]
)
def test_cast(from_type: int, to_type: int):
    """Test case for cast op."""
    cast_node = helper.make_node("Cast", ["a"], ["a_cast"], to=to_type)

    graph = helper.make_graph(
        [cast_node],
        "cast_test",
        inputs=[helper.make_tensor_value_info("a", from_type, [24, 32])],
        outputs=[helper.make_tensor_value_info("a_cast", to_type, [24, 32])],
    )

    model = helper.make_model(graph, producer_name="cast_test")
    assert tvm.ir.structural_equal(
        relax.from_onnx(model),
        eval(f"CastModule_{from_type}_{to_type}"),  # pylint: disable=eval-used
    )


@pytest.mark.parametrize("axis", [0, 1, 2, 3])
def test_gather(axis: int):
    """Test case for gather op."""
    gather_node = helper.make_node("Gather", ["data", "indices"], ["output"], axis=axis)

    data_shape = [5, 4, 3, 2]
    indices_shape = [2]
    output_shape = data_shape
    output_shape[axis] = indices_shape[0]
    graph = helper.make_graph(
        [gather_node],
        "gather_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, data_shape),
            helper.make_tensor_value_info("indices", TensorProto.INT32, indices_shape),
        ],
        outputs=[helper.make_tensor_value_info("output", TensorProto.FLOAT, output_shape)],
    )

    model = helper.make_model(graph, producer_name="gather_test")
    assert tvm.ir.structural_equal(
        relax.from_onnx(model),
        eval(f"GatherModule_{axis}"),  # pylint: disable=eval-used
    )


@pytest.mark.parametrize("alpha", [None, 0.25])
@pytest.mark.parametrize("beta", [None, 0.35])
@pytest.mark.parametrize("trans_a", [False, True])
@pytest.mark.parametrize("trans_b", [False, True])
@pytest.mark.parametrize("use_c", [False, True])
def test_gemm(
    alpha: Optional[float],
    beta: Optional[float],
    trans_a: bool,
    trans_b: bool,
    use_c: bool,
):
    """Test case for gemm op."""
    if use_c:
        gemm_node = helper.make_node(
            "Gemm", ["a", "b", "c"], ["y"], alpha=alpha, beta=beta, transA=trans_a, transB=trans_b
        )
    else:
        gemm_node = helper.make_node(
            "Gemm", ["a", "b"], ["y"], alpha=alpha, beta=beta, transA=trans_a, transB=trans_b
        )

    shape_a = [3, 4]
    shape_b = [4, 5]
    if trans_a:
        shape_a = shape_a[::-1]
    if trans_b:
        shape_b = shape_b[::-1]

    inputs = [
        helper.make_tensor_value_info("a", TensorProto.FLOAT, shape_a),
        helper.make_tensor_value_info("b", TensorProto.FLOAT, shape_b),
    ]
    if use_c:
        inputs.append(helper.make_tensor_value_info("c", TensorProto.FLOAT, [1, 5]))

    graph = helper.make_graph(
        [gemm_node],
        "gemm_test",
        inputs=inputs,
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 5])],
    )

    model = helper.make_model(graph, producer_name="gemm_test")
    assert tvm.ir.structural_equal(
        relax.from_onnx(model),
        eval(  # pylint: disable=eval-used
            f"GemmModule_{str(alpha).replace('.', '_')}"
            + f"_{str(beta).replace('.', '_')}_{trans_a}_{trans_b}_{use_c}"
        ),
    )


def test_mul():
    """Test case for mul op."""
    mul_node = helper.make_node("Mul", ["a", "b"], ["c"])

    graph = helper.make_graph(
        [mul_node],
        "mul_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [24, 32]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [24, 32]),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, [24, 32])],
    )

    model = helper.make_model(graph, producer_name="mul_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), MulModule)


@pytest.mark.parametrize(
    "case_num, in_shape, shape, out_shape",
    [(0, [7, 32, 32, 8], [224, 256], [224, 256]), (1, [7, 32, 32, 8], [-1, 8192], [7, 8192])],
)
def test_reshape(case_num: int, in_shape: Tuple[int], shape: Tuple[int], out_shape: Tuple[int]):
    """Test case for reshape op."""
    reshape_node = helper.make_node("Reshape", ["data", "shape"], ["reshaped"])

    graph = helper.make_graph(
        [reshape_node],
        "reshape_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, in_shape),
        ],
        initializer=[helper.make_tensor("shape", TensorProto.INT64, [len(shape)], shape)],
        outputs=[helper.make_tensor_value_info("reshaped", TensorProto.FLOAT, out_shape)],
    )

    model = helper.make_model(graph, producer_name="reshape_test")
    assert tvm.ir.structural_equal(
        relax.from_onnx(model),
        eval(f"ReshapeModuleConstant_{case_num}"),  # pylint: disable=eval-used
    )


def test_shape():
    """Test case for shape op."""
    shape_node = helper.make_node("Shape", ["data"], ["shape"])

    graph = helper.make_graph(
        [shape_node],
        "shape_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [7, 32, 32, 8]),
        ],
        outputs=[helper.make_tensor_value_info("shape", TensorProto.INT64, [4])],
    )

    model = helper.make_model(graph, producer_name="shape_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), ShapeModule)


def test_clip():
    """Test case for clip op."""
    clip_node = helper.make_node("Clip", ["data", "min", "max"], ["clipped"])

    graph = helper.make_graph(
        [clip_node],
        "clip_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [7, 32, 32, 8]),
        ],
        initializer=[
            # TODO: TVMScript not supporting float as prim value here
            helper.make_tensor("min", TensorProto.INT64, (), [2]),
            helper.make_tensor("max", TensorProto.INT64, (), [3]),
        ],
        outputs=[helper.make_tensor_value_info("clipped", TensorProto.FLOAT, [7, 32, 32, 8])],
    )

    model = helper.make_model(graph, producer_name="clip_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), ClipModule)


def test_div():
    """Test case for div op."""
    div_node = helper.make_node("Div", ["a", "b"], ["c"])

    graph = helper.make_graph(
        [div_node],
        "div_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [24, 32]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [24, 32]),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, [24, 32])],
    )

    model = helper.make_model(graph, producer_name="div_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), DivModule)


def test_expand():
    """Test case for expand op."""
    in_shape = [3, 1]
    out_shape = [3, 4]
    expand_node = helper.make_node("Expand", ["data", "shape"], ["expanded"])

    graph = helper.make_graph(
        [expand_node],
        "expand_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, in_shape),
            helper.make_tensor_value_info("shape", TensorProto.INT64, [len(out_shape)]),
        ],
        # output shape depends on input of shape
        outputs=[helper.make_tensor_value_info("expanded", TensorProto.FLOAT, out_shape)],
    )

    model = helper.make_model(graph, producer_name="expand_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), ExpandModule)


@pytest.mark.parametrize("input_shape", [[8, 7, 6, 5], [1, 2, 3, 4]])
@pytest.mark.parametrize(
    "axes",
    [
        [0],
        [1],
        [2],
        [3],
        [0, 1],
        [0, 2],
        [0, 3],
        [1, 2],
        [1, 3],
        [2, 3],
        [0, 1, 2],
        [0, 1, 3],
        [0, 2, 3],
        [1, 2, 3],
        [0, 1, 2, 3],
    ],
)
@pytest.mark.parametrize("keepdims", [0, 1])
def test_reduce_mean(input_shape: List[int], axes: List[int], keepdims: int):
    """Test case for reduce_mean op."""
    reduce_mean_node = helper.make_node(
        "ReduceMean", ["data"], ["reduced"], axes=axes, keepdims=keepdims
    )
    output_shape = []
    for i, shape_var in enumerate(input_shape):
        if i not in axes or keepdims == 1:
            output_shape.append(shape_var)
    graph = helper.make_graph(
        [reduce_mean_node],
        "reduce_mean_test",
        inputs=[
            helper.make_tensor_value_info(
                "data",
                TensorProto.FLOAT,
                input_shape,
            ),
        ],
        outputs=[helper.make_tensor_value_info("reduced", TensorProto.FLOAT, [24])],
    )

    model = helper.make_model(graph, producer_name="reduce_mean_test")
    assert tvm.ir.structural_equal(
        relax.from_onnx(model),
        eval(  # pylint: disable=eval-used
            f"ReduceMeanModule__{'_'.join([str(x) for x in input_shape])}"
            f"__{'_'.join([str(x) for x in axes])}__{keepdims}"
        ),
    )


def test_sigmoid():
    """Test case for sigmoid op."""
    sigmoid_node = helper.make_node("Sigmoid", ["data"], ["sigmoid"])

    graph = helper.make_graph(
        [sigmoid_node],
        "sigmoid_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [7, 32, 32, 8]),
        ],
        outputs=[helper.make_tensor_value_info("sigmoid", TensorProto.FLOAT, [7, 32, 32, 8])],
    )

    model = helper.make_model(graph, producer_name="sigmoid_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), SigmoidModule)


def test_slice():
    """Test case for slice op."""

    def verify_slice(
        case_num: int,
        data_shape: List[int],
        output_shape: List[int],
        starts: Union[List[int], np.ndarray],
        ends: Union[List[int], np.ndarray],
        axes: Union[List[int], np.ndarray] = None,
        steps: Union[List[int], np.ndarray] = None,
    ):
        if isinstance(starts, list):
            starts = np.array(starts, "int64")
        if isinstance(ends, list):
            ends = np.array(ends, "int64")
        if isinstance(axes, list):
            axes = np.array(axes, "int64")
        if isinstance(steps, list):
            steps = np.array(steps, "int64")

        slice_inputs = ["x", "starts", "ends"]
        initializer = [
            helper.make_tensor("starts", TensorProto.INT64, starts.shape, starts),
            helper.make_tensor("ends", TensorProto.INT64, ends.shape, ends),
        ]

        if axes is not None:
            initializer.append(helper.make_tensor("axes", TensorProto.INT64, axes.shape, axes))
            slice_inputs.append("axes")
        if steps is not None:
            initializer.append(helper.make_tensor("steps", TensorProto.INT64, steps.shape, steps))
            slice_inputs.append("steps")

        slice_node = helper.make_node("Slice", inputs=slice_inputs, outputs=["y"])

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
        assert tvm.ir.structural_equal(
            relax.from_onnx(model),
            eval("SliceModule_" + str(case_num)),  # pylint: disable=eval-used
        ), (
            "Case " + str(case_num) + " failed"
        )

    # Test with all parameters set.
    verify_slice(0, [20, 10, 5], [3, 10, 5], starts=[0, 0], ends=[3, 10], axes=[0, 1], steps=[1, 1])
    # Test with default axes and steps.
    verify_slice(1, [20, 10, 5], [3, 10, 5], starts=[0, 0], ends=[3, 10])
    # Test with negative steps.
    verify_slice(
        2,
        [20, 10, 5],
        [19, 3, 2],
        starts=[20, 10, 4],
        ends=[0, 0, 1],
        steps=[-1, -3, -2],
        axes=[0, 1, 2],
    )
    verify_slice(3, [20, 10, 5], [10, 5], starts=[0, 0], ends=[3, 10], axes=[1, 2])
    verify_slice(4, [20, 10, 5], [19, 3, 2], starts=[20, 10, 4], ends=[0, 0, 1], steps=[-1, -3, -2])


def test_softmax():
    """Test case for softmax op."""
    softmax_node = helper.make_node("Softmax", ["a"], ["b"])

    graph = helper.make_graph(
        [softmax_node],
        "softmax_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [7, 32, 32, 8]),
        ],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [7, 32, 32, 8])],
    )

    model = helper.make_model(graph, producer_name="softmax_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), SoftmaxModule)


def test_transpose():
    """Test case for transpose op."""
    transpose_node = helper.make_node("Transpose", ["a"], ["b"], perm=[1, 0, 2])

    graph = helper.make_graph(
        [transpose_node],
        "transpose_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [7, 32, 8]),
        ],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [32, 7, 8])],
    )

    model = helper.make_model(graph, producer_name="transpose_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), TransposeModule)


def test_unsqueeze():
    """Test case for unsqueeze op."""
    unsqueeze_node = helper.make_node("Unsqueeze", ["a", "axes"], ["b"])

    graph = helper.make_graph(
        [unsqueeze_node],
        "unsqueeze_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [7, 32, 8]),
        ],
        initializer=[
            helper.make_tensor("axes", TensorProto.INT64, [2], [0, 4]),
        ],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [1, 7, 32, 8, 1])],
    )

    model = helper.make_model(graph, producer_name="unsqueeze_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), UnsqueezeModule)


def test_bias_gelu():
    """Test case for bias_gelu op."""
    bias_gelu_node = helper.make_node("BiasGelu", ["a", "b"], ["c"])

    graph = helper.make_graph(
        [bias_gelu_node],
        "bias_gelu_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [7, 32, 8]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [8]),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, [7, 32, 8])],
    )

    model = helper.make_model(graph, producer_name="bias_gelu_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), BiasGeluModule)


def test_conv():
    """Test case for conv op."""
    conv_node = helper.make_node("Conv", ["x", "w", "b"], ["y"], auto_pad="SAME_UPPER")
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
    assert tvm.ir.structural_equal(relax.from_onnx(model), ConvModule)


def test_equal():
    """Test case for equal op."""
    equal_node = helper.make_node("Equal", ["a", "b"], ["c"])

    graph = helper.make_graph(
        [equal_node],
        "equal_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [7, 32, 8]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [7, 32, 8]),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.BOOL, [7, 32, 8])],
    )

    model = helper.make_model(graph, producer_name="equal_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), EqualModule)


def test_erf():
    """Test case for erf op."""
    erf_node = helper.make_node("Erf", ["a"], ["b"])

    graph = helper.make_graph(
        [erf_node],
        "erf_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [7, 32, 8]),
        ],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [7, 32, 8])],
    )

    model = helper.make_model(graph, producer_name="erf_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), ErfModule)


def test_not():
    """Test case for not op."""
    not_node = helper.make_node("Not", ["a"], ["b"])

    graph = helper.make_graph(
        [not_node],
        "not_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.BOOL, [7, 32, 8]),
        ],
        outputs=[helper.make_tensor_value_info("b", TensorProto.BOOL, [7, 32, 8])],
    )

    model = helper.make_model(graph, producer_name="not_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), NotModule)


def test_pow():
    """Test case for pow op."""
    pow_node = helper.make_node("Pow", ["a", "b"], ["c"])

    graph = helper.make_graph(
        [pow_node],
        "pow_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [7, 32, 8]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [7, 32, 8]),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, [7, 32, 8])],
    )

    model = helper.make_model(graph, producer_name="pow_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), PowModule)


def test_sqrt():
    """Test case for sqrt op."""
    sqrt_node = helper.make_node("Sqrt", ["a"], ["b"])

    graph = helper.make_graph(
        [sqrt_node],
        "sqrt_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [7, 32, 8]),
        ],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [7, 32, 8])],
    )

    model = helper.make_model(graph, producer_name="sqrt_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), SqrtModule)


def test_skip_layer_normalization():
    """Test case for skip_layer_normalization op."""
    skip_layer_normalization_node = helper.make_node(
        "SkipLayerNormalization", ["input", "skip", "gamma", "beta", "bias"], ["y"], epsilon=1e-5
    )

    graph = helper.make_graph(
        [skip_layer_normalization_node],
        "skip_layer_normalization_test",
        inputs=[
            helper.make_tensor_value_info("input", TensorProto.FLOAT, [3, 4, 5]),
            helper.make_tensor_value_info("skip", TensorProto.FLOAT, [3, 4, 5]),
            helper.make_tensor_value_info("gamma", TensorProto.FLOAT, [5]),
            helper.make_tensor_value_info("beta", TensorProto.FLOAT, [5]),
            helper.make_tensor_value_info("bias", TensorProto.FLOAT, [5]),
        ],
        outputs=[helper.make_tensor_value_info("y", TensorProto.FLOAT, [3, 4, 5])],
    )

    model = helper.make_model(graph, producer_name="skip_layer_normalization_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), SkipLayerNormModule)


def test_squeeze():
    """Test case for squeeze op."""
    squeeze_node = helper.make_node("Squeeze", ["a", "axes"], ["b"])

    graph = helper.make_graph(
        [squeeze_node],
        "squeeze_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [1, 3, 1, 5]),
        ],
        initializer=[
            helper.make_tensor("axes", TensorProto.INT64, [2], [0, 2]),
        ],
        outputs=[helper.make_tensor_value_info("b", TensorProto.INT64, [3, 5])],
    )

    model = helper.make_model(graph, producer_name="squeeze_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), SqueezeModule)


def test_sub():
    """Test case for sub op."""
    sub_node = helper.make_node("Sub", ["a", "b"], ["c"])

    graph = helper.make_graph(
        [sub_node],
        "sub_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [7, 32, 8]),
            helper.make_tensor_value_info("b", TensorProto.FLOAT, [7, 32, 8]),
        ],
        outputs=[helper.make_tensor_value_info("c", TensorProto.FLOAT, [7, 32, 8])],
    )

    model = helper.make_model(graph, producer_name="sub_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), SubModule)


def test_tanh():
    """Test case for tanh op."""
    tanh_node = helper.make_node("Tanh", ["a"], ["b"])

    graph = helper.make_graph(
        [tanh_node],
        "tanh_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [7, 32, 8]),
        ],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [7, 32, 8])],
    )

    model = helper.make_model(graph, producer_name="tanh_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), TanhModule)


def test_where():
    """Test case for where op."""
    where_node = helper.make_node("Where", ["condition", "x", "y"], ["z"])

    graph = helper.make_graph(
        [where_node],
        "where_test",
        inputs=[
            helper.make_tensor_value_info("condition", TensorProto.BOOL, [7, 32, 8]),
            helper.make_tensor_value_info("x", TensorProto.FLOAT, [7, 32, 8]),
            helper.make_tensor_value_info("y", TensorProto.FLOAT, [7, 32, 8]),
        ],
        outputs=[helper.make_tensor_value_info("z", TensorProto.FLOAT, [7, 32, 8])],
    )

    model = helper.make_model(graph, producer_name="where_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), WhereModule)


def test_const():
    """Test case for constant op."""
    shape = [32, 16]
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
    relax.from_onnx(model)

    # TODO: figure out how to get metadata to work with Relax TVMScript
    # @tvm.script.ir_module
    # class Module:
    #     @R.function
    #     def main() -> R.Tensor((32, 16), dtype="float32"):
    #         # block 0
    #         with R.dataflow():
    #             gv: R.Tensor((32, 16), dtype="float32") = metadata["relax.expr.Constant"][0]
    #             R.output(gv)
    #         return gv


def test_relu():
    """Test case for relu op."""
    relu_node = helper.make_node("Relu", ["a"], ["b"])

    graph = helper.make_graph(
        [relu_node],
        "relu_test",
        inputs=[
            helper.make_tensor_value_info("a", TensorProto.FLOAT, [7, 32, 8]),
        ],
        outputs=[helper.make_tensor_value_info("b", TensorProto.FLOAT, [7, 32, 8])],
    )

    model = helper.make_model(graph, producer_name="relu_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), ReluModule)


def test_split():
    """Test case for split op."""
    split_node = helper.make_node("Split", ["data", "split"], ["y0", "y1"])

    graph = helper.make_graph(
        [split_node],
        "split_test",
        inputs=[
            helper.make_tensor_value_info("data", TensorProto.FLOAT, [7, 32, 8]),
        ],
        initializer=[
            helper.make_tensor("split", TensorProto.INT64, [2], [2, 5]),
        ],
        outputs=[
            helper.make_tensor_value_info("y0", TensorProto.FLOAT, [2, 32, 8]),
            helper.make_tensor_value_info("y1", TensorProto.FLOAT, [5, 32, 8]),
        ],
    )

    model = helper.make_model(graph, producer_name="split_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), SplitModule)


def test_embed_layer_normalization():
    """Test case for embed layer normalization op."""
    embed_node = helper.make_node(
        "EmbedLayerNormalization",
        [
            "input_ids",
            "segment_ids",
            "word_embedding",
            "position_embedding",
            "segment_embedding",
            "gamma",
            "beta",
            "mask",
            "position_ids",
        ],
        ["output", "mask_index", "embedding_sum"],
    )

    graph = helper.make_graph(
        [embed_node],
        "embed_layer_normalization_test",
        inputs=[
            helper.make_tensor_value_info("input_ids", TensorProto.INT64, [8, 64]),
            helper.make_tensor_value_info("segment_ids", TensorProto.INT64, [8, 64]),
            helper.make_tensor_value_info("word_embedding", TensorProto.FLOAT, [30522, 128]),
            helper.make_tensor_value_info("position_embedding", TensorProto.FLOAT, [512, 128]),
            helper.make_tensor_value_info("segment_embedding", TensorProto.FLOAT, [2, 128]),
            helper.make_tensor_value_info("gamma", TensorProto.FLOAT, [128]),
            helper.make_tensor_value_info("beta", TensorProto.FLOAT, [128]),
            helper.make_tensor_value_info("mask", TensorProto.BOOL, [8, 64]),
            helper.make_tensor_value_info("position_ids", TensorProto.INT64, [8, 64]),
        ],
        outputs=[
            helper.make_tensor_value_info("output", TensorProto.FLOAT, [8, 64, 128]),
            helper.make_tensor_value_info("mask_index", TensorProto.INT64, [8]),
            helper.make_tensor_value_info("embedding_sum", TensorProto.FLOAT, [8, 64, 128]),
        ],
    )

    model = helper.make_model(graph, producer_name="embed_layer_normalization_test")
    assert tvm.ir.structural_equal(relax.from_onnx(model), EmbedLayerNormModule)


if __name__ == "__main__":
    tvm.testing.main()
