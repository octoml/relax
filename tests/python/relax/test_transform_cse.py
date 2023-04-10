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
"""Test eliminate common subexpr pass"""
import tvm
import tvm.testing
from tvm.relax.transform import EliminateCommonSubexpr
from tvm.script.parser import ir as I, relax as R, tir as T

import numpy as np


def verify(input, expected):
    tvm.ir.assert_structural_equal(EliminateCommonSubexpr()(input), expected)


def test_simple():
    @I.ir_module
    class Before:
        @R.function
        def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")):
            with R.dataflow():
                lv0 = R.add(x, y)
                lv1 = R.add(x, y)
                gv = R.multiply(lv0, lv1)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def foo(x: R.Tensor((2, 3), dtype="float32"), y: R.Tensor((2, 3), dtype="float32")):
            with R.dataflow():
                lv0 = R.add(x, y)
                # can combine with canonicalizing bindings
                # and getting rid of unused bindings to eliminate this line too
                lv1 = lv0
                gv = R.multiply(lv0, lv1)
                R.output(gv)
            return gv

    verify(Before, Expected)


def test_constants():
    @I.ir_module
    class Before:
        @R.function
        def foo() -> R.Tuple(R.Tensor((), dtype="int32"), R.Tensor((2, 2), dtype="int32")):
            with R.dataflow():
                # we are not going to bind the constant 1 to a var
                lv0 = R.add(R.const(1, dtype="int32"), R.const(1, dtype="int32"))
                # we expect to bind the repeated large constants
                lv1 = R.add(
                    R.const(tvm.nd.array(np.zeros((2, 2), dtype="int32"))),
                    R.const(tvm.nd.array(np.zeros((2, 2), dtype="int32"))),
                )
                gv = (lv0, lv1)
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def foo() -> R.Tuple(R.Tensor((), dtype="int32"), R.Tensor((2, 2), dtype="int32")):
            with R.dataflow():
                lv0 = R.add(R.const(1, dtype="int32"), R.const(1, dtype="int32"))
                lv1 = R.const(tvm.nd.array(np.zeros((2, 2), dtype="int32")))
                lv2 = R.add(lv1, lv1)
                gv = (lv0, lv2)
                R.output(gv)
            return gv

    verify(Before, Expected)


def test_repeated_inner_tuples():
    @I.ir_module
    class Before:
        @R.function
        def foo(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
            with R.dataflow():
                # repeated units: (x, x), (x, (x, x)), ((x, x), (x, (x, x)))
                tup = (((x, x), (x, (x, x))), ((x, x), (x, (x, x))), (x, (x, x)))
                gv = tup[0][0][1]
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def foo(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
            with R.dataflow():
                t1 = (x, x)
                t2 = (x, t1)
                t3 = (t1, t2)
                t4 = (t3, t3, t2)
                gv = t4[0][0][1]
                R.output(gv)
            return gv

    verify(Before, Expected)


def test_inner_function():
    @I.ir_module
    class Before:
        @R.function
        def foo(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
            with R.dataflow():
                # we are going to do CSE inside the local function
                @R.function
                def bar(y: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
                    # not in dataflow: should not be touched
                    z = R.add(R.add(y, y), R.add(y, y))
                    with R.dataflow():
                        # writing this out in ANF to illustrate why CSE behaves as it does
                        # result of ANF transforming R.add(R.add(y, y), R.add(y, y))
                        lv0 = R.add(y, y)
                        lv1 = R.add(y, y)
                        lv2 = R.add(lv0, lv1)
                        gv = lv2
                        R.output(gv)
                    return R.add(z, gv)

                # also making the ANF explicit to better illustrate the result of CSE
                # result of ANF transforming R.add(R.add(bar(x), bar(x)), R.add(bar(x), bar(x)))
                lv0 = bar(x)
                lv1 = bar(x)
                lv2 = R.add(lv0, lv1)
                lv3 = bar(x)
                lv4 = bar(x)
                lv5 = R.add(lv3, lv4)
                lv6 = R.add(lv2, lv5)
                gv = lv6
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def foo(x: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
            with R.dataflow():

                @R.function
                def bar(y: R.Tensor((), dtype="int32")) -> R.Tensor((), dtype="int32"):
                    z = R.add(R.add(y, y), R.add(y, y))
                    with R.dataflow():
                        lv0 = R.add(y, y)
                        lv1 = lv0
                        lv2 = R.add(lv0, lv1)
                        gv = lv2
                        R.output(gv)
                    return R.add(z, gv)

                # can further clean this up
                # using canonicalize bindings, eliminate unused bindings, and CSE again
                lv0 = bar(x)
                lv1 = lv0
                lv2 = R.add(lv0, lv1)
                lv3 = lv0
                lv4 = lv0
                lv5 = R.add(lv3, lv4)
                lv6 = R.add(lv2, lv5)
                gv = lv6
                R.output(gv)
            return gv

    verify(Before, Expected)


if __name__ == "__main__":
    tvm.testing.main()
