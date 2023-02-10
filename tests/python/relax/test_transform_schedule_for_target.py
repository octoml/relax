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

from typing import List
import tempfile

import tvm
import tvm.testing
import tvm.meta_schedule as ms
from tvm import relax
from tvm.ir import transform
from tvm.ir.module import IRModule
from tvm.ir.transform import PassContext
from tvm.relax.transform.tuning_api import Trace
from tvm.script import relax as R
from tvm.script import tir as T


@tvm.script.ir_module
class InputModule:
    @T.prim_func
    def tir_matmul(x: T.handle, y: T.handle, z: T.handle) -> None:
        T.func_attr({"global_symbol": "tir_matmul"})
        k = T.var("int32")
        A = T.match_buffer(x, (32, 32))
        B = T.match_buffer(y, (32, 32))
        C = T.match_buffer(z, (32, 32))

        for (i0, j0, k0) in T.grid(32, 32, 32):
            with T.block():
                i, j, k = T.axis.remap("SSR", [i0, j0, k0])
                with T.init():
                    C[i, j] = 0.0
                C[i, j] += A[i, k] * B[j, k]

    @T.prim_func
    def tir_relu(x: T.handle, y: T.handle):
        T.func_attr({"global_symbol": "tir_relu"})
        A = T.match_buffer(x, (32, 32))
        B = T.match_buffer(y, (32, 32))
        for (i, j) in T.grid(32, 32):
            with T.block():
                vi, vj = T.axis.remap("SS", [i, j])
                B[vi, vj] = T.max(A[vi, vj], 0.0)

    @R.function
    def main(x: R.Tensor((32, 32), "float32"), w: R.Tensor((32, 32), "float32")) -> R.Tensor:
        with R.dataflow():
            lv0 = R.call_tir(tir_matmul, (x, w), R.Tensor((32, 32), dtype="float32"))
            lv1 = R.call_tir(tir_relu, (lv0), R.Tensor((32, 32), dtype="float32"))
            R.output(lv1)
        return lv1


@tvm.testing.parametrize_targets(
    "llvm --num-cores=2",
    "cuda -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -registers_per_block=65536 -thread_warp_size=32",
)
def test_schedule_for_target(target, dev):
    # Check that input device is valid on current machine.
    if dev in [enabled_dev for _, enabled_dev in tvm.testing.enabled_targets()]:
        mod = InputModule
        assert isinstance(mod, IRModule)
        out_mod = relax.transform.ScheduleForTarget(target)(mod)
        assert not tvm.ir.structural_equal(mod, out_mod)

        # Perform an additional check depending on the target.
        if dev == tvm.cpu():
            # On CPU, parallelization should have been inserted.
            assert "parallel" in out_mod.astext()
        if dev == tvm.gpu():
            # On GPU, thread bindings should have been inserted.
            assert "thread_binding" in out_mod.astext()


if __name__ == "__main__":
    tvm.testing.main()
