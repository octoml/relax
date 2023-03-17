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
# pylint: disable=invalid-name, abstract-method, unused-argument
"Scheduling transformation for thrust cumsum on cuda."
import tvm
from tvm import relax, topi
from tvm.relax.expr_functor import PyExprMutator


@tvm.ir.transform.module_pass(opt_level=0)
class ScheduleCumsum(tvm.transform.Pass):
    """Transformation pass that legalizes cumsum for gpu and schedules it."""

    def transform_module(self, mod: tvm.IRModule, ctx: tvm.transform.PassContext) -> tvm.IRModule:
        """
        Expand any relax.cumsum operators into TIR and schedule them.
        Should only be used on GPU.
        """

        @relax.expr_functor.visitor
        class RewriteCumsum(PyExprMutator):
            """Visitor that expands cumsum into tensorir implemenation."""

            def visit_call_(self, op):
                if op.op.name == "relax.cumsum":
                    # Convert high level relax call to topi.
                    op = self.builder_.call_te(
                        topi.cuda.cumsum, op.args[0], op.attrs.axis, op.attrs.dtype
                    )
                return super().visit_call_(op)

            def transform(self):
                """Iterate over functions and expand any cumsum ops found."""
                for gv, func in mod.functions.items():
                    if isinstance(func, relax.Function):
                        func = self.visit_expr(func)
                    self.builder_.update_func(gv, func)
                new_mod = self.builder_.get()
                new_mod = new_mod.with_attrs(mod.attrs) if mod.attrs else new_mod
                return new_mod

        new_mod = RewriteCumsum().transform()
        # Bind cumsum add to cuda threads.
        sch = tvm.tir.Schedule(new_mod)
        for gv, _ in new_mod.functions.items():
            if "cumsum" in gv.name_hint:
                sch.work_on(gv.name_hint)
                t_add = sch.get_block("T_add")
                i0, i1 = sch.get_loops(t_add)
                sch.bind(i0, "blockIdx.x")
                sch.bind(i1, "threadIdx.x")
        return sch.mod
