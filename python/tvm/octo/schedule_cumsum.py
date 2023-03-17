import tvm
from tvm import relax, topi
from tvm.relax.expr_functor import PyExprMutator


@tvm.ir.transform.module_pass(opt_level=0)
class ScheduleCumsum(tvm.transform.Pass):
    def transform_module(self, mod, ctx):
        @relax.expr_functor.visitor
        class RewriteCumsum(PyExprMutator):
            def visit_call_(self, op):
                if op.op.name == "relax.cumsum":
                    # Convert high level relax call to topi.
                    op = self.builder_.call_te(
                        topi.cuda.cumsum, op.args[0], op.attrs.axis, op.attrs.dtype
                    )
                return super().visit_call_(op)

            def transform(self):
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
