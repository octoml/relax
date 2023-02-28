import tvm
from tvm import tir
from typing import Dict, Optional
from tvm.ir.module import IRModule
from ..expr_functor import mutator, PyExprMutator
from ..expr import Call, Constant, Expr, Function, ShapeExpr, Tuple, TupleGetItem, Var
from tvm import relax
from ..block_builder import BlockBuilder


def _rd_reshape(bb: BlockBuilder, call: Call) -> Expr:
    data = call.args[0]
    shape = call.args[1]
    if isinstance(shape, relax.expr.Constant):
        const_shape = shape.data.numpy().tolist()
        return relax.op.reshape(data, const_shape)
    else:
        return call


conversion_map = {"relax.rd_reshape": _rd_reshape}


@tvm.transform.module_pass(opt_level=0, name="convert_rd")
class ConvertRDOps:
    def __init__(self):
        pass

    def transform_module(self, mod: IRModule, ctx: tvm.transform.PassContext) -> IRModule:
        @mutator
        class OperatorConverter(PyExprMutator):
            def __init__(self, mod: IRModule):
                super().__init__(mod)
                self.mod = mod

            def _convert_op(self, call: Call) -> Expr:
                if call.op.name in conversion_map:
                    return conversion_map[call.op.name](self.builder_, call)

                return call

            def transform(self) -> IRModule:
                for global_var, func in self.mod.functions.items():
                    if not isinstance(func, Function):
                        continue
                    updated_func = self.visit_expr(func)
                    # updated_func = remove_all_unused(updated_func)
                    self.builder_.update_func(global_var, updated_func)

                return self.builder_.get()

            def visit_call_(self, call):  # pylint: disable=arguments-differ
                call = self.visit_expr_post_order(call)
                if not isinstance(call.op, tir.op.Op):
                    return call
                return self._convert_op(call)

        return OperatorConverter(mod).transform()
