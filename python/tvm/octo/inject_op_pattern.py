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
"Operator based rewriting of fusion pattern."
import re
import enum
import tvm
from tvm import relax
from tvm.relax.expr_functor import PyExprMutator


class OpPatternKind(enum.IntEnum):
    """Helpful naming for each fusion pattern"""

    kElemWise = 0
    kBroadcast = 1
    kInjective = 2
    kCommReduce = 3
    kOutEWiseFusable = 4
    kTuple = 7
    kOpaque = 8


@tvm.ir.transform.module_pass(opt_level=0)
class InjectOpPattern(tvm.transform.Pass):
    """Transformation pass that changes the fusion pattern for certain ops."""

    def transform_module(self, mod: tvm.IRModule, ctx: tvm.transform.PassContext) -> tvm.IRModule:
        """
        Iterate through a module and explicitly adjust the fusion pattern for some ops.
        """

        @relax.expr_functor.visitor
        class AdjustOpPattern(PyExprMutator):
            """Visitor that changes the fusion pattern of specified operators."""

            def __init__(self, mod):
                self.mod = mod
                # Create a map between functions and the desired pattern to register.
                self.pattern_map = {
                    "layer_norm": OpPatternKind.kInjective,
                    "take": OpPatternKind.kInjective,
                    "split": OpPatternKind.kInjective,
                }
                # Keep track of specific variable names and their new pattern.
                self.pattern_update_map = {}
                super().__init__()

            def visit_global_var_(self, op):
                # Sanitize global var name.
                simple_name = re.sub(r"[0-9]", "", op.name_hint)
                # If we've specified a rewrite of op pattern, denote it.
                if simple_name in self.pattern_map:
                    self.pattern_update_map[op.name_hint] = self.pattern_map[simple_name]
                return super().visit_global_var_(op)

            def transform(self):
                """Iterate over functions to build up a map of which should be updated."""
                for gv, func in mod.functions.items():
                    if isinstance(func, relax.Function):
                        func = self.visit_expr(func)
                    self.builder_.update_func(gv, func)
                new_mod = self.builder_.get()
                new_mod = new_mod.with_attrs(mod.attrs) if mod.attrs else new_mod
                # Apply all found updates.
                for func_name, op_pattern in self.pattern_update_map.items():
                    new_mod[func_name] = new_mod[func_name].with_attr("op_pattern", op_pattern)
                return new_mod

        return AdjustOpPattern(mod).transform()
