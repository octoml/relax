import pytest

import onnx, json
from onnx import helper, TensorProto, ModelProto, ValueInfoProto, mapping
import onnxruntime

import tvm
from tvm import tir, relax
from tvm.script import ir as I
from tvm.script import relax as R


def test_reshape(in_shape, shape, out_shape):
    metadata = tvm.ir.load_json(
        json.dumps(
            {
                "root": 1,
                "nodes": [
                    {"type_key": ""},
                    {"type_key": "Map", "keys": ["relax.expr.Constant"], "data": [2]},
                    {"type_key": "Array", "data": [3]},
                    {
                        "type_key": "relax.expr.Constant",
                        "attrs": {
                            "_checked_type_": "10",
                            "data": "0",
                            "span": "0",
                            "struct_info_": "4",
                        },
                    },
                    {
                        "type_key": "relax.TensorStructInfo",
                        "attrs": {"dtype": "int64", "ndim": "1", "shape": "5", "span": "0"},
                    },
                    {
                        "type_key": "relax.expr.ShapeExpr",
                        "attrs": {
                            "_checked_type_": "9",
                            "span": "0",
                            "struct_info_": "8",
                            "values": "6",
                        },
                    },
                    {"type_key": "Array", "data": [7]},
                    {
                        "type_key": "IntImm",
                        "attrs": {"dtype": "int64", "span": "0", "value": "2"},
                    },
                    {
                        "type_key": "relax.ShapeStructInfo",
                        "attrs": {"ndim": "1", "span": "0", "values": "6"},
                    },
                    {"type_key": "relax.ShapeType", "attrs": {"ndim": "1", "span": "0"}},
                    {
                        "type_key": "relax.DynTensorType",
                        "attrs": {"dtype": "int64", "ndim": "1", "span": "0"},
                    },
                ],
                "b64ndarrays": [
                    "P6G0lvBAXt0AAAAAAAAAAAEAAAAAAAAAAQAAAABAAQACAAAAAAAAABAAAAAAAAAA4AAAAAAAAAAAAQAAAAAAAA=="
                ],
                "attrs": {"tvm_version": "0.11.dev0"},
            }
        )
    )

    @I.ir_module
    class Module:
        @R.function
        def main(
            data: R.Tensor((7, 32, 32, 8), dtype="float32")
        ) -> R.Tensor(dtype="float32", ndim=2):
            with R.dataflow():
                gv: R.Tensor(dtype="float32", ndim=2) = R.rd_reshape(
                    data, metadata["relax.expr.Constant"][0]
                )
                R.output(gv)
            return gv

    @I.ir_module
    class Expected:
        @R.function
        def main(
            data: R.Tensor((7, 32, 32, 8), dtype="float32")
        ) -> R.Tensor((224, 256), dtype="float32"):
            with R.dataflow():
                gv: R.Tensor((224, 256), dtype="float32") = R.reshape(data, R.shape([224, 256]))
                R.output(gv)
            return gv

    mod = Module
    new_mod = relax.transform.ConvertRDOps()(mod)
    assert tvm.ir.structural_equal(new_mod, Expected)


test_reshape([7, 32, 32, 8], [224, 256], [224, 256])
