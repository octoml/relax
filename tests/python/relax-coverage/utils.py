import subprocess
import json
import time
from typing import Tuple
from pathlib import Path


def sha256sum(model_file_name: str):
    proc = subprocess.run(
        ["sha256sum", model_file_name], stdout=subprocess.PIPE, check=True, encoding="utf-8"
    )
    return proc.stdout.strip().split()[0]


def git_info() -> Tuple[str, str]:
    """
    Determine the git branch and sha
    """
    proc = subprocess.run(
        ["git", "rev-parse", "--abbrev-ref", "HEAD"],
        stdout=subprocess.PIPE,
        check=True,
        encoding="utf-8",
    )
    branch = proc.stdout.strip()
    proc = subprocess.run(
        ["git", "rev-parse", "--verify", "HEAD"],
        stdout=subprocess.PIPE,
        check=True,
        encoding="utf-8",
    )
    sha = proc.stdout.strip()
    return branch, sha


_manifest = None


def manifest():
    global _manifest
    if _manifest is None:
        with open(Path(__file__).resolve().parent / "ONNX_HUB_MANIFEST.json") as f:
            _manifest = json.load(f)

    return _manifest


def find_model(name):
    for model in manifest():
        if model["model_path"] == name:
            return model
    raise ValueError(f"{name} not found")


def infer_inputs(model):
    """
    Infers this model's input shapes and input dtypes.

    :return: input shapes and input dtypes of this Hulk ONNX Model.
    :raises: ONNXInferInputsUnknownDataType when dtype is unknown.
    """
    # N.B. Defer the import so as not to unconditionally require other runtimes.
    from tvm import relay
    from tvm.tir import Any as Any

    input_shapes = {}
    input_dtypes = {}
    initializer_names = [n.name for n in model.graph.initializer]
    # The inputs contains both the inputs and parameters. We are just interested in the
    # inputs so skip all parameters listed in graph.initializer
    for input_info in model.graph.input:
        if input_info.name not in initializer_names:
            name, shape, dtype, axis_names = relay.frontend.onnx.get_info(input_info)
            if dtype is None:
                raise RuntimeError(
                    f"Unknown dtype on input '{input_info.name}' is not supported. inputs: '{input_info.name}'",
                )

            # Normalize the shape dimensions to integers
            assert isinstance(input_shapes, dict)
            new_shape = []
            for value, name in zip(shape, axis_names):
                # print("Unspecified shape, assuming 1", value)
                value = int(value) if not isinstance(value, Any) else 1
                # new_shape[name] = value
                new_shape.append(value)
            input_shapes.update({input_info.name: new_shape})
            input_dtypes.update({input_info.name: dtype})

    return input_shapes, input_dtypes


class Timer(object):
    def __enter__(self):
        self.start = time.perf_counter_ns()
        return self

    def __exit__(self, *args):
        self.end = time.perf_counter_ns()
        self.ms_duration = (self.end - self.start) / 1000 / 1000
