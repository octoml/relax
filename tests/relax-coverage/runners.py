import sys
import importlib

from utils import *
from typing import *

import onnxruntime
import onnxruntime as ort
import onnx
import tvm
import numpy as np

from tvm import relay
from tvm import relax
from tvm.relax.transform.tuning_api import Trace


def load_onnx(use_nightly: bool, nightly_path: str):
    """
    Hack to get 2 different versions of onnx running within the same Python
    process (at mutually exclusive different times)
    """
    if use_nightly:
        # ensure that 'nightly_path' is in 'sys.path'
        if nightly_path not in sys.path:
            sys.path.insert(0, nightly_path)
    else:
        # ensure that 'nightly_path' is NOT in 'sys.path'
        if nightly_path in sys.path:
            sys.path.remove(nightly_path)

    importlib.reload(onnxruntime)


def run_onnx(use_nightly: bool, nightly_path: str, providers: List[str]):
    def fn(*args, **kwargs):
        load_onnx(use_nightly=use_nightly, nightly_path=nightly_path)
        return run_onnx_impl(*args, **kwargs, providers=providers)

    return fn


def run_onnx_impl(
    model: onnx.ModelProto,
    inputs,
    run_config: BenchmarkConfig,
    shapes,
    dtypes,
    providers: List[str],
    gold_results: List[np.ndarray],
):
    """
    Run an onnx `model` with onnxruntime's TensorRT EP
    """
    runtimes_ms = []
    output_deltas = []
    sess_opt = ort.SessionOptions()

    # Set up an onnx inference on GPU
    sess = ort.InferenceSession(
        model.SerializeToString(), sess_options=sess_opt, providers=providers
    )

    # Unwrap input if necessary
    if isinstance(inputs, list) and len(inputs) == 1:
        inputs = inputs[0]
    else:
        inputs = inputs

    output_names = []
    compile_time_ms = 0
    for i in range(run_config.warmup_runs):
        print(
            f"[{i + 1} / {run_config.warmup_runs}][onnx] Warmup {run_config.config.id()}"
        )
        sess.run(output_names, inputs)

    # Run the model a few times and record the end to end execution time
    for i in range(run_config.test_runs):
        print(
            f"[{i + 1} / {run_config.test_runs}][onnx] Running {run_config.config.id()}"
        )
        with Timer() as timer:
            output = sess.run(output_names, inputs)
            print(output)

        # Stash the runtime
        runtimes_ms.append(timer.ms_duration)

        # Check accuracy
        output_deltas.append([gold_results[i] - output[i] for i in range(len(output))])

    return None, None, compile_time_ms, runtimes_ms, output_deltas


def byoc_cutlass(model, inputs, run_config, shapes, dtypes):
    raise NotImplementedError()
    assert has_cutlass()
    mod, params = relay.frontend.from_onnx(
        model, shapes, opset=None, convert_config=None
    )
    params = inputs
    vm, dev, num_cutlass = profile_and_build(mod=mod, sm=80, params=params)
    return [1, 2, 3, 4]


def tvm_relay_msat(model, inputs, run_config, shapes, dtypes):
    raise NotImplementedError()
    return [1, 2, 3, 4]


def tvm_llvm(
    model: onnx.ModelProto,
    inputs,
    run_config: BenchmarkConfig,
    shapes,
    dtypes,
    gold_results: List[np.ndarray],
):
    """
    Run an onnx `model` with the vm executor
    """
    print("Loading relay")
    mod, params = relay.frontend.from_onnx(
        model, shapes, opset=None, convert_config=None
    )
    print(relax.frontends)

    params = inputs
    dev = tvm.cpu()
    print("Loaded model")

    target = "llvm"
    print(mod, dev, target)
    print("========")
    print(mod.show())
    exit(0)
    executor = relay.create_executor("vm", mod=mod, device=dev, target=target)
    runner = executor.evaluate()
    runtimes_ms = []
    print(list(inputs.values()))
    input_data = list(inputs.values())
    for i in range(test_runs):
        print(f"[{i} / {test_runs}] tvm llvm running {id}")
        with Timer() as timer:
            result = runner(*input_data, **params)

        runtimes_ms.append(timer.ms_duration)
    return runtimes_ms


def relax_native(
    model: onnx.ModelProto,
    inputs,
    run_config: BenchmarkConfig,
    shapes,
    dtypes,
    gold_results: List[np.ndarray],
):
    runtimes_ms = []
    output_deltas = []

    with Timer() as compile_timer:
        try:
            tvm_model = relax.from_onnx(model, shape=shapes)
        except Exception as e:
            return e, ImportError.FAILED_ONNX_IMPORT, 0, [], []

        try:
            with tvm.transform.PassContext(opt_level=3):
                # 93s
                # ex = relax.vm.build(tvm_model, target="llvm -mcpu=znver2")

                # 93s
                # ex = relax.vm.build(tvm_model, target="llvm -mcpu=znver2 -mtriple=x86_64-pc-linux-gnu -mattr=+cx8,+cmov,+mmx,+fxsr,+sse,+sse2,+nopl,+ssse3,+fma,+cx16,+movbe,+popcnt,+aes,+xsave,+avx,+f16c,+sse4a,+fsgsbase,+avx2,+bmi2,+rdseed,+adx,+clflushopt,+clwb,+xsaveopt,+xsavec,+xsaves,+clzero,+rdpru,+rdpid")

                # 50s
                ex = relax.vm.build(tvm_model, target="llvm -mcpu=core-avx2")

                # 54s
                # ex = relax.vm.build(tvm_model, target="llvm -mcpu=core-avx2 -mtriple=x86_64-pc-linux-gnu -mattr=+cx8,+cmov,+mmx,+fxsr,+sse,+sse2,+nopl,+ssse3,+fma,+cx16,+movbe,+popcnt,+aes,+xsave,+avx,+f16c,+sse4a,+fsgsbase,+avx2,+bmi2,+rdseed,+adx,+clflushopt,+clwb,+xsaveopt,+xsavec,+xsaves,+clzero,+rdpru,+rdpid")
                vm = relax.VirtualMachine(ex, tvm.cpu())
        except Exception as e:
            return e, ImportError.FAILED_RELAX_BUILD, 0, [], []

    compile_time_ms = compile_timer.ms_duration
    vm.set_input("main", **inputs)

    for i in range(run_config.warmup_runs):
        print(
            f"[{i + 1} / {run_config.warmup_runs}][relax_native] Warmup {run_config.config.id()}"
        )
        vm.invoke_stateful("main")

    # Run the model a few times and record the end to end execution time
    for i in range(run_config.test_runs):
        print(
            f"[{i + 1} / {run_config.test_runs}][relax_native] Running {run_config.config.id()}"
        )
        try:
            with Timer() as timer:
                vm.invoke_stateful("main")

            # Stash the runtime
            runtimes_ms.append(timer.ms_duration)
            output = vm.get_outputs("main")
        except Exception as e:
            return e, ImportError.FAILED_EXECUTION, 0, [], []

        # Check accuracy
        output_deltas.append(
            [gold_results[i] - output[i].numpy() for i in range(len(output))]
        )

    return None, None, compile_time_ms, runtimes_ms, output_deltas


def relax_cuda(
    model: onnx.ModelProto,
    inputs,
    run_config: BenchmarkConfig,
    shapes,
    dtypes,
    gold_results: List[np.ndarray],
):
    runtimes_ms = []
    output_deltas = []
    with Timer() as compile_timer:
        try:
            tvm_model = relax.from_onnx(model, shape=shapes)
        except Exception as e:
            return e, ImportError.FAILED_ONNX_IMPORT, 0, [], []

        try:
            mod = tvm_model
            target = tvm.target.Target(
                "cuda -libs=thrust -arch=sm_75 -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -thread_warp_size=32 -registers_per_block=65536"
            )

            with target, tvm.transform.PassContext(opt_level=3):
                out_mod = relax.transform.ScheduleForTarget(target)(mod)
                ex = relax.vm.build(out_mod, target=target)
                vm = relax.VirtualMachine(ex, tvm.cuda(0))
        except Exception as e:
            return e, ImportError.FAILED_RELAX_BUILD, 0, [], []

    compile_time_ms = compile_timer.ms_duration
    vm.set_input("main", **inputs)

    for i in range(run_config.warmup_runs):
        print(
            f"[{i + 1} / {run_config.warmup_runs}][relax_native] Warmup {run_config.config.id()}"
        )
        vm.invoke_stateful("main")

    # Run the model a few times and record the end to end execution time
    for i in range(run_config.test_runs):
        print(
            f"[{i + 1} / {run_config.test_runs}][relax_native] Running {run_config.config.id()}"
        )
        try:
            with Timer() as timer:
                vm.invoke_stateful("main")

            # Stash the runtime
            runtimes_ms.append(timer.ms_duration)
            output = vm.get_outputs("main")
        except Exception as e:
            return e, ImportError.FAILED_EXECUTION, 0, [], []

        # Check accuracy
        output_deltas.append(
            [gold_results[i] - output[i].numpy() for i in range(len(output))]
        )

    return None, None, compile_time_ms, runtimes_ms, output_deltas
