#!/usr/bin/env python3
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

from benchmarking_utils import (
    BenchmarkConfig,
    ModelConfig,
    eprint,
    extract_framework_ops,
)
from base import flush_result
from pathlib import Path
from typing import *

import numpy as np

import importlib
import re
import json
import argparse
import warnings
import sys

np.set_printoptions(threshold=5, precision=4)
# warnings.filterwarnings(action="ignore", category=DeprecationWarning, module=r".*")
# warnings.filterwarnings(action="error", category=UserWarning, module=r".*")


def find_inputs_and_outputs(dir: Path):
    input_dir = Path(dir)

    all_inputs = []
    gold_results = []
    eprint(f"Loading sample inputs from {dir}...")
    for input_path in input_dir.glob("sample_input*.npy"):
        output_path = input_path.parent / input_path.name.replace("input", "output")
        all_inputs.append(np.load(input_path, allow_pickle=True).item())
        gold_results.append(np.load(output_path, allow_pickle=True))

    if len(all_inputs) == 0:
        eprint(f"No sample inputs (e.g. files named sample_input0.npy found in '{dir}')")
        exit(1)
    elif len(all_inputs) > 1:
        eprint(f"Found multiple input files in '{dir}', use --input to choose a specific one")
        exit(1)

    return all_inputs, gold_results


def run(args, runner):
    """
    Run the benchmark as defined by the CLI args
    """

    # Determine where the model inputs should come from
    if args.random_inputs:
        # No specific input to use, make one
        all_inputs = runner.generate_inputs(1)
        gold_results = None
    elif args.input is not None:
        # A specific file has been chosen, use it
        all_inputs = [np.load(Path(args.input), allow_pickle=True).item()]
        gold_results = None
        if args.output is not None:
            gold_results = np.load(Path(args.output), allow_pickle=True)
    elif args.input_dir is not None:
        # Find a file in a particular directory
        all_inputs, gold_results = find_inputs_and_outputs(args.input_dir)
    else:
        # Find an input file in the same directory as the model.onnx file
        all_inputs, gold_results = find_inputs_and_outputs(
            runner.benchmark_config.config.model_dir()
        )

    # Only one input is used per run, multiple inputs should be specified as
    # separate runs
    inputs = all_inputs[0]

    # Trigger the onnx.load call
    eprint("Loading model...")
    try:
        onnx_model = runner.load_model()
    except Exception as error:
        flush_result(
            result_directory=None,
            run_config=runner.benchmark_config,
            runtimes_ms=[],
            shapes=[],
            import_error="Failed ONNX load",
            compile_time_ms=[],
            output_deltas=[],
            relay_ops=[],
            framework_ops=[],
            runtime_metadata=runner.metadata(),
        )
        raise error

    if gold_results is None:
        # Generate the expected results if necessary
        eprint("Generating expected results at runtime")
        gold_results = runner.run_onnx_cpu_inference(inputs)

    # Run the model a few times and extract timings
    error, import_error, compile_time_ms, runtimes_ms, output_deltas = runner.run(
        inputs=inputs,
        gold_results=gold_results,
    )

    try:
        framework_ops = extract_framework_ops(onnx_model)
    except Exception as e:
        framework_ops = []
        error = e

    # TODO: relay ops

    # Send the output results to a JSON file on disk
    flush_result(
        result_directory=None,
        run_config=runner.benchmark_config,
        runtimes_ms=runtimes_ms,
        shapes=None,
        import_error=import_error,
        compile_time_ms=compile_time_ms,
        output_deltas=output_deltas,
        relay_ops=[],
        framework_ops=framework_ops,
        runtime_metadata=runner.metadata(),
    )

    # Re-raise any failures
    if error is not None:
        raise error


def generate(args, runner):
    """
    Generate pairs of sample inputs and outputs
    """
    import numpy as np

    np.random.seed(int(args.seed))
    n = int(args.n)
    all_inputs = runner.generate_inputs(n=n)
    output_dir = Path(args.result_directory)

    eprint("Loading ONNX model...")
    onnx_model = runner.load_model()

    should_generate_outputs = not args.skip_run

    for i, inputs in enumerate(all_inputs):
        input_path = output_dir / f"sample_input{i}.npy"
        output_path = output_dir / f"sample_output{i}.npy"

        if input_path.exists() and not args.force:
            eprint(f"Refusing to overwrite {input_path} since --force was not used")
            exit(1)

        if output_path.exists() and not args.force:
            eprint(f"Refusing to overwrite {output_path} since --force was not used")
            exit(1)

        if should_generate_outputs:
            desc = f"{input_path.name}, {output_path.name}"
        else:
            desc = f"{input_path.name}"

        eprint(f"[{i + 1} / {n}] Generating input and output ({desc})")

        if should_generate_outputs:
            outputs = runner.run_onnx_cpu_inference(inputs)

        np.save(input_path, inputs)

        if should_generate_outputs:
            np.save(output_path, outputs)


def parse_args(valid_executors: List[str]):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(help="benchmarking utilities", dest="command")

    def add_shared_arguments(sub):
        sub.add_argument(
            "--cuda-sm",
            type=int,
            default=75,
            help="CUDA target sm level (default: 75, compute capability for Tesla T4)",
        )
        sub.add_argument(
            "-m",
            "--model",
            required=True,
            help="the model slug to run (e.g. oss-onnx.t5-encoder-12@1)",
        )
        sub.add_argument(
            "--sha",
            help="the model's sha256 to use to verify file integrity",
        )
        sub.add_argument("--input-scale", help="scalar to scale np.random results by", default=1.0)
        sub.add_argument(
            "--shapes",
            help="shapes as JSON (will be inferred if not provided), the $axes key can be used to fill in dynamic shapes by axis name",
        )
        sub.add_argument(
            "--files",
            help="comma separated list of files to download",
        )
        sub.add_argument(
            "--tuning-steps",
            help="if tuning should be used, the number of steps",
        )
        sub.add_argument(
            "--dtypes",
            help="comma separated list of dtypes (will be inferred if not provided)",
        )

    # CLI for running models
    run = subparsers.add_parser("run", help="run the benchmark")
    run.add_argument("-i", "--input", help=".npy file to use for input")
    run.add_argument("-o", "--output", help=".npy file to use for output")
    add_shared_arguments(run)

    run.add_argument("--runs", help="number of test runs (default: 1)", default=1)
    run.add_argument(
        "--warmup-runs",
        help="number of warmup runs (default: 0)",
        default=0,
    )
    run.add_argument(
        "--toposort",
        action="store_true",
        help="toposort nodes in model before running",
    )
    run.add_argument(
        "--atol",
        help="absolute tolerance (default: 0.0001)",
        default=0.0001,
    )
    run.add_argument(
        "--rtol",
        help="relative tolerance (default: 0.0001)",
        default=0.0001,
    )
    run.add_argument(
        "--input-dir",
        help="directory of sample_inputN.npy and sample_outputN.npy files",
    )
    run.add_argument(
        "--random-inputs",
        action="store_true",
        help="generate random values for inputs, execute on CPU to generate expected results at runtime",
    )
    run.add_argument(
        "-e",
        "--executor",
        required=True,
        help=f"executor to use (options are {', '.join(valid_executors)})",
    )

    # CLI for generating inputs for a model
    generate = subparsers.add_parser(
        "generate", help="generate new output results for a set of inputs"
    )
    generate.add_argument("--shape", help="input shapes as JSON")
    generate.add_argument(
        "--skip-run",
        action="store_true",
        help="only generate inputs, skip running the model and generating outputs",
    )
    generate.add_argument(
        "-f",
        "--force",
        action="store_true",
        help="overwrite existing files",
    )
    generate.add_argument("--seed", help="int to use for np.random.seed (default=0)", default=0)
    generate.add_argument("-n", help="how many inputs to generate (default=5)", default=5)
    generate.add_argument(
        "-r",
        "--result-directory",
        required=True,
        help="directory to store resulting .npy files in",
    )

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    # Find the possible values for --executor (i.e. the modules that have a .Runner attribute)
    ignored_files = set(
        [
            "all.py",
            "cli.py",
            "base.py",
            "benchmarking_utils.py",
        ]
    )
    executors = [
        x.stem for x in Path(__file__).resolve().parent.glob("*.py") if x.name not in ignored_files
    ]

    args = parse_args(executors)

    # Break apart the model slug
    m = re.match(pattern=r"(.+)\.(.+)@(\d+)", string=args.model)
    if m is None:
        eprint(
            f"--model must match the pattern '<model set>.<model name>@<version number>' (e.g. 'oss-onnx.t5-encoder-12@1'), found {args.model}"
        )
        exit(1)
    set, name, version = m.groups()

    # Find what should run the model
    if hasattr(args, "executor"):
        executor = importlib.import_module(args.executor)
    else:
        executor = importlib.import_module("onnx-nightly-cpu")

    # Check if shapes or dtypes were provided
    shapes = None
    if args.shapes is not None:
        shapes = json.loads(args.shapes)

    dtypes = None
    if args.dtypes is not None:
        dtypes = [d.strip() for d in args.dtypes.split(",")]

    files = []
    if args.files is not None:
        files = [x.strip() for x in args.files.split(",")]

    # Instantiate the runner
    runner_cls = getattr(executor, "Runner")
    runner = runner_cls(
        benchmark_config=BenchmarkConfig(
            config=ModelConfig(
                **{
                    "set": set,
                    "name": name,
                    "sha256": args.sha,
                    "version": version,
                    "flow_config": getattr(args, "executor", None),
                    "requires_toposort": args.toposort,
                    "tuning_steps": None if args.tuning_steps is None else int(args.tuning_steps),
                    "input_scale": float(args.input_scale),
                    "shapes": shapes,
                    "dtypes": dtypes,
                    "files": files,
                }
            ),
            warmup_runs=int(args.warmup_runs),
            test_runs=int(args.runs),
            check_accuracy=True,
            atol=float(args.atol),
            rtol=float(args.rtol),
            cuda_sm=args.cuda_sm,
        ),
    )

    # Run the specified CLI command
    if args.command == "generate":
        generate(args, runner)
    elif args.command == "run":
        run(args, runner)
    else:
        eprint("Unknown command")
        exit(1)
