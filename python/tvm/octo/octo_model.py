# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
# pylint: disable=invalid-name, wrong-import-position
"""Wrapper class for compiled models."""
import json
import tarfile
from pathlib import Path
from typing import Optional, Union, Dict, Tuple, List
import numpy as np
import tvm
from tvm import relax
from tvm.contrib import utils


class OctoModel(object):
    """A compiled model wrapper that provides helpful utilities for execution and serialization.

    Parameters
    ----------
    exe : Optional[relax.Executable]
        A compiled executable that can be loaded and run by a relax VM.
    input_info : Optional[Dict[str, Tuple[List, str]]]
        Information about the input names, shapes, and types for the VM.
    model_path : Optional[Union[str, Path]]
        The path to a saved OctoModel, one of exe and model_path must
        be specified.
    target : Optional[tvm.target.Target]
        The target being compiled for.
    """

    def __init__(
        self,
        exe: Optional[relax.Executable] = None,
        input_info: Optional[Dict[str, Tuple[List, str]]] = None,
        model_path: Optional[Union[str, Path]] = None,
        target: Optional[tvm.target.Target] = None,
    ):
        self.target = target

        if exe is None and model_path is None:
            raise ValueError("One of vm and model_path must be provided.")

        self._tmp_dir = utils.tempdir()

        if model_path is not None:
            exe, input_info = self.load(model_path)

        self.dev = tvm.device(self.target.get_target_device_type())
        self.exe = exe
        self.input_info = input_info

        # Create a vm from exe.
        self.vm = relax.VirtualMachine(self.exe, self.dev, profile=True)

    def save(self, model_path: Union[str, Path]):
        """Save the OctoModel to disk.

        The current format used is a simple tar of the exported model library (exe.so),
        the input information of the model (input_info.json), and a metadata
        file containing strings such as the target.

        Parameters
        ----------
        model_path : Union[str, Path]
            A full path to save this OctoModel to including the output file name.
            The file will be saved as a tar file so using a ".tar" extension is advised.
        """
        # Only two artifacts need to be saved, the exe and the input struct info.
        # Serialize both to a temp directory.
        exe_path = self._tmp_dir.relpath("exe.so")
        self.exe.mod.export_library(exe_path)
        input_info_path = self._tmp_dir.relpath("input_info.json")
        with open(input_info_path, "w") as fo:
            json.dump(self.input_info, fo)

        # Save additional metadata.
        metadata = {"target": str(self.target)}
        metadata_path = self._tmp_dir.relpath("metadata.json")
        with open(metadata_path, "w") as fo:
            json.dump(metadata, fo)

        # Tar the tempfile and save to the designated model_path.
        with tarfile.open(model_path, "w") as tar:
            tar.add(exe_path, "exe.so")
            tar.add(input_info_path, "input_info.json")
            tar.add(metadata_path, "metadata.json")

    def load(self, model_path: Union[str, Path]) -> Tuple[relax.Executable, Dict[List, str]]:
        """Load a saved OctoModel back into memory.

        Parameters
        ----------
        model_path : Union[str, Path]
            The path to the saved OctoModel that will be loaded.

        Returns
        -------
        exe : relax.Executable
            A compiled executable that can be loaded and run by a relax VM.
        input_info : Dict[str, Tuple[List, str]]
            Information about the input names, shapes, and types for the VM.
            Will be loaded from memory if possible.
        """
        t = tarfile.open(model_path)
        t.extractall(self._tmp_dir.relpath("."))

        # Load executable.
        exe_path = self._tmp_dir.relpath("exe.so")
        exe = relax.Executable(tvm.runtime.load_module(exe_path))

        # Load input info.
        input_info_path = self._tmp_dir.relpath("input_info.json")
        with open(input_info_path, "r") as fi:
            input_info = json.load(fi)

        # load other metadata.
        metadata_path = self._tmp_dir.relpath("metadata.json")
        with open(metadata_path, "r") as fi:
            metadata = json.load(fi)
        self.target = tvm.target.Target(metadata["target"])

        return exe, input_info

    def generate_inputs(self) -> Dict[str, np.array]:
        """Generate random inputs based on 'self.input_info' for inference or benchmarking

        Returns
        -------
        input_dict : Dict[str, np.array]
        """
        input_dict = {}
        for name, (shape, dtype) in self.input_info.items():
            input_dict[name] = np.random.normal(size=shape).astype(dtype)
        return input_dict

    def run(self, inputs: Optional[Dict[str, np.array]] = None) -> List[np.array]:
        """Perform an inference of the model.

        Parameters
        ----------
        inputs : Optional[Dict[str, np.array]]
            An optional input dictionary containing the values to perform
            inference with. If not provided, random values will be generated
            instead.

        Returns
        -------
        outputs : List[np.array]
            The output values from the inference.
        """
        # Generate random inputs if none are provided.
        if inputs is None:
            inputs = self.generate_inputs()

        # Assign inputs.
        self.vm.set_input("main", **inputs)
        # Run the modeel.
        self.vm.invoke_stateful("main")
        # Get and return the outputs.
        outputs = self.vm.get_outputs("main")
        if isinstance(outputs, tuple):
            outputs = [output.numpy() for output in outputs]
        else:
            outputs = [outputs.numpy()]
        return outputs

    def profile(self) -> tvm.runtime.profiling.Report:
        """Measures the model's performance.

        Returns
        -------
        report : tvm.runtime.profiling.Report
            A breakdown of the runtime and per layer metrics.
        """
        inputs = self.generate_inputs()
        self.vm.set_input("main", **inputs)
        report = self.vm.profile("main")
        return report
