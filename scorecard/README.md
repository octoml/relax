<!--- Licensed to the Apache Software Foundation (ASF) under one -->
<!--- or more contributor license agreements.  See the NOTICE file -->
<!--- distributed with this work for additional information -->
<!--- regarding copyright ownership.  The ASF licenses this file -->
<!--- to you under the Apache License, Version 2.0 (the -->
<!--- "License"); you may not use this file except in compliance -->
<!--- with the License.  You may obtain a copy of the License at -->

<!---   http://www.apache.org/licenses/LICENSE-2.0 -->

<!--- Unless required by applicable law or agreed to in writing, -->
<!--- software distributed under the License is distributed on an -->
<!--- "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY -->
<!--- KIND, either express or implied.  See the License for the -->
<!--- specific language governing permissions and limitations -->
<!--- under the License. -->

# Relax Scorecard

The Relax scorecard is intended benchmark Relax against other frameworks. The models are defined externally in `models.yaml` in https://gitlab.com/octoml/relax-scorecard. The test bench code here in [`relax-coverage`](./relax-coverage/) uses `models.yaml` to determine which frameworks to use, models to run, input shapes, and other per-model options.

## Local Usage

### One-Time Setup

#### Docker

First, ensure that `nvidia-docker2` is set up and configured so that Docker can access GPUs. Run the installation steps if you have not already.

```bash
sudo apt update
sudo apt install -y docker.io

# See https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
      && curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
      && curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt update
sudo apt install -y nvidia-docker2
```

Then check to see that `nvidia-smi` can correctly run in a Docker container.

```bash
$ docker run --gpus all -it nvidia/cuda:11.7.1-devel-ubuntu22.04 nvidia-smi
+-----------------------------------------------------------------------------+
| NVIDIA-SMI 470.82.01    Driver Version: 470.82.01    CUDA Version: 11.7     |
|-------------------------------+----------------------+----------------------+
...
```

#### AWS

The relevant Docker images are stored in AWS's Elastic Container Registry (ECR). You must authenticate first in order to access the images.

```bash
# Get the AWS CLI
apt install -y awscli

# Authenticate with AWS via SSO and Docker
docker/login.py
```

Note you need to click the https://device.sso.us-west-2.amazonaws.com link above
and complete auth in the browser in order for the command to complete
successfully.


### Build the Docker Image

Scorecard CI will build the Docker image each time before it runs benchmarks and store the resulting image in AWS's ECR. You can pull this image locally after using `docker/login.py` via the image at the end of the logs of any `docker_build` job in a scorecard GitLab pipeline ([example](https://gitlab.com/octoml/relax-scorecard-ci2/-/jobs/4061474766)). `local.py` will also determine an image to use from ECR if none is specified, so if you want to use those images you can skip this step.

You can also build the image locally, though you will still need to authorize with AWS to download models if you do not have them on disk.

```bash
export NO_CACHE=1
# builds a Docker image and tags it as 'scorecard:latest'
./scorecard/docker/build.sh
```

### Getting Models

The scorecard testbench will automatically download models from AWS S3 if they are not found on disk. The models on disk must be in a format that matches S3 for the scorecard infrastructure to locate them: `<model dir>/<model set>/<model name>@<version>/model.onnx`.

```bash
$ find ~/.scorecard-models -type f
/home/homeuser/.scorecard-models/oss-onnx/Tiny-Random-GPTJForCausalLM@1/model.onnx
...
```

If this file does not exist the scorecard will use the AWS CLI to download the models from S3. For this to work ensure that you have run `docker/login.py` and signed in with SSO to AWS.

### Getting `models.yaml`

The model descriptions live outside of the Relax repository in GitLab in [`models.yaml`](https://gitlab.com/octoml/relax-scorecard/-/blob/main/models.yaml). This file is necessary for the scorecard to do useful work.

```bash
# This is used in the next step to provide the path for --models-yaml
git clone --depth=1 git@gitlab.com:octoml/relax-scorecard.git gitlab-scorecard
```

### Run the Scorecard

The scorecard runs as a pytest test suite inside a Docker container, and all usual pytest features are available. [`local.py`](./local.py) provides an interface to easily mount your local files and work with the scorecard. Unlike the TVM and Relax CI images, the `scorecard` Docker image contains a fully built version of Relax, which allows results to be easily replicated. `local.py` takes care of mounting your local Relax version on top of this one and building it when launched.

```bash
# Mount the local Relax and drop into a shell in the Docker container
python3 local.py

# Skip the local checkout of Relax and use the one included in the image
python3 local.py --prebuilt-relax

# Use a different Docker image, in this case a locally built version
python3 local.py --image scorecard:latest

# Use a different Docker image, in this case the latest scorecard image in ECR
python3 local.py --image latest

# Run a command other than 'bash'
python3 local.py pytest relax-coverage/ -k 'stable-diffusion and relax-cuda'
```

## Adding a Model

To add a model:

1. Upload the `.onnx` and associated files to the S3 bucket ([`scorecard-models`](https://s3.console.aws.amazon.com/s3/buckets/scorecard-models)) so the testbench and other users can access it in CI. The S3 key used should be of the format `<model set>/<model name>@<version>/model.onnx`.

```bash
# an example of uploading a model to s3 via the aws CLI
aws s3 cp the_model.onnx s3://scorecard-models/my-model-set/my-model-name@1/model.onnx`
```

2. Configure it in [`models.yaml`](https://gitlab.com/octoml/relax-scorecard/-/blob/main/models.yaml) via a merge request on GitLab to https://gitlab.com/octoml/relax-scorecard/-/blob/main/models.yaml. See the other configurations and the loading code in [`test_coverage.py`](relax-coverage/test_coverage.py) for a full listing of possible options. Required keys in an entry in [`models.yaml`](https://gitlab.com/octoml/relax-scorecard/-/blob/main/models.yaml) are:

* `set`: (string) The model set
* `name`: (string) The model name
* `version`: (number) The model version
* `sha256`: (string) The sha256 of the `.onnx` file, used to verify file integrity

And optional keys:

* `configs`: (list) Configs to run this model under (possible options are `relax-cuda` and `onnx-trt`)
* `shapes`: (dictionary) The shapes to hardcode rather than inferring them from the ONNX model. These shapes always supersede those that can be inferred from the ONNX model. The special name `$axes` may be used to specify a common value for dynamic axes referenced in multiple shapes in the ONNX model, commonly things like `encoder_sequence_length` or `batch_size`. This can help decrease boilerplate and copy-paste when specifying shapes.
* `tuning_steps`: (int) When unset, no tuning will be used (the default behavior). When set, Relax will run this many tuning steps before execution (only applies to the `relax-*` configs, has no effect on others).
* `requires-toposort`: (boolean) When `true`, ONNX models are topologically sorted immediately after loading.
* `cuda-sm`: (int) The CUDA compute capability to specify in the Relax target (only applies to the `relax-*` configs, has no effect on others).


Once the merge request from (2) is merged, the scorecard CI testbench will pick up the changes to `models.yaml` and start running the tests as configured.

### Special Considerations

When adding a model, you should profile it locally to check the expected load, compile, and run time. If these in sum are excessive (i.e. over an hour), the model should be run in parallel with the others rather than serially. Make changes to [`.gitlab-ci.yml`](../.gitlab-ci.yml) as necessary.

## Configuring an Existing Model

To change how an existing model runs, you may need to edit either a specific runner or the model's configuration. To determine which, see the section above about adding a model to see if the change required should go in [`models.yaml`](https://gitlab.com/octoml/relax-scorecard/-/blob/main/models.yaml). If so, make your change on GitLab and trigger a CI run on the scorecard by pushing an empty commit or re-running the latest job. If you need to make a change to the runner (e.g. a specific change to how all models are executed), see the relevant code in [`runners/`](./relax-coverage/runners/).

## Adding a backend

To add a backend (e.g. a new `config` accessible from `models.yaml`), create a file in [`runners/`](./relax-coverage/runners/) named for the backend (e.g. `my-backend.py`). In this file, you must define a subclass of [`BaseRunner`](./relax-coverage/runners/benchmarking_utils.py) and implement the `run()` method. See [`relax-cuda`](./relax-coverage/runners/relax-cuda.py) and [`onnx-trt`](./relax-coverage/runners/onnx-trt.py) for examples. Once complete, you can test the runner with the low level testbench interface in [`cli.py`](./relax-coverage/runners/cli.py). `cli.py` will directly import your new runner by name and execute the `run()` method.

```bash
python relax-coverage/runners/cli.py --executor my-backend --model my-model-set.my-model-name@version --random-inputs
```

See the help for `cli.py` for more details and a full listing of available options.

```bash
python relax-coverage/runners/cli.py --help
```

Once `cli.py` can correctly use your backend, you can add reference your backend from [`models.yaml`](https://gitlab.com/octoml/relax-scorecard/-/blob/main/models.yaml) and the testbench will dispatch to your new code.

## Configuring the Relax runner

The Relax runner is used to compile and execute the models from [`models.yaml`](https://gitlab.com/octoml/relax-scorecard/-/blob/main/models.yaml). To make changes to the runner, see `run()` in [`relax_base.py`](./relax-coverage/runners/relax_base.py).
