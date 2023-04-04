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

import argparse
import getpass
import grp
import os
import subprocess
import shlex
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCORECARD_DIR = REPO_ROOT / "scorecard"
DEFAULT_AWS_DIR = Path("~").expanduser() / ".aws"
SANDBOX_PROFILE = "Sandbox-Developer-186900524924"


def touch(filename: Path):
    subprocess.check_call(["touch", str(filename)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--prebuilt-relax",
        action="store_true",
        help="use the relax version baked into the scorecard image",
    )
    parser.add_argument(
        "--image",
        help="Docker image to use (default: 186900524924.dkr.ecr.us-west-2.amazonaws.com/scorecard:latest)",
        default="186900524924.dkr.ecr.us-west-2.amazonaws.com/scorecard:latest",
    )
    parser.add_argument(
        "--aws-dir",
        help="Directory with AWS configuration (default ~/.aws)",
    )
    parser.add_argument(
        "--shell",
        default="bash",
        help="Shell to use for interactive sessions (default: bash)",
    )
    parser.add_argument(
        "--models-yaml",
        default="gitlab-scorecard/models.yaml",
        help="Path to models.yaml (default: gitlab-scorecard/models.yaml)",
    )
    parser.add_argument(
        "--models-dir",
        help="Docker image to use (default: ~/.scorecard-models)",
        default="~/.scorecard-models",
    )
    parser.add_argument("cmd", nargs="*")
    args, other = parser.parse_known_args()

    models_dir = Path(args.models_dir).expanduser().resolve()
    models_dir.mkdir(exist_ok=True, parents=True)

    aws_dir = DEFAULT_AWS_DIR if args.aws_dir is None else Path(args.aws_dir)
    aws_dir = aws_dir.resolve()

    cmd_args = args.cmd + other
    if len(cmd_args) == 0:
        user_cmd = [args.shell]
    else:
        user_cmd = cmd_args

    models_yaml_path = Path(args.models_yaml)
    if not models_yaml_path.exists():
        print(
            f"--models-yaml path: {models_yaml_path} does not exist, no tests will be available. Get it "
            "from GitLab:\n    git clone --depth=1 git@gitlab.com:octoml/relax-scorecard.git gitlab-scorecard"
        )
        exit(1)

    volumes = []

    if args.prebuilt_relax:
        # Just mount the scorecard testbench
        volumes.append(
            (SCORECARD_DIR, SCORECARD_DIR, "rw"),
        )

        git_dir = REPO_ROOT / ".git"
        if git_dir.is_file():
            git_common_dir = REPO_ROOT.joinpath(
                subprocess.check_output(
                    ["git", "rev-parse", "--git-common-dir"], encoding="utf-8", cwd=args.relax
                ).rstrip(" \n")
            ).resolve()
            if git_common_dir != git_dir:
                volumes.append((git_common_dir, git_common_dir, "rw"))
    else:
        # Mount the whole repo
        volumes.append((REPO_ROOT, "/opt/scorecard", "rw"))
        # Rebuild relax before starting the shell
        user_cmd = [
            "bash",
            "-c",
            f"scorecard/docker/build_relax.sh && {' '.join(shlex.quote(u) for u in user_cmd)}",
        ]

    # models.yaml
    volumes.append((models_yaml_path.resolve(), "/opt/scorecard/scorecard/models.yaml", "ro"))

    # Welcome message
    volumes.append((SCORECARD_DIR / "bashrc.sh", "/root/.bashrc", "ro"))

    # Model files
    volumes.append((models_dir, "/opt/scorecard/model-data", "rw"))

    if aws_dir.exists():
        volumes.append((aws_dir, "/root/.aws", "rw"))

    if args.shell == "fish":
        touch(REPO_ROOT / ".fish_history")
        volumes.append(
            (REPO_ROOT / ".fish_history", "/opt/scorecard/.local/share/fish/fish_history", "rw")
        )

    docker_cmd = ["docker", "run", "--gpus", "all"]
    for host_dir, mount_dir, mode in volumes:
        volume = f"{host_dir.resolve()}:{mount_dir}"
        if mode != "rw":
            volume = f"{volume}:{mode}"

        docker_cmd += ["-v", volume]

    env = os.environ.copy()
    env["AWS_PROFILE"] = env.get("AWS_PROFILE", SANDBOX_PROFILE)
    for var in ("UPLOAD_GCP", "UPLOAD_PG", "TEST_RUNS", "WARMUP_RUNS", "AWS_PROFILE"):
        docker_cmd += ["--env", var]

    # settings for docker/with_the_same_user. These allow the host filesystem to be read and written from
    # inside the docker container.
    def _set_env(k, v):
        docker_cmd.append("-e")
        docker_cmd.append(f"{k}={v}")

    _set_env("CI_BUILD_GID", os.getgid())
    _set_env("CI_BUILD_GROUP", grp.getgrgid(os.getgid()).gr_name)
    _set_env("CI_BUILD_HOME", Path("~").expanduser())
    _set_env("CI_BUILD_UID", str(os.getgid()))
    _set_env("CI_BUILD_USER", getpass.getuser())

    docker_cmd += [
        "--rm",
        "-it",
        args.image,
        SCORECARD_DIR.relative_to(REPO_ROOT) / "docker" / "with_the_same_user",
    ] + user_cmd

    if "/" in args.image:
        # Always try to pull images from an external repository
        pull_cmd = ["docker", "pull", args.image]
        print(" ".join(pull_cmd))
        proc = subprocess.run(pull_cmd)
        if proc.returncode != 0:
            print(f"Unable to pull image {args.image}, maybe you need to run docker/login.py?")
            exit(1)
        subprocess.check_call(pull_cmd)

    docker_cmd = [str(c) for c in docker_cmd]
    print(" ".join(docker_cmd))
    proc = subprocess.run(docker_cmd, env=env, check=False)
    exit(proc.returncode)
