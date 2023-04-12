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

import configparser
import os
import json
import pathlib
import re
import shutil
import subprocess
import sys


SSO_PROFILE_CONFIG = {
    "sso_start_url": "https://octoml.awsapps.com/start",
    "sso_region": "us-west-2",
    "sso_account_id": "186900524924",
    "sso_role_name": "Sandbox-Developer",
    "region": "us-west-2",
    "output": "json",
}


REQUIRED_MATCHING_KEYS = ("sso_account_id", "sso_role_name")


PROFILE_RE = re.compile(r"profile (?P<profile>.*)")


AWSCLI_VERSION_RE = re.compile(r"aws-cli/(?P<version>[0-9.]+).*")


def check_aws_cli_version():
    output = subprocess.check_output([shutil.which("aws"), "--version"], encoding="utf-8")
    m = AWSCLI_VERSION_RE.search(output)
    assert m is not None, f"error: can't find aws-cli/<ver> string in {output}"
    if not m.group("version").startswith("2."):
        print("You need to upgrade to AWS CLI v2. Follow the instructions here:")
        print("https://docs.aws.amazon.com/cli/latest/userguide/getting-started-install.html")
        print()
        print("At the time of writing, this amounted to:")
        print('  curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip"')
        print("  unzip awscliv2.zip")
        print("  sudo ./aws/install")
        sys.exit(2)


class NoAwsProfileError(Exception):
    """Raised when ~/.aws/config is missing the necessary SSO profile."""


def find_sso_profile(append_if_missing=False):
    aws_config_path = pathlib.Path.home() / ".aws" / "config"
    profile = None
    if aws_config_path.exists():
        cp = configparser.ConfigParser()
        assert cp.read(aws_config_path), f"failed to parse {aws_config_path}"
        for s in cp.sections():
            if all(cp[s].get(k) == SSO_PROFILE_CONFIG[k] for k in REQUIRED_MATCHING_KEYS):
                profile = PROFILE_RE.match(s).group("profile")
                print(f"NOTE: reusing existing SSO profile {profile}")
                break

    if profile is None:
        if not append_if_missing:
            raise NoAwsProfileError(
                f"find_sso_profile did not find an AWS profile with keys "
                f"{f'{k}={SSO_PROFILE_CONFIG[k]}' for k in REQUIRED_MATCHING_KEYS}"
            )

        print(f"NOTE: Appending SSO config to {aws_config_path}")
        profile = f"{SSO_PROFILE_CONFIG['sso_role_name']}-{SSO_PROFILE_CONFIG['sso_account_id']}"
        aws_config_path.parent.mkdir(parents=True, exist_ok=True)
        with open(aws_config_path, "a+") as aws_config_f:
            aws_config_f.write(f"\n[profile {profile}]\n")
            for k, v in SSO_PROFILE_CONFIG.items():
                aws_config_f.write(f"{k} = {v}\n")

    return profile


DOCKER_REGISTRY_SERVER = (
    f"{SSO_PROFILE_CONFIG['sso_account_id']}.dkr.ecr.{SSO_PROFILE_CONFIG['region']}.amazonaws.com"
)


DOCKER_CONFIG = pathlib.Path.home() / ".docker" / "config.json"


# Error messages which mean we should rerun aws sso login.
EXPECTED_AWS_ECR_ERRORS = (
    b"Error loading SSO Token",
    b"The SSO session associated with this profile has expired or is otherwise invalid",
)


def test_pull():
    subprocess.run(
        ["docker", "pull", f"{DOCKER_REGISTRY_SERVER}/auth-test:latest"],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=True,
    )


def main():
    if DOCKER_CONFIG.exists():
        with open(DOCKER_CONFIG, "r") as docker_config_f:
            docker_config = json.load(docker_config_f)

        if DOCKER_REGISTRY_SERVER in docker_config["auths"]:
            try:
                test_pull()
                print(f"Already logged in to docker registry {DOCKER_REGISTRY_SERVER}.")
                print(
                    f"Run 'docker logout {DOCKER_REGISTRY_SERVER}', then rerun this command to re-auth."
                )
                sys.exit(0)
            except subprocess.CalledProcessError as cpe:
                if (
                    b"Your authorization token has expired. Reauthenticate and try again."
                    in cpe.stderr
                ):
                    print("AWS auth token expired; logging you out to reauth")
                    subprocess.check_call(["docker", "logout", DOCKER_REGISTRY_SERVER])
                else:
                    raise cpe

    check_aws_cli_version()

    sso_profile_name = find_sso_profile(append_if_missing=True)
    did_login = False
    num_tries = 0
    docker_password = ""
    while num_tries == 0 or not did_login:
        num_tries += 0
        try:
            docker_password = subprocess.check_output(
                [
                    shutil.which("aws"),
                    "ecr",
                    "get-login-password",
                    f"--profile={sso_profile_name}",
                ],
                stderr=subprocess.PIPE,
            )
            break
        except subprocess.CalledProcessError as cpe:
            if any(e in cpe.stderr for e in EXPECTED_AWS_ECR_ERRORS):
                subprocess.check_call(
                    [
                        shutil.which("aws"),
                        "sso",
                        "login",
                        "--profile",
                        sso_profile_name,
                        "--no-browser",
                    ]
                )
                did_login = True
                continue

            print(f"An exception occurred invoking {' '.join(cpe.cmd)}. Stdout:")
            print(str(cpe.stdout, "utf-8"))
            print("\nStderr:")
            print(str(cpe.stderr, "utf-8"))
            sys.exit(2)

    if docker_password == "":
        print("ERROR: aws ecr get-login-password did not return any data.")
        sys.exit(2)

    proc = subprocess.run(
        [
            shutil.which("docker"),
            "login",
            "--username",
            "AWS",
            "--password-stdin",
            DOCKER_REGISTRY_SERVER,
        ],
        input=docker_password,
        check=True,
    )


if __name__ == "__main__":
    main()
