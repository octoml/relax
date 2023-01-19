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
import os
import sys
import random
import json
import logging
from pathlib import Path
from datetime import datetime

# Hackery to enable importing of utils from ci/scripts/jenkins
REPO_ROOT = Path(__file__).resolve().parent.parent.parent.parent
sys.path.append(str(REPO_ROOT / "ci" / "scripts" / "jenkins"))

from cmd_utils import Sh, init_log


sh = Sh(env={"EDITOR": "true"})

DEBUG = os.getenv("DEBUG", "0") == "1"
if DEBUG:
    SLACK_WEBHOOK_URL = "test"
    GITHUB_RUN_URL = "test"
    TVM_BOT_GITHUB_TOKEN = "test"
else:
    SLACK_WEBHOOK_URL = os.environ["SLACK_WEBHOOK_URL"]
    GITHUB_RUN_URL = os.environ["GITHUB_RUN_URL"]
    TVM_BOT_GITHUB_TOKEN = os.environ["TVM_BOT_GITHUB_TOKEN"]

# To add people to this list, go to
# your Slack Profile > Profile > 3 dots > Copy Member ID
users = [
    "U013D814ZLJ",
]


def slack_message(message: str):
    if DEBUG:
        print(f"Slack mesaging: {message}")
        return

    data = {"text": message}
    sh.run(
        [
            "curl",
            "-X",
            "POST",
            "-H",
            "Content-type: application/json",
            "--data",
            json.dumps(data),
            SLACK_WEBHOOK_URL,
        ],
        shell=False,
    )


def take_theirs(path):
    sh.run(f"git checkout --theirs -- {path}")
    sh.run(f"git add {path}")


def rebase(base: str) -> bool:
    proc = sh.run(f"git rebase tlcpack/relax", check=False)

    if proc.returncode == 0:
        return True

    # rebase failed, begin resolutions
    # Ignore any changes to the CI directory and take whatever is in octoml/relax
    take_theirs("ci/.")

    proc = sh.run("git rebase --continue", check=False)
    return proc.returncode == 0


def save_current_ref():
    # Find an unused branch name for the backup branch (use the date, if that's
    # taken, add a counter)
    backup_name = datetime.now().strftime("staging-backup-%m-%d-%Y")
    proc = sh.run(f"git rev-parse --verify origin/{backup_name}", check=False)
    i = 1
    while proc.returncode == 0:
        backup_name = datetime.now().strftime(f"staging-backup-%m-%d-%Y.{i}")
        i += 1
        proc = sh.run(f"git rev-parse --verify origin/{backup_name}", check=False)

    _, old_branch = sh.tee("git rev-parse --abbrev-ref HEAD")
    old_branch = old_branch.strip()
    sh.run(f"git checkout -b {backup_name}")
    prefix = ""
    if DEBUG:
        prefix = "echo "
    sh.run(f"{prefix}git push origin {backup_name}")
    sh.run(f"git checkout {old_branch}")


def main():
    if not DEBUG:
        sh.run("git config --global rerere.enabled true")
        sh.run("git config --global user.name tvm-bot")
        sh.run("git config --global user.email 95660001+tvm-bot@users.noreply.github.com")
    sh.run("git remote add tlcpack https://github.com/tlc-pack/relax.git")
    sh.run("git fetch tlcpack")

    # Replace the origin remote with one with tvm-bot's token
    sh.run("git remote remove origin")
    sh.run(
        f"git remote add origin https://tvm-bot:{TVM_BOT_GITHUB_TOKEN}@github.com/octoml/relax.git"
    )
    sh.run("git fetch origin")

    # Show some info about HEAD
    sh.run("git status")
    sh.run("git branch")
    sh.run("git branch -r")

    # Determine the last 'base' commit from tlc-pack/relax to use (i.e. grab the
    # most recent branch named something like tlcpack/staging-backup-01-12-2023)
    _, base = sh.tee(
        "git branch -r | grep 'tlcpack/staging' | sort -b -t- -k 5,5 -k 3,3 -k 4,4 | tail -n 1"
    )
    base = base.strip()
    logging.info(f"Rebasing onto base commit {base}")

    # Back up the current history just in case
    save_current_ref()

    # Run the base, using the 'base' as the start for 'git rebase --onto'
    if rebase(base):
        logging.info("Successfully rebased")
        prefix = ""
        if DEBUG:
            prefix = "echo "
        sh.run(f"{prefix}git checkout -B relax")
        sh.run(f"{prefix}git push origin relax --force")
        slack_message("Successfully force pushed a rebase")
    else:
        sh.run("git status")
        sh.run("git diff")
        slack_message(
            f"Failed to rebase, <@{random.choice(users)}> please take a look (<{GITHUB_RUN_URL}|failed run>)"
        )
        exit(1)


if __name__ == "__main__":
    init_log()

    try:
        main()
    except Exception as e:
        slack_message(
            f"Failed to run rebase workflow (`{str(e)}`), <@{random.choice(users)}> please take a look (<{GITHUB_RUN_URL}|failed run>)"
        )
        exit(1)
