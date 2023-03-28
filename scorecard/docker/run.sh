#!/bin/bash
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

set -euxo pipefail

# NB: Also source MODEL_DATA_DIR and GCP_AUTH_JSON from a .env or whatever
# is relevant for the running platform
export UPLOAD_GCP=1
export UPLOAD_PG=0
mkdir -p model-data

IMAGE="${IMAGE:-186900524924.dkr.ecr.us-west-2.amazonaws.com/scorecard:latest}"

docker run \
    --gpus all \
    --env TEST_RUNS=10 \
    --env WARMUP_RUNS=3 \
    --env UPLOAD_GCP=1 \
    --env AWS_ACCESS_KEY_ID \
    --env AWS_SECRET_ACCESS_KEY \
    --env AWS_DEFAULT_REGION=us-west-2 \
    -v $(pwd)/model-data:/opt/scorecard/model-data \
    -v $GCP_AUTH_JSON:/opt/scorecard/gcp_auth.json:ro \
    -v $(pwd)/.coverage_results:/opt/scorecard/.coverage_results \
    $IMAGE \
    pytest --tb=native -v -s -q relax-coverage/
