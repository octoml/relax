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
set +x
UPLOAD_GCP="${UPLOAD_GCP:=0}"
UPLOAD_PG="${UPLOAD_PG:=0}"
TEST_RUNS="${TEST_RUNS:=1}"
WARMUP_RUNS="${WARMUP_RUNS:=0}"
IMAGE="${IMAGE:=scorecard}"
MODEL_DATA_DIR="${MODEL_DATA_DIR:=model-data}"
GCP_AUTH_JSON="${GCP_AUTH_JSON:=none.json}"
PWD=$(pwd)

touch .fish_history
sudo rm -rf doc-relax
mkdir -p doc-relax
mkdir -p onnx-hub-cache
mkdir -p model-data

set -x

docker run \
    --gpus all \
    --env TEST_RUNS=$TEST_RUNS \
    --env WARMUP_RUNS=$WARMUP_RUNS \
    --env UPLOAD_GCP=$UPLOAD_GCP \
    --env UPLOAD_PG=$UPLOAD_PG \
    -v $PWD/$MODEL_DATA_DIR:/opt/scorecard/model-data \
    -v $GCP_AUTH_JSON:/opt/scorecard/gcp_auth.json:ro \
    -v $PWD/.coverage_results:/opt/scorecard/.coverage_results \
    -v $PWD/.tuning_records:/opt/scorecard/.tuning_records \
    -v $PWD/.fish_history:/root/.local/share/fish/fish_history \
    -v $PWD/relax-coverage:/opt/scorecard/relax-coverage \
    -v $PWD/schema:/opt/scorecard/schema \
    -v $PWD/scripts:/opt/scorecard/scripts \
    -v $PWD/models.yaml:/opt/scorecard/models.yaml \
    -v $PWD/hub_models.yaml:/opt/scorecard/hub_models.yaml \
    --mount type=volume,dst=/opt/scorecard/relax,volume-driver=local,volume-opt=type=none,volume-opt=o=bind,volume-opt=device=$PWD/doc-relax \
    --mount type=volume,dst=/root/.cache/onnx/hub,volume-driver=local,volume-opt=type=none,volume-opt=o=bind,volume-opt=device=$PWD/onnx-hub-cache \
    -it $IMAGE \
    fish
