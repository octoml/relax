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

set -eux

set +x
source docker/retry.sh
set -x

PUSH_TO_ECR="${PUSH_TO_ECR:=0}"
NO_CACHE="${NO_CACHE:=0}"
TVM_BUILT_AT="${TVM_BUILT_AT:=0}"
RETRIES="${RETRIES:=5}"
IMAGE_NAME="${IMAGE_NAME:=scorecard}"

CACHE_ARG=""
if [ "$NO_CACHE" == "1" ]; then
    CACHE_ARG="--no-cache"
fi

retry $RETRIES docker build . --build-arg TVM_BUILT_AT=$TVM_BUILT_AT -f docker/Dockerfile.${IMAGE_NAME} $CACHE_ARG --tag ${IMAGE_NAME}:latest

# # testing code to skip the docker build but still have an image to work with
# docker pull hello-world
# docker tag hello-world scorecard:latest

if [ "$PUSH_TO_ECR" == "1" ]; then
    DATE=$(date '+%Y-%m-%d')
    HASH=${GIT_COMMIT_SHA:0:7}
    TAG="$DATE-$HASH"

    REGION="us-west-2"
    ACCOUNT_ID="186900524924"

    # Make 'docker push' authenticated with ECR
    aws ecr get-login-password --region $REGION | docker login --username AWS --password-stdin $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com

    # Push to ECR registry (latest)
    retry 5 docker tag ${IMAGE_NAME}:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${IMAGE_NAME}:latest
    retry 5 docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${IMAGE_NAME}:latest

    # Push to ECR registry (fixed tag)
    retry 5 docker tag ${IMAGE_NAME}:latest $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${IMAGE_NAME}:$TAG
    retry 5 docker push $ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${IMAGE_NAME}:$TAG

    # Save the tag so it can be used later
    echo "TAG=$TAG" >> output.env
    echo "ECR_IMAGE=$ACCOUNT_ID.dkr.ecr.$REGION.amazonaws.com/${IMAGE_NAME}:$TAG" >> output.env
fi
