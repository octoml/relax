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
git config --global --add safe.directory /opt/scorecard
git submodule update --init --recursive --jobs 0
mkdir -p build-scorecard
cd build-scorecard
cmake -GNinja \
    -DCMAKE_LINKER=/usr/bin/lld-15 \
    -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DUSE_LLVM=llvm-config-15 \
    -DSUMMARIZE=1 \
    -DUSE_CUDA=1 \
    -DUSE_MICRO=1 \
    -DCMAKE_BUILD_TYPE=RelWithDebInfo \
    -DUSE_CUTLASS=1 \
    -DUSE_THRUST=1 \
    ..
cmake --build .
