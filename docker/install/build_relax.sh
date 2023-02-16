#!/bin/bash
set -euxo pipefail
cd relax
mkdir -p build
cd build
cmake -GNinja \
    -DCMAKE_LINKER=/usr/bin/lld-15 \
    -DCMAKE_CUDA_ARCHITECTURES=75 \
    -DUSE_LLVM=llvm-config-15 \
    -DSUMMARIZE=1 \
    -DUSE_CUDA=1 \
    -DUSE_MICRO=1 \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_CUTLASS=1 \
    -DUSE_THRUST=1 \
    ..
cmake --build . --
