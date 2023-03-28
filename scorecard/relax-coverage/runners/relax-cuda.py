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

from typing import *


from relax_base import RelaxBase


class RelaxCuda(RelaxBase):
    name = "relax-cuda"

    def __init__(self, *args, **kwargs):
        cuda_sm = kwargs["benchmark_config"].cuda_sm
        super().__init__(
            target=f"cuda -libs=thrust -arch=sm_{cuda_sm} -max_shared_memory_per_block=49152 -max_threads_per_block=1024 -thread_warp_size=32 -registers_per_block=65536",
            *args,
            **kwargs,
        )


Runner = RelaxCuda
