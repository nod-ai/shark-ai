# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import math
from .config_struct import ModelParams, ServerParams

class RateLimiter:
    def __init__(
        self,
        *,
        model_params: ModelParams,
        server_params: ServerParams,
    ):
        self.model_params = model_params
        self.server_params = server_params

    def check_memory_availability(self, *, input_token_ids_len: int, available_pages: int):
        stride = self.model_params.paged_kv_cache.block_seq_stride
        total_requested_beams = self.server_params.decode_config.num_beams
        needed_pages = math.ceil(input_token_ids_len / stride) + total_requested_beams - 1

        if needed_pages <= available_pages:
            return True

        return False
