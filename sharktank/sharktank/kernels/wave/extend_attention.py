# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from sharktank.kernels.base import *
from sharktank.kernels.mlir_kernel import *
from sharktank.kernels.wave.utils import get_wave_module_body_asm, mangle
from wave_lang.kernel.wave.templates.extend_attention import (
    get_extend_attention_kernel,
)
from wave_lang.kernel.wave.scheduling.schedule import SchedulingType
from wave_lang.kernel.wave.compile import wave_compile, WaveCompileOptions
from wave_lang.kernel.wave.templates.attention_common import AttentionShape
from wave_lang.kernel.wave.constraints import MMAType
from wave_lang.kernel.wave.utils.general_utils import (
    get_default_scheduling_params,
)
from wave_lang.kernel.wave.utils.run_utils import (
    set_default_run_config,
)
from iree.compiler.ir import (
    Module,
    Context,
)
import torch
from dataclasses import replace


__all__ = [
    "wave_extend_attention",
]


def get_wave_extend_attention_asm(
    target_function_name: str,
    shape: AttentionShape,
    mfma_variant: tuple[MMAType, MMAType],
    q_shape: tuple[int],
    k_shape: tuple[int],
    v_shape: tuple[int],
    k_cache_shape: tuple[int],
    v_cache_shape: tuple[int],
    o_shape: tuple[int],
    input_dtype: torch.dtype = torch.float16,
    output_dtype: torch.dtype = torch.float32,
    size_dtype: torch.dtype = torch.int32,
    is_causal: bool = False,
    logit_cap: float = 0.0,
    layer_scaling: Optional[float] = None,
    num_waves: int = 4,
    use_custom_mask: bool = False,
) -> str:
    assert not (
        is_causal and use_custom_mask
    ), "Causal and custom mask cannot be True simultaneously"

    assert shape.num_query_heads % shape.num_kv_heads == 0
    (
        q_extend,
        k_extend,
        v_extend,
        k_buffer,
        v_buffer,
        b_req_idx,
        b_seq_len,
        qo_indptr,
        kv_indptr,
        kv_indices,
        custom_mask,
        mask_offsets,
        b_start_loc,
        b_seq_len_prefix,
        extend_token_num,
        max_len_extend,
        logit_cap,
        _,
        _,
    ) = create_inputs(shape, dtype)
    shape = replace(shape, max_seq_len=max_len_extend)
    if mfma_variant == MMAType.F32_16x16x16_F16:
        num_waves = 4
    if mfma_variant == MMAType.F32_32x32x8_F16:
        num_waves = 2

    output = torch.empty(
        extend_token_num, shape.num_query_heads, shape.head_size, dtype=torch.float32
    )
    (extend_attention, hyperparams, dynamic_symbols,) = get_extend_attention_kernel(
        shape,
        mfma_variant,
        q_extend.shape,
        k_extend.shape,
        v_extend.shape,
        k_buffer.shape,
        v_buffer.shape,
        output.shape,
        is_causal=is_causal,
        logit_cap=logit_cap,
        num_waves=num_waves,
        use_custom_mask=use_custom_mask,
    )
    hyperparams.update(get_default_scheduling_params())
    options = WaveCompileOptions(
        subs=hyperparams,
        canonicalize=True,
        schedule=enable_scheduling,
        dynamic_symbols=dynamic_symbols,
        use_buffer_ops=use_buffer_ops,
        func_name=target_function_name,
        compile_to_mlir=True,
        iree_launch_async=False,
    )
    options = set_default_run_config(options)

    with Context() as ctx:
        extend_attention = wave_compile(options, extend_attention)

    asm = extend_attention.asm
    return asm
