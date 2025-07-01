# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Paged attention specifically designed for the decoding step of inference.
"""

from sharktank.kernels.mlir_kernel import (
    mlir_kernel,
    Dtype,
    StaticDim,
    DynDim,
    MLIRTensor,
    MLIRSpec,
)
from iree.turbine.kernel.wave.templates.paged_decode_attention import (
    paged_decode_attention_shape,
    get_paged_decode_attention_kernels,
    get_paged_decode_intermediate_arrays_shapes,
)
from sharktank.kernels.wave.utils import get_wave_module_body_asm, mangle
from iree.turbine.kernel.wave.constraints import MMAType, GenericDot, MMAOperand
from iree.turbine.kernel.wave.utils.general_utils import get_default_scheduling_params
from iree.turbine.kernel.wave.compile import wave_compile, WaveCompileOptions
from iree.turbine.kernel.wave.scheduling import SchedulingType
from iree.turbine.kernel.wave.utils.run_utils import set_default_run_config
import torch
from iree.compiler.ir import Module, Context

# Dimensions are taken from Wave's `templates.paged_decode_attention.phase_0`
S = DynDim.NUM_SEQUENCES
N_KV = DynDim.SUM_KV_SEQ_LENS
K2 = DynDim.KV_BLOCK_TABLE_LEN
B = StaticDim.NUM_QUERY_HEADS
K1 = StaticDim.QUERY_HEAD_DIM
BH = StaticDim.NUM_KV_HEADS
N = StaticDim.KV_HEAD_DIM
U = StaticDim.NUM_KV_SPLITS

F16 = Dtype.F16(
    torch.float16
)  # TODO: for compat with existing shark tank code, but should be using fp4 in the future right?
F32 = Dtype.F32(torch.float32)
I32 = Dtype.I32(torch.int32)


@mlir_kernel(
    inputs=(
        MLIRTensor[S, B, K1, F16],
        MLIRTensor[N_KV, BH, K1, F16],
        MLIRTensor[N_KV, BH, N, F16],
        MLIRTensor[S, B, N, F32],
        MLIRTensor[S, I32],
        MLIRTensor[K2, I32],
        MLIRTensor[U, S, B, N, F32],
        MLIRTensor[U, S, B, F32],
    ),
    results=(MLIRTensor[S, B, N, F32],),
)
def paged_decode_attention(
    query,
    key,
    value,
    output_buf,
    request_indices,
    kv_indices,
    phase_0_logits,
    phase_0_logits_max,
    result=None,
):
    """
    TODO: would it make more sense to allocate request_indices, kv_indices, phase_0_logits* inside this func?
    """
    num_sequences, num_query_heads, query_head_dim = query.type.shape
    sum_kv_seq_lens, num_kv_heads, kv_head_dim = value.type.shape
    shape = paged_decode_attention_shape(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=query_head_dim,
        head_size_kv=kv_head_dim,
        block_size=32,  # TODO: Turns out I can't pass block_size into this function, since the inputs to this end up in the MLIR
        num_seqs=num_sequences,
    )

    num_kv_splits, _, _, _ = phase_0_logits.type.shape

    use_multi_head_attention = shape.num_query_heads == shape.num_kv_heads
    if use_multi_head_attention:
        mfma_variant = (
            GenericDot(along_dim=MMAOperand.M, k_vec_size=4, k_mult=1),
            GenericDot(along_dim=MMAOperand.M, k_vec_size=1, k_mult=64),
        )
    else:
        mfma_variant = (MMAType.F32_16x16x1_F16, MMAType.F32_16x16x1_F16)

    # TODO: should num_kv_splits be kv_lens?
    # TODO: input and output dtype should be FP4?
    # TODO: what should logit_cap be?
    (
        phase_0,
        phase_1,
        hyperparams_0,
        hyperparams_1,
        dynamic_symbols_0,
        dynamic_symbols_1,
    ) = get_paged_decode_attention_kernels(
        shape,
        mfma_variant,
        num_kv_splits,
        input_dtype=torch.float16,
        output_dtype=torch.float32,
        logit_cap=0.0,
    )

    # TODO: Do we need block_size and logit_cap?
    kernel_params = {
        B.name: num_query_heads,
        K1.name: query_head_dim,
        BH.name: num_kv_heads,
        N.name: kv_head_dim,
        U.name: num_kv_splits,
        "input_dtype": F16.name,
        "output_dtype": F32.name,
    }

    hyperparams_0.update(get_default_scheduling_params())
    hyperparams_1.update(get_default_scheduling_params())

    name_0 = mangle("wave_paged_attention_decode_phase_0", **kernel_params)
    options_0 = WaveCompileOptions(
        subs=hyperparams_0,
        canonicalize=True,
        schedule=SchedulingType.NONE,
        dynamic_symbols=dynamic_symbols_0,
        func_name=name_0,
        compile_to_mlir=True,
    )
    options_0 = set_default_run_config(options_0)
    with Context() as _:
        phase_0 = wave_compile(options_0, phase_0)

    name_1 = mangle("wave_paged_attention_decode_phase_1", **kernel_params)
    options_1 = WaveCompileOptions(
        subs=hyperparams_1,
        canonicalize=True,
        schedule=SchedulingType.NONE,
        dynamic_symbols=dynamic_symbols_1,
        func_name=name_1,
        compile_to_mlir=True,
    )
    options_1 = set_default_run_config(options_1)
    with Context() as _:
        phase_1 = wave_compile(options_1, phase_1)

    phase_0_body = get_wave_module_body_asm(Module.parse(phase_0.asm))
    phase_1_body = get_wave_module_body_asm(Module.parse(phase_1.asm))

    # `{% raw %}` disables Jinja templating for the contained text
    mlir = f"""
module {{

{{% raw %}}
{phase_0_body}
{phase_1_body}
{{% endraw %}}

    util.func private @{{{{kernel_name}}}}(
        %query: !query, %key: !key, %value: !value, %output_buf: !output_buf,
        %request_indices: !request_indices, %kv_indices: !kv_indices,
        %phase_0_logits: !phase_0_logits, %phase_0_logits_max: !phase_0_logits_max
    ) -> !result {{
        %c0 = arith.constant 0 : index

        %num_sequences = tensor.dim %query, %c0 : !query
        %sum_kv_seq_lens = tensor.dim %key, %c0 : !key
        %kv_block_table_len = tensor.dim %kv_indices, %c0 : !kv_indices

        func.call @{name_0}(
            %query, %key, %value,
            %request_indices, %kv_indices,
            %phase_0_logits, %phase_0_logits_max,
            %kv_block_table_len, %sum_kv_seq_lens, %num_sequences
        ) : (
            !query, !key, !value,
            !request_indices, !kv_indices,
            !phase_0_logits, !phase_0_logits_max,
            index, index, index
        ) -> (!phase_0_logits, !phase_0_logits_max)
        %result = func.call @{name_1}(
            %phase_0_logits, %phase_0_logits_max,
            %request_indices, %output_buf,
            %num_sequences
        ) : (
            !phase_0_logits, !phase_0_logits_max,
            !request_indices, !output_buf,
            index
        ) -> !result
        util.return %result : !result
    }}
}}
"""

    return MLIRSpec(mlir)
