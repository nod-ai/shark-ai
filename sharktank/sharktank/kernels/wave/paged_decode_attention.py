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
from torch._prims_common import DeviceLikeType
from typing import Callable
from iree.compiler.ir import Module, Context

# TODO: really wish Python had an equivalent to Rust's std::sync::OnceLock/OnceCell
phase_0_global: Callable | None = None
phase_1_global: Callable | None = None

# Dimensions are taken from Wave's `templates.paged_decode_attention`
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


def paged_decode_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    sequence_lengths: torch.Tensor,
    device: DeviceLikeType,
    block_size: int = 32,
    logit_cap: float = 0.0,
    num_kv_splits: int = 8,
) -> torch.Tensor:
    """
    TODO: docs
    """
    def phase_0_once() -> Callable:
        @mlir_kernel(
            inputs=(
                MLIRTensor[S, B, K1, F16],
                MLIRTensor[N_KV, BH, K1, F16],
                MLIRTensor[N_KV, BH, N, F16],
                MLIRTensor[S, I32],
                MLIRTensor[K2, I32],
                MLIRTensor[U, S, B, N, F32],
                MLIRTensor[U, S, B, F32],
            ),
            results=(MLIRTensor[U, S, B, N, F32], MLIRTensor[U, S, B, F32]),
        )
        def phase_0_wrapper(
            query,
            key,
            value,
            request_indices,
            kv_indices,
            logits_buf,
            logits_max_buf,
            phase_0_logits=None,
            phase_0_logits_max=None,
        ):
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
            name = mangle("wave_paged_attention_decode_phase_0", **kernel_params)
            options = WaveCompileOptions(
                subs=hyperparams_0,
                canonicalize=True,
                schedule=SchedulingType.NONE,
                dynamic_symbols=dynamic_symbols_0,
                func_name=name,
                compile_to_mlir=True,
            )
            options = set_default_run_config(options)
            with Context() as _:
                nonlocal phase_0_kernel
                phase_0_kernel = wave_compile(options, phase_0_kernel)

            phase_0_body = get_wave_module_body_asm(Module.parse(phase_0_kernel.asm))

            # `{% raw %}` disables Jinja templating for the contained text
            mlir = f"""
module {{
{{% raw %}}
    {phase_0_body}
{{% endraw %}}
    util.func private @{{{{kernel_name}}}}(
        %query: !query,
        %key: !key,
        %value: !value,
        %request_indices: !request_indices,
        %kv_indices: !kv_indices,
        %logits_buf: !logits_buf,
        %logits_max_buf: !logits_max_buf
    ) -> (!phase_0_logits, !phase_0_logits_max) {{
        %c0 = arith.constant 0 : index

        %num_sequences = tensor.dim %query, %c0 : !query
        %sum_kv_seq_lens = tensor.dim %key, %c0 : !key
        %kv_block_table_len = tensor.dim %kv_indices, %c0 : !kv_indices

        %phase_0_logits, %phase_0_logits_max = func.call @{name}(
            %query, %key, %value,
            %request_indices, %kv_indices,
            %logits_buf, %logits_max_buf,
            %kv_block_table_len, %sum_kv_seq_lens, %num_sequences
        ) : (
            !query, !key, !value,
            !request_indices, !kv_indices,
            !logits_buf, !logits_max_buf,
            index, index, index
        ) -> (!phase_0_logits, !phase_0_logits_max)

        util.return %phase_0_logits, %phase_0_logits_max : !phase_0_logits, !phase_0_logits_max
    }}
}}
"""

            return MLIRSpec(mlir)
        return phase_0_wrapper

    def phase_1_once() -> Callable:
        @mlir_kernel(
            inputs=(
                MLIRTensor[U, S, B, N, F32],
                MLIRTensor[U, S, B, F32],
                MLIRTensor[S, I32],
                MLIRTensor[S, B, N, F32],
            ),
            results=(MLIRTensor[S, B, N, F32],),
        )
        def phase_1_wrapper(
            phase_0_logits,
            phase_0_logits_max,
            request_indices,
            output_buf,
            result=None,
        ):
            """
            TODO: docs
            """
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
            name = mangle("wave_paged_attention_decode_phase_1", **kernel_params)
            options = WaveCompileOptions(
                subs=hyperparams_1,
                canonicalize=True,
                schedule=SchedulingType.NONE,
                dynamic_symbols=dynamic_symbols_1,
                func_name=name,
                compile_to_mlir=True,
            )
            options = set_default_run_config(options)
            with Context() as _:
                nonlocal phase_1_kernel
                phase_1_kernel = wave_compile(options, phase_1_kernel)

            phase_1_body = get_wave_module_body_asm(Module.parse(phase_1_kernel.asm))

            # `{% raw %}` disables Jinja templating for the contained text
            mlir = f"""
module {{
{{% raw %}}
    {phase_1_body}
{{% endraw %}}
    util.func private @{{{{kernel_name}}}}(
        %phase_0_logits: !phase_0_logits,
        %phase_0_logits_max: !phase_0_logits_max,
        %request_indices: !request_indices,
        %output_buf: !output_buf
    ) -> !result {{
        %c0 = arith.constant 0 : index
        %num_sequences = tensor.dim %request_indices, %c0 : !request_indices

        %result = func.call @{name}(
            %phase_0_logits, %phase_0_logits_max, %request_indices,
            %output_buf, %num_sequences
        ) : (
            !phase_0_logits, !phase_0_logits_max, !request_indices,
            !output_buf, index
        ) -> !result

        util.return %result : !result
    }}
}}
"""

            return MLIRSpec(mlir)
        return phase_1_wrapper

    num_sequences, num_query_heads, query_head_dim = query.shape
    _, num_kv_heads, kv_head_dim = value.shape

    shape = paged_decode_attention_shape(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        head_size=query_head_dim,
        head_size_kv=kv_head_dim,
        block_size=block_size,
        num_seqs=num_sequences,
    )

    request_indices = torch.cat(
        (
            torch.zeros((1,), dtype=torch.int32, device=device),
            torch.cumsum(sequence_lengths[:-1], dim=0, dtype=torch.int32),
        )
    )
    assert request_indices.shape == (num_sequences,)

    seq_lens_sum = torch.sum(sequence_lengths).item()

    # Aka the block table
    kv_indices = torch.arange(
        seq_lens_sum, dtype=torch.int32, device=device
    )
    logits_shape, logits_max_shape = get_paged_decode_intermediate_arrays_shapes(
        shape,
        num_kv_splits,
    )
    logits_buf = torch.empty(
        logits_shape,
        dtype=torch.float32,
        device=device,
    )
    logits_max_buf = logits_buf.new_empty(
        logits_max_shape,
    )
    output_buf = torch.empty(
        (num_sequences, num_query_heads, kv_head_dim),
        dtype=torch.float32,
        device=device,
    )

    # TODO: should num_kv_splits be kv_lens?
    # TODO: input and output dtype should be FP4?
    # TODO: what should logit_cap be?

    use_multi_head_attention = shape.num_query_heads == shape.num_kv_heads
    if use_multi_head_attention:
        mfma_variant = (
            GenericDot(along_dim=MMAOperand.M, k_vec_size=4, k_mult=1),
            GenericDot(along_dim=MMAOperand.M, k_vec_size=1, k_mult=64),
        )
    else:
        mfma_variant = (MMAType.F32_16x16x1_F16, MMAType.F32_16x16x1_F16)

    (
        phase_0_kernel,
        phase_1_kernel,
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

    hyperparams_0.update(get_default_scheduling_params())
    hyperparams_1.update(get_default_scheduling_params())

    global phase_0_global
    if phase_0_global is None:
        phase_0_global = phase_0_once()

    global phase_1_global
    if phase_1_global is None:
        phase_1_global = phase_1_once()

    assert callable(phase_0_global)
    assert callable(phase_1_global)

    phase_0_logits, phase_0_logits_max = phase_0_global(
        query,
        key,
        value,
        request_indices,
        kv_indices,
        logits_buf,
        logits_max_buf,
    )
    output = phase_1_global(phase_0_logits, phase_0_logits_max, request_indices, output_buf)

    return output
