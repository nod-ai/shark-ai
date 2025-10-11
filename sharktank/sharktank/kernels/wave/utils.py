# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from iree.compiler.ir import (
    Module,
    StringAttr,
)
import math
import re
import functools
import torch
from torch.nn import functional as F
from wave_lang.kernel.wave.templates.attention_common import AttentionShape
from wave_lang.kernel.wave.utils.torch_utils import (
    device_randn,
    device_zeros,
    device_empty,
    device_arange,
    device_randint,
    device_full,
)
from enum import Enum


class ScoreMod(Enum):
    SoftCap = 0
    RPE = 1


def get_wave_module_body_asm(module: Module) -> str:
    """
    Concatenates the MLIR of all operations within the
    body region of the top-level wave_compile() module and modifies the
    visibility of the top-level public FuncOp generated in wave_compile()
    to private, so that it gets removed when inlined.
    """
    block = module.operation.regions[0].blocks[0]
    ops_asm = []
    for op in block.operations:
        if op.operation.name == "func.func":
            op.attributes["sym_visibility"] = StringAttr.get("private")
        ops_asm.append(op.get_asm())

    return "\n".join(ops_asm)


# Disallowed characters in an MLIR suffix-id
_DISALLOWED = re.compile(r"[^A-Za-z0-9\$\._-]")


def mangle(base_name: str, **kwargs) -> str:
    r"""
    Build a readable, deterministic MLIR kernel name (note the double underscore
    after `base_name`):
    ```
    base_name__key1_val1_key2_val2_...
    ```
    Make sure the `kwargs` uniquely identify the kernel for any shapes or dtypes
    it can take. TODO: is this the right defn of unique?
    Keys are sorted so the output is stable.
    According to the MLIR LangRef, only characters matching the regex
    `[A-Za-z0-9\$\._-]` are allowed in an unquoted suffix-id. Any other
    characters are simply removed.
    """
    parts: list[str] = [base_name, ""]

    for key in kwargs:
        val = kwargs[key]
        parts.append(f"{str(key)}_{str(val)}")

    return re.sub(_DISALLOWED, "", "_".join(parts))


def create_extend_attention_inputs(
    shape: AttentionShape,
    dtype: torch.dtype,
):

    N_CTX = shape.context_len
    B = shape.num_seqs
    H_KV = shape.num_kv_heads
    H_Q = shape.num_query_heads
    D = shape.head_size
    b_seq_len_prefix = device_randint(1, N_CTX // 2, (B,), dtype=torch.int32)
    if shape.fixed_seq_len_prefix:
        b_seq_len_prefix.fill_(shape.fixed_seq_len_prefix)
    b_seq_len_extend = device_randint(1, N_CTX // 2, (B,), dtype=torch.int32)
    if shape.fixed_seq_len_extend:
        b_seq_len_extend.fill_(shape.fixed_seq_len_extend)
    b_seq_len = b_seq_len_prefix + b_seq_len_extend

    b_req_idx = device_arange(B, dtype=torch.int32)
    b_start_loc = device_zeros((B,), dtype=torch.int32)
    b_start_loc[1:] = torch.cumsum(b_seq_len[:-1], 0)
    b_start_loc_extend = device_zeros((B,), dtype=torch.int32)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)

    kv_indptr = device_zeros((B + 1,), dtype=torch.int32)
    kv_indptr[1 : B + 1] = torch.cumsum(b_seq_len_prefix[:B], dim=0)
    kv_indices = device_zeros((b_seq_len_prefix.sum().item(),), dtype=torch.int32)

    for i in range(B):
        kv_indices[kv_indptr[i] : kv_indptr[i + 1]] = torch.arange(
            b_start_loc[i], b_start_loc[i] + b_seq_len_prefix[i]
        )
    total_token_num = torch.sum(b_seq_len).item()
    extend_token_num = torch.sum(b_seq_len_extend).item()
    k_buffer = device_empty((total_token_num, H_KV, D), dtype=dtype).normal_(
        mean=0.1, std=0.2
    )
    v_buffer = device_empty((total_token_num, H_KV, D), dtype=dtype).normal_(
        mean=0.1, std=0.2
    )

    k_extend = device_empty((extend_token_num, H_KV, D), dtype=dtype)
    v_extend = device_empty((extend_token_num, H_KV, D), dtype=dtype)
    q_extend = device_empty((extend_token_num, H_Q, D), dtype=dtype)
    for i in range(B):
        extend_start_in_buffer = b_start_loc[i] + b_seq_len_prefix[i]
        extend_end_in_buffer = b_start_loc[i] + b_seq_len[i]
        extend_start = b_start_loc_extend[i]
        extend_end = b_start_loc_extend[i] + b_seq_len_extend[i]
        k_extend[extend_start:extend_end] = k_buffer[
            extend_start_in_buffer:extend_end_in_buffer
        ]
        v_extend[extend_start:extend_end] = v_buffer[
            extend_start_in_buffer:extend_end_in_buffer
        ]
        q_extend[extend_start:extend_end] = device_empty(
            (b_seq_len_extend[i], H_Q, D), dtype=dtype
        ).normal_(mean=0.1, std=0.2)

    b_seq_len_extend = b_seq_len - b_seq_len_prefix
    b_start_loc_extend = torch.zeros_like(b_seq_len)
    b_start_loc_extend[1:] = torch.cumsum(b_seq_len_extend[:-1], 0)
    max_len_extend = torch.max(b_seq_len_extend, 0)[0].item()
    qo_indptr = device_zeros((B + 1,), dtype=torch.int32)
    qo_indptr[1 : B + 1] = torch.cumsum(b_seq_len_extend[:B], dim=0)
    logit_cap = 30.0

    b_seq_mask_len = b_seq_len_extend * b_seq_len
    # NOTE: Custom mask is of causal nature in this test. Random mask numerics
    # is not tested.
    custom_mask = device_full(
        (b_seq_mask_len.sum().item(),), fill_value=1, dtype=torch.int8
    )
    mask_offsets = device_zeros((B + 1,), dtype=torch.int32)
    mask_offsets[1 : B + 1] = torch.cumsum(b_seq_mask_len[:B], dim=0)
    for i in range(B):
        causal_mask = (
            torch.tril(
                device_full(
                    (b_seq_len_extend[i], b_seq_len_extend[i]),
                    fill_value=1,
                    dtype=torch.int8,
                ),
                diagonal=0,
            )
            == 1
        )
        prefix_mask = device_full(
            (b_seq_len_extend[i], b_seq_len_prefix[i]), fill_value=1, dtype=torch.int8
        )
        mask_flatten = torch.cat([prefix_mask, causal_mask], dim=1).flatten()
        custom_mask[mask_offsets[i] : mask_offsets[i + 1]] = mask_flatten

    max_rpe_context_length = 10
    rpe_bias = device_zeros(max_rpe_context_length + 1, dtype=torch.float32)
    rpe_bias.copy_(device_randn(max_rpe_context_length + 1, dtype=torch.float32))
    rpe_bias[max_rpe_context_length] = 0

    return (
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
        rpe_bias,
        max_rpe_context_length,
    )


def context_attention_fwd(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    o: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    max_len_extend: int,
    is_causal: bool = False,
    logit_cap: float = 0.0,
    rpe_bias: torch.Tensor = None,
    score_mod: ScoreMod = ScoreMod.SoftCap,
    max_rpe_context_length: int = 0,
):

    cu_seq_lens = [0] * (len(b_seq_len) + 1)
    for i, seq_len in enumerate(b_seq_len):
        cu_seq_lens[i + 1] = cu_seq_lens[i] + seq_len

    for i in range(len(b_seq_len)):
        start, end = cu_seq_lens[i], cu_seq_lens[i + 1]
        qkv_len = end - start
        Q = q[start:end].permute(1, 0, 2)
        K = k[start:end].permute(1, 0, 2)
        K = K.repeat_interleave(Q.shape[0] // K.shape[0], dim=0)
        V = v[start:end].permute(1, 0, 2)
        V = V.repeat_interleave(Q.shape[0] // V.shape[0], dim=0)
        dk_sqrt = math.sqrt(1.0 / Q.shape[-1])
        a = torch.bmm(Q * dk_sqrt, K.transpose(-1, -2))
        if score_mod == ScoreMod.SoftCap:
            a = a / logit_cap
            a = torch.tanh(a)
            a = a * logit_cap
        else:
            rpe_cond = t5_rpe_masked_cond(
                rpe_bias,
                max_rpe_context_length=max_rpe_context_length,
                sequence_length=K.shape[1],
            )
            rpe_cond = rpe_cond.unsqueeze(0)
            rpe_cond = rpe_cond.expand(Q.shape[0], *rpe_cond.shape[1:])
            a = a + rpe_cond
        if is_causal:
            # Create a mask for the upper triangular part (excluding the diagonal)
            mask = (
                torch.triu(torch.ones(a.shape[-2:]), diagonal=1)
                .unsqueeze(0)
                .expand(a.shape)
            )
            # Apply the mask to set the upper triangular part to -infinity
            a[mask == 1] = float("-inf")
        reference = torch.bmm(F.softmax(a, dim=-1).to(dtype=V.dtype), V)
        reference = reference.squeeze(0).permute(1, 0, 2)
        o[start:end] = reference

    return o


# From: https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/layers/attention/triton_ops/extend_attention.py#L369
def ref_extend_attn(
    q_extend: torch.Tensor,
    k_buffer: torch.Tensor,
    v_buffer: torch.Tensor,
    b_req_idx: torch.Tensor,
    b_start_loc: torch.Tensor,
    b_seq_len: torch.Tensor,
    b_seq_len_prefix: torch.Tensor,
    max_len_extend: int,
    extend_token_num: int,
    dtype: torch.dtype,
    is_causal: bool = False,
    logit_cap: float = 0.0,
    rpe_bias: torch.Tensor = None,
    score_mod: ScoreMod = ScoreMod.SoftCap,
    max_rpe_context_length: int = 0,
) -> torch.Tensor:
    total_token_num = k_buffer.shape[0]
    B, H_Q, D = b_req_idx.shape[0], q_extend.shape[-2], q_extend.shape[-1]
    q_buffer = device_empty(
        (total_token_num, H_Q, D), dtype=q_extend.dtype, device=q_extend.device
    )
    o_extend = device_empty((extend_token_num, H_Q, D), dtype=dtype)

    pt = 0
    for i in range(B):
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]
        q_buffer[pl:pr] = q_extend[pt : pt + cur_seq_len_extend]
        pt += cur_seq_len_extend

    o_buffer = torch.empty_like(q_buffer)
    context_attention_fwd(
        q_buffer,
        k_buffer,
        v_buffer,
        o_buffer,
        b_start_loc,
        b_seq_len,
        max_len_extend,
        is_causal,
        logit_cap=logit_cap,
        rpe_bias=rpe_bias,
        score_mod=score_mod,
        max_rpe_context_length=max_rpe_context_length,
    )

    pt = 0
    for i in range(B):
        cur_seq_len_extend = b_seq_len[i] - b_seq_len_prefix[i]
        pl, pr = b_start_loc[i] + b_seq_len_prefix[i], b_start_loc[i] + b_seq_len[i]
        o_extend[pt : pt + cur_seq_len_extend] = o_buffer[pl:pr]
        pt += cur_seq_len_extend

    return o_extend


def create_causal_mask(seq_len: int, dtype: torch.dtype, device: str):
    # Create a simple attention mask with shape [1, 1, seq_len, seq_len]
    # This broadcasts across all batches and heads
    mask = torch.triu(torch.ones(seq_len, seq_len) * float("-inf"), diagonal=1)
    mask = mask.unsqueeze(0).unsqueeze(0)
    mask = mask.to(dtype).to(device=device)
    return mask
