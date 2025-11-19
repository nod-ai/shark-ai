# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from .base import *

import torch

__all__ = [
    "einsum_2args_q4",
]


def einsum_util(einsum_str):
    es_in, es_out = einsum_str.split("->")
    es_in0, es_in1 = es_in.split(",")
    es_set = set(es_out)
    es_set = es_set.union(es_in0)
    es_set = es_set.union(es_in1)
    size = len(es_set)
    imap = dict()
    lmap = dict()
    for i in range(len(es_out)):
        imap[i] = es_out[i]
        lmap[es_out[i]] = i
    count = len(es_out)
    for c in es_set:
        if c not in lmap:
            imap[count] = c
            lmap[c] = count
            count += 1

    assert count == len(es_set)

    in0_idx = [lmap[i] for i in es_in0]
    in1_idx = [lmap[i] for i in es_in1]
    out_idx = [lmap[i] for i in es_out]

    input_idx_str = ", ".join(["d" + str(i) for i in range(size)])
    in0_idx_str = ", ".join(["d" + str(i) for i in in0_idx])
    in1_idx_str = ", ".join(["d" + str(i) for i in in1_idx])
    out_idx_str = ", ".join(["d" + str(i) for i in out_idx])

    iterators = ", ".join(
        ['"parallel"' if i in out_idx else '"reduction"' for i in range(size)]
    )

    affine_map_in0 = f"affine_map<({input_idx_str}) -> ({in0_idx_str})>"
    affine_map_in1 = f"affine_map<({input_idx_str}) -> ({in1_idx_str})>"
    affine_map_out = f"affine_map<({input_idx_str}) -> ({out_idx_str})>"

    indexing_maps = f"""{affine_map_in0},
    {affine_map_in1},
    {affine_map_out}
"""

    out_dyn_dim_size_str = ""
    for c in es_out:
        if c in es_in0:
            out_dyn_dim_size_str += "%a" + str(es_in0.find(c)) + ","
        elif c in es_in1:
            if es_in1.find(c) == len(es_in1) - 1:
                out_dyn_dim_size_str += "%b_unblocked_dim,"
            else:
                out_dyn_dim_size_str += "%b" + str(es_in1.find(c)) + ","
        else:
            raise Exception("Invalid einsum string")
    out_dyn_dim_size_str = out_dyn_dim_size_str[:-1]
    return (
        (in0_idx, in1_idx, out_idx),
        iterators,
        indexing_maps,
        out_dyn_dim_size_str,
    )


@CustomOp.register(library=LIBRARY)
class einsum_2args_q4(CustomOp):
    """Einsum that takes two tensor inputs and returns one tensor.

    The first input is expected to be a normal tensor.

    The second input corresponds to the BlockScaledLayout and operates on planar `d`
    and `qs` tensors as specified there:

    * `d`: `[..., K // BLOCK_SIZE, 1]`
    * `qs`: `[..., K // BLOCK_SIZE, BLOCK_SIZE // 2]` (of uint8)
    * `m`: `[..., K // BLOCK_SIZE, 1]`
    """

    signature = (
        "einsum_2args_q4(Tensor a, Tensor d, Tensor qs, Tensor m, str es) -> (Tensor)"
    )

    def select(self, ksel: KernelSelection):
        a_desc = ksel.arg_tensor(0)  # Shape [b, ] m, k
        d_desc = ksel.arg_tensor(1)  # Shape [N, K // BLOCK_SIZE, 1]
        qs_desc = ksel.arg_tensor(2)  # Shape [N, K // BLOCK_SIZE, BLOCK_SIZE // 2]
        m_desc = ksel.arg_tensor(3)  # Shape [N, K // BLOCK_SIZE, 1]
        einsum_str = ksel.attr_str(4).v

        # a arg
        a_dims = a_desc.t.shape
        torch._check(
            a_desc.t.dtype.is_floating_point,
            lambda: f"einsum_2args_q4 arg 'a': Expected floating point (got {a_desc.t.dtype})",
        )

        # qs arg
        *qs_dims, qs_group0, qs_bs_div_2 = qs_desc.t.shape
        block_size = qs_bs_div_2 * 2

        # d arg
        *d_dims, d_group0, d_one = d_desc.t.shape
        torch._check(
            d_group0 == qs_group0 and d_one == 1 and len(d_dims) == len(qs_dims),
            lambda: f"einsum_2args_q4 arg 'd': Incorrect shape (got {d_desc.t.shape})",
        )

        # m arg
        *m_dims, m_group0, m_one = m_desc.t.shape
        torch._check(
            m_desc.t.dtype == d_desc.t.dtype and len(m_dims) == len(qs_dims),
            lambda: f"einsum_2args_q4 arg 'm': Incorrect dtype (got {m_desc.t.dtype})",
        )
        # einsum_str
        torch._check(
            einsum_str.count(",") == 1 and einsum_str.count("->") == 1,
            lambda: f"einsum_2args_q4 arg 'einsum_str': Expected format '{{}},{{}}->{{}}' (got '{einsum_str}')",
        )

        es_in, es_out = einsum_str.split("->")
        es_in0, es_in1 = es_in.split(",")
        es_set = set(es_out)

        shp = qs_desc.t.shape
        b_dims = list(shp[:-2]) + [shp[-2] * block_size]
        torch._check(
            len(es_in0) == len(a_desc.t.shape)
            and len(es_in1)
            == len(qs_desc.t.shape)
            - 1,  # The quantized shape is larger until the blocks are collapsed
            lambda: f"einsum_2args_q4 arg 'einsum_str': Einsum str dimensions do not match input dimensions (got '{einsum_str}' with inputs: {a_desc.t.shape} and {b_dims})",
        )
        torch._check(
            len(es_in0) == len(set(es_in0))
            and len(es_in1) == len(set(es_in1))
            and len(es_in0) != 0
            and len(es_in1) != 0,
            lambda: f"einsum_2args_q4 arg 'einsum_str': Unsupported einsum str (got '{einsum_str}')",
        )

        # Check corresponding dimensions match
        for i in range(len(es_in0)):
            a_dim = a_dims[i]
            c = es_in0[i]
            pos = es_in1.find(c)
            if pos >= 0:
                b_dim = b_dims[pos]
                torch._check(
                    a_dim == b_dim,
                    lambda: f"einsum_2args_q4 arg 'einsum_str': Einsum str dimensions do not match input dim for idx {c} (got '{einsum_str}' with inputs: {a_desc.t.shape} and {b_dims})",
                )

        # Determine the output shape by referencing corresponding input shapes
        out_dims = []
        for c in es_out:
            pos0 = es_in0.find(c)
            pos1 = es_in1.find(c)
            a_dim = a_dims[pos0]
            b_dim = b_dims[pos1]
            if pos0 >= 0:
                out_dims.append(a_dim)
            elif pos1 >= 0:
                out_dims.append(b_dim)
            else:
                torch._check(
                    False,
                    lambda: f"einsum_2args_q4 arg 'einsum_str': output indices must be in input indices (got '{einsum_str}')",
                )

        # Specialize on BS
        qs_desc.specialize_dims(-1)
        d_desc.specialize_dims(-1)
        m_desc.specialize_dims(-1)

        # Shape batch..., m, n
        c_desc = ksel.return_new_tensor(out_dims, dtype=a_desc.t.dtype)

    def generate(self, ksel: KernelSelection, kb: KernelBuilder):
        a = kb.arg_value(0)
        a_tensor_type = RankedTensorType(a.type)
        d = kb.arg_value(1)
        d_tensor_type = RankedTensorType(d.type)
        qs = kb.arg_value(2)
        qs_tensor_type = RankedTensorType(qs.type)
        einsum_str = ksel.arg_descs[4].v
        # einsum_str = "mek,menk->men"

        es_in, es_out = einsum_str.split("->")
        es_in0, es_in1 = es_in.split(",")

        es_name = "_".join([es_in0, es_in1, es_out])

        (
            (es_0, es_1, es_2),
            einsum_iterators,
            einsum_indexing_maps,
            oddss,
        ) = einsum_util(einsum_str)

        rank1 = len(es_1)
        dequant_iterators = ", ".join(
            ['"parallel"' for i in range(rank1 + 1)]
        )  # rank + 1 because of the group dimensions
        input_idx_str = ", ".join(["d" + str(i) for i in range(rank1 + 1)])
        broadcast_idx_str = ", ".join(
            ["d" + str(i) if i != rank1 else "0" for i in range(rank1 + 1)]
        )
        affine_map_parallel = f"affine_map<({input_idx_str}) -> ({input_idx_str})>"
        affine_map_broadcast = f"affine_map<({input_idx_str}) -> ({broadcast_idx_str})>"
        dequant_indexing_maps = f"""{affine_map_broadcast},
        {affine_map_broadcast},
        {affine_map_parallel},
        {affine_map_parallel}"""

        size_str = "x".join("?" for i in range(rank1 - 2))

        rank = a_tensor_type.rank
        *n_dims, group0, bs_i8 = qs_tensor_type.shape
        bs = bs_i8 * 2  # 2 nibbles per byte.
        group = group0 * bs
        a_type_str = str(a_tensor_type.element_type)
        scale_type_str = str(d_tensor_type.element_type)

        template_file = "einsum_2args_q4.mlir"
        target_function_name = (
            f"amdsharktank_einsum_2args_q4_{es_name}_{bs}_{a_type_str}"
        )

        target_function = inline_template_function(
            kb,
            template_file,
            target_function_name,
            bs=bs,
            bs_i8=bs_i8,
            a_type=a_type_str,
            scale_type=scale_type_str,
            dequant_indexing_maps=dequant_indexing_maps,
            dequant_iterator_types=dequant_iterators,
            einsum_indexing_maps=einsum_indexing_maps,
            einsum_iterator_types=einsum_iterators,
            es_name=es_name,
            a_size=len(es_in0),
            b_size=len(es_in1),
            c_size=len(es_out),
            out_dyn_dim_size_str=oddss,
        )
        kb.yield_results(*call_function(target_function, *kb.arg_bindings))
