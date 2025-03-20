// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

!q_type = tensor<{{b1}}x{{b2}}x{{l}}x{{d}}x{{i_dtype}}>
!k_type = tensor<{{b1}}x{{b2}}x{{s}}x{{d}}x{{i_dtype}}>
!v_type = tensor<{{b1}}x{{b2}}x{{s}}x{{e}}x{{i_dtype}}>
!a_type = tensor<{{l}}x{{s}}x{{a_dtype}}>
!trans_v_type = tensor<{{b1}}x{{b2}}x{{e}}x{{s}}x{{i_dtype}}>
!o_type = tensor<{{b1}}x{{b2}}x{{l}}x{{e}}x{{o_dtype}}>
!o_dyn_type = tensor<?x?x?x{{o_dtype}}>
!o_collapsed_type = tensor<{{b}}x{{l}}x{{e}}x{{o_dtype}}>
!q_collapsed_type = tensor<{{b}}x{{l}}x{{d}}x{{i_dtype}}>
!k_collapsed_type = tensor<{{b}}x{{s}}x{{d}}x{{i_dtype}}>
!v_collapsed_type = tensor<{{b}}x{{s}}x{{e}}x{{i_dtype}}>
!a_collapsed_type = tensor<{{l}}x{{s}}x{{a_dtype}}>
!s_type = tensor<{{scale_dtype}}>

module {

util.func private @{{func_name}}(
    %q: !q_type,
    %k: !k_type,
    %v: !v_type,
    %s: !s_type,
    %a: !a_type) -> !o_type {

        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %c2 = arith.constant 2 : index
        %c3 = arith.constant 3 : index
        %b0 = arith.constant {{b}} : index


        %l = tensor.dim %q, %c2 : !q_type
        %e = tensor.dim %v, %c3 : !v_type

        %scale = tensor.extract %s[] : !s_type
        %empty_dyn = tensor.empty(%b0, %l, %e) : !o_dyn_type
        %empty = tensor.cast %empty_dyn : !o_dyn_type to !o_collapsed_type

        %collapsed_q = tensor.collapse_shape %q [[0, 1], [2], [3]] : !q_type into !q_collapsed_type
        %collapsed_k = tensor.collapse_shape %k [[0, 1], [2], [3]] : !k_type into !k_collapsed_type
        %collapsed_v = tensor.collapse_shape %v [[0, 1], [2], [3]] : !v_type into !v_collapsed_type

        %atten = iree_linalg_ext.attention {indexing_maps = [
                    affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d3)>,
                    affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d3)>,
                    affine_map<(d0, d1, d2, d3, d4) -> (d0, d4, d2)>,
                    affine_map<(d0, d1, d2, d3, d4) -> ()>,
                    affine_map<(d0, d1, d2, d3, d4) -> (d1, d4)>,
                    affine_map<(d0, d1, d2, d3, d4) -> (d0, d1, d2)>]}
                    ins(%collapsed_q, %collapsed_k, %collapsed_v, %scale, %a : !q_collapsed_type, !k_collapsed_type, !v_collapsed_type, {{scale_dtype}}, !a_collapsed_type) outs(%empty : !o_collapsed_type) {
                      ^bb0(%score: {{o_dtype}}):
                        iree_linalg_ext.yield %score : {{o_dtype}}
                    } -> !o_collapsed_type
        %expanded_o = tensor.expand_shape %atten [[0,1], [2], [3]] output_shape [{{b1}}, {{b2}}, %l, %e] : !o_collapsed_type into !o_type
        util.return %expanded_o : !o_type
    }
}
