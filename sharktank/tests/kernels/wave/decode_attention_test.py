# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import logging
import unittest

import torch
from sharktank.kernels import wave
from iree.turbine import aot


logging.basicConfig(level=logging.DEBUG)

# TODO(paulzzy): Currently fails with `torch._dynamo.exc.Unsupported: call_function UserDefinedClassVariable(<class 'sympy.core.assumptions.StdFactKB'>) [ConstDictVariable()] {}`


class WaveDecodeAttention(unittest.TestCase):
    def test_generate_mlir(self):
        class WaveDecodeAttentionModule(torch.nn.Module):
            def forward(
                self,
                query,
                key,
                value,
                sequence_lengths,
                input_dtype,
                output_dtype,
                device,
            ):
                return wave.decode_attention(
                    query,
                    key,
                    value,
                    sequence_lengths,
                    input_dtype,
                    output_dtype,
                    device,
                )

        NUM_SEQUENCES = 32
        NUM_QUERY_HEADS = 128
        NUM_KV_HEADS = 2
        QUERY_HEAD_DIMENSION = 512
        KV_HEAD_DIMENSION = 512
        KV_SEQUENCE_LEN = 500

        fxb = aot.FxProgramsBuilder(WaveDecodeAttentionModule())

        args = (
            torch.empty(
                (NUM_SEQUENCES, NUM_QUERY_HEADS, QUERY_HEAD_DIMENSION),
                dtype=torch.float16,
            ),
            torch.empty(
                (
                    NUM_SEQUENCES * KV_SEQUENCE_LEN,
                    NUM_KV_HEADS,
                    QUERY_HEAD_DIMENSION,
                ),
                dtype=torch.float16,
            ),
            torch.empty(
                (NUM_SEQUENCES * KV_SEQUENCE_LEN, NUM_KV_HEADS, KV_HEAD_DIMENSION),
                dtype=torch.float16,
            ),
            torch.empty((NUM_SEQUENCES,), dtype=torch.int32),
        )

        @fxb.export_program(name="wave_decode_attention", args=args, strict=False)
        def _(
            model,
            query,
            key,
            value,
            sequence_lengths,
        ):
            return model.forward(
                query,
                key,
                value,
                sequence_lengths,
                query.dtype,
                torch.float32,
                torch.device("cpu"),
            )

        output = aot.export(fxb, import_symbolic_shape_expressions=True)
        output.verify()

        mlir = str(output.mlir_module)
        self.assertIn(
            ("func.func @wave_decode_attention"),
            mlir,
        )
        self.assertIn(
            ("stream.executable private @phase_0"),
            mlir,
        )
        self.assertIn(
            ("stream.executable private @phase_1"),
            mlir,
        )
        self.assertIn(
            (
                "func.func private @wave_attention_decode_phase_0__KV_HEAD_DIM_512_NUM_KV_HEADS_2_NUM_KV_SPLITS_8_NUM_QUERY_HEADS_128_QUERY_HEAD_DIM_512_input_dtype_torch.float16_output_dtype_torch.float32"
            ),
            mlir,
        )
        self.assertIn(
            (
                "func.func private @wave_attention_decode_phase_1__KV_HEAD_DIM_512_NUM_KV_HEADS_2_NUM_KV_SPLITS_8_NUM_QUERY_HEADS_128_QUERY_HEAD_DIM_512_input_dtype_torch.float16_output_dtype_torch.float32"
            ),
            mlir,
        )
        self.assertIn(
            (
                "util.func private @phase_0_wrapper_NUM_SEQUENCES_NUM_QUERY_HEADS_128_QUERY_HEAD_DIM_512_f16_SUM_KV_SEQ_LENS_NUM_KV_HEADS_2_QUERY_HEAD_DIM_512_f16_SUM_KV_SEQ_LENS_NUM_KV_HEADS_2_KV_HEAD_DIM_512_f16_NUM_SEQUENCES_i32_KV_BLOCK_TABLE_LEN_i32_NUM_KV_SPLITS_8_NUM_SEQUENCES_NUM_QUERY_HEADS_128_KV_HEAD_DIM_512_f32_NUM_KV_SPLITS_8_NUM_SEQUENCES_NUM_QUERY_HEADS_128_f32_NUM_KV_SPLITS_8_NUM_SEQUENCES_NUM_QUERY_HEADS_128_KV_HEAD_DIM_512_f32_NUM_KV_SPLITS_8_NUM_SEQUENCES_NUM_QUERY_HEADS_128_f32"
            ),
            mlir,
        )
        self.assertIn(
            (
                "util.func private @phase_1_wrapper_NUM_KV_SPLITS_8_NUM_SEQUENCES_NUM_QUERY_HEADS_128_KV_HEAD_DIM_512_f32_NUM_KV_SPLITS_8_NUM_SEQUENCES_NUM_QUERY_HEADS_128_f32_NUM_SEQUENCES_i32_NUM_SEQUENCES_NUM_QUERY_HEADS_128_KV_HEAD_DIM_512_f32_NUM_SEQUENCES_NUM_QUERY_HEADS_128_KV_HEAD_DIM_512_f32"
            ),
            mlir,
        )
