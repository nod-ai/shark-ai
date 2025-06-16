# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest

import torch

from sharktank.types import *
from sharktank.types.tensors import REGISTERED_LAYOUT_CLASSES
from sharktank.types.layout_utils import (
    pack_fp4_e2m1_to_uint8,
    unpack_uint8_to_fp4_e2m1,
)
from sharktank.types.ocp_floats import (
    float32_to_fp4_e2m1,
    fp4_e2m1_to_float32,
)


class BlockScaledLayoutTest(unittest.TestCase):
    def testRegistered(self):
        self.assertIs(
            BlockScaledLayout,
            REGISTERED_LAYOUT_CLASSES[BlockScaledLayout.serialized_name()],
        )

    def testRoundtrip(self):
        n = 128
        k = 1024
        bs = 32

        l = BlockScaledLayout(
            [n, k],
            d=torch.empty(n, k // bs, 1, dtype=torch.float32),
            qs=torch.empty(n, k // bs, bs, dtype=torch.int8),
            m=torch.empty(n, k // bs, bs, dtype=torch.float32),
        )

        l_new = BlockScaledLayout.create(l.shape, l.metadata, l.planes)
        self.assertEqual(l.shape, l_new.shape)
        self.assertEqual(l.planes, l_new.planes)
        self.assertEqual(l.metadata, l_new.metadata)


class BlockScaledI4LayoutTest(unittest.TestCase):
    def testRegistered(self):
        self.assertIs(
            BlockScaledI4Layout,
            REGISTERED_LAYOUT_CLASSES[BlockScaledI4Layout.serialized_name()],
        )

    def testRoundtrip(self):
        n = 128
        k = 1024
        bs = 32

        l = BlockScaledI4Layout(
            [n, k],
            d=torch.empty(n, k // bs, 1, dtype=torch.float32),
            qs=torch.empty(n, k // bs, bs // 2, dtype=torch.int8),
            m=torch.empty(n, k // bs, bs, dtype=torch.float32),
            signed=True,
        )

        l_new = BlockScaledI4Layout.create(l.shape, l.metadata, l.planes)
        self.assertEqual(l.shape, l_new.shape)
        self.assertEqual(l.planes, l_new.planes)
        self.assertEqual(l.metadata, l_new.metadata)
        self.assertEqual(l.signed, l_new.signed)


class SuperBlockOffsetScaled_4_6_LayoutTest(unittest.TestCase):
    def testRegistered(self):
        self.assertIs(
            SuperBlockOffsetScaled_4_6_Layout,
            REGISTERED_LAYOUT_CLASSES[
                SuperBlockOffsetScaled_4_6_Layout.serialized_name()
            ],
        )

    def testRoundtrip(self):
        n = 128
        sup = 10
        sub = 8
        k = 2560
        bs = 32

        l = SuperBlockOffsetScaled_4_6_Layout(
            shape=[n, k],
            d=torch.empty(n, sup, 1, dtype=torch.float32),
            dmin=torch.empty(n, sup, 1, dtype=torch.float32),
            sb_scales_high=torch.empty(n, sup, sub // 4, dtype=torch.uint8),
            sb_scales_low=torch.empty(n, sup, sub // 2, dtype=torch.uint8),
            sb_mins_high=torch.empty(n, sup, sub // 4, dtype=torch.uint8),
            sb_mins_low=torch.empty(n, sup, sub // 2, dtype=torch.uint8),
            qs=torch.empty(n, sup, sub, bs // 2, dtype=torch.uint8),
        )

        l_new = SuperBlockOffsetScaled_4_6_Layout.create(l.shape, l.metadata, l.planes)
        self.assertEqual(l.shape, l_new.shape)
        self.assertEqual(l.planes, l_new.planes)
        self.assertEqual(l.metadata, l_new.metadata)


class TensorScaledLayoutTest(unittest.TestCase):
    def testRegistered(self):
        self.assertIs(
            TensorScaledLayout,
            REGISTERED_LAYOUT_CLASSES[TensorScaledLayout.serialized_name()],
        )

    def testRoundtripWithOffset(self):
        n = 128
        k = 2560
        l = TensorScaledLayout(
            shape=[n, k],
            d=torch.tensor(2.0, dtype=torch.float32),
            qs=torch.tensor([2.0, 3.0, 4.0], dtype=torch.float16),
            m=torch.tensor(5.0, dtype=torch.float32),
        )
        d = l.dequant()
        torch.testing.assert_close(
            d, torch.tensor([-6.0, -4.0, -2.0], dtype=torch.float32)
        )

        l_new = TensorScaledLayout.create(l.shape, l.metadata, l.planes)
        d = l_new.dequant()
        torch.testing.assert_close(
            d, torch.tensor([-6.0, -4.0, -2.0], dtype=torch.float32)
        )

    def testTensorScaledFp4Layout(self):
        original_data = torch.tensor([1.0, 2.0, 3.0, 4.0, 0.5, 1.5, -1.0, -2.0])
        fp4_indices = float32_to_fp4_e2m1(original_data)
        packed_fp4 = pack_fp4_e2m1_to_uint8(fp4_indices)

        l = TensorScaledLayout(
            shape=list(original_data.shape),
            d=torch.tensor(1.0),
            qs=packed_fp4,
            dtype=torch.float32,
        )

        self.assertEqual(l.qs.shape, (4,))
        self.assertEqual(l.shape, [8])

        unpacked_indices = unpack_uint8_to_fp4_e2m1(l.qs)
        fp4_values = fp4_e2m1_to_float32(unpacked_indices)
        dequantized = fp4_values * l.d

        torch.testing.assert_close(dequantized, original_data, atol=0.0, rtol=0.0)


if __name__ == "__main__":
    unittest.main()
