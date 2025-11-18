# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# For testing dataset transforms, we like to use an example CLI tool, invoked
# programmatically to test the entire interaction. See examples under the
# examples/ directory.

import unittest

import pytest
import torch

from amdsharktank.types import *
from amdsharktank.utils.testing import MainRunnerTestBase


class DatasetShardingTransformTest(MainRunnerTestBase):
    @pytest.mark.skip("Skipping sharding related tests for now")
    def testShardLlmDataset(self):
        orig_pts = [
            DefaultPrimitiveTensor(
                name="blk.1.attn_k.weight", data=torch.randn([32, 128])
            ),
            DefaultPrimitiveTensor(
                name="blk.2.attn_q.weight", data=torch.randn([48, 64])
            ),
        ]
        ds_orig = Dataset(
            {
                "general.architecture": "llama",
                "llama.attention.head_count": 1,
                "llama.context_length": 2,
                "llama.embedding_length": 3,
                "llama.block_count": 4,
                "llama.feed_forward_length": 5,
                "llama.attention.layer_norm_rms_epsilon": 0.1,
            },
            Theta(orig_pts),
        )
        input_path = self.save_dataset(ds_orig, "input")
        output_path = self.get_irpa_path("output")
        from amdsharktank.examples.sharding import shard_llm_dataset

        self.run_main(
            shard_llm_dataset.main,
            "--irpa-file",
            input_path,
            "--output-irpa-file",
            output_path,
            "--tensor-parallelism-size",
            8,
        )
        ds_tran = Dataset.load(output_path, mmap=False)

        ds_tran.properties["tensor_parallelism_size"] = 8

        # Verify.
        flat_sts = ds_tran.root_theta.flatten()
        self.assertEqual(2, len(flat_sts))
        st_1 = flat_sts["blk.1.attn_k.weight"]
        st_2 = flat_sts["blk.2.attn_q.weight"]
        self.assertIsInstance(st_1, SplitPrimitiveTensor)
        self.assertIsInstance(st_2, SplitPrimitiveTensor)
        self.assertListEqual(st_1.shape, [32, 128])
        self.assertListEqual(st_2.shape, [48, 64])

        # Verify component shapes for st_1.
        self.assertEqual(8, len(st_1.shards))
        self.assertTrue(all(pt.shape == [4, 128] for pt in st_1.shards))
        self.assertTrue(
            all(list(pt.as_torch().shape) == [4, 128] for pt in st_1.shards)
        )

        # Verify component shapes for st_2.
        self.assertEqual(8, len(st_2.shards))
        self.assertTrue(all(pt.shape == [6, 64] for pt in st_2.shards))
        self.assertTrue(all(list(pt.as_torch().shape) == [6, 64] for pt in st_2.shards))

        # Verify contents for one shard for sanity.
        new_t = st_1.shards[0].as_torch()
        torch.testing.assert_close(new_t, orig_pts[0].as_torch().split(4, dim=0)[0])


if __name__ == "__main__":
    unittest.main()
