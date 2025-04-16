# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import unittest
import pytest
from typing import Any, Tuple, OrderedDict
from sharktank.models.llm import *
import sharktank.ops as ops
from sharktank.types import Dataset, UnreducedTensor, SplitPrimitiveTensor
from sharktank.models.llama.testing import make_random_llama_theta
from sharktank.utils.testing import (
    assert_cosine_similarity_close,
    get_iree_compiler_flags,
    is_hip_condition,
)
from sharktank.models.llama.sharding import shard_theta
from sharktank.layers.configs import LlamaHParams, LlamaModelConfig
from sharktank.utils.math import round_up_to_multiple_of
from sharktank.utils import iterables_equal
from sharktank.utils.iree import (
    with_iree_device_context,
    get_iree_devices,
    load_iree_module,
    run_iree_module_function,
    prepare_iree_module_function_args,
    call_torch_module_function,
    iree_to_torch,
)
import sharktank.ops as ops
from sharktank.utils.export import export as sharktank_export

import tempfile
import torch
from copy import deepcopy
from iree.turbine.aot import FxProgramsBuilder, export
import iree
import numpy as np
import os


@pytest.mark.usefixtures("caching", "path_prefix", "get_iree_flags")
class ShardedLlamaTest(unittest.TestCase):
    def setUp(self):
        torch.random.manual_seed(123456)
        self.dtype = torch.float32
        torch.set_default_dtype(self.dtype)
        self.batch_size = 3
        self.attention_head_count_kv = 4
        self.attention_head_count = self.attention_head_count_kv * 5
        self.vocabulary_size = 19
        self.rope_dimension_count = 7 * 2
        self.attn_head_dim = self.rope_dimension_count
        self.block_seq_stride = 13
        self.cache_page_count = 11
        self.config = LlamaModelConfig(
            hp=LlamaHParams(
                context_length=self.block_seq_stride * 2,
                embedding_length=self.attention_head_count * self.attn_head_dim,
                block_count=3,
                feed_forward_length=23,
                rope_dimension_count=self.rope_dimension_count,
                rope_freq_base=500000.0,
                attention_head_count=self.attention_head_count,
                attn_head_dim=self.attn_head_dim,
                attention_layer_norm_rms_epsilon=0.01,
                attention_head_count_kv=self.attention_head_count_kv,
                expert_count=0,
                expert_used_count=0,
                model_arch="llama",
            ),
            block_seq_stride=self.block_seq_stride,
            activation_dtype=self.dtype,
            attention_dtype=self.dtype,
        )
        self.sharded_config = deepcopy(self.config)
        self.sharded_config.tensor_parallelism_size = 2
        self.theta = make_random_llama_theta(
            config=self.config,
            vocab_size=self.vocabulary_size,
        )
        self.prefill_seq_lens = torch.tensor(
            [14, 9, self.block_seq_stride - 1], dtype=torch.int64
        )

    def make_prefill_args(self, model: PagedLlmModelV1) -> OrderedDict[str, Any]:
        batch_seq_len = round_up_to_multiple_of(
            int(torch.max(self.prefill_seq_lens)), model.cache.pad_sequence_stride
        )
        token_ids = torch.randint(
            low=0,
            high=self.vocabulary_size,
            size=[self.batch_size, batch_seq_len],
            dtype=torch.int32,
        )
        attention_mask = model.attention_mask(
            model.input_mask(self.prefill_seq_lens, batch_seq_len)
        )
        seq_block_ids = torch.arange(
            self.batch_size * batch_seq_len // self.config.block_seq_stride
        ).view(self.batch_size, -1)
        cache_state = model.cache.allocate(page_count=self.cache_page_count)
        cache_state = [torch.rand_like(cache_state[0])]
        return OrderedDict(
            [
                ("tokens", token_ids),
                ("attention_mask", attention_mask),
                ("seq_block_ids", seq_block_ids),
                ("cache_state", cache_state),
            ]
        )

    def make_equal_unsharded_and_sharded_prefill_args(
        self, model: PagedLlmModelV1, sharded_model: PagedLlmModelV1
    ) -> Tuple[OrderedDict[str, Any], OrderedDict[str, Any]]:
        prefill_kwargs = self.make_prefill_args(model)
        sharded_cache_state = sharded_model.cache.allocate(
            page_count=self.cache_page_count
        )
        assert iterables_equal(
            prefill_kwargs["cache_state"][0].shape, sharded_cache_state[0].shape
        )
        sharded_prefill_kwargs = deepcopy(prefill_kwargs)
        sharded_cache_state = sharded_model.cache.shard_state(
            sharded_prefill_kwargs["cache_state"]
        )
        sharded_prefill_kwargs["cache_state"] = sharded_cache_state

        sharding = sharded_model.config.tensor_parallelism_size
        for k in sharded_prefill_kwargs:
            if k == "cache_state":
                continue
            sharded_prefill_kwargs[k] = ops.replicate(
                sharded_prefill_kwargs[k], count=sharding
            )

        return prefill_kwargs, sharded_prefill_kwargs

    def make_decode_args(self, model: PagedLlmModelV1) -> OrderedDict[str, Any]:
        start_positions = self.prefill_seq_lens.clone()
        seq_lens = self.prefill_seq_lens + 1
        batch_seq_len = round_up_to_multiple_of(
            int(torch.max(seq_lens)), model.cache.pad_sequence_stride
        )
        decode_token_ids = torch.randint(
            low=0,
            high=self.vocabulary_size,
            size=[self.batch_size, 1],
            dtype=torch.int32,
        )
        attention_mask = model.decode_attention_mask(
            model.input_mask(seq_lens, batch_seq_len)
        )
        seq_block_ids = torch.arange(
            self.batch_size * batch_seq_len // self.config.block_seq_stride
        ).view(self.batch_size, -1)
        cache_state = model.cache.allocate(page_count=self.cache_page_count)
        cache_state = [torch.rand_like(cache_state[0])]
        return OrderedDict(
            [
                ("tokens", decode_token_ids),
                ("attention_mask", attention_mask),
                ("start_positions", start_positions),
                ("seq_block_ids", seq_block_ids),
                ("cache_state", cache_state),
            ]
        )

    def make_equal_unsharded_and_sharded_decode_args(
        self, model: PagedLlmModelV1, sharded_model: PagedLlmModelV1
    ) -> Tuple[OrderedDict[str, Any], OrderedDict[str, Any]]:
        decode_kwargs = self.make_decode_args(model)
        sharded_decode_kwargs = deepcopy(decode_kwargs)
        sharded_decode_kwargs["cache_state"] = sharded_model.cache.shard_state(
            sharded_decode_kwargs["cache_state"]
        )

        sharding = sharded_model.config.tensor_parallelism_size
        for k in sharded_decode_kwargs:
            if k == "cache_state":
                continue
            sharded_decode_kwargs[k] = ops.replicate(
                sharded_decode_kwargs[k], count=sharding
            )

        return decode_kwargs, sharded_decode_kwargs

    def testCompareToySizedModelToUnsharded(self):
        """Run a sharded variant of a toy model size and compare it against the
        unsharded variant."""
        model = PagedLlmModelV1(self.theta, self.config)
        sharded_theta = shard_theta(self.theta, self.sharded_config)
        sharded_model = PagedLlmModelV1(sharded_theta, self.sharded_config)

        # Verify prefill step.
        (
            prefill_kwargs,
            sharded_prefill_kwargs,
        ) = self.make_equal_unsharded_and_sharded_prefill_args(model, sharded_model)

        expected_prefill_result = model.prefill(**prefill_kwargs)
        sharded_prefill_result = sharded_model.prefill(**sharded_prefill_kwargs)
        sharded_prefill_result = ops.unshard(sharded_prefill_result)
        # The errors are quite high, but for float64 both errors drop to < 1e-12.
        # The numerics are probably correct.
        torch.testing.assert_close(
            sharded_prefill_result, expected_prefill_result, atol=1e-3, rtol=1e-2
        )
        expected_cache_state = prefill_kwargs["cache_state"][0]
        actual_cache_state = ops.unshard(
            sharded_model.cache.unflatten_page_table(
                sharded_prefill_kwargs["cache_state"]
            )
        ).flatten(start_dim=1)
        torch.testing.assert_close(
            actual_cache_state, expected_cache_state, atol=1e-4, rtol=1e-1
        )

        # Verify decode step.
        (
            decode_kwargs,
            sharded_decode_kwargs,
        ) = self.make_equal_unsharded_and_sharded_decode_args(model, sharded_model)
        expected_decode_result = model.decode(**decode_kwargs)
        sharded_decode_result = sharded_model.decode(**sharded_decode_kwargs)
        sharded_decode_result = ops.unshard(sharded_decode_result)
        torch.testing.assert_close(
            sharded_decode_result, expected_decode_result, atol=1e-4, rtol=1e-5
        )
        expected_decode_cache_state = decode_kwargs["cache_state"][0]
        actual_decode_cache_state = ops.unshard(
            sharded_model.cache.unflatten_page_table(
                sharded_decode_kwargs["cache_state"]
            )
        ).flatten(start_dim=1)
        # TODO: investigate why the Windows machine CI is producing a larger numerical
        # error.
        # The Ubuntu CI runs fine with default tolerances.
        torch.testing.assert_close(
            actual_decode_cache_state, expected_decode_cache_state, atol=1e-4, rtol=1e-4
        )

    @pytest.mark.xfail(
        is_hip_condition,
        raises=RuntimeError,
        match="Compilation failed",
        strict=True,
        reason="IREE regression https://github.com/iree-org/iree/issues/20365",
    )
    def testExportAndRunToySizedModelWithIree(self):
        """Test exporting to MLIR and compiling with IREE the sharded Llama model.
        Test numerical accuracy of the IREE module against PyTorch."""

        if self.path_prefix is not None:
            self.runTestExportAndRunToySizedModelWithIree(
                path_prefix=self.path_prefix, dump_enabled=True
            )
        else:
            with tempfile.TemporaryDirectory() as temp_dir:
                self.runTestExportAndRunToySizedModelWithIree(
                    path_prefix=f"{temp_dir}/", dump_enabled=False
                )

    def runTestExportAndRunToySizedModelWithIree(
        self, path_prefix: str, dump_enabled: bool
    ):
        sharded_theta = shard_theta(self.theta, self.sharded_config)
        sharded_theta.rename_tensors_to_paths()
        sharded_dataset = Dataset({}, sharded_theta)
        sharded_parameters_path = f"{path_prefix}parameters.irpa"
        sharded_dataset.save(sharded_parameters_path)
        sharded_dataset = Dataset.load(sharded_parameters_path, mmap=False)

        model = PagedLlmModelV1(self.theta, self.config)
        sharded_model = PagedLlmModelV1(sharded_dataset.root_theta, self.sharded_config)
        (
            _,
            sharded_prefill_kwargs,
        ) = self.make_equal_unsharded_and_sharded_prefill_args(model, sharded_model)
        (
            _,
            sharded_decode_kwargs,
        ) = self.make_equal_unsharded_and_sharded_decode_args(model, sharded_model)

        iree_module_path = f"{path_prefix}program.vmfb"
        if not self.caching or not os.path.exists(iree_module_path):
            # Export and compile the IREE module.
            sharded_fxb = FxProgramsBuilder(sharded_model)

            @sharktank_export(
                fx_builder=sharded_fxb,
                name="prefill",
                kwargs=sharded_prefill_kwargs,
                strict=False,
            )
            def _(model, *args, **kwargs) -> torch.Tensor:
                return model.prefill(*args, **kwargs)

            # TODO: remove strict=False when
            # https://github.com/pytorch/pytorch/issues/136757
            # is resolved.
            @sharktank_export(
                fx_builder=sharded_fxb,
                name="decode",
                kwargs=sharded_decode_kwargs,
                strict=False,
            )
            def _(model, *args, **kwargs) -> torch.Tensor:
                return model.decode(*args, **kwargs)

            output = export(sharded_fxb)
            if dump_enabled:
                output.save_mlir(f"{path_prefix}program.mlir")
            output.session.set_flags(
                *get_iree_compiler_flags(
                    self, self.sharded_config.tensor_parallelism_size
                )
            )
            output.compile(
                save_to=iree_module_path,
                target_backends=None,
            )

        expected_prefill_result = call_torch_module_function(
            module=sharded_model,
            function_name="prefill",
            kwargs=sharded_prefill_kwargs,
            trace_path_prefix=f"{path_prefix}expected_" if dump_enabled else None,
        )
        expected_decode_result = call_torch_module_function(
            module=sharded_model,
            function_name="decode",
            kwargs=sharded_decode_kwargs,
            trace_path_prefix=f"{path_prefix}expected_" if dump_enabled else None,
        )

        iree_devices = get_iree_devices(
            device=self.iree_device,
            device_count=self.sharded_config.tensor_parallelism_size,
        )

        def run_iree_module(iree_devices: list[iree.runtime.HalDevice]):
            iree_module, vm_context, vm_instance = load_iree_module(
                module_path=iree_module_path,
                devices=iree_devices,
                parameters_path=sharded_parameters_path,
            )

            # Run prefill step.
            prefill_iree_args = prepare_iree_module_function_args(
                args=deepcopy(sharded_prefill_kwargs).values(), devices=iree_devices
            )
            for i, arg in enumerate(prefill_iree_args):
                np.save(f"{path_prefix}prefill_arg{i}.npy", arg.to_host())
            prefill_iree_result = run_iree_module_function(
                args=prefill_iree_args,
                function_name="prefill",
                module=iree_module,
                vm_context=vm_context,
                device=iree_devices[0],
                trace_path_prefix=path_prefix if dump_enabled else None,
            )
            prefill_iree_result = UnreducedTensor(
                ts=[t.clone() for t in iree_to_torch(*prefill_iree_result)]
            )
            prefill_iree_cache_state_shards = prefill_iree_args[
                -self.config.tensor_parallelism_size - 1 :
            ]
            prefill_iree_cache_state = SplitPrimitiveTensor(
                ts=[t.clone() for t in iree_to_torch(*prefill_iree_cache_state_shards)],
                shard_dim=sharded_prefill_kwargs["cache_state"][0].shard_dim,
            )

            # Run decode step.
            decode_iree_args = prepare_iree_module_function_args(
                args=deepcopy(sharded_decode_kwargs).values(), devices=iree_devices
            )
            decode_iree_result = run_iree_module_function(
                args=decode_iree_args,
                function_name="decode",
                module=iree_module,
                vm_context=vm_context,
                device=iree_devices[0],
                trace_path_prefix=path_prefix if dump_enabled else None,
            )
            decode_iree_result = UnreducedTensor(
                ts=[t.clone() for t in iree_to_torch(*decode_iree_result)]
            )
            decode_iree_cache_state_shards = decode_iree_args[
                -self.config.tensor_parallelism_size - 1 :
            ]
            decode_iree_cache_state = SplitPrimitiveTensor(
                ts=[t.clone() for t in iree_to_torch(*decode_iree_cache_state_shards)],
                shard_dim=sharded_decode_kwargs["cache_state"][0].shard_dim,
            )

            return (
                prefill_iree_result,
                prefill_iree_cache_state,
                decode_iree_result,
                decode_iree_cache_state,
            )

        (
            prefill_iree_result,
            prefill_iree_cache_state,
            decode_iree_result,
            decode_iree_cache_state,
        ) = with_iree_device_context(run_iree_module, iree_devices)

        atol = 1e-5
        assert_cosine_similarity_close(
            ops.unshard(prefill_iree_result),
            ops.unshard(expected_prefill_result),
            dim=-1,
            atol=atol,
        )

        actual_prefill_cache_state = ops.unshard(
            sharded_model.cache.unflatten_page_table([prefill_iree_cache_state])
        )
        expected_prefill_cache_state = ops.unshard(
            sharded_model.cache.unflatten_page_table(
                sharded_prefill_kwargs["cache_state"]
            )
        )
        assert_cosine_similarity_close(
            actual_prefill_cache_state, expected_prefill_cache_state, dim=-1, atol=atol
        )

        assert_cosine_similarity_close(
            ops.unshard(decode_iree_result),
            ops.unshard(expected_decode_result),
            dim=-1,
            atol=atol,
        )

        actual_decode_cache_state = ops.unshard(
            sharded_model.cache.unflatten_page_table([decode_iree_cache_state])
        )
        expected_decode_cache_state = ops.unshard(
            sharded_model.cache.unflatten_page_table(
                sharded_decode_kwargs["cache_state"]
            )
        )
        assert_cosine_similarity_close(
            actual_decode_cache_state, expected_decode_cache_state, dim=-1, atol=atol
        )
