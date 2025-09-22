# Copyright 2025 Advanced Micro Devices, Inc
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception


import unittest

import pytest
import torch


from copy import deepcopy
from pathlib import Path
from iree.turbine.support.torch import has_torch_device
from iree.turbine.ops.iree import IreeDeviceAffinityToTorchDevice
from iree.turbine.aot import DeviceAffinity
from sharktank.layers.configs import ParallelismConfig
from sharktank.models.llm.llm import PagedLlmModelV1
from sharktank.models.llm.testing import make_random_prefill_args
from sharktank.models.llama.toy_llama import generate
from sharktank.types.pipelining import pipeline_parallelize_llm_theta
from sharktank.utils.export_artifacts import IreeCompileException
from sharktank.utils.testing import (
    is_mi300x,
    IreeVsEagerLLMTester,
    TempDirTestBase,
)


class CrossEntropyTest(unittest.TestCase):
    def testUnsharded(self):
        torch.set_default_dtype(torch.float32)
        theta, config = generate(12345)
        model = PagedLlmModelV1(theta=theta, config=config)

        ids = [0, 208, 214, 29, 19, 86, 176, 120, 120, 80, 120, 208, 37, 157, 191, 137]
        seq_len = len(ids)

        blocks = (seq_len - 1) // config.block_seq_stride
        blocks = blocks + 1
        padded_length = blocks * config.block_seq_stride
        padding = padded_length - seq_len
        ids = ids + [0] * padding

        ids = torch.asarray([ids], dtype=torch.int64)
        block_ids = torch.asarray([[i for i in range(blocks)]]).to(torch.int64)

        cache_state = model.cache.allocate(
            page_count=config.hp.context_length // config.block_seq_stride
        )

        logits = model.prefill(
            tokens=ids,
            seq_lens=torch.tensor([seq_len]),
            cache_state=cache_state,
            seq_block_ids=block_ids,
        )

        # Remove padding
        ids = ids[:, :seq_len]
        logits = logits[:, :seq_len, :]

        ids = ids[0, 1:]
        logits = logits[0, :-1].to(torch.float32)
        cross_entropy = torch.nn.functional.cross_entropy(logits, ids)
        assert pytest.approx(0.583, 1e-2) == cross_entropy


@pytest.mark.usefixtures("iree_flags", "device")
@is_mi300x
class LlamaIreeVsEagerTest(TempDirTestBase):
    @pytest.mark.xfail(
        raises=IreeCompileException,
        reason="https://github.com/iree-org/iree/issues/21462, https://github.com/nod-ai/shark-ai/issues/1758",
    )
    def testUnshardedToyIreeVsEager(self):
        theta, config = generate(12345)

        tester = IreeVsEagerLLMTester(
            work_dir=self._temp_dir,
            theta=theta,
            config=config,
            torch_device=self.device,
            iree_device=self.iree_device,
            iree_hip_target=self.iree_hip_target,
            iree_hal_target_device=self.iree_hal_target_device,
        )
        tester.run_and_compare_iree_vs_eager()


@pytest.mark.parametrize(
    "torch_devices",
    [
        [torch.device("cpu"), torch.device("cpu")],
        pytest.param(
            [torch.device("cpu"), torch.device("cuda")],
            marks=pytest.mark.skipif(
                not has_torch_device("cuda"),
                reason="Test is disabled if no CUDA device is available",
            ),
        ),
        pytest.param(
            [torch.device("cuda"), torch.device("cpu")],
            marks=pytest.mark.skipif(
                not has_torch_device("cuda"),
                reason="Test is disabled if no CUDA device is available",
            ),
        ),
    ],
)
def test_eager_pipeline_parallel_toy_llama(
    deterministic_random_seed, torch_devices: list[torch.device]
):
    pp_size = len(torch_devices)
    theta, config = generate(12345)
    # Make the leading device the model device. It used later to allocate some
    # arguments on that device.
    # This is a bit of a abuse as the model is not really on 1 device.
    config.device = torch_devices[0]

    pp_config = deepcopy(config)
    pp_config.parallelism_config = ParallelismConfig.default_config(
        block_count=config.hp.block_count,
        pp=pp_size,
    )

    iree_device_affinity_to_torch_device_map = {
        DeviceAffinity(i): torch_devices[i] for i in range(len(torch_devices))
    }
    with IreeDeviceAffinityToTorchDevice(iree_device_affinity_to_torch_device_map):
        pp_theta = deepcopy(theta)
        pipeline_parallelize_llm_theta(pp_theta, pp_config.parallelism_config)
        pp_model = PagedLlmModelV1(pp_theta, pp_config)
        prefill_kwargs = make_random_prefill_args(pp_model, batch_size=1)
        res = pp_model.prefill(**prefill_kwargs)


@pytest.mark.expensive
def test_import_llama3_8B_instruct(tmp_path: Path):
    from sharktank.tools.import_hf_dataset_from_hub import main

    irpa_path = tmp_path / "model.irpa"
    main(
        [
            "--revision=0e9e39f249a16976918f6564b8830bc894c89659",
            f"--output-irpa-file={irpa_path}",
            "meta-llama/Llama-3.1-8B-Instruct",
        ]
    )
    assert irpa_path.exists()
