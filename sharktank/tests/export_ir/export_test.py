# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest

from amdsharktank.types import (
    ReplicatedTensor,
    SplitPrimitiveTensor,
    DefaultPrimitiveTensor,
    unbox_tensor,
)
from amdsharktank.utils.export import (
    export,
    flatten_signature,
    get_argument_flat_device_affinities,
)
from amdsharktank import ops
from amdsharktank.utils.testing import (
    assert_equal,
    assert_iterables_equal,
    assert_dicts_equal,
)
from iree.turbine.aot import DeviceAffinity, FxProgramsBuilder
from iree.turbine import aot
from unittest import TestCase
import torch


class ExportTest(TestCase):
    def testFlattenSignature(self):
        expected_a = [ReplicatedTensor(ts=[torch.tensor([1])])]
        expected_b = {"element": DefaultPrimitiveTensor(data=torch.tensor([2]))}
        expected_c = torch.tensor([3])

        @flatten_signature(expected_a, expected_b, expected_c)
        def f(
            a: list[ReplicatedTensor],
            b: dict[str, DefaultPrimitiveTensor],
            c: torch.Tensor,
        ):
            assert_iterables_equal(a, expected_a, elements_equal=ops.equal)
            assert_dicts_equal(b, expected_b, values_equal=ops.equal)
            assert_equal(c, expected_c, equal=ops.equal)

        f(
            unbox_tensor(expected_a[0].shards[0]),
            expected_b["element"].as_torch(),
            expected_c,
        )

    def testGetFlatArgumentDeviceAffinities(self):
        args = [
            {
                "a": [
                    SplitPrimitiveTensor(
                        ts=[torch.tensor([1]), torch.tensor([2])], shard_dim=0
                    )
                ]
            },
            torch.tensor([3]),
            ReplicatedTensor(ts=[torch.tensor([4]), torch.tensor([5])]),
        ]
        affinities = get_argument_flat_device_affinities(*args)
        expected_affinities = {
            0: DeviceAffinity("0"),
            1: DeviceAffinity("1"),
            3: DeviceAffinity("0"),
            4: DeviceAffinity("1"),
        }
        assert_dicts_equal(affinities, expected_affinities)

    def testExportWithArgumentDeviceAffinities(self):
        args = (ReplicatedTensor(ts=[torch.tensor([1])]), torch.tensor([[2]]))

        class Module(torch.nn.Module):
            def f(self, a, b):
                return a, b

        module = Module()
        fxb = FxProgramsBuilder(module)
        export(
            Module.f,
            fx_builder=fxb,
            args=args,
            strict=False,
        )
        export_output = aot.export(
            fxb,
        )
        asm = str(export_output.mlir_module)
        print(asm)
        self.assertRegex(
            asm,
            expected_regex=(
                "func.func @f\\("
                "%.+: !torch.vtensor<\\[1\\],si64> "
                "{iree.abi.affinity = #hal.device.promise<@__device_0>}, "
                "%.+: !torch.vtensor<\\[1,1\\],si64>\\)"
            ),
        )
