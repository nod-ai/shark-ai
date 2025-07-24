# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#
# Significant portions of this implementation were derived from diffusers,
# licensed under Apache2: https://github.com/huggingface/diffusers
# While much was a simple reverse engineering of the config.json and parameters,
# code was taken where appropriate.

from typing import Any, Sequence

from dataclasses import dataclass, asdict
import inspect
import warnings

__all__ = [
    "HParams",
]


@dataclass
class HParams:
    act_fn: str = "silu"
    block_out_channels: Sequence[int] = (128, 256, 512, 512)
    in_channels: int = 3
    out_channels: int = 3
    up_block_types: Sequence[str] = (
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
        "UpDecoderBlock2D",
    )
    layers_per_block: int = 2
    latent_channels: int = 4
    norm_num_groups: int = 32
    scaling_factor: float = 0.13025
    use_post_quant_conv: bool = True
    shift_factor: float = 0.0
    sample_size: int | tuple[int, int] | list[int, int] = 1024

    def assert_default_values(self, attr_names: Sequence[str]):
        for name in attr_names:
            actual = getattr(self, name)
            required = getattr(HParams, name)
            if actual != required:
                raise ValueError(
                    f"NYI: HParams.{name} != {required!r} (got {actual!r})"
                )

    @classmethod
    def from_dict(cls, d: dict):
        if "layers_per_block" not in d:
            d["layers_per_block"] = 2

        allowed = inspect.signature(cls).parameters
        declared_kwargs = {k: v for k, v in d.items() if k in allowed}
        extra_kwargs = [k for k in d.keys() if k not in allowed]
        if extra_kwargs:
            warnings.warn(f"Unhandled vae.HParams: {extra_kwargs}")
        return cls(**declared_kwargs)

    def asdict(self) -> dict[str, Any]:
        return asdict(self)

    def to_hugging_face_config(self) -> dict[str, Any]:
        return self.asdict()
