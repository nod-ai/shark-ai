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

from typing import List, Optional, Sequence, Tuple, Union

from dataclasses import dataclass
import inspect
import warnings

__all__ = [
    "HParams",
]


@dataclass
class HParams:
    # Per block sequences. These are normalized from either an int (duplicated
    # to the number of down_blocks) or a list.
    layers_per_block: Tuple[int]
    cross_attention_dim: Tuple[int]
    num_attention_heads: Tuple[int]
    # Per down-block, per attention layer.
    transformer_layers_per_block: Tuple[Union[int, Tuple[int, ...]]]

    act_fn: str = "silu"
    addition_embed_type: str = "text_time"
    addition_embed_type_num_heads: int = 64
    addition_time_embed_dim: Optional[int] = None
    block_out_channels: Sequence[int] = (320, 640, 1280, 1280)
    class_embed_type: Optional[str] = None
    class_embeddings_concat: bool = False
    center_input_sample: bool = False
    conv_in_kernel: int = 3
    conv_out_kernel: int = 3
    down_block_types: Sequence[str] = ()
    downsample_padding: int = 1
    dropout: float = 0.0
    dual_cross_attention: bool = False
    encoder_hid_dim: Optional[str] = None
    encoder_hid_dim_type: Optional[str] = None
    flip_sin_to_cos: bool = True
    freq_shift: int = 0
    in_channels: int = 4
    mid_block_scale_factor: float = 1.0
    mid_block_type: str = "UNetMidBlock2DCrossAttn"
    norm_eps: float = 1e-5
    norm_num_groups: int = 32
    only_cross_attention: bool = False
    projection_class_embeddings_input_dim: Optional[int] = (None,)
    resnet_out_scale_factor: float = 1.0
    resnet_time_scale_shift: str = "default"
    time_embedding_act_fn: Optional[str] = None
    time_embedding_dim: Optional[int] = None
    time_embedding_type: str = "positional"
    timestep_post_act: Optional[str] = None
    up_block_types: Sequence[str] = ()
    upcast_attention: bool = False
    use_linear_projection: bool = False

    def __post_init__(self):
        # Normalize some.
        if self.upcast_attention is None:
            self.upcast_attention = False

        # Normalize per-block.
        block_arity = len(self.down_block_types)
        self.layers_per_block = _normalize_int_arity(self.layers_per_block, block_arity)
        self.cross_attention_dim = _normalize_int_arity(
            self.cross_attention_dim, block_arity
        )
        self.num_attention_heads = _normalize_int_arity(
            self.num_attention_heads, block_arity
        )
        self.transformer_layers_per_block = _normalize_int_arity(
            self.transformer_layers_per_block, block_arity
        )

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
        # Pre-process the dict to account for some name drift.
        if "attention_head_dim" in d:
            assert (
                d.get("num_attention_heads") is None
            ), "HParam rename cannot have both of attention_head_dim and num_attention_heads"
            d["num_attention_heads"] = d["attention_head_dim"]
            del d["attention_head_dim"]

        # Per block defaults.
        if "layers_per_block" not in d:
            d["layers_per_block"] = 2
        if "cross_attention_dim" not in d:
            d["cross_attention_dim"] = 1280
        if "num_attention_heads" not in d:
            d["num_attention_heads"] = 8
        if "transformer_layers_per_block" not in d:
            d["transformer_layers_per_block"] = 1

        allowed = inspect.signature(cls).parameters
        declared_kwargs = {k: v for k, v in d.items() if k in allowed}
        extra_kwargs = [k for k in d.keys() if k not in allowed]
        if extra_kwargs:
            # TODO: Consider making this an error once bringup is done and we
            # handle everything.
            warnings.warn(f"Unhandled punet.HParams: {extra_kwargs}")
        return cls(**declared_kwargs)


def _normalize_int_arity(v, arity) -> tuple:
    if isinstance(v, int):
        return tuple([v] * arity)
    else:
        assert len(v) == arity
        return tuple(v)
