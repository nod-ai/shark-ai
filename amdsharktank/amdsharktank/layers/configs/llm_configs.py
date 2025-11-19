# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Structured configuration objects for various LLMs.

This draws heavily from the work that ggml has done to systematize the state
of the world for GGUF files:
  https://github.com/ggerganov/ggml/blob/master/docs/gguf.md

When in question, we draw from the vocabulary and normalization they have done
(and indeed, can bootstrap these off of GGUF files).
"""

from typing import TYPE_CHECKING, ClassVar, Any, Optional, List
from collections import defaultdict
from os import PathLike
from dataclasses import asdict, dataclass, field, fields
import torch
from transformers import T5Config as T5ConfigHf
from iree.turbine.aot import DeviceAffinity
from .config import ModelConfig
from amdsharktank.utils import parse_version
from amdsharktank.types import Dataset
from amdsharktank.types.tensors import (
    serialized_name_to_dtype,
    dtype_to_serialized_name,
)

if TYPE_CHECKING:
    import transformers
    from amdsharktank.types import PropertyValueType

__all__ = [
    "ClipTextConfig",
    "LlamaHParams",
    "LlamaModelConfig",
    "ParallelismConfig",
    "T5Config",
    "is_hugging_face_llama3_config",
]


def is_hugging_face_llama3_config(hf_config: dict[str, Any]) -> bool:
    if "architectures" not in hf_config or "model_type" not in hf_config:
        return False
    return (
        "LlamaForCausalLM" in hf_config["architectures"]
        and hf_config["model_type"] == "llama"
    )


@dataclass
class LlamaHParams:
    """This was originally designed to correspond 1:1 with the 'LLM'
    section of the GGUF docs. It has experienced some semantic drift,
    and now additionally collects additional model parameters needed
    for model export. Currently there are some "subsections" that
    refer to a specific model, but this is not ideal due to the
    iterative nature of model arch design, so don't follow this
    trend when adding more Params.

    See the optional_keys under the get_custom_configs function
    for an example of how to add support for custom Params without
    basing conditionals on a specific name prefix. The ideal is
    to have a robust list of reusable params across models when
    possible.
    """

    # Attention config
    model_arch: str
    context_length: int
    embedding_length: int
    block_count: int
    feed_forward_length: int
    attention_head_count: int
    attn_head_dim: int
    attention_layer_norm_rms_epsilon: float
    attention_head_count_kv: Optional[int] = None
    # The size of the model's vocabulary.
    vocab_size: Optional[int] = None

    # Which blocks share kv cache entries
    share_kv_schedule: Optional[list[int]] = None
    # Which layers have global vs windowed context
    attention_global_layer_schedule: Optional[list[int]] = None

    # Deepseek Multi-Latent Attention config
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None

    # Grok Attention config
    attention_softcap: Optional[float] = None

    # YaRN configurations for context window expansion
    yarn_beta_slow: Optional[float] = None
    yarn_beta_fast: Optional[float] = None
    yarn_factor: Optional[float] = None
    yarn_original_context_len: Optional[int] = None

    # RoPE config
    rope_dimension_count: Optional[int] = None
    rope_freq_base: Optional[float] = None
    rope_interleave_emb: bool = False

    # MoE config
    expert_count: Optional[int] = None
    expert_used_count: Optional[int] = None
    expert_feed_forward_length: Optional[int] = None
    expert_shared_feed_forward_length: Optional[int] = None

    # Specifies the interval at which Mixture of Experts (MoE) layers are inserted among the model's layers.
    # For example:
    #     1 - every layer is an MoE layer.
    #     3 - every third layer (layers 0, 3, 6, ...) is an MoE layer; others are dense layers.
    # If None, no specific interleaving pattern is applied.
    interleave_moe_layer_step: Optional[int] = None

    # Deepseek MoE config
    expert_shared_count: Optional[int] = None
    moe_intermediate_size: Optional[int] = None
    """Size of the MoE experts feed forward network hidden dimension."""
    n_expert_groups: Optional[int] = None
    n_limited_groups: Optional[int] = None
    n_dense_layers: Optional[int] = None
    route_scale: Optional[float] = None

    # Llama 4 configs
    # Ensure all layers are rope if `nope_layer_interval` is None.
    no_rope_layer_step: Optional[list[int]] = None

    # In HuggingFace transformers, this field is represented as an int, but it is only ever used as a boolean.
    # For clarity and correctness, it should be a bool: if True, enables attention temperature tuning.
    attn_temperature_tuning: Optional[bool] = None

    # Scaling factor applied to attention scores.
    attention_scale: Optional[float] = None

    # Scaling factor applied as a floor value in attention computations.
    floor_scale: Optional[int] = None

    # gpt-oss configs
    sliding_window: int = 0  # 0 = no sliding window, >0 = window size
    swiglu_limit: Optional[float] = None  # Limit for swiglu activation function
    use_base_frequency_scaling: bool = False  # Whether to use base^(i/d) frequency scaling for RoPE, applies base frequency YaRN variant
    topk_then_softmax: bool = (
        False  # Whether to use topk then softmax MoE routing without expert groups
    )
    use_residual_moe: bool = False  # Whether to use residual connection after MoE
    # FFN processing configuration
    moe_block_type: str = "DenseFFNMOE"  # DenseFFNMOE, PreGatherFFNMOE
    use_fused_qkv: bool = False  # Whether to use fused QKV for attention
    use_moe_swiglu: bool = False  # Whether to use fused SwiGLU for MoE

    @staticmethod
    def from_gguf_props(p: dict[str, Any]):
        name_prefix = p.get("general.architecture", "llama")
        default_expert_count = 0
        default_expert_used_count = 0
        default_interleave_moe_layer_step = None
        default_rope_freq_base = 500000.0
        default_rope_dimension_count = 128
        attention_head_count = _int_prop(p, f"{name_prefix}.attention.head_count")
        default_rope_interleave_emb = "schema" in p and p["schema"] == "GGUF"
        rope_dimension_count = _optional_int_prop(
            p, f"{name_prefix}.rope.dimension_count", default_rope_dimension_count
        )
        expert_count = _optional_int_prop(
            p, f"{name_prefix}.expert_count", default_expert_count
        )

        custom_config = get_custom_configs(p, name_prefix)

        if custom_config["attn_head_dim"] is None:
            custom_config["attn_head_dim"] = rope_dimension_count

        if custom_config["rope_interleave_emb"] is None:
            custom_config["rope_interleave_emb"] = _optional_bool_prop(
                p, f"{name_prefix}.rope.interleave_emb", default_rope_interleave_emb
            )

        return LlamaHParams(
            model_arch=name_prefix,
            vocab_size=_optional_int_prop(p, f"{name_prefix}.vocab_size", None),
            context_length=_int_prop(p, f"{name_prefix}.context_length"),
            embedding_length=_int_prop(p, f"{name_prefix}.embedding_length"),
            block_count=_int_prop(p, f"{name_prefix}.block_count"),
            feed_forward_length=_int_prop(p, f"{name_prefix}.feed_forward_length"),
            attention_head_count=attention_head_count,
            attention_layer_norm_rms_epsilon=_float_prop(
                p, f"{name_prefix}.attention.layer_norm_rms_epsilon"
            ),
            attention_head_count_kv=_optional_int_prop(
                p, f"{name_prefix}.attention.head_count_kv", attention_head_count
            ),
            expert_count=expert_count,
            expert_used_count=_optional_int_prop(
                p, f"{name_prefix}.expert_used_count", default_expert_used_count
            ),
            moe_intermediate_size=_optional_int_prop(
                p, f"{name_prefix}.moe_intermediate_size", None
            ),
            rope_dimension_count=rope_dimension_count,
            rope_freq_base=_optional_float_prop(
                p, f"{name_prefix}.rope.freq_base", default_rope_freq_base
            ),
            no_rope_layer_step=_optional_int_prop(
                p, f"{name_prefix}.no_rope_layer_step", None
            ),
            attn_temperature_tuning=_optional_bool_prop(
                p, f"{name_prefix}.attn_temperature_tuning", None
            ),
            attention_scale=_optional_float_prop(p, f"{name_prefix}.attn_scale", None),
            floor_scale=_optional_int_prop(p, f"{name_prefix}.floor_scale", None),
            **custom_config,
        )

    def to_gguf_props(self) -> dict[str, Any]:
        res = {
            "general.architecture": self.model_arch,
            f"{self.model_arch}.context_length": self.context_length,
            f"{self.model_arch}.embedding_length": self.embedding_length,
            f"{self.model_arch}.block_count": self.block_count,
            f"{self.model_arch}.feed_forward_length": self.feed_forward_length,
            f"{self.model_arch}.attention.head_count": self.attention_head_count,
            f"{self.model_arch}.attention.layer_norm_rms_epsilon": self.attention_layer_norm_rms_epsilon,
            f"{self.model_arch}.attention.head_count_kv": self.attention_head_count_kv,
        }
        if self.vocab_size is not None:
            res[f"{self.model_arch}.vocab_size"] = self.vocab_size
        if self.attn_head_dim is not None:
            res[f"{self.model_arch}.attention.head_dim"] = self.attn_head_dim
        if self.qk_rope_head_dim is not None:
            res[f"{self.model_arch}.attention.qk_rope_head_dim"] = self.qk_rope_head_dim
        if self.qk_nope_head_dim is not None:
            res[f"{self.model_arch}.attention.qk_nope_head_dim"] = self.qk_nope_head_dim
        if self.v_head_dim is not None:
            res[f"{self.model_arch}.attention.v_head_dim"] = self.v_head_dim
        if self.q_lora_rank is not None:
            res[f"{self.model_arch}.attention.q_lora_rank"] = self.q_lora_rank
        if self.kv_lora_rank is not None:
            res[f"{self.model_arch}.attention.kv_lora_rank"] = self.kv_lora_rank
        if self.route_scale is not None:
            res[f"{self.model_arch}.expert_weights_scale"] = self.route_scale
        if self.n_dense_layers is not None:
            res[f"{self.model_arch}.leading_dense_block_count"] = self.n_dense_layers
        if self.expert_count is not None:
            res[f"{self.model_arch}.expert_count"] = self.expert_count
        if self.expert_used_count is not None:
            res[f"{self.model_arch}.expert_used_count"] = self.expert_used_count
        if self.expert_shared_count is not None:
            res[f"{self.model_arch}.expert_shared_count"] = self.expert_shared_count
        if self.n_expert_groups is not None:
            res[f"{self.model_arch}.n_expert_groups"] = self.n_expert_groups
        if self.n_limited_groups is not None:
            res[f"{self.model_arch}.n_limited_groups"] = self.n_limited_groups
        if self.moe_intermediate_size is not None:
            res[f"{self.model_arch}.moe_intermediate_size"] = self.moe_intermediate_size
        if self.rope_dimension_count is not None:
            res[f"{self.model_arch}.rope.dimension_count"] = self.rope_dimension_count
        if self.rope_freq_base is not None:
            res[f"{self.model_arch}.rope.freq_base"] = self.rope_freq_base
        if self.rope_interleave_emb is not None:
            res[f"{self.model_arch}.rope.interleave_emb"] = self.rope_interleave_emb
        if self.expert_feed_forward_length is not None:
            res[
                f"{self.model_arch}.expert_feed_forward_length"
            ] = self.expert_feed_forward_length
        if self.expert_shared_feed_forward_length is not None:
            res[
                f"{self.model_arch}.expert_shared_feed_forward_length"
            ] = self.expert_shared_feed_forward_length
        if self.interleave_moe_layer_step is not None:
            res[
                f"{self.model_arch}.interleave_moe_layer_step"
            ] = self.interleave_moe_layer_step
        if self.vocab_size is not None:
            res[f"{self.model_arch}.vocab_size"] = self.vocab_size

        if self.no_rope_layer_step is not None:
            res[f"{self.model_arch}.no_rope_layer_step"] = self.no_rope_layer_step
        if self.attn_temperature_tuning is not None:
            res[
                f"{self.model_arch}.attn_temperature_tuning"
            ] = self.attn_temperature_tuning
        if self.floor_scale is not None:
            res[f"{self.model_arch}.floor_scale"] = self.floor_scale
        if self.attention_scale is not None:
            res[f"{self.model_arch}.attn_scale"] = self.attention_scale

        if self.sliding_window > 0:
            res[f"{self.model_arch}.sliding_window"] = self.sliding_window
        if self.swiglu_limit is not None:
            res[f"{self.model_arch}.swiglu_limit"] = self.swiglu_limit
        if self.yarn_original_context_len is not None:
            res[
                f"{self.model_arch}.yarn_original_context_len"
            ] = self.yarn_original_context_len
        if self.yarn_factor is not None:
            res[f"{self.model_arch}.yarn_factor"] = self.yarn_factor
        if self.yarn_beta_slow is not None:
            res[f"{self.model_arch}.yarn_beta_slow"] = self.yarn_beta_slow
        if self.yarn_beta_fast is not None:
            res[f"{self.model_arch}.yarn_beta_fast"] = self.yarn_beta_fast

        return res


def get_custom_configs(p: dict[str, Any], name_prefix: str):
    res = defaultdict(lambda: None)

    optional_keys = ["attention.global_layer_schedule", "share_kv_schedule"]
    for key in optional_keys:
        if f"{name_prefix}.{key}" in p.keys():
            res[key.replace(".", "_")] = p[f"{name_prefix}.{key}"]

    if name_prefix == "grok":
        res["attention_softcap"] = 30.0

    if name_prefix == "llama3":
        res["yarn_beta_slow"] = 1
        res["yarn_beta_fast"] = 4
        res["yarn_factor"] = 8
        res["yarn_original_context_len"] = 8192

    if name_prefix == "deepseek2":
        res["qk_rope_head_dim"] = _optional_int_prop(
            p, f"{name_prefix}.attention.qk_rope_head_dim", 64
        )
        res["qk_nope_head_dim"] = _optional_int_prop(
            p, f"{name_prefix}.attention.qk_nope_head_dim", 128
        )
        res["v_head_dim"] = _optional_int_prop(
            p, f"{name_prefix}.attention.v_head_dim", 128
        )
        res["q_lora_rank"] = _int_prop(p, f"{name_prefix}.attention.q_lora_rank")
        res["kv_lora_rank"] = _int_prop(p, f"{name_prefix}.attention.kv_lora_rank")
        res["route_scale"] = _float_prop(p, f"{name_prefix}.expert_weights_scale")
        res["n_expert_groups"] = _optional_int_prop(
            p, f"{name_prefix}.n_expert_groups", 8
        )
        res["n_limited_groups"] = _optional_int_prop(
            p, f"{name_prefix}.n_limited_groups", 4
        )
        res["expert_shared_count"] = _int_prop(p, f"{name_prefix}.expert_shared_count")
        res["attn_head_dim"] = res["qk_nope_head_dim"] + res["qk_rope_head_dim"]
        res["n_dense_layers"] = _int_prop(p, f"{name_prefix}.leading_dense_block_count")

    if name_prefix == "llama4":
        res["interleave_moe_layer_step"] = _int_prop(
            p, f"{name_prefix}.interleave_moe_layer_step"
        )
        res["expert_shared_count"] = _int_prop(p, f"{name_prefix}.expert_shared_count")
        res["expert_feed_forward_length"] = _int_prop(
            p, f"{name_prefix}.expert_feed_forward_length"
        )
        res["expert_shared_feed_forward_length"] = _int_prop(
            p, f"{name_prefix}.expert_shared_feed_forward_length"
        )

    if name_prefix == "gpt-oss":
        res["attn_head_dim"] = _optional_int_prop(
            p, f"{name_prefix}.attention.head_dim", None
        )
        res["yarn_original_context_len"] = _optional_int_prop(
            p, f"{name_prefix}.yarn_original_context_len", None
        )
        res["yarn_factor"] = _optional_float_prop(p, f"{name_prefix}.yarn_factor", None)
        res["yarn_beta_slow"] = _optional_float_prop(
            p, f"{name_prefix}.yarn_beta_slow", None
        )
        res["yarn_beta_fast"] = _optional_float_prop(
            p, f"{name_prefix}.yarn_beta_fast", None
        )
        res["sliding_window"] = _optional_int_prop(
            p, f"{name_prefix}.sliding_window", 128
        )  # Default for gpt-oss
        res["swiglu_limit"] = _float_prop(p, f"{name_prefix}.swiglu_limit")
        res["use_moe_swiglu"] = True
        res["moe_block_type"] = "PreGatherFFNMOE"
        res["use_residual_moe"] = True
        res["use_base_frequency_scaling"] = True
        res["use_fused_qkv"] = True
        res["topk_then_softmax"] = True
        res["rope_interleave_emb"] = False

    return res


def _float_prop(p: dict[str, Any], name: str) -> float:
    try:
        return float(p[name])
    except ValueError as e:
        raise ValueError(f"Property '{name}' expected to be a float and was not") from e
    except KeyError:
        raise KeyError(f"Property '{name}' not found (among keys {p.keys()})")


def _int_prop(p: dict[str, Any], name: str) -> int:
    try:
        return int(p[name])
    except ValueError as e:
        raise ValueError(f"Property '{name}' expected to be an int and was not") from e
    except KeyError:
        raise KeyError(f"Property '{name}' not found (among keys {p.keys()})")


def _str_prop(p: dict[str, Any], name: str) -> str:
    try:
        return str(p[name])
    except ValueError as e:
        raise ValueError(f"Property '{name}' expected to be an str and was not") from e
    except KeyError:
        raise KeyError(f"Property '{name}' not found (among keys {p.keys()})")


def _optional_bool_prop(
    p: dict[str, Any],
    name: str,
    default_value: bool | None,
) -> bool | None:
    value = p.get(name, default_value)

    if value is None:
        return None
    try:
        return bool(value)
    except ValueError as e:
        raise ValueError(f"Property '{name}' expected to be a bool and was not") from e


def _optional_float_prop(
    p: dict[str, Any], name: str, default_value: float | None
) -> float | None:
    value = p.get(name, default_value)
    if value is None:
        return None
    try:
        return float(value)
    except ValueError as e:
        raise ValueError(f"Property '{name}' expected to be a float and was not") from e


def _optional_int_prop(
    p: dict[str, Any], name: str, default_value: int | None
) -> int | None:
    value = p.get(name, default_value)
    if value is None:
        return None
    try:
        return int(value)
    except ValueError as e:
        raise ValueError(f"Property '{name}' expected to be an int and was not") from e


@dataclass
class ParallelismConfig:
    # Mapping between a transformer block and the pipeline it's a part of.
    block_to_pipeline_map: tuple[int, ...] | None = None

    # Mapping between a pipeline and the IREE devices it's executed on.
    pipeline_to_device_map: tuple[tuple[int, ...], ...] | None = None

    @staticmethod
    def default_config(
        *, block_count: int, pp: int = 1, tp: int = 1
    ) -> "ParallelismConfig":
        assert tp == 1, "Tensor parallelism not yet supported."

        from amdsharktank.types.pipelining import (
            distribute_blocks_uniformly_over_pipeline_stages,
        )

        (
            block_to_pipeline_map,
            pipeline_to_device_map,
        ) = distribute_blocks_uniformly_over_pipeline_stages(block_count, pp, tp)
        return ParallelismConfig(block_to_pipeline_map, pipeline_to_device_map)

    @property
    def num_blocks_per_pipeline(self) -> List[int]:
        if self.block_to_pipeline_map is None:
            raise ValueError(
                "block_to_pipeline_map must be set to use num_blocks_per_pipeline"
            )
        counts = [0] * self.pipeline_size
        for p in self.block_to_pipeline_map:
            counts[p] += 1
        return counts

    @property
    def pipeline_size(self) -> int:
        return (
            1
            if self.pipeline_to_device_map is None
            else len(self.pipeline_to_device_map)
        )

    @property
    def tensor_parallelism_size(self) -> int:
        """
        How many devices are involved for tensor parallel sharding.
        If greater than 1, the model will expect sharded model parameters and function
        arguments.
        """
        return len(self.devices_for_pipeline(0))

    def device_affinity_for_pipeline(self, pipeline: int) -> DeviceAffinity:
        devices = self.devices_for_pipeline(pipeline)
        assert len(devices) == 1, "Tensor parallelism not supported"
        return DeviceAffinity(devices[0])

    def first_block_for_pipeline(self, pipeline: int) -> int:
        if self.block_to_pipeline_map is None:
            if pipeline == 0:
                return 0
            raise ValueError("No pipeline mapping, only pipeline 0 is valid.")
        return self.block_to_pipeline_map.index(pipeline)

    def first_block_in_pipeline_for_block(self, block: int) -> int:
        if self.block_to_pipeline_map is None:
            return 0
        pipeline = self.pipeline_for_block(block)
        return self.block_to_pipeline_map.index(pipeline)

    def pipeline_for_block(self, block: int) -> int:
        if self.block_to_pipeline_map is None:
            return 0
        return self.block_to_pipeline_map[block]

    def devices_for_pipeline(self, pipeline: int) -> tuple[int, ...]:
        if self.pipeline_to_device_map is None:
            return (0,)
        return self.pipeline_to_device_map[pipeline]

    def to_properties(self) -> "PropertyValueType":
        res: dict[str, Any] = {}
        res["block_to_pipeline_map"] = self.block_to_pipeline_map
        res["pipeline_to_device_map"] = self.pipeline_to_device_map
        return res

    @staticmethod
    def from_properties(properties: "PropertyValueType") -> "ParallelismConfig":
        kwargs = dict(properties)
        fields_name_set = set(field.name for field in fields(ParallelismConfig))
        kwargs = {k: v for k, v in kwargs.items() if k in fields_name_set}

        if len(kwargs) != 0:
            return ParallelismConfig(**kwargs)

        return ParallelismConfig.default_config(
            block_count=LlamaHParams.from_gguf_props(properties).block_count
        )


@dataclass
class LlamaModelConfig:
    hp: LlamaHParams

    # Block sequence stride for a paged KV cache. This must divide evenly
    # into the context length.
    block_seq_stride: int = 32

    # amdsharktank supports only "paged"
    kv_cache_type: str = "paged"

    # If None will use attention_dtype.
    kv_cache_dtype: Optional[torch.dtype] = None

    # The device on which to place intermediate state.
    device: Optional[torch.device] = None

    # Dtype to use for general FP activations not otherwise configured.
    activation_dtype: torch.dtype = torch.float16

    # Dtype to use for attention.
    attention_dtype: torch.dtype = torch.float16

    # fake quant determines the mode the Layer Thetas operate w.r.t quantized tensors.
    fake_quant: bool = True

    # Configuration info for pipeline and tensor parallelism.
    parallelism_config: ParallelismConfig = None

    # Which attention kernel to use.
    attention_kernel: str = "torch"

    # Which matmul kernel to use.
    matmul_kernel: str = "*"

    # A list of layer indices where chunked attention is applied instead of full attention.
    chunked_attention_layers: Optional[set[int]] = None

    # Specifies the size of each chunk used during chunked attention computation.
    attention_chunk_size: Optional[int] = None

    # If True, applies normalization to the query and key vectors in attention.
    use_qk_norm: bool = False

    # Indices of layers that are MoE.
    moe_layers: Optional[list[int]] = None

    # Indices of layers for rope for llama4
    rope_layers: Optional[list[int]] = None

    # The default data type to use for model parameters and computations.
    dtype: Optional[torch.dtype] = None

    @property
    def pipeline_parallelism_size(self) -> int:
        return self.parallelism_config.pipeline_size

    @property
    def tensor_parallelism_size(self) -> int:
        return self.parallelism_config.tensor_parallelism_size

    @property
    def pipeline_to_device_map(self) -> tuple[tuple[int, ...], ...] | None:
        return self.parallelism_config.pipeline_to_device_map

    @property
    def block_to_pipeline_map(self) -> tuple[int, ...] | None:
        return self.parallelism_config.block_to_pipeline_map

    def __post_init__(self):
        if self.parallelism_config is None:
            self.parallelism_config = ParallelismConfig.default_config(
                block_count=self.hp.block_count
            )

        if self.moe_layers is None:
            if self.hp.interleave_moe_layer_step is None:
                self.moe_layers = []
            else:
                self.moe_layers = list(
                    range(
                        self.hp.interleave_moe_layer_step - 1,
                        self.hp.block_count,
                        self.hp.interleave_moe_layer_step,
                    )
                )
        else:
            if self.hp.interleave_moe_layer_step is not None:
                raise ValueError(
                    "moe_layers and hp.interleave_moe_layer_step are mutually exclusive."
                )

        if isinstance(self.dtype, str):
            self.dtype = serialized_name_to_dtype(self.dtype)

        if self.hp.no_rope_layer_step is not None:
            self.rope_layers = [
                i
                for i in range(self.hp.block_count)
                if int((i + 1) % self.hp.no_rope_layer_step != 0)
            ]

    def to_properties(self) -> "PropertyValueType":
        res = self.hp.to_gguf_props()
        res["kv_cache_type"] = self.kv_cache_type
        res["block_seq_stride"] = self.block_seq_stride
        if self.kv_cache_dtype is not None:
            res["kv_cache_dtype"] = dtype_to_serialized_name(self.kv_cache_dtype)
        res["activation_dtype"] = dtype_to_serialized_name(self.activation_dtype)
        res["attention_dtype"] = dtype_to_serialized_name(self.attention_dtype)
        res["fake_quant"] = self.fake_quant
        res["tensor_parallelism_size"] = self.tensor_parallelism_size
        res["block_to_pipeline_map"] = self.block_to_pipeline_map
        res["pipeline_to_device_map"] = self.pipeline_to_device_map
        res["attention_kernel"] = self.attention_kernel
        res["use_qk_norm"] = self.use_qk_norm
        res["attention_chunk_size"] = self.attention_chunk_size
        if self.chunked_attention_layers is not None:
            res["chunked_attention_layers"] = list(self.chunked_attention_layers)
        res["parallelism_config"] = self.parallelism_config.to_properties()
        return res

    @staticmethod
    def from_properties(properties: "PropertyValueType") -> "LlamaModelConfig":
        kwargs = dict(properties)
        fields_name_set = set(field.name for field in fields(LlamaModelConfig))
        kwargs = {k: v for k, v in kwargs.items() if k in fields_name_set}
        kwargs["hp"] = LlamaHParams.from_gguf_props(properties)
        kwargs["parallelism_config"] = ParallelismConfig.from_properties(properties)
        if "kv_cache_dtype" in kwargs:
            kwargs["kv_cache_dtype"] = serialized_name_to_dtype(
                kwargs["kv_cache_dtype"]
            )
        if "activation_dtype" in kwargs:
            kwargs["activation_dtype"] = serialized_name_to_dtype(
                kwargs["activation_dtype"]
            )
        if "attention_dtype" in kwargs:
            kwargs["attention_dtype"] = serialized_name_to_dtype(
                kwargs["attention_dtype"]
            )
        if "chunked_attention_layers" in kwargs:
            kwargs["chunked_attention_layers"] = set(kwargs["chunked_attention_layers"])
        return LlamaModelConfig(**kwargs)

    @staticmethod
    def from_dataset(dataset: Dataset, **kwargs):
        hparams = LlamaHParams.from_gguf_props(dataset.properties)

        default_dtype = torch.float16
        attn_k_theta = dataset.root_theta.optional_tensor("blk", 0, "attn_k")
        if attn_k_theta is not None and "q_output" in attn_k_theta:
            q_output = attn_k_theta["q_output"]
            default_dtype = q_output.dtype

        if "kv_cache_dtype" not in kwargs or kwargs["kv_cache_dtype"] is None:
            kwargs["kv_cache_dtype"] = default_dtype

        if "attention_dtype" not in kwargs or kwargs["attention_dtype"] is None:
            kwargs["attention_dtype"] = default_dtype

        return LlamaModelConfig(hp=hparams, **kwargs)

    @staticmethod
    def from_hugging_face_config(hf_config: dict[str, Any]) -> "LlamaModelConfig":
        if is_hugging_face_llama3_config(hf_config):
            return LlamaModelConfig.from_hugging_face_llama3_config(hf_config)

        raise ValueError(
            f"Could not convert Hugging Face config to amdsharktank LlamaModelConfig. No known conversion for config\n{hf_config}"
        )

    @staticmethod
    def from_hugging_face_llama3_config(
        hf_config: dict[str, Any]
    ) -> "LlamaModelConfig":
        assert is_hugging_face_llama3_config(hf_config)

        context_length = hf_config.get("max_position_embeddings", 2048)
        embedding_length = hf_config.get("hidden_size", 4096)
        block_count = hf_config.get("num_hidden_layers", 32)
        feed_forward_length = hf_config.get("intermediate_size", 11008)
        attention_head_count = hf_config.get("num_attention_heads", 32)
        attn_head_dim = hf_config.get("head_dim", 128)
        attention_layer_norm_rms_epsilon = hf_config.get("rms_norm_eps", 1e-06)
        attention_head_count_kv = hf_config.get("num_key_value_heads", 32)
        vocab_size = hf_config.get("vocab_size", 32000)
        rope_dimension_count = attn_head_dim
        rope_freq_base = hf_config.get("rope_theta", 10000.0)
        hp = LlamaHParams(
            model_arch="llama",
            context_length=context_length,
            embedding_length=embedding_length,
            block_count=block_count,
            feed_forward_length=feed_forward_length,
            attention_head_count=attention_head_count,
            attn_head_dim=attn_head_dim,
            attention_layer_norm_rms_epsilon=attention_layer_norm_rms_epsilon,
            attention_head_count_kv=attention_head_count_kv,
            vocab_size=vocab_size,
            rope_dimension_count=rope_dimension_count,
            rope_freq_base=rope_freq_base,
        )

        # The default in HF transformers is float32.
        dtype = getattr(torch, hf_config.get("torch_dtype", "float32"))
        activation_dtype = dtype
        attention_dtype = dtype
        return LlamaModelConfig(
            hp=hp,
            activation_dtype=activation_dtype,
            attention_dtype=attention_dtype,
            dtype=dtype,
        )


@dataclass
class T5Config:
    return_dict: bool = True
    output_hidden_states: bool = False
    output_attentions: bool = False
    is_encoder_decoder: bool = True
    is_decoder: bool = False
    vocab_size: int = 32128
    context_length: int = 512
    d_model: int = 512
    d_kv: int = 64
    d_ff: int = 2048
    num_layers: int = 6
    num_decoder_layers: int = 6
    num_heads: int = 8
    relative_attention_num_buckets: int = 32
    relative_attention_max_distance: int = 128
    layer_norm_epsilon: float = 1e-6
    feed_forward_proj: str = "relu"
    is_gated_act: bool = field(init=False)
    activation_dtype: torch.dtype = torch.float32
    dense_act_fn: str = field(init=False)
    use_cache: bool = True
    pad_token_id: int = 0
    eos_token_id: int = 1
    decoder_start_token_id: int = 0
    context_length_padding_block_size: int = 16

    def __post_init__(self):
        self.is_gated_act = self.feed_forward_proj.startswith("gated-")
        self.dense_act_fn = (
            self.feed_forward_proj.split("-")[1]
            if "-" in self.feed_forward_proj
            else self.feed_forward_proj
        )
        if self.dense_act_fn == "gelu":
            self.dense_act_fn = "gelu_new"

    @staticmethod
    def from_hugging_face_config(
        config: T5ConfigHf, tokenizer_config: dict[str, Any], **kwargs
    ) -> "T5Config":
        all_kwargs = {}
        for filed in fields(T5Config):
            if hasattr(config, filed.name):
                all_kwargs[filed.name] = getattr(config, filed.name)
        all_kwargs["context_length"] = tokenizer_config["model_max_length"]
        del all_kwargs["is_gated_act"]
        del all_kwargs["dense_act_fn"]
        all_kwargs.update(kwargs)
        return T5Config(**all_kwargs)

    def to_hugging_face_config(self) -> T5ConfigHf:
        kwargs = asdict(self)
        del kwargs["activation_dtype"]
        return T5ConfigHf(dropout_rate=0, **kwargs)

    @staticmethod
    def from_properties(properties: dict[str, Any]) -> "T5Config":
        kwargs = dict(properties)
        if "amdshark_DATASET_VERSION" in kwargs:
            kwargs.pop("amdshark_DATASET_VERSION")
        if "activation_dtype" in kwargs and kwargs["activation_dtype"] is not None:
            kwargs["activation_dtype"] = serialized_name_to_dtype(
                kwargs["activation_dtype"]
            )
        if "is_gated_act" in kwargs:
            kwargs.pop("is_gated_act")
        if "dense_act_fn" in kwargs:
            kwargs.pop("dense_act_fn")

        return T5Config(**kwargs)

    def to_properties(self) -> dict[str, Any]:
        res = asdict(self)
        if self.activation_dtype is not None:
            res["activation_dtype"] = dtype_to_serialized_name(self.activation_dtype)
        return res


@dataclass(kw_only=True)
class ClipTextConfig(ModelConfig):
    current_clip_config_version: ClassVar[str] = "0.1.0"
    vocab_size: int = 49408
    hidden_size: int = 512
    intermediate_size: int = 2048
    projection_dim: int = 512
    num_hidden_layers: int = 12
    num_attention_heads: int = 8
    max_position_embeddings: int = 77
    hidden_act: str = "quick_gelu"
    layer_norm_eps: float = 1e-5
    # This differs from `CLIPTokenizer`'s default and from openai/clip
    # See https://github.com/huggingface/transformers/pull/24773#issuecomment-1632287538
    pad_token_id: int = 1
    bos_token_id: int = 49406
    eos_token_id: int = 49407
    output_attentions: bool = False
    output_hidden_states: bool = False
    use_return_dict: bool = True
    dtype: torch.dtype = torch.float32

    def __post_init__(self):
        from amdsharktank.models.clip import ClipTextModel

        self.model_type = ClipTextModel
        super().__post_init__()

        self.layer_norm_eps = float(self.layer_norm_eps)
        if isinstance(self.dtype, str):
            self.dtype = serialized_name_to_dtype(self.dtype)

    @staticmethod
    def from_hugging_face_clip_text_model_config(
        config: "transformers.CLIPTextConfig",
    ) -> "ClipTextConfig":
        from amdsharktank.models.clip import ClipTextModel
        from amdsharktank.layers.base import get_model_type_id

        return ClipTextConfig(
            model_type=get_model_type_id(ClipTextModel),
            **ClipTextConfig.translate_hugging_face_config_dict_into_init_kwargs(
                config.to_dict()
            ),
        )

    @classmethod
    def translate_hugging_face_config_dict_into_init_kwargs(
        cls, properties: dict[str, Any], /
    ) -> dict[str, Any]:
        architectures: list[str] = properties["architectures"]
        if architectures is not None and architectures.count("CLIPModel") < 1:
            raise ValueError(
                f"Could not translate Hugging Face Clip text model config, unknown architectures {architectures}"
            )
        import transformers

        hf_config = transformers.CLIPTextConfig(**properties)
        res = {
            name: getattr(hf_config, hf_name)
            for name, hf_name in cls.get_config_name_to_hugging_face_map().items()
        }
        res["dtype"] = res["dtype"] or torch.float32
        return res

    def to_hugging_face_clip_text_model_config(self) -> "transformers.CLIPTextConfig":
        kwargs = {
            hf_name: getattr(self, name)
            for name, hf_name in self.get_config_name_to_hugging_face_map().items()
        }
        from transformers import CLIPTextConfig

        return CLIPTextConfig(**kwargs)

    @staticmethod
    def from_properties(properties: dict[str, Any]) -> "ClipTextConfig":
        kwargs = dict(properties)
        if "amdshark_DATASET_VERSION" in kwargs:
            kwargs.pop("amdshark_DATASET_VERSION")

        return ClipTextConfig(**kwargs)

    def asdict_for_saving(
        self, config_path: PathLike | None = None, /
    ) -> dict[str, Any]:
        res = super().asdict_for_saving(config_path)
        if res["dtype"] == torch.float32:
            del res["dtype"]
        if "dtype" in res:
            res["dtype"] = dtype_to_serialized_name(self.dtype)
        res["clip_config_version"] = self.current_clip_config_version
        return res

    @classmethod
    def get_config_name_to_hugging_face_map(cls) -> dict[str, str]:
        return {
            "vocab_size": "vocab_size",
            "hidden_size": "hidden_size",
            "intermediate_size": "intermediate_size",
            "projection_dim": "projection_dim",
            "num_hidden_layers": "num_hidden_layers",
            "num_attention_heads": "num_attention_heads",
            "max_position_embeddings": "max_position_embeddings",
            "hidden_act": "hidden_act",
            "layer_norm_eps": "layer_norm_eps",
            "pad_token_id": "pad_token_id",
            "bos_token_id": "bos_token_id",
            "eos_token_id": "eos_token_id",
            "output_attentions": "output_attentions",
            "output_hidden_states": "output_hidden_states",
            "use_return_dict": "return_dict",
            "dtype": "torch_dtype",
        }

    @classmethod
    def parse_for_init_kwargs(cls, **config_dict) -> dict[str, Any]:
        config_dict = super().parse_for_init_kwargs(**config_dict)
        cls._check_clip_config_version(config_dict)
        config_dict.pop("clip_config_version")
        return config_dict

    @classmethod
    def _check_clip_config_version(cls, config_dict: dict[str, Any], /):
        version = config_dict.get("clip_config_version")
        if version is None:
            raise ValueError("Missing CLIP config version.")
        if parse_version(version) != parse_version(cls.current_clip_config_version):
            raise ValueError(
                f"Could not load config with a CLIP config version {version},"
                f"expected version is {parse_version(cls.current_clip_config_version)}"
            )
