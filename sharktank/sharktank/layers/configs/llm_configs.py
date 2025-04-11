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

from typing import TYPE_CHECKING, ClassVar
from dataclasses import asdict, dataclass, field, fields
from typing import Any, Optional
import torch
from transformers import T5Config as T5ConfigHf
from .config import ModelConfig
from ...utils import parse_version
from os import PathLike

from ...types.tensors import serialized_name_to_dtype, dtype_to_serialized_name

if TYPE_CHECKING:
    import transformers

__all__ = ["ClipTextConfig", "LlamaHParams", "LlamaModelConfig", "T5Config"]


@dataclass
class LlamaHParams:
    """Corresponds 1:1 with the 'LLM' section of the GGUF docs.

    Comments are only provided if they differ from this source.
    """

    model_arch: str
    context_length: int
    embedding_length: int
    block_count: int
    feed_forward_length: int
    attention_head_count: int
    attn_head_dim: int
    attention_layer_norm_rms_epsilon: float
    attention_head_count_kv: int
    rope_dimension_count: Optional[int] = None
    rope_freq_base: Optional[float] = None
    expert_count: Optional[int] = None
    expert_used_count: Optional[int] = None

    # Deepseek Multi-Latent Attention config
    q_lora_rank: Optional[int] = None
    kv_lora_rank: Optional[int] = None
    qk_nope_head_dim: Optional[int] = None
    qk_rope_head_dim: Optional[int] = None
    v_head_dim: Optional[int] = None

    # Deepseek RoPE+YaRN config
    rope_scaling_type: Optional[str] = None
    rope_scaling_factor: Optional[float] = None
    rope_scaling_original_context_length: Optional[int] = None
    rope_scaling_yarn_log_multiplier: Optional[float] = None

    # Deepseek MoE config
    expert_shared_count: Optional[int] = None
    route_scale: Optional[float] = None
    n_expert_groups: Optional[int] = None
    n_limited_groups: Optional[int] = None
    n_dense_layers: Optional[int] = None

    # Grok configurations
    attention_softcap: Optional[float] = None

    @staticmethod
    def from_gguf_props(p: dict[str, Any]):
        name_prefix = p.get("general.architecture", "llama")
        default_expert_count = 0
        default_expert_used_count = 0
        default_rope_freq_base = 500000.0
        default_rope_dimension_count = 128
        defaut_n_dense_layers = 0
        attention_head_count = _int_prop(p, f"{name_prefix}.attention.head_count")
        rope_dimension_count = _optional_int_prop(
            p, f"{name_prefix}.rope.dimension_count", default_rope_dimension_count
        )

        attention_softcap = 30.0 if name_prefix == "grok" else None

        if name_prefix == "deepseek2":
            q_lora_rank = _int_prop(p, f"{name_prefix}.attention.q_lora_rank")
            kv_lora_rank = _int_prop(p, f"{name_prefix}.attention.kv_lora_rank")
            route_scale = _float_prop(p, f"{name_prefix}.expert_weights_scale")
            expert_shared_count = _int_prop(p, f"{name_prefix}.expert_shared_count")

            rope_scaling_type = _str_prop(p, f"{name_prefix}.rope.scaling.type")
            rope_scaling_factor = _float_prop(p, f"{name_prefix}.rope.scaling.factor")
            rope_scaling_original_context_length = _int_prop(
                p, f"{name_prefix}.rope.scaling.original_context_length"
            )
            rope_scaling_yarn_log_multiplier = _float_prop(
                p, f"{name_prefix}.rope.scaling.yarn_log_multiplier"
            )

            qk_nope_head_dim = 128
            qk_rope_head_dim = 64
            v_head_dim = 128
            n_expert_groups = 8
            n_limited_groups = 4
            attn_head_dim = qk_nope_head_dim + qk_rope_head_dim
        else:
            attn_head_dim = rope_dimension_count
            route_scale = None
            attention_softcap = None
            n_expert_groups = None
            n_limited_groups = None
            expert_shared_count = None
            q_lora_rank = None
            kv_lora_rank = None
            qk_nope_head_dim = None
            qk_rope_head_dim = None
            v_head_dim = None
            rope_scaling_type = None
            rope_scaling_factor = None
            rope_scaling_original_context_length = None
            rope_scaling_yarn_log_multiplier = None

        return LlamaHParams(
            model_arch=name_prefix,
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
            attn_head_dim=attn_head_dim,
            rope_dimension_count=rope_dimension_count,
            rope_freq_base=_optional_float_prop(
                p, f"{name_prefix}.rope.freq_base", default_rope_freq_base
            ),
            expert_count=_optional_int_prop(
                p, f"{name_prefix}.expert_count", default_expert_count
            ),
            expert_used_count=_optional_int_prop(
                p, f"{name_prefix}.expert_used_count", default_expert_used_count
            ),
            route_scale=route_scale,
            n_dense_layers=_optional_int_prop(
                p, f"{name_prefix}.leading_dense_block_count", defaut_n_dense_layers
            ),
            attention_softcap=attention_softcap,
            n_expert_groups=n_expert_groups,
            n_limited_groups=n_limited_groups,
            expert_shared_count=expert_shared_count,
            q_lora_rank=q_lora_rank,
            kv_lora_rank=kv_lora_rank,
            qk_nope_head_dim=qk_nope_head_dim,
            qk_rope_head_dim=qk_rope_head_dim,
            v_head_dim=v_head_dim,
            rope_scaling_type=rope_scaling_type,
            rope_scaling_factor=rope_scaling_factor,
            rope_scaling_original_context_length=rope_scaling_original_context_length,
            rope_scaling_yarn_log_multiplier=rope_scaling_yarn_log_multiplier,
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
        if self.rope_dimension_count is not None:
            res[f"{self.model_arch}.rope.dimension_count"] = self.rope_dimension_count
        if self.rope_freq_base is not None:
            res[f"{self.model_arch}.rope.freq_base"] = self.rope_freq_base
        if self.expert_count is not None:
            res[f"{self.model_arch}.expert_count"] = self.expert_count
        if self.expert_used_count is not None:
            res[f"{self.model_arch}.expert_used_count"] = self.expert_used_count
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


def _optional_float_prop(p: dict[str, Any], name: str, default_value: float) -> float:
    value = p.get(name, default_value)
    try:
        return float(value)
    except ValueError as e:
        raise ValueError(f"Property '{name}' expected to be a float and was not") from e


def _optional_int_prop(p: dict[str, Any], name: str, default_value: int) -> int:
    value = p.get(name, default_value)
    try:
        return int(value)
    except ValueError as e:
        raise ValueError(f"Property '{name}' expected to be an int and was not") from e


@dataclass
class LlamaModelConfig:
    hp: LlamaHParams

    # Block sequence stride for a paged KV cache. This must divide evenly
    # into the context length.
    block_seq_stride: int = 32

    # Either "paged" or "direct".
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

    # How many devices are involved for tensor parallel sharding.
    # If greater than 1, the model will expect sharded model parameters and function
    # arguments.
    tensor_parallelism_size: int = 1

    # Which attention kernel to use.
    attention_kernel: str = "torch"

    # Indicates if running with HuggingFace implementation and ensures
    # numerical equivalency to HuggingFace's LLaMa if true (by modifying
    # rotary embedding).
    use_hf: bool = False

    # If true, then the model may pre-initialize certain tables during
    # init. This can be better for eager execution but when capturing a program,
    # it is often better to preserve the calculation explicitly and rely on
    # the compiler to transform it to an initialization time step. This can
    # be the difference of many gigabytes of static data being embedded in
    # the program and not.
    static_tables: bool = True


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
        if "SHARK_DATASET_VERSION" in kwargs:
            kwargs.pop("SHARK_DATASET_VERSION")
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
        from ...models.clip import ClipTextModel

        self.model_type = ClipTextModel
        super().__post_init__()

        self.layer_norm_eps = float(self.layer_norm_eps)
        if isinstance(self.dtype, str):
            self.dtype = serialized_name_to_dtype(self.dtype)

    @staticmethod
    def from_hugging_face_clip_text_model_config(
        config: "transformers.CLIPTextConfig",
    ) -> "ClipTextConfig":
        from ...models.clip import ClipTextModel
        from ..base import get_model_type_id

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
        if "SHARK_DATASET_VERSION" in kwargs:
            kwargs.pop("SHARK_DATASET_VERSION")

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
