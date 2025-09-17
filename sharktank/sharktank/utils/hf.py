# Copyright 2025 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any, Callable, Optional, Sequence, TYPE_CHECKING
from os import PathLike
import re
import os
import json
import logging
import torch
from pathlib import Path

from huggingface_hub import snapshot_download
from sharktank.layers.configs import LlamaHParams, LlamaModelConfig
from sharktank.types import *
from sharktank.utils.functools import compose
from sharktank.transforms.dataset import wrap_in_list_if_inference_tensor

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sharktank.types.theta import InferenceTensorTransform

MetadataTransform = Callable[[dict[str, Any]], dict[str, Any]]


def is_hf_llama3_config(hf_config: dict[str, Any]) -> bool:
    if "architectures" not in hf_config or "model_type" not in hf_config:
        return False
    return (
        "LlamaForCausalLM" in hf_config["architectures"]
        and hf_config["model_type"] == "llama"
    )


def make_metadata_transform(hf_config: dict[str, Any]) -> MetadataTransform:
    def default_metadata_transform(metadata: dict[str, Any]) -> dict[str, Any]:
        # Separate meta parameters (prefixed with _) from hparams.
        meta_params = {k: v for k, v in metadata.items() if k.startswith("_")}
        hparams = {k: v for k, v in metadata.items() if not k.startswith("_")}
        return {
            "meta": meta_params,
            "hparams": hparams,
        }

    if is_hf_llama3_config:
        return llama3_hf_config_to_sharktank
    return default_metadata_transform


def make_tensor_transform(hf_config: dict[str, Any]) -> "InferenceTensorTransform":
    if is_hf_llama3_config:
        return transform_llama3_tensor_name_to_sharktank
    return lambda x: x


def import_hf_dataset(
    config_json_path: PathLike,
    param_paths: list[PathLike],
    output_irpa_file: Optional[PathLike] = None,
    target_dtype=None,
    tensor_transform: Optional["InferenceTensorTransform"] = None,
    metadata_transform: MetadataTransform | None = None,
) -> Optional[Dataset]:
    import safetensors

    with open(config_json_path, "rb") as f:
        hf_config = json.load(f)
    if metadata_transform is None:
        metadata_transform = make_metadata_transform(hf_config)
    props = metadata_transform(hf_config)

    if tensor_transform is None:
        tensor_transform = make_tensor_transform(hf_config)
    tensor_transform = compose(tensor_transform, wrap_in_list_if_inference_tensor)

    tensors = []
    for params_path in param_paths:
        with safetensors.safe_open(params_path, framework="pt", device="cpu") as st:
            for name in st.keys():
                tensor = DefaultPrimitiveTensor(
                    name=name, data=st.get_tensor(name).to(target_dtype)
                )
                transformed_tensors = tensor_transform(tensor)
                if transformed_tensors is None:
                    continue
                tensors.extend(transformed_tensors)

    theta = Theta(tensors)

    dataset = Dataset(props, theta)
    if output_irpa_file is not None:
        dataset.save(output_irpa_file, io_report_callback=logger.debug)
    return dataset


def import_hf_dataset_from_hub(
    repo_id_or_path: str,
    *,
    revision: str | None = None,
    subfolder: str | None = None,
    config_subpath: str | None = None,
    output_irpa_file: PathLike | None = None,
) -> Dataset | None:
    model_dir = Path(repo_id_or_path)
    if not model_dir.exists():
        model_dir = Path(snapshot_download(repo_id=repo_id_or_path, revision=revision))

    if subfolder is not None:
        model_dir /= subfolder
    if config_subpath is None:
        config_json_path = model_dir / "config.json"
    else:
        config_json_path = model_dir / config_subpath
    file_paths = [
        model_dir / file_name
        for file_name in os.listdir(model_dir)
        if (model_dir / file_name).is_file()
    ]
    param_paths = [p for p in file_paths if p.is_file() and p.suffix == ".safetensors"]

    return import_hf_dataset(
        config_json_path=config_json_path,
        param_paths=param_paths,
        output_irpa_file=output_irpa_file,
    )


_llama3_hf_to_sharktank_tensor_name_map: dict[str, str] = {
    "model.embed_tokens.weight": "token_embd.weight",
    "lm_head.weight": "output.weight",
    "model.norm.weight": "output_norm.weight",
    "model.layers.{layer_idx}.input_layernorm.weight": "blk.{layer_idx}.attn_norm.weight",
    "model.layers.{layer_idx}.mlp.down_proj.weight": "blk.{layer_idx}.ffn_down.weight",
    "model.layers.{layer_idx}.mlp.gate_proj.weight": "blk.{layer_idx}.ffn_gate.weight",
    "model.layers.{layer_idx}.mlp.up_proj.weight": "blk.{layer_idx}.ffn_up.weight",
    "model.layers.{layer_idx}.post_attention_layernorm.weight": "blk.{layer_idx}.ffn_norm.weight",
    "model.layers.{layer_idx}.self_attn.k_proj.weight": "blk.{layer_idx}.attn_k.weight",
    "model.layers.{layer_idx}.self_attn.o_proj.weight": "blk.{layer_idx}.attn_output.weight",
    "model.layers.{layer_idx}.self_attn.q_proj.weight": "blk.{layer_idx}.attn_q.weight",
    "model.layers.{layer_idx}.self_attn.v_proj.weight": "blk.{layer_idx}.attn_v.weight",
}


def transform_llama3_tensor_name_to_sharktank(
    tensor: InferenceTensor,
) -> None | InferenceTensor | Sequence[InferenceTensor]:
    layer_idx = None
    layer_idx_pattern = "(model\.layers\.)(\d+)(\..*)"
    re_match = re.match(layer_idx_pattern, tensor.name)
    name_to_map = tensor.name
    if re_match:
        layer_idx = re_match.group(2)
        name_to_map = re.sub(layer_idx_pattern, r"\1{layer_idx}\3", tensor.name)

    sharktank_name = _llama3_hf_to_sharktank_tensor_name_map[name_to_map]
    sharktank_name = sharktank_name.format(layer_idx=layer_idx)
    return DefaultPrimitiveTensor(data=tensor.as_torch(), name=sharktank_name)


def llama3_hf_config_to_sharktank(hf_config: dict[str, Any]) -> dict[str, Any]:
    assert is_hf_llama3_config(hf_config)

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
    config = LlamaModelConfig(
        hp=hp,
        activation_dtype=activation_dtype,
        attention_dtype=attention_dtype,
        use_hf=True,
        dtype=dtype,
    )

    return config.to_properties()
