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
import shutil
import torch
from pathlib import Path

from huggingface_hub import snapshot_download
from amdsharktank.layers.configs import (
    LlamaModelConfig,
    is_hugging_face_llama3_config,
)
from amdsharktank.types import *
from amdsharktank.utils import verify_exactly_one_is_not_none
from amdsharktank.utils.functools import compose
from amdsharktank.utils.logging import get_logger
from amdsharktank.transforms.dataset import wrap_in_list_if_inference_tensor

logger = get_logger(__name__)

if TYPE_CHECKING:
    from amdsharktank.types.theta import InferenceTensorTransform

MetadataTransform = Callable[[dict[str, Any]], dict[str, Any]]


def make_metadata_transform(hf_config: dict[str, Any]) -> MetadataTransform:
    def default_metadata_transform(metadata: dict[str, Any]) -> dict[str, Any]:
        # Separate meta parameters (prefixed with _) from hparams.
        meta_params = {k: v for k, v in metadata.items() if k.startswith("_")}
        hparams = {k: v for k, v in metadata.items() if not k.startswith("_")}
        return {
            "meta": meta_params,
            "hparams": hparams,
        }

    if is_hugging_face_llama3_config(hf_config):
        return llama3_hf_config_to_amdsharktank
    return default_metadata_transform


def make_tensor_transform(hf_config: dict[str, Any]) -> "InferenceTensorTransform":
    if is_hugging_face_llama3_config(hf_config):
        return transform_llama3_tensor_name_to_amdsharktank
    return lambda x: x


def import_hf_dataset(
    config_json_path: PathLike,
    param_paths: list[PathLike],
    output_irpa_file: Optional[PathLike] = None,
    target_dtype=None,
    tensor_transform: Optional["InferenceTensorTransform"] = None,
    metadata_transform: MetadataTransform | None = None,
    file_copy_map: dict[PathLike, PathLike] | None = None,
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

    if file_copy_map is not None:
        for src, dst in file_copy_map.items():
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(src, dst)

    dataset = Dataset(props, theta)
    if output_irpa_file is not None:
        Path(output_irpa_file).parent.mkdir(parents=True, exist_ok=True)
        dataset.save(output_irpa_file, io_report_callback=logger.debug)
    return dataset


def import_hf_dataset_from_hub(
    repo_id_or_path: str | None = None,
    *,
    revision: str | None = None,
    subfolder: str | None = None,
    config_subpath: str | None = None,
    output_irpa_file: PathLike | None = None,
    target_dtype: torch.dtype | None = None,
    file_copy_map: dict[PathLike, PathLike] | None = None,
    hf_dataset: str | None = None,
    preset: str | None = None,
) -> Dataset | None:
    verify_exactly_one_is_not_none(
        repo_id_or_path=repo_id_or_path, preset=preset, hf_dataset=hf_dataset
    )
    if preset is not None:
        return import_hf_dataset_from_hub(**get_dataset_import_preset_kwargs(preset))

    if hf_dataset is not None:
        from amdsharktank.utils.hf_datasets import get_dataset

        download_result_dict = get_dataset(hf_dataset).download()
        downloaded_file_paths = [
            p for paths in download_result_dict.values() for p in paths
        ]
        if len(downloaded_file_paths) > 1 or downloaded_file_paths[0].is_file():
            assert (
                subfolder is None
            ), "Not robust in determining the model dir if doing a non-single model snapshot download and subfolder is specified."
        model_dir = Path(os.path.commonpath([str(p) for p in downloaded_file_paths]))
    else:
        model_dir = Path(repo_id_or_path)
        if not model_dir.exists():
            model_dir = Path(
                snapshot_download(repo_id=repo_id_or_path, revision=revision)
            )

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

    if file_copy_map is not None:
        file_copy_map = {model_dir / src: dst for src, dst in file_copy_map.items()}

    return import_hf_dataset(
        config_json_path=config_json_path,
        param_paths=param_paths,
        output_irpa_file=output_irpa_file,
        target_dtype=target_dtype,
        file_copy_map=file_copy_map,
    )


dataset_import_presets: dict[str, dict[str, Any]] = {}
"""Declarative specification on how to import a HF dataset."""


def register_default_llama_dataset_preset(
    name: str,
    *,
    hf_dataset: str,
    output_prefix_path: str,
    target_dtype: torch.dtype | None = None,
):
    output_prefix_path = Path(output_prefix_path)
    dataset_import_presets[name] = {
        "hf_dataset": hf_dataset,
        "output_irpa_file": output_prefix_path / "model.irpa",
        "target_dtype": target_dtype,
        "file_copy_map": {
            "tokenizer.json": output_prefix_path / "tokenizer.json",
            "tokenizer_config.json": output_prefix_path / "tokenizer_config.json",
            "LICENSE": output_prefix_path / "LICENSE",
        },
    }


def register_all_dataset_import_presets():
    register_default_llama_dataset_preset(
        name="meta_llama3_1_8b_instruct_f16",
        hf_dataset="meta-llama/Llama-3.1-8B-Instruct",
        output_prefix_path="llama3.1/8b/instruct/f16",
        target_dtype=torch.float16,
    )
    register_default_llama_dataset_preset(
        name="meta_llama3_1_70b_instruct_f16",
        hf_dataset="meta-llama/Llama-3.1-70B-Instruct",
        output_prefix_path="llama3.1/70b/instruct/f16",
        target_dtype=torch.float16,
    )
    register_default_llama_dataset_preset(
        name="meta_llama3_1_405b_instruct_f16",
        hf_dataset="meta-llama/Llama-3.1-405B-Instruct",
        output_prefix_path="llama3.1/405b/instruct/f16",
        target_dtype=torch.float16,
    )


register_all_dataset_import_presets()


def get_dataset_import_preset_kwargs(preset: str) -> dict[str, Any]:
    return dataset_import_presets[preset]


_llama3_hf_to_amdsharktank_tensor_name_map: dict[str, str] = {
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


def transform_llama3_tensor_name_to_amdsharktank(
    tensor: InferenceTensor,
) -> None | InferenceTensor | Sequence[InferenceTensor]:
    layer_idx = None
    layer_idx_pattern = "(model\.layers\.)(\d+)(\..*)"
    re_match = re.match(layer_idx_pattern, tensor.name)
    name_to_map = tensor.name
    if re_match:
        layer_idx = re_match.group(2)
        name_to_map = re.sub(layer_idx_pattern, r"\1{layer_idx}\3", tensor.name)

    amdsharktank_name = _llama3_hf_to_amdsharktank_tensor_name_map[name_to_map]
    amdsharktank_name = amdsharktank_name.format(layer_idx=layer_idx)
    return DefaultPrimitiveTensor(data=tensor.as_torch(), name=amdsharktank_name)


def llama3_hf_config_to_amdsharktank(hf_config: dict[str, Any]) -> dict[str, Any]:
    config = LlamaModelConfig.from_hugging_face_llama3_config(hf_config)
    return config.to_properties()
