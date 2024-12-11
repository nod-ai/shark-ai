# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import functools
import torch
from os import PathLike, makedirs
from typing import Union, Optional
from copy import copy
from iree.turbine.aot.params import ParameterArchiveBuilder

from ...layers.configs.llm_configs import ClipTextConfig
from .clip import ClipTextModel
from ...types.theta import Theta, Dataset
from ...types.tensors import dtype_to_serialized_short_name
from ...utils.io import save_tensor_as_irpa
from .export import (
    clip_text_model_to_dataset,
    hugging_face_clip_text_model_to_theta,
    export_clip_text_model_to_iree,
)
from ...transforms.dataset import set_float_dtype


def clip_toy_text_model_config(dtype: Optional[torch.dtype] = None) -> ClipTextConfig:
    num_attention_heads = 5
    vocab_size = 11
    return ClipTextConfig(
        vocab_size=vocab_size,
        hidden_size=13 * num_attention_heads,
        intermediate_size=7,
        projection_dim=3,
        num_attention_heads=num_attention_heads,
        max_position_embeddings=17,
        layer_norm_eps=1e-4,
        num_hidden_layers=2,
        bos_token_id=vocab_size - 2,
        eos_token_id=vocab_size - 1,
        dtype=dtype,
    )


def export_clip_toy_text_model_default_iree_test_data(output_dir: PathLike):
    makedirs(output_dir, exist_ok=True)

    # We want to always export the same without interfering with RNG for the rest of
    # the program.
    rng_state = torch.get_rng_state()
    torch.random.manual_seed(12345)

    reference_dtype = torch.float32
    target_dtypes = [torch.float32, torch.bfloat16]
    target_iree_parameters_output_paths = []
    target_mlir_output_paths = []
    batch_size = 4
    for dtype in target_dtypes:
        prefix = output_dir / f"{dtype_to_serialized_short_name(dtype)}"
        target_iree_parameters_output_paths.append(f"{prefix}_parameters.irpa")
        target_mlir_output_paths.append(f"{prefix}.mlir")
    call_prefix = output_dir / f"forward_bs{batch_size}"
    input_ids_output_path = f"{call_prefix}_arg0_input_ids.irpa"
    expected_last_hidden_state_output_path = (
        f"{call_prefix}_expected_result0_last_hidden_state_"
        f"{dtype_to_serialized_short_name(reference_dtype)}.irpa"
    )
    export_clip_toy_text_model_iree_test_data(
        reference_dtype=reference_dtype,
        target_dtypes=target_dtypes,
        batch_size=batch_size,
        input_ids_output_path=input_ids_output_path,
        expected_last_hidden_state_output_path=expected_last_hidden_state_output_path,
        target_iree_parameters_output_paths=target_iree_parameters_output_paths,
        target_mlir_output_paths=target_mlir_output_paths,
    )

    torch.set_rng_state(rng_state)


def export_clip_toy_text_model_iree_test_data(
    reference_dtype: torch.dtype,
    target_dtypes: list[torch.dtype],
    batch_size: int,
    target_iree_parameters_output_paths: list[PathLike],
    target_mlir_output_paths: list[PathLike],
    input_ids_output_path: PathLike,
    expected_last_hidden_state_output_path: PathLike,
):
    reference_config = clip_toy_text_model_config(reference_dtype)
    input_ids = make_random_input_token_sequences(
        batch_size=batch_size, config=reference_config
    )
    reference_theta = make_clip_text_model_random_theta(reference_config)
    reference_model = ClipTextModel(theta=reference_theta, config=reference_config)
    for i, (
        target_dtype,
        target_iree_parameters_output_path,
        target_mlir_output_path,
    ) in enumerate(
        zip(
            target_dtypes,
            target_iree_parameters_output_paths,
            target_mlir_output_paths,
            strict=True,
        )
    ):
        current_input_ids_output_path = None
        current_expected_last_hidden_state_output_path = None
        if i == 0:
            current_input_ids_output_path = input_ids_output_path
            current_expected_last_hidden_state_output_path = (
                expected_last_hidden_state_output_path
            )
        export_clip_text_model_iree_test_data(
            reference_model=reference_model,
            target_dtype=target_dtype,
            input_ids=input_ids,
            target_iree_parameters_output_path=target_iree_parameters_output_path,
            target_mlir_output_path=target_mlir_output_path,
            input_ids_output_path=current_input_ids_output_path,
            expected_last_hidden_state_output_path=current_expected_last_hidden_state_output_path,
        )


def export_clip_text_model_iree_test_data(
    reference_model: ClipTextModel,
    target_dtype: torch.dtype,
    input_ids: torch.LongTensor,
    target_mlir_output_path: PathLike,
    target_iree_parameters_output_path: PathLike,
    input_ids_output_path: Optional[PathLike] = None,
    expected_last_hidden_state_output_path: Optional[PathLike] = None,
):
    batch_size = input_ids.shape[0]
    reference_dataset = clip_text_model_to_dataset(reference_model)
    target_config = copy(reference_model.config)
    target_config.dtype = target_dtype
    target_dataset = Dataset(
        root_theta=reference_dataset.root_theta.transform(
            functools.partial(set_float_dtype, dtype=target_dtype)
        ),
        properties=target_config.to_properties(),
    )
    target_model = ClipTextModel(theta=target_dataset.root_theta, config=target_config)
    export_clip_text_model_to_iree(
        target_model,
        batch_sizes=[batch_size],
        mlir_output_path=target_mlir_output_path,
        parameters_output_path=target_iree_parameters_output_path,
    )

    if input_ids_output_path is not None:
        save_tensor_as_irpa(input_ids, input_ids_output_path)

    if expected_last_hidden_state_output_path is None:
        return

    expected_last_hidden_state = reference_model(input_ids=input_ids)[
        "last_hidden_state"
    ]
    save_tensor_as_irpa(
        expected_last_hidden_state, expected_last_hidden_state_output_path
    )


def make_clip_text_model_random_theta(config: ClipTextConfig) -> Theta:
    from transformers import CLIPTextModel as HfCLIPTextModel

    hf_config = config.to_hugging_face_clip_text_model_config()
    model = HfCLIPTextModel(hf_config)
    return hugging_face_clip_text_model_to_theta(model)


def make_random_input_token_sequences(
    batch_size: int, config: ClipTextConfig
) -> torch.LongTensor:
    sequence_lens = torch.randint(
        low=1, high=config.max_position_embeddings + 1, size=(batch_size,)
    )
    sequences = torch.full(
        size=(batch_size, config.max_position_embeddings),
        fill_value=config.eos_token_id,
        dtype=torch.long,
    )
    for batch_idx, l in enumerate(sequence_lens):
        sequences[batch_idx][0:l] = torch.randint(
            low=0, high=config.vocab_size - 1, size=(l,), dtype=torch.long
        )
    return sequences
