# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Contains utilities for fetching datasets from huggingface.

There is nothing special about this mechanism, but it gives us a common
place to stash dataset information for testing and examples.

This can be invoked as a tool in order to fetch a local dataset.
"""

from typing import Dict, Optional, Sequence, Tuple, Union

import argparse
from dataclasses import dataclass
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


################################################################################
# Dataset support
################################################################################


@dataclass
class RemoteFile:
    file_id: str
    repo_id: str
    revision: str | None = None
    filename: str | None = None
    allow_patterns: list[str] | str | None = None
    ignore_patterns: list[str] | str | None = None
    extra_filenames: Sequence[str] = ()

    def download(self, *, local_dir: Optional[Path] = None) -> list[Path]:
        if self.filename is None:
            return self._download_snapshot(local_dir=local_dir)
        else:
            return self._download_files(local_dir=local_dir)

    def _download_files(self, *, local_dir: Optional[Path] = None) -> list[Path]:
        res = []
        res.append(
            Path(
                hf_hub_download(
                    repo_id=self.repo_id, filename=self.filename, local_dir=local_dir
                )
            )
        )
        for extra_filename in self.extra_filenames:
            res.append(
                Path(
                    hf_hub_download(
                        repo_id=self.repo_id,
                        revision=self.revision,
                        filename=extra_filename,
                        local_dir=local_dir,
                    )
                )
            )
        return res

    def _download_snapshot(self, *, local_dir: Optional[Path] = None) -> list[Path]:
        return [
            Path(
                snapshot_download(
                    repo_id=self.repo_id,
                    revision=self.revision,
                    allow_patterns=self.allow_patterns,
                    ignore_patterns=self.ignore_patterns,
                )
            )
        ]


@dataclass
class Dataset:
    name: str
    files: Tuple[RemoteFile]
    revision: str | None = None

    def __post_init__(self):
        if self.name in ALL_DATASETS:
            raise KeyError(f"Duplicate dataset name '{self.name}'")

        # Set the revision of all files if a Dataset revision is provided.
        if self.revision is not None:
            for f in self.files:
                assert f.revision is None
                f.revision = self.revision

        ALL_DATASETS[self.name] = self

    def alias_to(self, to_name: str) -> "Dataset":
        alias_dataset(self.name, to_name)
        return self

    def download(self, *, local_dir: Optional[Path] = None) -> Dict[str, list[Path]]:
        return {f.file_id: f.download(local_dir=local_dir) for f in self.files}


ALL_DATASETS: Dict[str, Dataset] = {}


def get_dataset(name: str) -> Dataset:
    try:
        return ALL_DATASETS[name]
    except KeyError:
        raise KeyError(f"Dataset {name} not found (available: {ALL_DATASETS.keys()})")


def alias_dataset(from_name: str, to_name: str):
    if to_name in ALL_DATASETS:
        raise KeyError(f"Cannot alias dataset: {to_name} already exists")
    ALL_DATASETS[to_name] = get_dataset(from_name)


################################################################################
# Dataset definitions
################################################################################

Dataset(
    "meta-llama/Llama-3.1-8B-Instruct",
    (
        RemoteFile(
            "all",
            repo_id="meta-llama/Llama-3.1-8B-Instruct",
            ignore_patterns="original/*",
        ),
    ),
    revision="0e9e39f249a16976918f6564b8830bc894c89659",
)

Dataset(
    "meta-llama/Llama-3.1-70B-Instruct",
    (
        RemoteFile(
            "all",
            repo_id="meta-llama/Llama-3.1-70B-Instruct",
            ignore_patterns="original/*",
        ),
    ),
    revision="1605565b47bb9346c5515c34102e054115b4f98b",
)

Dataset(
    "meta-llama/Llama-3.1-405B-Instruct",
    (
        RemoteFile(
            "all",
            repo_id="meta-llama/Llama-3.1-405B-Instruct",
            ignore_patterns="original/*",
        ),
    ),
    revision="be673f326cab4cd22ccfef76109faf68e41aa5f1",
)

Dataset(
    "SanctumAI/Meta-Llama-3.1-8B-Instruct-GGUF",
    (
        RemoteFile(
            "gguf",
            "SanctumAI/Meta-Llama-3.1-8B-Instruct-GGUF",
            "meta-llama-3.1-8b-instruct.f16.gguf",
        ),
        RemoteFile(
            "tokenizer_config.json",
            "NousResearch/Meta-Llama-3-8B-Instruct",
            "tokenizer_config.json",
            extra_filenames=["tokenizer.json"],
        ),
    ),
).alias_to("llama3_8B_fp16")

Dataset(
    "QuantFactory/Llama-3-8B_q4_1_gguf",
    (
        RemoteFile(
            "gguf",
            "QuantFactory/Meta-Llama-3-8B-GGUF",
            "Meta-Llama-3-8B.Q4_1.gguf",
        ),
        RemoteFile(
            "tokenizer_config.json",
            "NousResearch/Meta-Llama-3-8B",
            "tokenizer_config.json",
            extra_filenames=["tokenizer.json"],
        ),
    ),
).alias_to("llama3_8B_q4_1")

Dataset(
    "QuantFactory/Llama-3-8B_q8_0_gguf",
    (
        RemoteFile(
            "gguf",
            "QuantFactory/Meta-Llama-3-8B-GGUF",
            "Meta-Llama-3-8B.Q8_0.gguf",
        ),
        RemoteFile(
            "tokenizer_config.json",
            "NousResearch/Meta-Llama-3-8B",
            "tokenizer_config.json",
            extra_filenames=["tokenizer.json"],
        ),
    ),
).alias_to("llama3_8B_q8_0")

Dataset(
    "SlyEcho/open_llama_3b_v2_f16_gguf",
    (
        RemoteFile(
            "gguf", "SlyEcho/open_llama_3b_v2_gguf", "open-llama-3b-v2-f16.gguf"
        ),
        RemoteFile(
            "tokenizer_config.json",
            "openlm-research/open_llama_3b_v2",
            "tokenizer_config.json",
            extra_filenames=["tokenizer.model"],
        ),
    ),
).alias_to("open_llama_3b_v2_f16_gguf")

Dataset(
    "SlyEcho/open_llama_3b_v2_q8_0_gguf",
    (
        RemoteFile(
            "gguf", "SlyEcho/open_llama_3b_v2_gguf", "open-llama-3b-v2-q8_0.gguf"
        ),
        RemoteFile(
            "tokenizer_config.json",
            "openlm-research/open_llama_3b_v2",
            "tokenizer_config.json",
            extra_filenames=["tokenizer.model"],
        ),
    ),
).alias_to("open_llama_3b_v2_q8_0_gguf")

Dataset(
    "SlyEcho/open_llama_3b_v2_q4_1_gguf",
    (
        RemoteFile(
            "gguf", "SlyEcho/open_llama_3b_v2_gguf", "open-llama-3b-v2-q4_1.gguf"
        ),
        RemoteFile(
            "tokenizer_config.json",
            "openlm-research/open_llama_3b_v2",
            "tokenizer_config.json",
            extra_filenames=["tokenizer.model"],
        ),
    ),
).alias_to("open_llama_3b_v2_q4_1_gguf")

Dataset(
    "TheBloke/Llama-2-70B-GGUF_q4_k_m",
    (
        RemoteFile(
            "gguf",
            "TheBloke/Llama-2-70B-GGUF",
            "llama-2-70b.Q4_K_M.gguf",
        ),
        RemoteFile(
            "tokenizer_config.json",
            "TheBloke/Llama-2-70B-fp16",
            "tokenizer_config.json",
            extra_filenames=["tokenizer.model"],
        ),
    ),
).alias_to("llama2_70b_q4_k_m_gguf")

Dataset(
    "TheBloke/Llama-2-70B-GGUF_q4_k_s",
    (
        RemoteFile(
            "gguf",
            "TheBloke/Llama-2-70B-GGUF",
            "llama-2-70b.Q4_K_S.gguf",
        ),
        RemoteFile(
            "tokenizer_config.json",
            "TheBloke/Llama-2-70B-fp16",
            "tokenizer_config.json",
            extra_filenames=["tokenizer.model"],
        ),
    ),
).alias_to("llama2_70b_q4_k_s_gguf")

Dataset(
    "TheBloke/Mistral-7B-v0.1-GGUF_q4_k_m",
    (
        RemoteFile(
            "gguf",
            "TheBloke/Mistral-7B-v0.1-GGUF",
            "mistral-7b-v0.1.Q4_K_M.gguf",
        ),
        RemoteFile(
            "tokenizer_config.json",
            "TheBloke/Mistral-7B-v0.1-GPTQ",
            "tokenizer_config.json",
            extra_filenames=["tokenizer.model"],
        ),
    ),
).alias_to("mistral_7b_q4_k_m_gguf")

Dataset(
    "TheBloke/Mistral-7B-v0.1-GGUF_q8_0",
    (
        RemoteFile(
            "gguf",
            "TheBloke/Mistral-7B-v0.1-GGUF",
            "mistral-7b-v0.1.Q8_0.gguf",
        ),
        RemoteFile(
            "tokenizer_config.json",
            "TheBloke/Mistral-7B-v0.1-GPTQ",
            "tokenizer_config.json",
            extra_filenames=["tokenizer.model"],
        ),
    ),
).alias_to("mistral_7b_q8_0_gguf")

Dataset(
    "TheBloke/Mixtral-8x7B-v0.1-GGUF",
    (
        RemoteFile(
            "gguf",
            "TheBloke/Mixtral-8x7B-v0.1-GGUF",
            "mixtral-8x7b-v0.1.Q8_0.gguf",
        ),
        RemoteFile(
            "tokenizer_config.json",
            "TheBloke/Mixtral-8x7B-v0.1-GPTQ",
            "tokenizer_config.json",
            extra_filenames=["tokenizer.model"],
        ),
    ),
).alias_to("mixtral_8x7b_q8_0_gguf")

Dataset(
    "amd-shark/llama3.1-8B",
    (
        RemoteFile(
            "gguf",
            "amd-shark/llama-quant-models",
            "llama3.1-8b/llama8b_f16.gguf",
        ),
        RemoteFile(
            "tokenizer_config.json",
            "amd-shark/llama-quant-models",
            "llama3.1-8b/tokenizer_config.json",
            extra_filenames=["llama3.1-8b/tokenizer.json"],
        ),
    ),
).alias_to("llama3_8B_f16")

Dataset(
    "amd-shark/llama2-7B",
    (
        RemoteFile(
            "gguf",
            "amd-shark/llama-quant-models",
            "llama2-7b/llama2_7b_f16.gguf",
        ),
        RemoteFile(
            "tokenizer_config.json",
            "amd-shark/llama-quant-models",
            "llama2-7b/tokenizer_config.json",
            extra_filenames=["llama2-7b/tokenizer.json"],
        ),
    ),
).alias_to("llama2_7B_f16")

Dataset(
    "google/t5-v1_1-small",
    (
        RemoteFile(
            "config",
            "google/t5-v1_1-small",
            "config.json",
            extra_filenames=["generation_config.json", "special_tokens_map.json"],
        ),
        RemoteFile(
            "tokenizer_config.json",
            "google/t5-v1_1-small",
            "tokenizer_config.json",
            extra_filenames=["spiece.model"],
        ),
        RemoteFile(
            "pytorch_model.bin",
            "google/t5-v1_1-small",
            "pytorch_model.bin",
        ),
    ),
)

Dataset(
    "google/t5-v1_1-xxl",
    (
        RemoteFile(
            "config",
            "google/t5-v1_1-xxl",
            "config.json",
            extra_filenames=["generation_config.json", "special_tokens_map.json"],
        ),
        RemoteFile(
            "tokenizer_config.json",
            "google/t5-v1_1-xxl",
            "tokenizer_config.json",
            extra_filenames=["spiece.model"],
        ),
        RemoteFile(
            "pytorch_model.bin",
            "google/t5-v1_1-xxl",
            "pytorch_model.bin",
        ),
    ),
)

Dataset(
    "openai/clip-vit-large-patch14",
    (
        RemoteFile(
            "config",
            "openai/clip-vit-large-patch14",
            "config.json",
            extra_filenames=[
                "model.safetensors",
                "preprocessor_config.json",
                "special_tokens_map.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "vocab.json",
            ],
        ),
    ),
)

Dataset(
    "stabilityai/stable-diffusion-xl-base-1.0",
    (
        RemoteFile(
            "config",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "unet/config.json",
        ),
        RemoteFile(
            "parameters",
            "stabilityai/stable-diffusion-xl-base-1.0",
            "unet/diffusion_pytorch_model.fp16.safetensors",
        ),
    ),
)

# The Flux transformer is in 2 formats.
# This is used in diffusers.FluxTransformer2DModel
Dataset(
    "black-forest-labs/FLUX.1-schnell/transformer",
    (
        RemoteFile(
            "config",
            "black-forest-labs/FLUX.1-schnell",
            "transformer/config.json",
        ),
        RemoteFile(
            "parameters",
            "black-forest-labs/FLUX.1-schnell",
            "transformer/diffusion_pytorch_model-00001-of-00003.safetensors",
            extra_filenames=[
                "transformer/diffusion_pytorch_model-00002-of-00003.safetensors",
                "transformer/diffusion_pytorch_model-00003-of-00003.safetensors",
            ],
        ),
        RemoteFile(
            "parameters-index",
            "black-forest-labs/FLUX.1-schnell",
            "transformer/diffusion_pytorch_model.safetensors.index.json",
        ),
    ),
)

# The Flux transformer is in 2 formats.
# This is used in the Black Forest's Flux repo.
# https://github.com/black-forest-labs/flux
# We have based our implementation on that.
Dataset(
    "black-forest-labs/FLUX.1-schnell/black-forest-labs-transformer",
    (
        RemoteFile(
            "config",
            "black-forest-labs/FLUX.1-schnell",
            "transformer/config.json",
        ),
        RemoteFile(
            "parameters",
            "black-forest-labs/FLUX.1-schnell",
            "flux1-schnell.safetensors",
        ),
    ),
)
Dataset(
    "black-forest-labs/FLUX.1-dev/black-forest-labs-transformer",
    (
        RemoteFile(
            "config",
            "black-forest-labs/FLUX.1-dev",
            "transformer/config.json",
        ),
        RemoteFile(
            "parameters",
            "black-forest-labs/FLUX.1-dev",
            "flux1-dev.safetensors",
        ),
    ),
)

# tiny llama2-25m model for testing; has tokenizer
Dataset(
    "Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA",
    files=[
        RemoteFile(
            file_id="model.safetensors",
            filename="model.safetensors",
            repo_id="Mxode/TinyStories-LLaMA2-25M-256h-4l-GQA",
            extra_filenames=(
                "config.json",
                "tokenizer.json",
                "tokenizer_config.json",
            ),
        ),
    ],
)


################################################################################
# Tool entrypoint
################################################################################


def main():
    parser = argparse.ArgumentParser("hf_datasets")
    parser.add_argument(
        "dataset_name",
        nargs="+",
        help=f"Dataset to request (available = {list(ALL_DATASETS.keys())})",
    )
    parser.add_argument(
        "--local-dir", type=Path, help="Link all files to a local directory"
    )
    args = parser.parse_args()

    if args.local_dir:
        args.local_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in args.dataset_name:
        print(f"Downloading dataset {dataset_name}")
        ds = get_dataset(dataset_name).download(local_dir=args.local_dir)
        for key, paths in ds.items():
            print(f"  {key}: {paths}")


if __name__ == "__main__":
    main()
