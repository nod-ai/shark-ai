# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""
Configuration objects.

Classes:
- ModelParams: for reading and managing config keys specified in `config.json` files exported by `python -m amdsharktank.examples.export_paged_llm_v1`
- ServerParams: for specifying config keys needed by `python -m shortfin_apps.llm.server`
"""

import dataclasses_json

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from dataclasses_json import dataclass_json, Undefined

from .decode_config import DecodeConfig
from .decode_config import LogitsNormalization

import shortfin.array as sfnp


def _decode_dtype(name: str) -> sfnp.DType:
    obj = getattr(sfnp, name, None)
    if not isinstance(obj, sfnp.DType):
        raise ValueError(f"{name} is not a recognized dtype")

    return obj


dataclasses_json.cfg.global_config.encoders[sfnp.DType] = lambda dt: dt.name
dataclasses_json.cfg.global_config.decoders[sfnp.DType] = _decode_dtype


@dataclass_json(undefined=Undefined.EXCLUDE)
@dataclass
class PagedKVCacheParams:
    """Parameters for the paged KV cache.

    In a typical transformer model, the KV cache is organized similar to (mapped to
    our parameter names below):
        k = tensor.empty(transformer_block_count, batch_size, seq,
                        attn_head_count_kv, attn_head_dim)
        v = ...

    For context, a popular model has parameters of:
        attn_dtype_size = 2  # (fp16)
        max_seq_len = 2048
        transformer_block_count = 32
        attn_head_count_kv = 32
        attn_head_dim = 128   # (dim / head_count)

    If paging, then we primarily care about the organization of a single block, where
    a block represents a single position in the sequence for a single item in the batch.
    Therefore, it will be organized like:
        block = torch.empty(transformer_block_count, 2, attn_head_count_kv, attn_head_dim)

    In this scenario, we declare that one block holds the KV cache for all transformer
    block layers because it reduces the accounting. As such, for the above example,
    a single position in the sequence will be 524,288 bytes, assuming a 2-byte element
    type. If we choose to block by block_stride=16 positions, each block will be 8MiB.
    Assuming we wanted to dedicate 12GiB to the block cache, this would equate to 1536
    blocks for a total number of sequence positions of 24,576.

    These are well-known numbers but are derived above to give a sense of scale.

    In order to indirect through to the block cache, we have to provide the index map
    to specific invocations:

    * Prefill: Prefill is only writing to the blocks from [0:prompt_len], so it will
        need write indices of [batch_size, prompt_len // block_stride + 1].
    * Decode step: Decode is auto-regressive, and needs to first compute the new kv
        row and then attend over all rows in the cache up to this point in the sequence.

    If wanting to avoid dynamic allocation of transients, we can also pool the index
    tables based on the maximum batch size and maximum sequence length. Since all
    block cache sizes are well within the range of an i16, we will use that for storage.
    Therefore, each batch invocation would need a block lookup table of:

        byte_size = max_batch_size * (max_seq_len // block_stride) * sizeof(int16_t)

    For a max_batch_size of 16, this is 4KiB of block index table lookups per
    invocation. We don't have to statically allocate this, but the system is more
    predictable if we just reserve what we need. Again, numbers are given to give a
    sense of scale only: real workloads will vary.
    """

    # Tokens per page.
    block_seq_stride: int

    # Number of attention heads per block. This can be different from the model's
    # attention head count due to sharing.
    attention_head_count_kv: int

    # Size of the cache on each device.
    # Default: 256
    device_block_count: int

    # Element type of the KVCache
    kv_cache_dtype: sfnp.DType

    # Number of blocks per device for kvcache
    paged_kv_block_size_elements_per_device: list[int] | None = None


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class ModelParams:
    """
    Parameters for a specific compiled model, sufficient to do cache planning and
    invocations.

    Compatibility should be maintained with function generate_params_json in

    amdsharktank/amdsharktank/examples/export_paged_llm_v1.py
    """

    # Maximum length of a sequence including prompt and output.
    max_seq_len: int

    # Number of transformer layers (aka attention blocks / transformer blocks).
    transformer_block_count: int

    # Dimensionality of each attention head. This is the dimensionality of the
    # key and value vectors. AKA rope_dimension_count from the GGUF props.
    attn_head_dim: int

    # Batch sizes that the prefill stage is compiled for. These are expected to be
    # functions exported from the model with suffixes of "_bs{batch_size}". Must
    # be in ascending order.
    prefill_batch_sizes: list[int]

    # Similarly, batch sizes that the decode stage is compiled for.
    decode_batch_sizes: list[int]

    # Whether the model was exported with `start_positions` for prefill.
    has_prefill_position: bool = False

    # Name of the IREE module implementing the model.
    module_name: str = "module"

    # ABI of the module.
    module_abi_version: int = 1

    logits_normalization: LogitsNormalization = LogitsNormalization.NONE

    # The element type of the attention caches.
    attn_dtype: sfnp.DType = sfnp.float16

    # Define the `top_k` option that model was exported with for prefill/decode.
    # If `top_k` is None, only logits are returned.
    # If `top_k == 1`, logits/indices from `argmax` are returned.
    # If `top_k` > 1, logits/indices from `top_k` are returned.
    top_k: int | None = None

    # Cache parameters.
    paged_kv_cache: PagedKVCacheParams | None = None

    def __post_init__(self):
        if self.top_k is None or self.top_k >= 1:
            return

        raise ValueError(f"Currently, only `top_k >= 1` is supported.")

    # Size in bytes of the KV cache dtype.
    @property
    def attn_dtype_size(self) -> int:
        assert sfnp.DType.is_byte_aligned()
        return sfnp.DType.dense_byte_count()

    @property
    def max_prefill_batch_size(self) -> int:
        return self.prefill_batch_sizes[-1]

    @property
    def max_decode_batch_size(self) -> int:
        return self.decode_batch_sizes[-1]

    @property
    def max_batch_size(self):
        return max(self.max_prefill_batch_size, self.max_decode_batch_size)

    @property
    def has_paged_kv_cache(self):
        return self.paged_kv_cache is not None

    @property
    def paged_kv_unit_size_elements(self) -> int:
        """Size in elements of each cache line in the attention cache.
        Each cache line can store a unit position stride.
        """
        import warnings

        warnings.warn(
            "Using an old model which relies of deprecated features that will be removed in a future update, please re-export the model.",
            DeprecationWarning,
            stacklevel=2,
        )
        assert self.has_paged_kv_cache
        size = 1
        size *= self.transformer_block_count
        size *= 2  # K and V cache line
        size *= self.paged_kv_cache.attention_head_count_kv
        size *= self.attn_head_dim
        return size

    @property
    def paged_kv_block_size_elements(self) -> int:
        """Size in elements of each attention block of {block_position_stride}
        positions.
        """
        assert self.paged_kv_cache is not None
        return self.paged_kv_unit_size_elements * self.paged_kv_cache.block_seq_stride

    @staticmethod
    def load_json(path: Path | str):
        with open(path, "rt") as f:
            json_text = f.read()
        return ModelParams.from_json(json_text)


@dataclass_json(undefined=Undefined.RAISE)
@dataclass
class ServerParams:
    """
    Parameters relevant to a specific server launch.

    shortfin_apps.llm.server accepts an optional json server_config file for this + commandline arguments for setting attributes of this class.

    Commandline args take priority over server_config.json, which takes precedence over defaults defined in this dataclass.
    """

    # KV cache configuration
    prefix_sharing_algorithm: str = "none"  # none or trie

    # Program isolation configuration
    program_isolation: str = "per_call"

    decode_config: DecodeConfig | None = None

    use_native_impls: bool = False

    chunk_block_size: Optional[int] = None

    # Device configuration
    device_ids: list[str] = field(default_factory=list)
    amdgpu_async_allocations: bool = False
    amdgpu_async_caching: bool = False
    amdgpu_allocators: Optional[str] = None
    amdgpu_allow_device_reuse: bool = False

    @staticmethod
    def load(config_path: Optional[Path] = None) -> "ServerParams":
        """Create a new ServerParams object by overriding defaults from a `server_config.json` file.

        Args:
            config_path: Path to config file.

        Returns:
            ServerParams instance with defaults or loaded values
        """
        params = ServerParams()
        if config_path and config_path.exists():
            with open(config_path) as f:
                file_params = ServerParams.from_json(f.read())
                # Update only non-None values from file
                for field in params.__dataclass_fields__:
                    file_value = getattr(file_params, field)
                    if file_value is not None:
                        setattr(params, field, file_value)
        return params

    def update_from_args(self, args) -> None:
        """Update configuration from command line arguments.

        Args:
            args: Parsed command line arguments

        Command line arguments take highest priority.
        """
        for field in self.__dataclass_fields__:
            if hasattr(args, field):
                arg_value = getattr(args, field)
                if (
                    arg_value is not None
                ):  # Only override if a cmdline arg of the same name was provided
                    setattr(self, field, arg_value)
