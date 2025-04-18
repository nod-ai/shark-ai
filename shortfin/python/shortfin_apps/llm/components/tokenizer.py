# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path

import tokenizers

import shortfin as sf
import shortfin.array as sfnp

# Type alias from the backing library.
Encoding = tokenizers.Encoding


class Tokenizer:
    def __init__(
        self, raw_tk: tokenizers.Tokenizer, pad_id: int = 0, eos_token: str = None
    ):
        self.pad_id = pad_id
        self.eos_token = eos_token
        self.eos_token_id = (
            raw_tk.token_to_id(eos_token) if eos_token is not None else None
        )
        self._raw = raw_tk
        self._raw.enable_padding(pad_id=pad_id)

    @staticmethod
    def from_pretrained(name: str) -> "Tokenizer":
        raw_tk = tokenizers.Tokenizer.from_pretrained(name)
        return Tokenizer(raw_tk)

    @staticmethod
    def from_tokenizer_json_file(json_path: Path | str, eos_token: str):
        return Tokenizer(
            tokenizers.Tokenizer.from_file(str(json_path)), eos_token=eos_token
        )

    def encode(self, texts: list[str]) -> list[tokenizers.Encoding]:
        """Encodes a batch of texts, applying no padding."""
        return self._raw.encode_batch(texts)

    def decode(self, sequences) -> list[str]:
        """Decodes a batch of sequences to text."""
        return self._raw.decode_batch(sequences)

    def encoding_length(self, enc: tokenizers.Encoding) -> int:
        """Gets the length of an encoding."""
        return len(enc.ids)

    def post_process_encodings(
        self, encs: list[tokenizers.Encoding], batch_seq_len: int
    ):
        """Truncates and pads to a requested size."""
        for enc in encs:
            enc.truncate(batch_seq_len)
            enc.pad(batch_seq_len)

    def encodings_to_array(
        self,
        device: sf.ScopedDevice,
        encs: list[tokenizers.Encoding],
        batch_seq_len: int,
        *,
        dtype: sfnp.DType = sfnp.int32,
    ):
        """Creates a device_array with the contents of a batch of encodings.

        It is expected that the user has called post_process_encodings with
        the same batch_seq_len in order to properly truncate/pad.
        """
        ary = sfnp.device_array.for_host(device, [len(encs), batch_seq_len], dtype)
        for i, enc in enumerate(encs):
            ary.view(i).items = enc.ids
        return ary

    def attention_masks_to_array(
        self,
        device: sf.ScopedDevice,
        encs: list[tokenizers.Encoding],
        batch_seq_len: int,
        *,
        dtype: sfnp.DType = sfnp.int32,
    ):
        ary = sfnp.device_array.for_host(device, [len(encs), batch_seq_len], dtype)
        for i, enc in enumerate(encs):
            ary.view(i).items = enc.attention_mask
        return ary
