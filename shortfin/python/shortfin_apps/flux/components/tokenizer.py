# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from transformers import CLIPTokenizer, T5Tokenizer, BatchEncoding

import shortfin as sf
import shortfin.array as sfnp


class Tokenizer:
    def __init__(
        self,
        raw_tk: CLIPTokenizer,
        max_length: int = 77,
        pad_id: int = 0,
        attn_mask=False,
    ):
        self.pad_id = pad_id
        self._raw = raw_tk
        self.max_length = max_length
        self.return_attention_mask = attn_mask

    @staticmethod
    def from_pretrained(name: str, subfolder: str) -> "Tokenizer":
        if subfolder == "tokenizer_2":
            raw_tk = T5Tokenizer.from_pretrained(name, subfolder=subfolder)
            max_length = 512
        else:
            raw_tk = CLIPTokenizer.from_pretrained(name, subfolder=subfolder)
            max_length = 77
        return Tokenizer(raw_tk, max_length=max_length)

    def encode(self, texts: list[str]):
        """Encodes a batch of texts, applying no padding."""
        return self._raw(
            texts,
            padding="max_length",
            max_length=self.max_length,
            truncation=True,
            return_tensors="np",
            return_attention_mask=self.return_attention_mask,
        )

    def encoding_length(self, enc: BatchEncoding) -> int:
        """Gets the length of an encoding."""
        return len(enc.input_ids)

    def encodings_to_array(
        self,
        device: sf.ScopedDevice,
        encs: dict[str, BatchEncoding],
        batch_seq_len: int,
        *,
        dtype: sfnp.DType = sfnp.int32,
    ):
        """Creates a device_array with the contents of a batch of encodings.

        It is expected that the user has called post_process_encodings with
        the same batch_seq_len in order to properly truncate/pad.
        """
        ary = sfnp.device_array.for_host(
            device, [len(encs.input_ids), batch_seq_len], dtype
        )
        for i, ids in enumerate(encs.input_ids):
            ary.view(i).items = ids
        return ary

    def attention_masks_to_array(
        self,
        device: sf.ScopedDevice,
        encs: list[BatchEncoding],
        batch_seq_len: int,
        *,
        dtype: sfnp.DType = sfnp.int32,
    ):
        ary = sfnp.device_array.for_host(
            device, [len(encs.attention_mask), batch_seq_len], dtype
        )
        for i, enc in enumerate(encs.attention_mask):
            ary.view(i).items = enc
        return ary
