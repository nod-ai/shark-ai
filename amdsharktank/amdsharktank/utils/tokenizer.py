# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Simple helpers for accessing tokenizers of various kinds."""

from abc import ABC, abstractmethod
from typing import Optional, Union
import os

from amdsharktank.utils.evaluate import *

__all__ = [
    "load_tokenizer",
    "InferenceTokenizer",
]


class InferenceTokenizer(ABC):
    """Simple inference tokenizer."""

    def encode(
        self,
        texts: list[str],
        pad_to_multiple_of: int = 1,
        add_start_token: bool = False,
    ) -> tuple[list[list[int]]]:
        """Encodes a list of texts into a padded list of tokens.

        Returns a list of list of tokens and a list of unpadded lengths.
        """
        raw_rows = self._encode(texts, add_start_token)
        raw_rows, lengths = pad_tokens(
            token_ids=raw_rows, pad_to_multiple_of=pad_to_multiple_of
        )
        return raw_rows, lengths

    def decode(self, tokens: Union[list[list[int]]], lens: Optional[list[int]] = None):
        """Decodes a list of tokens."""
        if lens is not None:
            tokens = list(tokens)
            for i, row_length in enumerate(lens):
                tokens[i] = tokens[i][0:row_length]
        return self._decode(tokens)

    @property
    def vocab_size(self):
        return self._vocab_size()

    @abstractmethod
    def _encode(self, texts: list[str]) -> list[list[int]]:
        ...

    @abstractmethod
    def _decode(self, tokens: list[list[int]]) -> list[str]:
        ...


class FakeTokenizer(InferenceTokenizer):
    def _encode(self, texts: list[str], add_start_token: bool) -> list[list[int]]:
        encoded = []
        for text in texts:
            encoded.append([int(t) for t in text.split(" ")])
        return encoded

    def _decode(self, tokens: list[list[int]]) -> list[str]:
        strings = []
        for token in tokens:
            strings.append(" ".join([str(t) for t in token]))
        return strings


def fake_tokenizer():
    return FakeTokenizer()


def load_tokenizer(*posargs, tokenizer_type: str = "transformers", **kwargs):
    if tokenizer_type == "transformers":
        return _create_transformers_tokenizer(*posargs, **kwargs)


def _create_transformers_tokenizer(model_path: os.PathLike) -> InferenceTokenizer:
    from transformers import AutoTokenizer
    from transformers.tokenization_utils import PaddingStrategy, TruncationStrategy

    t = AutoTokenizer.from_pretrained(model_path, legacy=False)
    t.add_special_tokens({"pad_token": "0"})

    class _TransformersTokenizer(InferenceTokenizer):
        def __init__(self, t: AutoTokenizer):
            self._t = t

        def _encode(self, texts: list[str], add_start_token: bool) -> list[list[int]]:
            results = t.batch_encode_plus(
                texts,
                add_special_tokens=add_start_token,
                padding=False,
                truncation=False,
            )
            return results["input_ids"]

        def _decode(self, tokens: list[list[int]]) -> list[str]:
            return t.batch_decode(tokens)

        def _vocab_size(self):
            return t.vocab_size

    return _TransformersTokenizer(t)
