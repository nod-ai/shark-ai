# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


@pytest.fixture
def bert_tokenizer():
    import shortfin_apps.llm.components.tokenizer as tokenizer

    return tokenizer.Tokenizer.from_pretrained("bert-base-cased")


def test_tokenizers_lib(bert_tokenizer):
    enc0, enc1 = bert_tokenizer.encode(["This is sequence 1", "Sequence 2"])
    assert enc0.ids == [101, 1188, 1110, 4954, 122, 102]
    assert enc1.ids == [101, 22087, 25113, 123, 102]
    texts = bert_tokenizer.decode([enc0.ids, enc1.ids])
    assert texts == ["This is sequence 1", "Sequence 2"]

    # Test manual padding.
    enc0.pad(12)
    assert enc0.ids == [101, 1188, 1110, 4954, 122, 102, 0, 0, 0, 0, 0, 0]
    assert bert_tokenizer.encoding_length(enc0) == 12


def test_tokenizer_to_array(cpu_fiber, bert_tokenizer):
    batch_seq_len = 12
    encs = bert_tokenizer.encode(["This is sequence 1", "Sequence 2"])
    bert_tokenizer.post_process_encodings(encs, batch_seq_len)
    ary = bert_tokenizer.encodings_to_array(cpu_fiber.device(0), encs, batch_seq_len)
    print(ary)
    assert ary.view(0).items.tolist() == [
        101,
        1188,
        1110,
        4954,
        122,
        102,
        0,
        0,
        0,
        0,
        0,
        0,
    ]
    assert ary.view(1).items.tolist() == [
        101,
        22087,
        25113,
        123,
        102,
        0,
        0,
        0,
        0,
        0,
        0,
        0,
    ]

    masks = bert_tokenizer.attention_masks_to_array(
        cpu_fiber.device(0), encs, batch_seq_len
    )
    print(masks)
    assert masks.view(0).items.tolist() == [1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
    assert masks.view(1).items.tolist() == [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0]
