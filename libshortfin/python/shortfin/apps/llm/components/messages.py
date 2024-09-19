# Copyright 2024 Advanced Micro Devices, Inc.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions.
# See https://llvm.org/LICENSE.txt for license information.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import shortfin as sf

from .cache import AttnPageCache, AttnPageEntry


class PrefillRequest(sf.Message):
    """Performs a prefill operation."""

    def __init__(self, input_token_ids: list[int]):
        super().__init__()
        self.input_token_ids = input_token_ids
        self.done = sf.VoidFuture()

        # Cache pages that have been locked for this request.
        self._cache: AttnPageCache | None = None
        self._locked_pages: list[AttnPageEntry] | None = None

    def free_cache_pages(self):
        cache = self._cache
        if cache:
            pages = self._locked_pages
            self._cache = None
            self._locked_pages = None
            cache.release_pages(pages)

    def lock_cache_pages(self, cache: AttnPageCache, pages: list[AttnPageCache]):
        assert not self._cache
        self._cache = cache
        self._locked_pages = pages


class StrobeMessage(sf.Message):
    """Sent to strobe a queue with fake activity (generate a wakeup)."""

    ...
