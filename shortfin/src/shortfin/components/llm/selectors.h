// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_COMPONENTS_LLM_SELECTORS_H
#define SHORTFIN_COMPONENTS_LLM_SELECTORS_H

#include <vector>

#include "shortfin/components/llm/data.h"

namespace shortfin::llm {

SHORTFIN_API void SelectTokens(const std::vector<float> &scores,
                               const DecodeConfig &config,
                               std::vector<int> &selected_tokens,
                               std::vector<float> &selected_scores) noexcept;

}  // namespace shortfin::llm

#endif
