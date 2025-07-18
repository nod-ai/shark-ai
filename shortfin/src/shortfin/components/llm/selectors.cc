// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/components/llm/selectors.h"

#include <algorithm>
#include <numeric>
#include <vector>

namespace shortfin::llm {

void SelectTokensTopK(const std::vector<float> &scores,
                      const DecodeConfig &config,
                      std::vector<int> &selected_tokens,
                      std::vector<float> &selected_scores) noexcept {
  int num_select = config.num_beams;
  int vocab_size = scores.size();

  // Clear output vectors
  selected_tokens.clear();
  selected_scores.clear();

  if (num_select >= vocab_size) {
    // Select all tokens
    selected_tokens.resize(vocab_size);
    std::iota(selected_tokens.begin(), selected_tokens.end(), 0);
    selected_scores = scores;
  } else {
    // Create indices and sort by scores in descending order
    std::vector<int> indices(vocab_size);
    std::iota(indices.begin(), indices.end(), 0);

    // Get top k elements
    std::partial_sort(
        indices.begin(), indices.begin() + num_select, indices.end(),
        [&scores](int i, int j) { return scores[i] > scores[j]; });

    // Extract top k tokens and scores
    selected_tokens.resize(num_select);
    selected_scores.resize(num_select);

    for (int i = 0; i < num_select; ++i) {
      selected_tokens[i] = indices[i];
      selected_scores[i] = scores[indices[i]];
    }
  }
}

void SelectTokensGreedy(const std::vector<float> &scores,
                        const DecodeConfig &config,
                        std::vector<int> &selected_tokens,
                        std::vector<float> &selected_scores) noexcept {
  // Find the token with the highest score
  auto max_it = std::max_element(scores.begin(), scores.end());
  int argmax = std::distance(scores.begin(), max_it);

  // Clear output vectors and add the single best token
  selected_tokens.clear();
  selected_scores.clear();

  selected_tokens.push_back(argmax);
  selected_scores.push_back(*max_it);
}

void SelectTokens(const std::vector<float> &scores, const DecodeConfig &config,
                  std::vector<int> &selected_tokens,
                  std::vector<float> &selected_scores) noexcept {
  if (config.use_beam_search) {
    SelectTokensTopK(scores, config, selected_tokens, selected_scores);
  } else {
    SelectTokensGreedy(scores, config, selected_tokens, selected_scores);
  }
}

}  // namespace shortfin::llm
