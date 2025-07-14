// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/components/llm/beam_scorer.h"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <numeric>
#include <random>

namespace shortfin::llm {

// BaseBeamScorer helper methods
void BaseBeamScorer::ApplyLengthPenalty(BeamState& beam) {
  if (config_.length_penalty != 1.0f && beam.exec_req) {
    // TODO(@zeeshanhaque21): Access the exec_req to get the input_token_ids and
    // prompt_length
    float length_factor =
        std::pow(2.0f, config_.length_penalty);  // Placeholder
    beam.score /= length_factor;
  }
}

std::vector<float> BaseBeamScorer::Softmax(const std::vector<float>& logits) {
  std::vector<float> result(logits.size());
  float max_val = *std::max_element(logits.begin(), logits.end());

  float sum = 0.0f;
  for (size_t i = 0; i < logits.size(); ++i) {
    result[i] = std::exp(logits[i] - max_val);
    sum += result[i];
  }

  for (size_t i = 0; i < result.size(); ++i) {
    result[i] /= sum;
  }

  return result;
}

std::vector<float> BaseBeamScorer::LogSoftmax(
    const std::vector<float>& logits) {
  std::vector<float> result(logits.size());
  float max_val = *std::max_element(logits.begin(), logits.end());

  float log_sum = 0.0f;
  for (float logit : logits) {
    log_sum += std::exp(logit - max_val);
  }
  log_sum = std::log(log_sum);

  for (size_t i = 0; i < logits.size(); ++i) {
    result[i] = logits[i] - max_val - log_sum;
  }

  return result;
}

std::vector<float> BaseBeamScorer::ApplyTemperature(
    const std::vector<float>& logits, float temperature) {
  if (temperature == 1.0f) {
    return logits;
  }

  std::vector<float> result(logits.size());
  for (size_t i = 0; i < logits.size(); ++i) {
    result[i] = logits[i] / temperature;
  }
  return result;
}

// DefaultScorer implementation
DefaultScorer::DefaultScorer(const DecodeConfig& config)
    : BaseBeamScorer(config) {}

void DefaultScorer::UpdateScore(BeamState& beam, float value) {
  // Default scorer doesn't accumulate scores
}

void DefaultScorer::FinalizeScore(BeamState& beam) {
  // Default scorer doesn't need finalization
}

void DefaultScorer::Reset() {
  // Default scorer doesn't need state reset
}

std::vector<BeamState> DefaultScorer::SelectBeams(
    const std::vector<BeamState>& active_beams,
    const std::vector<BeamState>& completed_beams) {
  std::vector<BeamState> selections;
  selections.reserve(active_beams.size());

  // Sample logits for each active beam for it to select its next token.
  for (const auto& beam : active_beams) {
    // Create a copy of the beam for the selection
    BeamState selected_beam = beam;

    // In the Python implementation, this calls
    // beam.sample_logits(len(completed_beams)) For now, we'll use a simple
    // token sampling approach In a real implementation, this would get logits
    // from the inference request
    int sampled_token = SampleToken(LogitsData({0.1f, 0.2f, 0.7f}));
    selected_beam.last_token = sampled_token;

    selections.push_back(selected_beam);
  }

  return selections;
}

int DefaultScorer::SampleToken(const LogitsData& logits_data) {
  std::vector<float> processed_logits =
      ApplyTemperature(logits_data.logits, config_.temperature);

  if (config_.top_k > 0 && config_.top_p > 0.0f) {
    // Apply both top-k and top-p
    auto top_k_result = SampleTopK(processed_logits, config_.top_k);
    auto top_p_result = SampleTopP(top_k_result.second, config_.top_p);
    return top_p_result.first.empty() ? 0 : top_p_result.first[0];
  } else if (config_.top_k > 0) {
    // Apply only top-k
    auto result = SampleTopK(processed_logits, config_.top_k);
    return result.first.empty() ? 0 : result.first[0];
  } else if (config_.top_p > 0.0f) {
    // Apply only top-p
    auto result = SampleTopP(processed_logits, config_.top_p);
    return result.first.empty() ? 0 : result.first[0];
  } else {
    // Greedy selection
    return SelectGreedy(processed_logits);
  }
}

int DefaultScorer::SelectGreedy(const std::vector<float>& logits) {
  auto max_it = std::max_element(logits.begin(), logits.end());
  return static_cast<int>(std::distance(logits.begin(), max_it));
}

std::pair<std::vector<int>, std::vector<float>> DefaultScorer::SampleTopK(
    const std::vector<float>& logits, int k) {
  // Create pairs of (index, value) and sort by value
  std::vector<std::pair<int, float>> indexed_logits;
  for (size_t i = 0; i < logits.size(); ++i) {
    indexed_logits.emplace_back(static_cast<int>(i), logits[i]);
  }

  // Sort by value in descending order
  std::partial_sort(
      indexed_logits.begin(),
      indexed_logits.begin() +
          std::min(k, static_cast<int>(indexed_logits.size())),
      indexed_logits.end(),
      [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
        return a.second > b.second;
      });

  // Extract top-k tokens and probabilities
  std::vector<int> tokens;
  std::vector<float> probs;
  int actual_k = std::min(k, static_cast<int>(indexed_logits.size()));

  for (int i = 0; i < actual_k; ++i) {
    tokens.push_back(indexed_logits[i].first);
    probs.push_back(indexed_logits[i].second);
  }

  // Convert to probabilities
  std::vector<float> softmax_probs = Softmax(probs);

  // Sample from the distribution
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> dist(softmax_probs.begin(), softmax_probs.end());

  int selected_idx = dist(gen);
  return {{tokens[selected_idx]}, {softmax_probs[selected_idx]}};
}

std::pair<std::vector<int>, std::vector<float>> DefaultScorer::SampleTopP(
    const std::vector<float>& logits, float p) {
  // Create pairs and sort by probability
  std::vector<std::pair<int, float>> indexed_logits;
  for (size_t i = 0; i < logits.size(); ++i) {
    indexed_logits.emplace_back(static_cast<int>(i), logits[i]);
  }

  std::sort(indexed_logits.begin(), indexed_logits.end(),
            [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
              return a.second > b.second;
            });

  // Convert to probabilities
  std::vector<float> values;
  for (const auto& pair : indexed_logits) {
    values.push_back(pair.second);
  }
  std::vector<float> probs = Softmax(values);

  // Find cumulative probability cutoff
  float cumsum = 0.0f;
  size_t cutoff = 0;
  for (size_t i = 0; i < probs.size(); ++i) {
    cumsum += probs[i];
    cutoff = i + 1;
    if (cumsum >= p) break;
  }

  // Sample from the truncated distribution
  std::vector<float> truncated_probs(probs.begin(), probs.begin() + cutoff);
  float sum =
      std::accumulate(truncated_probs.begin(), truncated_probs.end(), 0.0f);
  for (float& prob : truncated_probs) {
    prob /= sum;
  }

  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> dist(truncated_probs.begin(),
                                    truncated_probs.end());

  int selected_idx = dist(gen);
  return {{indexed_logits[selected_idx].first},
          {truncated_probs[selected_idx]}};
}

// BeamSearchScorer implementation
BeamSearchScorer::BeamSearchScorer(const DecodeConfig& config)
    : BaseBeamScorer(config),
      min_log_prob_(0.0f),
      top_score_(std::numeric_limits<float>::lowest()) {}

void BeamSearchScorer::UpdateScore(BeamState& beam, float log_prob) {
  if (log_prob < min_log_prob_) {
    min_log_prob_ = log_prob;
  }

  beam.score += log_prob;

  if (beam.score > top_score_) {
    top_score_ = beam.score;
    top_beam_ = std::make_shared<BeamState>(beam);
  }
}

void BeamSearchScorer::FinalizeScore(BeamState& beam) {
  beam.score = beam.score - beam.accumulated_normalization;
  ApplyLengthPenalty(beam);
}

void BeamSearchScorer::Reset() {
  min_log_prob_ = 0.0f;
  top_score_ = std::numeric_limits<float>::lowest();
  top_beam_.reset();
}

std::vector<BeamState> BeamSearchScorer::SelectBeams(
    const std::vector<BeamState>& active_beams,
    const std::vector<BeamState>& completed_beams) {
  int k = config_.num_beams - static_cast<int>(completed_beams.size());
  std::vector<BeamState> selections;

  // Parse each beam to select the next candidates
  for (const auto& beam : active_beams) {
    // Sample multiple tokens for this beam - in Python this calls
    // beam.sample_logits(len(completed_beams)) which returns top_tokens,
    // top_values
    auto sampled_tokens = SampleLogits(LogitsData({0.1f, 0.2f, 0.7f}), k);

    for (size_t i = 0; i < sampled_tokens.first.size(); ++i) {
      BeamState new_beam = beam;  // Clone the beam
      new_beam.last_token = sampled_tokens.first[i];
      UpdateScore(new_beam, std::log(sampled_tokens.second[i]));
      selections.push_back(new_beam);
    }
  }

  // Ensure we have enough beams to fill the num_beams requirement
  if (static_cast<int>(selections.size()) < k && top_beam_) {
    int beams_to_add = k - static_cast<int>(selections.size());
    for (int i = 0; i < beams_to_add; ++i) {
      BeamState new_beam = *top_beam_;  // Clone top beam
      selections.push_back(new_beam);
    }
  }

  // Select top-k beams and apply normalization
  auto result = ScoreBeams(selections, k);
  Reset();
  return result;
}

std::vector<BeamState> BeamSearchScorer::ScoreBeams(
    const std::vector<BeamState>& beams, int k) {
  std::vector<BeamState> scored_beams = beams;

  // Sort beams by score in descending order
  std::sort(
      scored_beams.begin(), scored_beams.end(),
      [](const BeamState& a, const BeamState& b) { return a.score > b.score; });

  // Take top-k beams
  int actual_k = std::min(k, static_cast<int>(scored_beams.size()));
  std::vector<BeamState> result(scored_beams.begin(),
                                scored_beams.begin() + actual_k);

  // Apply normalization to all selected beams
  for (auto& beam : result) {
    NormalizeScore(beam, min_log_prob_);
  }

  return result;
}

std::pair<std::vector<int>, std::vector<float>> BeamSearchScorer::SampleLogits(
    const LogitsData& logits_data, int num_beams) {
  std::vector<float> processed_logits =
      ApplyTemperature(logits_data.logits, config_.temperature);
  std::vector<float> probs = Softmax(processed_logits);

  // For beam search, we want deterministic top-k selection
  std::vector<std::pair<int, float>> indexed_probs;
  for (size_t i = 0; i < probs.size(); ++i) {
    indexed_probs.emplace_back(static_cast<int>(i), probs[i]);
  }

  std::sort(indexed_probs.begin(), indexed_probs.end(),
            [](const std::pair<int, float>& a, const std::pair<int, float>& b) {
              return a.second > b.second;
            });

  std::vector<int> tokens;
  std::vector<float> selected_probs;
  int actual_k = std::min(num_beams, static_cast<int>(indexed_probs.size()));

  for (int i = 0; i < actual_k; ++i) {
    tokens.push_back(indexed_probs[i].first);
    selected_probs.push_back(indexed_probs[i].second);
  }

  return {tokens, selected_probs};
}

void BeamSearchScorer::NormalizeScore(BeamState& beam, float min_log_prob) {
  beam.accumulated_normalization += std::abs(min_log_prob);
}

// Factory function
std::unique_ptr<BaseBeamScorer> CreateBeamScorer(const DecodeConfig& config) {
  if (config.use_beam_search && config.num_beams > 1) {
    return std::make_unique<BeamSearchScorer>(config);
  } else {
    return std::make_unique<DefaultScorer>(config);
  }
}

}  // namespace shortfin::llm
