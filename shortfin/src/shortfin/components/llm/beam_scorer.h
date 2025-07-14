// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_COMPONENTS_LLM_BEAM_SCORER_H
#define SHORTFIN_COMPONENTS_LLM_BEAM_SCORER_H

#include <functional>
#include <memory>
#include <optional>
#include <vector>
// #include "shortfin/components/llm/beam_processor.h"
#include "shortfin/support/api.h"

namespace shortfin::llm {

// Forward declaration for exec request
class LlmInferenceExecRequest;

// Configuration for decoding
struct SHORTFIN_API DecodeConfig {
  int num_beams = 1;
  float temperature = 1.0f;
  int top_k = -1;       // -1 means no top_k filtering
  float top_p = -1.0f;  // -1 means no top_p filtering
  bool use_beam_search = false;
  int eos_token_id = 2;
  float length_penalty = 1.0f;
};

// Logits data structure for processing
struct SHORTFIN_API LogitsData {
  std::vector<float> logits;
  std::vector<int> indices;  // Optional pre-selected indices
  int vocab_size;

  LogitsData(const std::vector<float>& logits_vec, int vocab_size = -1)
      : logits(logits_vec), vocab_size(vocab_size) {
    if (vocab_size == -1) {
      this->vocab_size = logits.size();
    }
  }
};

// Beam state structure representing a single beam in the search
struct SHORTFIN_API BeamState {
  std::shared_ptr<LlmInferenceExecRequest> exec_req;
  float score = 0.0f;
  float accumulated_normalization = 0.0f;
  std::optional<int> last_token;

  BeamState() = default;
  BeamState(std::shared_ptr<LlmInferenceExecRequest> req)
      : exec_req(std::move(req)) {}
};

// Base class for beam scoring strategies
class SHORTFIN_API BaseBeamScorer {
 public:
  BaseBeamScorer(const DecodeConfig& config) : config_(config) {}
  virtual ~BaseBeamScorer() = default;

  // Update beam score with a new value
  virtual void UpdateScore(BeamState& beam, float value) = 0;

  // Finalize score after beam completion
  virtual void FinalizeScore(BeamState& beam) = 0;

  // Reset scorer state for next iteration
  virtual void Reset() = 0;

  // Core selection method for beams
  virtual std::vector<BeamState> SelectBeams(
      const std::vector<BeamState>& active_beams,
      const std::vector<BeamState>& completed_beams) = 0;

 protected:
  DecodeConfig config_;

  // Helper methods for scoring
  void ApplyLengthPenalty(BeamState& beam);
  std::vector<float> Softmax(const std::vector<float>& logits);
  std::vector<float> LogSoftmax(const std::vector<float>& logits);
  std::vector<float> ApplyTemperature(const std::vector<float>& logits,
                                      float temperature);
};

// Default scorer for simple greedy/sampling strategies
class SHORTFIN_API DefaultScorer : public BaseBeamScorer {
 public:
  DefaultScorer(const DecodeConfig& config);

  void UpdateScore(BeamState& beam, float value) override;
  void FinalizeScore(BeamState& beam) override;
  void Reset() override;

  // Core selection method
  std::vector<BeamState> SelectBeams(
      const std::vector<BeamState>& active_beams,
      const std::vector<BeamState>& completed_beams) override;

 private:
  // Sample a single token from logits
  int SampleToken(const LogitsData& logits_data);

  // Greedy selection
  int SelectGreedy(const std::vector<float>& logits);

  // Top-k sampling
  std::pair<std::vector<int>, std::vector<float>> SampleTopK(
      const std::vector<float>& logits, int k);

  // Top-p sampling
  std::pair<std::vector<int>, std::vector<float>> SampleTopP(
      const std::vector<float>& logits, float p);
};

// Beam search scorer for multi-beam scenarios
class SHORTFIN_API BeamSearchScorer : public BaseBeamScorer {
 public:
  BeamSearchScorer(const DecodeConfig& config);

  void UpdateScore(BeamState& beam, float value) override;
  void FinalizeScore(BeamState& beam) override;
  void Reset() override;

  // Core selection method
  std::vector<BeamState> SelectBeams(
      const std::vector<BeamState>& active_beams,
      const std::vector<BeamState>& completed_beams) override;

 private:
  float min_log_prob_;
  float top_score_;
  std::shared_ptr<BeamState> top_beam_;

  // Score and rank beams
  std::vector<BeamState> ScoreBeams(const std::vector<BeamState>& beams, int k);

  // Sample multiple tokens for beam expansion
  std::pair<std::vector<int>, std::vector<float>> SampleLogits(
      const LogitsData& logits_data, int num_beams);

  // Normalize scores based on min log probability
  void NormalizeScore(BeamState& beam, float min_log_prob);
};

// Factory function to create appropriate scorer
SHORTFIN_API std::unique_ptr<BaseBeamScorer> CreateBeamScorer(
    const DecodeConfig& config);

}  // namespace shortfin::llm

#endif  // SHORTFIN_COMPONENTS_LLM_BEAM_SCORER_H
