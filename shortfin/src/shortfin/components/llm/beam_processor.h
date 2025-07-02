// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_COMPONENTS_LLM_BEAM_PROCESSOR_H
#define SHORTFIN_COMPONENTS_LLM_BEAM_PROCESSOR_H

#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "shortfin/support/api.h"

namespace shortfin::llm {

// Forward declarations
struct BeamState;
struct InferenceRequest;

// Represents the state of a single beam in beam search
struct SHORTFIN_API BeamState {
  std::shared_ptr<InferenceRequest> exec_req;
  int last_token = -1;
  float score = 0.0f;
  float accumulated_normalization = 0.0f;

  BeamState() = default;
  BeamState(std::shared_ptr<InferenceRequest> req) : exec_req(std::move(req)) {}
};

// Simplified representation of an inference request for C++ processing
struct SHORTFIN_API InferenceRequest {
  std::string instance_id;
  std::vector<int> input_token_ids;
  int start_position = 0;
  int prompt_length = 0;

  // Simplified copy method - in real implementation this would handle
  // more complex state like cache pages, device allocations, etc.
  static std::shared_ptr<InferenceRequest> CopyRequest(
      const std::shared_ptr<InferenceRequest>& source);

  // Simplified method to update request state
  void UpdateWithToken(int token);

  // Simplified cache cleanup - would interface with actual cache system
  void FreeCachePages();
};

// Result structure for process_beams operation
struct SHORTFIN_API ProcessBeamsResult {
  std::vector<BeamState> active_beams;
  std::vector<BeamState> completed_beams;
  bool success = true;
  std::string error_message;
};

// Selection callback type - matches Python interface
using BeamSelectionCallback = std::function<std::vector<BeamState>(
    const std::vector<BeamState>& active_beams,
    const std::vector<BeamState>& completed_beams)>;

// Main beam processor class
class SHORTFIN_API BeamProcessor {
 public:
  BeamProcessor(int eos_token_id, int num_beams,
                BeamSelectionCallback callback);
  ~BeamProcessor() = default;

  // Core process_beams function - equivalent to Python implementation
  ProcessBeamsResult ProcessBeams(
      const std::vector<BeamState>& current_active_beams,
      const std::vector<BeamState>& current_completed_beams);

  // Getters
  int eos_token_id() const { return eos_token_id_; }
  int num_beams() const { return num_beams_; }

  // Helper functions
  static bool ShouldCompleteBeam(int token, int eos_token_id);
  static void CleanupBeamResources(const std::vector<BeamState>& beams);

 private:
  int eos_token_id_;
  int num_beams_;
  BeamSelectionCallback selection_callback_;

  // Helper methods for the core algorithm
  void ProcessBeamSelection(
      const std::vector<BeamState>& beam_selections,
      std::vector<BeamState>& active_beams,
      std::vector<BeamState>& completed_beams,
      std::unordered_set<std::shared_ptr<InferenceRequest>>& active_reqs,
      std::unordered_set<std::shared_ptr<InferenceRequest>>& completed_reqs,
      std::unordered_map<std::string, std::shared_ptr<InferenceRequest>>&
          visited_reqs);

  void UpdateBeamStates(
      const std::vector<BeamState>& active_beams,
      const std::vector<BeamState>& completed_beams,
      const std::unordered_set<std::shared_ptr<InferenceRequest>>&
          completed_reqs);

  void CleanupUnusedRequests(
      const std::vector<BeamState>& original_active_beams,
      const std::unordered_set<std::shared_ptr<InferenceRequest>>& active_reqs,
      const std::unordered_set<std::shared_ptr<InferenceRequest>>&
          completed_reqs);
};

// Utility functions for parallel processing
namespace parallel {

// Process multiple beam groups in parallel using blocking executor
SHORTFIN_API std::vector<ProcessBeamsResult> ProcessBeamGroupsParallel(
    const std::vector<std::unique_ptr<BeamProcessor>>& processors,
    const std::vector<std::vector<BeamState>>& active_beams_batch,
    const std::vector<std::vector<BeamState>>& completed_beams_batch);

}  // namespace parallel

}  // namespace shortfin::llm

#endif  // SHORTFIN_COMPONENTS_LLM_BEAM_PROCESSOR_H
