// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/components/llm/beam_processor.h"

#include <gtest/gtest.h>

#include <memory>

namespace shortfin::llm {

class BeamProcessorTest : public testing::Test {
 protected:
  void SetUp() override {
    // Mock selection callback that simulates beam search selection
    selection_callback_ = [this](const std::vector<BeamState>& active_beams,
                                 const std::vector<BeamState>& completed_beams)
        -> std::vector<BeamState> {
      std::vector<BeamState> selections;

      // Simple mock logic: for each active beam, create a selection with
      // incremented token
      for (const auto& beam : active_beams) {
        BeamState selection = beam;
        selection.last_token = next_token_id_++;
        selections.push_back(selection);
      }

      return selections;
    };

    processor_ = std::make_unique<BeamProcessor>(eos_token_id_, num_beams_,
                                                 selection_callback_);
  }

  BeamSelectionCallback selection_callback_;
  std::unique_ptr<BeamProcessor> processor_;
  int eos_token_id_ = 2;  // Mock EOS token ID
  int num_beams_ = 2;
  int next_token_id_ = 10;  // Starting token ID for mock selections
};

TEST_F(BeamProcessorTest, ProcessBeamsBasic) {
  // Create mock inference requests
  auto req1 = std::make_shared<InferenceRequest>();
  req1->instance_id = "req_1";
  req1->input_token_ids = {1, 5, 8};
  req1->prompt_length = 2;

  auto req2 = std::make_shared<InferenceRequest>();
  req2->instance_id = "req_2";
  req2->input_token_ids = {1, 3, 7};
  req2->prompt_length = 2;

  // Create active beams
  std::vector<BeamState> active_beams;
  active_beams.emplace_back(req1);
  active_beams.emplace_back(req2);

  std::vector<BeamState> completed_beams;

  // Process beams
  auto result = processor_->ProcessBeams(active_beams, completed_beams);

  // Verify results
  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.active_beams.size(), 2);
  EXPECT_EQ(result.completed_beams.size(), 0);

  // Verify tokens were updated
  for (const auto& beam : result.active_beams) {
    EXPECT_GE(beam.last_token, 10);  // Should have received tokens >= 10
    EXPECT_EQ(beam.exec_req->input_token_ids.size(),
              4);  // Original 3 + 1 new token
  }
}

TEST_F(BeamProcessorTest, ProcessBeamsWithCompletion) {
  // Set up selection callback that returns EOS token for first beam
  selection_callback_ = [this](const std::vector<BeamState>& active_beams,
                               const std::vector<BeamState>& completed_beams)
      -> std::vector<BeamState> {
    std::vector<BeamState> selections;

    for (size_t i = 0; i < active_beams.size(); ++i) {
      BeamState selection = active_beams[i];
      // First beam gets EOS token, others get regular tokens
      selection.last_token = (i == 0) ? eos_token_id_ : next_token_id_++;
      selections.push_back(selection);
    }

    return selections;
  };

  processor_ = std::make_unique<BeamProcessor>(eos_token_id_, num_beams_,
                                               selection_callback_);

  // Create active beams
  auto req1 = std::make_shared<InferenceRequest>();
  req1->instance_id = "req_1";
  req1->input_token_ids = {1, 5, 8};

  auto req2 = std::make_shared<InferenceRequest>();
  req2->instance_id = "req_2";
  req2->input_token_ids = {1, 3, 7};

  std::vector<BeamState> active_beams;
  active_beams.emplace_back(req1);
  active_beams.emplace_back(req2);

  std::vector<BeamState> completed_beams;

  // Process beams
  auto result = processor_->ProcessBeams(active_beams, completed_beams);

  // Verify results
  EXPECT_TRUE(result.success);
  EXPECT_EQ(result.active_beams.size(), 1);     // One beam should be active
  EXPECT_EQ(result.completed_beams.size(), 1);  // One beam should be completed

  // Verify the completed beam has EOS token
  EXPECT_EQ(result.completed_beams[0].last_token, eos_token_id_);
}

TEST_F(BeamProcessorTest, HelperFunctions) {
  // Test ShouldCompleteBeam
  EXPECT_TRUE(BeamProcessor::ShouldCompleteBeam(eos_token_id_, eos_token_id_));
  EXPECT_FALSE(BeamProcessor::ShouldCompleteBeam(5, eos_token_id_));

  // Test CleanupBeamResources (shouldn't crash)
  std::vector<BeamState> test_beams;
  auto req = std::make_shared<InferenceRequest>();
  test_beams.emplace_back(req);

  BeamProcessor::CleanupBeamResources(test_beams);  // Should not crash
}

}  // namespace shortfin::llm
