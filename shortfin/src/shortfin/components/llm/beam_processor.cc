// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/components/llm/beam_processor.h"

#include <algorithm>
#include <future>
#include <iostream>
#include <thread>
#include <vector>

namespace shortfin::llm {

// InferenceRequest implementation
std::shared_ptr<InferenceRequest> InferenceRequest::CopyRequest(
    const std::shared_ptr<InferenceRequest>& source) {
  auto copy = std::make_shared<InferenceRequest>();
  copy->instance_id = source->instance_id;
  copy->input_token_ids = source->input_token_ids;
  copy->start_position = source->start_position;
  copy->prompt_length = source->prompt_length;
  return copy;
}

void InferenceRequest::UpdateWithToken(int token) {
  input_token_ids.push_back(token);
  start_position++;
}

void InferenceRequest::FreeCachePages() {
  // TODO: Implement cache page freeing
  // Add page ids to a list to be freed later
}

// BeamProcessor implementation
BeamProcessor::BeamProcessor(int eos_token_id, int num_beams,
                             BeamSelectionCallback callback)
    : eos_token_id_(eos_token_id),
      num_beams_(num_beams),
      selection_callback_(std::move(callback)) {}

ProcessBeamsResult BeamProcessor::ProcessBeams(
    const std::vector<BeamState>& current_active_beams,
    const std::vector<BeamState>& current_completed_beams) {
  ProcessBeamsResult result;

  try {
    // Step 1: Call selection callback to get beam selections
    std::vector<BeamState> beam_selections =
        selection_callback_(current_active_beams, current_completed_beams);

    // Step 2: Initialize tracking structures
    std::unordered_map<std::string, std::shared_ptr<InferenceRequest>>
        visited_reqs;
    std::vector<BeamState> active_beams;
    std::vector<BeamState> completed_beams;
    std::unordered_set<std::shared_ptr<InferenceRequest>> active_reqs;
    std::unordered_set<std::shared_ptr<InferenceRequest>> completed_reqs;

    // Step 3: Process beam selections
    ProcessBeamSelection(beam_selections, active_beams, completed_beams,
                         active_reqs, completed_reqs, visited_reqs);

    // Step 4: Update beam states and handle cache cleanup
    UpdateBeamStates(active_beams, completed_beams, completed_reqs);

    // Step 5: Cleanup unused requests
    CleanupUnusedRequests(current_active_beams, active_reqs, completed_reqs);

    // Step 6: Set results
    result.active_beams = std::move(active_beams);
    result.completed_beams = std::move(completed_beams);
    result.success = true;

  } catch (const std::exception& e) {
    result.success = false;
    result.error_message = e.what();
    std::cerr << "Error in ProcessBeams: " << e.what() << std::endl;
  }

  return result;
}

void BeamProcessor::ProcessBeamSelection(
    const std::vector<BeamState>& beam_selections,
    std::vector<BeamState>& active_beams,
    std::vector<BeamState>& completed_beams,
    std::unordered_set<std::shared_ptr<InferenceRequest>>& active_reqs,
    std::unordered_set<std::shared_ptr<InferenceRequest>>& completed_reqs,
    std::unordered_map<std::string, std::shared_ptr<InferenceRequest>>&
        visited_reqs) {
  // Process each beam selection - equivalent to Python loop:
  // for i in range(len(beam_selections)):
  //     beam = beam_selections[i]
  //     new_req, token = beam.exec_req, beam.last_token
  for (const auto& beam : beam_selections) {
    auto new_req = beam.exec_req;
    int token = beam.last_token;

    // Handle request deduplication
    auto visited_it = visited_reqs.find(new_req->instance_id);
    if (visited_it != visited_reqs.end()) {
      // Copy the visited request and update beam
      auto visited_req = visited_it->second;
      new_req = InferenceRequest::CopyRequest(visited_req);

      // Create new beam state with copied request
      BeamState updated_beam = beam;
      updated_beam.exec_req = new_req;
    }

    // Track the request
    visited_reqs[new_req->instance_id] = new_req;

    // Categorize beam based on token
    if (ShouldCompleteBeam(token, eos_token_id_)) {
      BeamState completed_beam = beam;
      completed_beam.exec_req = new_req;
      completed_beams.push_back(completed_beam);
      completed_reqs.insert(new_req);
    } else {
      BeamState active_beam = beam;
      active_beam.exec_req = new_req;
      active_beams.push_back(active_beam);
      active_reqs.insert(new_req);
    }
  }
}

void BeamProcessor::UpdateBeamStates(
    const std::vector<BeamState>& active_beams,
    const std::vector<BeamState>& completed_beams,
    const std::unordered_set<std::shared_ptr<InferenceRequest>>&
        completed_reqs) {
  // Update all beams (both active and completed)
  auto update_beam = [&](const BeamState& beam) {
    if (beam.last_token != -1) {
      beam.exec_req->UpdateWithToken(beam.last_token);
    }

    // Free cache pages for completed requests
    if (completed_reqs.find(beam.exec_req) != completed_reqs.end()) {
      beam.exec_req->FreeCachePages();
    }
  };

  // Apply updates to all beams
  std::for_each(completed_beams.begin(), completed_beams.end(), update_beam);
  std::for_each(active_beams.begin(), active_beams.end(), update_beam);
}

void BeamProcessor::CleanupUnusedRequests(
    const std::vector<BeamState>& original_active_beams,
    const std::unordered_set<std::shared_ptr<InferenceRequest>>& active_reqs,
    const std::unordered_set<std::shared_ptr<InferenceRequest>>&
        completed_reqs) {
  // Cleanup requests that are no longer active or completed
  for (const auto& beam : original_active_beams) {
    auto req = beam.exec_req;
    bool is_active = active_reqs.find(req) != active_reqs.end();
    bool is_completed = completed_reqs.find(req) != completed_reqs.end();

    if (!is_active && !is_completed) {
      req->FreeCachePages();
    }
  }
}

// Static helper functions
bool BeamProcessor::ShouldCompleteBeam(int token, int eos_token_id) {
  return token == eos_token_id;
}

void BeamProcessor::CleanupBeamResources(const std::vector<BeamState>& beams) {
  for (const auto& beam : beams) {
    if (beam.exec_req) {
      beam.exec_req->FreeCachePages();
    }
  }
}

// Parallel processing utilities
namespace parallel {

std::vector<ProcessBeamsResult> ProcessBeamGroupsParallel(
    const std::vector<std::unique_ptr<BeamProcessor>>& processors,
    const std::vector<std::vector<BeamState>>& active_beams_batch,
    const std::vector<std::vector<BeamState>>& completed_beams_batch) {
  if (processors.size() != active_beams_batch.size() ||
      processors.size() != completed_beams_batch.size()) {
    throw std::invalid_argument(
        "Mismatched batch sizes for parallel processing");
  }

  std::vector<std::future<ProcessBeamsResult>> futures;
  std::vector<std::thread> threads;
  std::vector<ProcessBeamsResult> results;

  // Submit tasks to separate threads
  for (size_t i = 0; i < processors.size(); ++i) {
    auto promise = std::make_shared<std::promise<ProcessBeamsResult>>();
    futures.push_back(promise->get_future());

    threads.emplace_back([&processors, &active_beams_batch,
                          &completed_beams_batch, i, promise]() {
      try {
        auto result = processors[i]->ProcessBeams(active_beams_batch[i],
                                                  completed_beams_batch[i]);
        promise->set_value(result);
      } catch (...) {
        promise->set_exception(std::current_exception());
      }
    });
  }

  // Wait for all threads to complete
  for (auto& thread : threads) {
    thread.join();
  }

  // Collect results
  results.reserve(futures.size());
  for (auto& future : futures) {
    results.push_back(future.get());
  }

  return results;
}

}  // namespace parallel

}  // namespace shortfin::llm
