// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/support/iree_helpers.h"

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <atomic>
#include <unordered_map>

#include "shortfin/support/iree_concurrency.h"
#include "shortfin/support/logging.h"

namespace shortfin::iree {

namespace detail {

#if SHORTFIN_IREE_LOG_RC

slim_mutex log_mutex;
std::unordered_map<std::string, int> app_ref_counts;

void LogIREERetain(const char *type_name, void *ptr) {
  slim_mutex_lock_guard g(log_mutex);
  std::string key = fmt::format("{}({})", type_name, ptr);
  int &rc = app_ref_counts[key];
  rc += 1;
  if (rc == 1) {
    logging::info("IREE new {}", key);
  } else {
    logging::info("IREE retain {} = {}", key, rc);
  }
}

void LogIREERelease(const char *type_name, void *ptr) {
  slim_mutex_lock_guard g(log_mutex);
  std::string key = fmt::format("{}({})", type_name, ptr);
  int &rc = app_ref_counts[key];
  rc -= 1;
  if (rc == 0) {
    logging::info("IREE delete {}", key);
  } else {
    logging::info("IREE release {} = {}", key, rc);
  }
}

void LogIREESteal(const char *type_name, void *ptr) {
  slim_mutex_lock_guard g(log_mutex);
  std::string key = fmt::format("{}({})", type_name, ptr);
  int &rc = app_ref_counts[key];
  rc += 1;
  if (rc == 1) {
    logging::info("IREE steal {}", key);
  } else {
    logging::info("IREE retain {} = {}", key, rc);
  }
}

void SHORTFIN_API LogLiveRefs() {
  slim_mutex_lock_guard g(log_mutex);
  bool logged_banner = false;
  for (auto &it : app_ref_counts) {
    if (it.second == 0) continue;
    if (it.second < 0) {
      logging::error("Shortfin IREE negative reference count: {} = {}",
                     it.first, it.second);
      continue;
    }
    if (!logged_banner) {
      logged_banner = true;
      logging::warn("Shortfin visible live IREE refs remain:");
    }
    logging::warn("  Live IREE ref {} = {}", it.first, it.second);
  }
}

#endif

}  // namespace detail

error::error(std::string message, iree_status_t failing_status)
    : code_(iree_status_code(failing_status)),
      message_(std::move(message)),
      failing_status_(failing_status) {
  message_.append(": ");
  AppendStatusMessage();
}

error::error(iree_status_t failing_status) : failing_status_(failing_status) {
  AppendStatusMessage();
}

void error::AppendStatusMessage() {
  iree_allocator_t allocator = iree_allocator_system();
  char *status_buffer = nullptr;
  iree_host_size_t length = 0;
  if (iree_status_to_string(failing_status_, &allocator, &status_buffer,
                            &length)) {
    message_.append(status_buffer, length);
    iree_allocator_free(allocator, status_buffer);
  } else {
    message_.append(": <<could not print iree_status_t>>");
  }
}

std::string DebugPrintSemaphoreList(iree_hal_semaphore_list_t &sl) {
  std::vector<std::string> parts;
  for (unsigned i = 0; i < sl.count; ++i) {
    parts.push_back(fmt::format("{}@{}", static_cast<void *>(sl.semaphores[i]),
                                sl.payload_values[i]));
  }
  return fmt::format("({})", fmt::join(parts, ", "));
}

}  // namespace shortfin::iree
