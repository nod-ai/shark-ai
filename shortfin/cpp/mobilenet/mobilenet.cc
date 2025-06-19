// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mobilenet.h"

#include "shortfin/local/async.h"
#include "shortfin/local/messaging.h"
#include "shortfin/local/worker.h"

namespace shortfin {
namespace cpp {
void InferenceProcess::ScheduleOnWorker() {
  local::Worker *worker = local::Worker::GetCurrent();
  worker->CallThreadsafe([this]() { this->Run(); });
  Terminate();
}

local::Coroutine<local::VoidFuture> InferenceProcess::Run() {
  while (true) {
    auto ref = co_await request_reader_.Read();
    InferenceRequest &request = *(static_cast<InferenceRequest *>(ref.get()));
  }
}
}  // namespace cpp
}  // namespace shortfin
