// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// WARNING: This is experimental code, written as a POC for the
// C++-only implementation of an app using the Shortfin API.
#ifndef SHORTFIN_CPP_MOBILENET_H
#define SHORTFIN_CPP_MOBILENET_H

#include <filesystem>
#include <memory>

#include "shortfin/array/array.h"
#include "shortfin/array/dtype.h"
#include "shortfin/local/async.h"
#include "shortfin/local/fiber.h"
#include "shortfin/local/messaging.h"
#include "shortfin/local/process.h"
#include "shortfin/local/program.h"
#include "shortfin/local/system.h"
#include "shortfin/local/systems/amdgpu.h"

namespace fs = std::filesystem;

namespace shortfin {
namespace cpp {

class InferenceRequest : public local::Message {
 public:
  InferenceRequest(std::vector<float> rawImageData) : raw_data_(rawImageData) {}

  std::vector<float> rawImageData() { return raw_data_; }

 private:
  std::vector<float> raw_data_;
};

class InferenceProcess : public local::Process {
  using device_array = shortfin::array::device_array;

 public:
  using local::detail::BaseProcess::Launch;
  InferenceProcess(std::shared_ptr<local::Fiber> fiber, local::Program program,
                   local::Queue &requestQueue, std::span<size_t> dims)
      : local::Process(fiber),
        program_(program),
        request_reader_(std::move(local::QueueReader(requestQueue))),
        device_(fiber->device(0)),
        input_(std::move(
            device_array::for_device(device_, std::span<size_t>{dims},
                                     shortfin::array::DType::float32()))),
        host_staging_(input_.for_transfer()) {}

  void ScheduleOnWorker();

  // void dump_result() { std::cerr << *result_.contents_to_s() << std::endl; }

  local::Coroutine<> Run();

 private:
  local::Program program_;
  local::QueueReader request_reader_;
  local::ScopedDevice device_;
  shortfin::array::device_array input_;
  shortfin::array::device_array host_staging_;
};

class Mobilenet {
 public:
  Mobilenet(const fs::path &modulePath)
      : system_(local::systems::AMDGPUSystemBuilder().CreateSystem()),
        module_(local::ProgramModule::Load(*system_, modulePath)) {
    queue_ = system_->CreateQueue();
    processes_per_worker_ = 4;
    processes_.reserve(processes_per_worker_);
  }

  void Shutdown() {
    system_->Shutdown();
    system_.reset();
  }

  local::Coroutine<> Run(std::vector<float> data);
  local::System &system() { return *system_; }

 private:
  local::SystemPtr system_;
  local::ProgramModule module_;
  local::QueuePtr queue_;
  int processes_per_worker_;
  std::vector<InferenceProcess *> processes_;
};
}  // namespace cpp
}  // namespace shortfin

#endif
