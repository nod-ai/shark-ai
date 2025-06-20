// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "mobilenet.h"

#include <variant>

#include "fmt/format.h"
#include "shortfin/array/array.h"
#include "shortfin/local/async.h"
#include "shortfin/local/messaging.h"
#include "shortfin/local/program.h"
#include "shortfin/local/program_interfaces.h"
#include "shortfin/local/worker.h"

namespace {
namespace local = shortfin::local;
namespace array = shortfin::array;

std::variant<array::device_array, array::storage> GetResultFromInvocationRef(
    local::ProgramInvocation::Ptr &invocation, ::iree::vm::opaque_ref ref,
    local::CoarseInvocationTimelineImporter *timeline_importer) {
  auto type = ref.get()->type;
  if (local::ProgramInvocationMarshalableFactory::invocation_marshalable_type<
          array::device_array>() == type) {
    return local::ProgramInvocationMarshalableFactory::
        CreateFromInvocationResultRef<array::device_array>(
            invocation.get(), timeline_importer, std::move(ref));
  } else if (local::ProgramInvocationMarshalableFactory::
                 invocation_marshalable_type<array::storage>() == type) {
    // storage
    return local::ProgramInvocationMarshalableFactory::
        CreateFromInvocationResultRef<array::storage>(
            invocation.get(), timeline_importer, std::move(ref));
  }

  throw std::invalid_argument(
      fmt::format("Could not marshal ref type {}",
                  shortfin::to_string_view(iree_vm_ref_type_name(type))));
}  // namespace shortfin::array
}  // namespace

namespace shortfin {
namespace cpp {
void InferenceProcess::ScheduleOnWorker() {
  local::Worker &worker = fiber()->worker();
  auto coro = Run();
  worker.CallThreadsafe([&]() { coro.resume(); });
  coro.wait();
}

local::Coroutine<> InferenceProcess::Run() {
  while (auto ref = co_await request_reader_.Read()) {
    auto host_staging = input_.for_transfer();
    InferenceRequest &request =
        *(static_cast<InferenceRequest *>(ref.release()));
    auto raw_data = request.rawImageData();
    {
      auto map = host_staging.typed_data_w<float>();
      std::copy(raw_data.begin(), raw_data.end(), map.begin());
    }

    input_.copy_from(host_staging);

    local::ProgramFunction function =
        *program_.LookupFunction("module.torch-jit-export");

    local::ProgramInvocation::Ptr invocation =
        function.CreateInvocation(fiber(), local::ProgramIsolation::PER_FIBER);

    input_.AddAsInvocationArgument(invocation.get(),
                                   local::ProgramResourceBarrier::DEFAULT);

    auto &ptr =
        co_await local::ProgramInvocation::Invoke(std::move(invocation));
    local::CoarseInvocationTimelineImporter::Options options;
    options.assume_no_alias = true;
    local::CoarseInvocationTimelineImporter timeline_importer(ptr.get(),
                                                              options);
    auto result_arr = std::get<array::device_array>(GetResultFromInvocationRef(
        ptr, ptr->result_ref(0), &timeline_importer));

    auto result = result_arr.for_transfer();
    result.copy_from(result_arr);
    co_await device_.OnSync();
    std::cerr << *result.contents_to_s() << "\n";
  }
  Terminate();
  co_return;
}

local::Coroutine<> Mobilenet::Run(std::vector<float> data) {
  auto device_span = [this]() -> std::vector<const local::Device *> {
    std::vector<const local::Device *> devices;
    for (auto device : system_->devices()) {
      devices.push_back(device);
    }
    return devices;
  }();

  local::Program::Options options = local::Program::Options();
  options.devices = device_span;
  options.isolation = local::ProgramIsolation::PER_FIBER;
  options.trace_execution = false;

  auto module_array = std::array<local::ProgramModule, 1>{module_};
  local::Program program = local::Program::Load(
      std::span<local::ProgramModule>{module_array}, std::move(options));

  std::array<size_t, 4> dims{1, 3, 224, 224};

  int count = 1;
  auto writer = local::QueueWriter(*queue_);

  while (count--) {
    InferenceRequest request = InferenceRequest(data);
    local::Message::Ref ref = local::Message::Ref(request);
    writer.Write(ref);
  }

  writer.Close();

  for (size_t i = 0; i < processes_per_worker_; ++i) {
    local::Worker::Options options(system_->host_allocator(),
                                   fmt::format("inference-worker-{}", i));
    auto fiber = system_->CreateFiber(system_->CreateWorker(std::move(options)),
                                      system_->devices());

    InferenceProcess process =
        InferenceProcess(fiber, program, *queue_, std::span<size_t>{dims});

    processes_.push_back(&process);
    process.Launch();
  }

  for (InferenceProcess *process : processes_) {
    co_await process->OnTermination();
  }
  co_return;
}
}  // namespace cpp
}  // namespace shortfin
