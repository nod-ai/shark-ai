// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local/system.h"

#include <fmt/core.h>
#include <fmt/xchar.h>

#include "iree/hal/utils/allocators.h"
#include "shortfin/local/fiber.h"
#include "shortfin/support/logging.h"

namespace shortfin::local {

// -------------------------------------------------------------------------- //
// System
// -------------------------------------------------------------------------- //

System::System(iree_allocator_t host_allocator)
    : host_allocator_(host_allocator) {
  SHORTFIN_TRACE_SCOPE_NAMED("System::System");
  logging::construct("System", this);
  SHORTFIN_THROW_IF_ERROR(iree_vm_instance_create(IREE_VM_TYPE_CAPACITY_DEFAULT,
                                                  host_allocator_,
                                                  vm_instance_.for_output()));
  // Register types for builtin modules we know we want to handle.
  SHORTFIN_THROW_IF_ERROR(iree_hal_module_register_all_types(vm_instance_));
}

System::~System() {
  SHORTFIN_TRACE_SCOPE_NAMED("System::~System");
  logging::destruct("System", this);
  bool needs_shutdown = false;
  {
    iree::slim_mutex_lock_guard guard(lock_);
    if (initialized_ && !shutdown_) {
      needs_shutdown = true;
    }
  }
  if (needs_shutdown) {
    logging::warn(
        "Implicit Shutdown from System destructor. Please call Shutdown() "
        "explicitly for maximum stability.");
    Shutdown();
  }

  // Orderly destruction of heavy-weight objects.
  // Shutdown order is important so we don't leave it to field ordering.
  vm_instance_.reset();

  // Devices.
  devices_.clear();
  named_devices_.clear();
  retained_devices_.clear();

  // HAL drivers.
  hal_drivers_.clear();

  // If support for logging refs was compiled in, report now.
  iree::detail::LogLiveRefs();
}

void System::Shutdown() {
  SHORTFIN_TRACE_SCOPE_NAMED("System::Shutdown");
  // Stop workers.
  std::vector<Worker *> local_workers;
  {
    iree::slim_mutex_lock_guard guard(lock_);
    if (!initialized_ || shutdown_) return;
    shutdown_ = true;
    for (auto &w : workers_) {
      local_workers.push_back(w.get());
    }
  }

  // Worker drain and shutdown.
  for (auto &worker : local_workers) {
    worker->Kill();
  }
  for (auto &worker : local_workers) {
    if (worker->options().owned_thread) {
      worker->WaitForShutdown();
    }
  }
  blocking_executor_.Kill();
}

std::shared_ptr<Fiber> System::CreateFiber(Worker &worker,
                                           std::span<Device *const> devices) {
  iree::slim_mutex_lock_guard guard(lock_);
  AssertRunning();
  return std::make_shared<Fiber>(shared_ptr(), worker, devices);
}

void System::InitializeNodes(int node_count) {
  iree::slim_mutex_lock_guard guard(lock_);
  AssertNotInitialized();
  if (!nodes_.empty()) {
    throw std::logic_error("System::InitializeNodes called more than once");
  }
  nodes_.reserve(node_count);
  for (int i = 0; i < node_count; ++i) {
    nodes_.emplace_back(i);
  }
}

QueuePtr System::CreateQueue(Queue::Options options) {
  if (options.name.empty()) {
    // Fast, lock-free path for anonymous queue creation.
    return Queue::Create(std::move(options));
  } else {
    // Lock and allocate a named queue.
    iree::slim_mutex_lock_guard guard(lock_);
    AssertRunning();
    if (queues_by_name_.count(options.name) != 0) {
      throw std::invalid_argument(fmt::format(
          "Cannot create queue with duplicate name '{}'", options.name));
    }
    queues_.push_back(Queue::Create(std::move(options)));
    Queue *unowned_queue = queues_.back().get();
    queues_by_name_[unowned_queue->options().name] = unowned_queue;
    return *unowned_queue;
  }
}

QueuePtr System::named_queue(std::string_view name) {
  iree::slim_mutex_lock_guard guard(lock_);
  auto it = queues_by_name_.find(name);
  if (it == queues_by_name_.end()) {
    throw std::invalid_argument(fmt::format("Queue '{}' not found", name));
  }
  return *it->second;
}

void System::AddWorkerInitializer(std::function<void(Worker &)> initializer) {
  iree::slim_mutex_lock_guard guard(lock_);
  if (!workers_.empty()) {
    throw std::logic_error(
        "AddWorkerInitializer can only be called before workers are created");
  }
  worker_initializers_.push_back(std::move(initializer));
}

void System::InitializeWorker(Worker &worker) {
  for (auto &f : worker_initializers_) {
    f(worker);
  }
}

Worker &System::CreateWorker(Worker::Options options) {
  Worker *unowned_worker;
  {
    iree::slim_mutex_lock_guard guard(lock_);
    AssertRunning();
    if (options.name == std::string_view("__init__")) {
      throw std::invalid_argument(
          "Cannot create worker '__init__' (reserved name)");
    }
    if (workers_by_name_.count(options.name) != 0) {
      throw std::invalid_argument(fmt::format(
          "Cannot create worker with duplicate name '{}'", options.name));
    }
    workers_.push_back(std::make_unique<Worker>(std::move(options)));
    unowned_worker = workers_.back().get();
    workers_by_name_[unowned_worker->name()] = unowned_worker;
  }
  InitializeWorker(*unowned_worker);
  if (unowned_worker->options().owned_thread) {
    unowned_worker->Start();
  }
  return *unowned_worker;
}

Worker &System::init_worker() {
  iree::slim_mutex_lock_guard guard(lock_);
  AssertRunning();
  auto found_it = workers_by_name_.find("__init__");
  if (found_it != workers_by_name_.end()) {
    return *found_it->second;
  }

  // Create.
  Worker::Options options(host_allocator(), "__init__");
  options.owned_thread = false;
  workers_.push_back(std::make_unique<Worker>(std::move(options)));
  Worker *unowned_worker = workers_.back().get();
  workers_by_name_[unowned_worker->name()] = unowned_worker;
  InitializeWorker(*unowned_worker);
  return *unowned_worker;
}

void System::InitializeHalDriver(std::string_view moniker,
                                 iree::hal_driver_ptr driver) {
  iree::slim_mutex_lock_guard guard(lock_);
  AssertNotInitialized();
  auto &slot = hal_drivers_[moniker];
  if (slot) {
    throw std::logic_error(fmt::format(
        "Cannot register multiple hal drivers with moniker '{}'", moniker));
  }
  slot = std::move(driver);
}

void System::InitializeHalDevice(std::unique_ptr<Device> device) {
  iree::slim_mutex_lock_guard guard(lock_);
  AssertNotInitialized();
  auto device_name = device->name();
  auto [it, success] = named_devices_.try_emplace(device_name, device.get());
  if (!success) {
    throw std::logic_error(
        fmt::format("Cannot register Device '{}' multiple times", device_name));
  }
  devices_.push_back(device.get());
  retained_devices_.push_back(std::move(device));
}

Device *System::FindDeviceByName(std::string_view name) {
  auto it = named_devices_.find(name);
  if (it == named_devices_.end()) {
    return nullptr;
  }
  return it->second;
}

void System::FinishInitialization() {
  iree::slim_mutex_lock_guard guard(lock_);
  AssertNotInitialized();
  initialized_ = true;
}

int64_t System::AllocateProcess(detail::BaseProcess *p) {
  iree::slim_mutex_lock_guard guard(lock_);
  AssertRunning();
  int pid = next_pid_++;
  processes_by_pid_[pid] = p;
  return pid;
}

void System::DeallocateProcess(int64_t pid) {
  iree::slim_mutex_lock_guard guard(lock_);
  processes_by_pid_.erase(pid);
}

// -------------------------------------------------------------------------- //
// SystemBuilder
// -------------------------------------------------------------------------- //

void SystemBuilder::ConfigureAllocators(const std::vector<std::string> &specs,
                                        iree_hal_device_t *device,
                                        std::string_view device_debug_desc) {
  if (specs.empty()) return;
  std::vector<iree_string_view_t> spec_views;
  spec_views.reserve(specs.size());
  for (auto &spec : specs) {
    spec_views.push_back(to_iree_string_view(spec));
  }

  logging::info("Configure allocator {} = [{}]", device_debug_desc,
                fmt::join(specs, " ; "));

  SHORTFIN_THROW_IF_ERROR(iree_hal_configure_allocator_from_specs(
      spec_views.size(), spec_views.data(), device));
}

std::vector<std::string> SystemBuilder::GetConfigAllocatorSpecs(
    std::optional<std::string_view> specific_config_key) {
  std::optional<std::string_view> value;
  if (specific_config_key) {
    value = config_options().GetOption(*specific_config_key);
  }
  if (!value) {
    value = config_options().GetOption("allocators");
  }
  if (!value) {
    return {};
  }

  auto split_views = ConfigOptions::Split(*value, ';');
  return std::vector<std::string>(split_views.begin(), split_views.end());
}

}  // namespace shortfin::local
