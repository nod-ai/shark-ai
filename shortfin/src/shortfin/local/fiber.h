// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_FIBER_H
#define SHORTFIN_LOCAL_FIBER_H

#include <functional>
#include <span>
#include <unordered_map>

#include "shortfin/local/async.h"
#include "shortfin/local/device.h"
#include "shortfin/local/program.h"
#include "shortfin/local/scheduler.h"
#include "shortfin/support/stl_extras.h"

namespace shortfin::local {

class Fiber;
class System;
class Worker;

// Wraps a Fiber and a DeviceAffinity together. This is used in all
// Fiber based APIs as a short-hand for "device" as it contains everything
// needed to do thing with some slice of device queues.
class SHORTFIN_API ScopedDevice {
 public:
  ScopedDevice() = default;
  ScopedDevice(Fiber &fiber, DeviceAffinity affinity)
      : fiber_(&fiber), affinity_(affinity) {}
  ScopedDevice(Fiber &fiber, Device *device)
      : fiber_(&fiber), affinity_(device) {}
  ScopedDevice(const ScopedDevice &other)
      : fiber_(other.fiber_), affinity_(other.affinity_) {}

  Fiber &fiber() const {
    assert(fiber_ && "fiber must not be null");
    return *fiber_;
  }
  DeviceAffinity affinity() const { return affinity_; }
  Device *raw_device() const { return affinity_.device(); }

  std::string to_s() const { return affinity().to_s(); }

  bool operator==(const ScopedDevice &other) const {
    return (fiber_ == other.fiber_) && affinity_ == other.affinity_;
  }

  // Returns a future which will be satisfied when the primary device timeline
  // of this affinity set progresses to "now". This will be true when all
  // currently queued work on the device has been completed.
  VoidFuture OnSync(bool flush = true);

 private:
  Fiber *fiber_ = nullptr;
  DeviceAffinity affinity_;
};

// A logical fiber of execution, consisting of participating devices,
// resources, and timelines. Most interaction with the compute resources
// is done on these instances.
//
// The fiber is generally instantiated with a slice of system resources,
// and produces an arrangement that is easy to use vs maximally diverse.
//
// Devices
// -------
// The fiber is initialized with a list of participating devices, which is
// a subset of all devices managed by the LocalSystem. Each device is given
// a logical name of the form `<device_class><index>`, by default using the
// DeviceAddress::logical_device_class as the `<device_class>`. In exotic
// situations, this can be customized. By default, devices are added in the
// order defined by the system and will have an `<index>` corresponding to
// their order. It is up to the constructor to produce a sensible arrangement.
class SHORTFIN_API Fiber : public std::enable_shared_from_this<Fiber> {
 public:
  // Initialize with devices using logical_device_class as the device class.
  Fiber(std::shared_ptr<System> system, Worker &worker,
        std::span<Device *const> devices);
  // Initialize with devices with custom device class names.
  Fiber(std::shared_ptr<System> system, Worker &worker,
        std::span<const std::pair<std::string_view, Device *>> devices);
  Fiber(const Fiber &) = delete;
  // Ensure polymorphic.
  virtual ~Fiber();
  std::string to_s() const;

  // All scopes are created as shared pointers.
  std::shared_ptr<Fiber> shared_ptr() { return shared_from_this(); }

  // The host allocator.
  iree_allocator_t host_allocator() { return host_allocator_; }

  // The worker that this fiber is bound to.
  Worker &worker() { return worker_; }

  // System that this fiber is bound to.
  System &system() { return *system_; }

  // Device access.
  // Throws std::invalid_argument on lookup failure.
  Device *raw_device(std::string_view name) const;
  Device *raw_device(std::size_t index) const;
  Device *raw_device(Device *device) const { return device; }
  std::span<const std::pair<std::string_view, Device *>> raw_devices() const {
    return devices_;
  }
  std::vector<std::string_view> device_names() const;

  // Variadic helper for making a DeviceAffinity from any of:
  //  * Explicit Device*
  //  * Device name (from a Fiber)
  //  * Device index (from a Fiber)
  // If at any point during accumulation, the DeviceAffinity would be invalid,
  // then a std::invalid_argument exception is thrown. Any failure to resolve
  // a name or index will also throw a std::invalid_argument.
  ScopedDevice device() { return ScopedDevice(*this, DeviceAffinity()); }
  template <typename T, typename... Args>
  ScopedDevice device(T first, Args... args) {
    return ScopedDevice(
        *this, device(args...).affinity() | DeviceAffinity(raw_device(first)));
  }
  ScopedDevice device(Device *d) {
    return ScopedDevice(*this, DeviceAffinity(d));
  }
  detail::Scheduler &scheduler() { return scheduler_; }
  detail::TimelineResource::Ref NewTimelineResource(
      detail::TimelineResourceDestructor destructor = nullptr) {
    return scheduler().NewTimelineResource(shared_ptr(), std::move(destructor));
  }

 private:
  void AddDevice(std::string_view device_class, Device *device);
  void Initialize();  // Called after all devices are added.

  // Back reference to owning system.
  std::shared_ptr<System> system_;
  string_interner interner_;
  iree_allocator_t host_allocator_;
  detail::Scheduler scheduler_;
  Worker &worker_;

  // Map of `<device_class>` to the count of that class contained.
  std::unordered_map<std::string_view, int> device_class_count_;
  // Ordered devices named as `<device_class><index>`.
  std::vector<std::pair<std::string_view, Device *>> devices_;

  // Program isolation control.
  // This data structure is manipulated by APIs on the Program class hierarchy.
  // It maps a parent context pointer to an isolate accounting struct. This
  // struct contains a strong reference to the parent_context and a vector
  // of fork contexts. For PER_FIBER invocations, there will only ever be either
  // zero or one fork_contexts: when no calls have been issued there will be one
  // and if a call is outstanding, there will be zero. This is used to guard
  // concurrent access. For PER_CALL invocations, there will be as many
  // fork_contexts as are needed to satisfy the peak number of calls in flight
  // at any time.
  // The program_isolate_mu_ must be held to manipulate the accounting structs.
  iree::slim_mutex program_isolate_mu_;
  std::unordered_map<iree_vm_context_t *,
                     std::unique_ptr<detail::ProgramIsolate>>
      program_isolates_;
  friend struct detail::ProgramIsolate;
};

}  // namespace shortfin::local

#endif  // SHORTFIN_LOCAL_FIBER_H
