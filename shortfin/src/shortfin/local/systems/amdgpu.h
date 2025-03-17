// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_SYSTEMS_AMDGPU_H
#define SHORTFIN_LOCAL_SYSTEMS_AMDGPU_H

#include <vector>

#include "iree/hal/drivers/hip/api.h"
#include "shortfin/local/system.h"
#include "shortfin/local/systems/host.h"
#include "shortfin/support/api.h"
#include "shortfin/support/iree_helpers.h"

namespace shortfin::local::systems {

// AMD GPU device subclass.
class SHORTFIN_API AMDGPUDevice : public Device {
 public:
  using Device::Device;
};

// System configuration for some subset of AMD GPUs connected to the local
// system. Note that this inherits from HostCPUSystemBuilder, amdgpu_allowing
// joint configuration of a heterogenous CPU/GPU system. Depending on the
// specific system, this can involve more than simple starting CPU drivers:
// datacenter GPU systems have specific NUMA configurations that need to be
// mated.
class SHORTFIN_API AMDGPUSystemBuilder : public HostCPUSystemBuilder {
 public:
  AMDGPUSystemBuilder(iree_allocator_t host_allocator,
                      ConfigOptions options = {});
  AMDGPUSystemBuilder() : AMDGPUSystemBuilder(iree_allocator_system()) {}
  ~AMDGPUSystemBuilder();

  SystemPtr CreateSystem() override;

  // Settings.
  bool &cpu_devices_enabled() { return cpu_devices_enabled_; }

  // See iree_hal_hip_driver_options_t::hip_lib_search_paths. Each is either
  // a directory or "file:" prefixed path to a specific HIP dynamic library.
  // This is typically libamdhip64.so or amdhip64.dll.
  // If the environment variable IREE_HIP_DYLIB_PATH is present, then it is
  // split on ';' and each entry added here (for compatibility with IREE
  // tools).
  // Changing these paths after enumeration has no effect.
  std::vector<std::string> &hip_lib_search_paths() {
    return hip_lib_search_paths_;
  }

  // If set, then the system will be created to only include devices with
  // the corresponding id (in the order listed).
  std::optional<std::vector<std::string>> &visible_devices() {
    return visible_devices_;
  };

  // Allocator specs to apply to amdgpu devices in this builder.
  std::vector<std::string> &amdgpu_allocator_specs() {
    return amdgpu_allocator_specs_;
  }

  // Whether to use async allocations if the device supports them (default
  // true). There are various reasons to disable this in different usage
  // scenarios.
  bool &async_allocations() { return default_device_params_.async_allocations; }

  // "amdgpu_tracing_level": Matches IREE flag --hip_tracing:
  // Permissible values are:
  //   0 : stream tracing disabled.
  //   1 : coarse command buffer level tracing enabled.
  //   2 : fine-grained kernel level tracing enabled.
  int32_t &tracing_level() { return default_device_params_.stream_tracing; }

  // The number of logical HAL devices to create per physical, visible device.
  // This form of topology can be useful in certain cases where we aim to have
  // oversubscription emulating what would usually be achieved with process
  // level isolation. Defaults to 1.
  size_t &logical_devices_per_physical_device() {
    return logical_devices_per_physical_device_;
  }

  // Gets all enumerated available device ids. This triggers enumeration, so
  // any settings required for that must already be set. This does no filtering
  // and will return all device ids.
  std::vector<std::string> GetAvailableDeviceIds();

 private:
  void InitializeDefaultSettings();
  // Triggers driver setup and initial device enumeration. No-op if already
  // done.
  void Enumerate();

  // Valid at construction time.
  iree_hal_hip_device_params_t default_device_params_;

  // Configuration.
  bool cpu_devices_enabled_ = false;
  bool amdgpu_allow_device_reuse_ = false;
  std::vector<std::string> hip_lib_search_paths_;
  std::optional<std::vector<std::string>> visible_devices_;
  size_t logical_devices_per_physical_device_ = 1;
  std::vector<std::string> amdgpu_allocator_specs_;

  // Valid post enumeration.
  iree::hal_driver_ptr hip_hal_driver_;
  iree_host_size_t available_devices_count_ = 0;
  iree::allocated_ptr<iree_hal_device_info_t> available_devices_;
};

}  // namespace shortfin::local::systems

#endif  // SHORTFIN_LOCAL_SYSTEMS_AMDGPU_H
