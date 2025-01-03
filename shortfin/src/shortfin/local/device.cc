// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local/device.h"

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fmt/xchar.h>

namespace shortfin::local {

// -------------------------------------------------------------------------- //
// DeviceAddress
// -------------------------------------------------------------------------- //

DeviceAddress::DeviceAddress(
    std::string_view system_device_class, std::string_view logical_device_class,
    std::string_view hal_driver_prefix, uint32_t instance_ordinal,
    uint32_t queue_ordinal,
    std::vector<iree_host_size_t> instance_topology_address)
    : system_device_class(std::move(system_device_class)),
      logical_device_class(std::move(logical_device_class)),
      hal_driver_prefix(std::move(hal_driver_prefix)),
      instance_ordinal(instance_ordinal),
      queue_ordinal(queue_ordinal),
      instance_topology_address(instance_topology_address),
      device_name(
          fmt::format("{}:{}:{}@{}", this->system_device_class,
                      this->instance_ordinal, this->queue_ordinal,
                      fmt::join(this->instance_topology_address, ","))) {}

// -------------------------------------------------------------------------- //
// DeviceAffinity
// -------------------------------------------------------------------------- //

void DeviceAffinity::ThrowIllegalDeviceAffinity(Device *first, Device *second) {
  throw std::invalid_argument(fmt::format(
      "Cannot combine unrelated devices into a DeviceAffinity: {} vs {}",
      first->name(), second->name()));
}

std::string DeviceAffinity::to_s() const {
  if (device()) {
    return fmt::format("DeviceAffinity({}[0x{:x}])", device()->name(),
                       queue_affinity());
  } else {
    return "DeviceAffinity(ANY)";
  }
}

// -------------------------------------------------------------------------- //
// Device
// -------------------------------------------------------------------------- //

Device::Device(DeviceAddress address, iree::hal_device_ptr hal_device,
               int node_affinity, uint32_t capabilities)
    : address_(std::move(address)),
      hal_device_(std::move(hal_device)),
      node_affinity_(node_affinity),
      capabilities_(capabilities) {
  assert(hal_device_ && "nullptr iree_hal_device_t");
}

Device::~Device() = default;

std::string Device::to_s() const {
  return fmt::format(
      "Device(name='{}', ordinal={}:{}, node_affinity={}, capabilities=0x{:x})",
      name(), address().instance_ordinal, address().queue_ordinal,
      node_affinity(), capabilities_);
}

}  // namespace shortfin::local
