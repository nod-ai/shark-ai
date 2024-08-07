// Copyright 2024 Advanced Micro Devices, Inc
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_DEVICE_H
#define SHORTFIN_LOCAL_DEVICE_H

#include <string>
#include <string_view>
#include <vector>

#include "iree/hal/api.h"
#include "shortfin/support/api.h"
#include "shortfin/support/iree_helpers.h"

namespace shortfin {

// NUMA node on the LocalSystem. There will always be at least one node, and
// not all NUMA nodes on the system may be included: only those applicable
// to device pinning/scheduling.
class SHORTFIN_API LocalNode {
 public:
  LocalNode(int node_num) : node_num_(node_num) {}

  int node_num() const { return node_num_; }

 private:
  int node_num_;
};

// Each device exists in the local system as part of some topology that consists
// of the following levels:
//
//   Level 0: User device category / system driver prefix
//     (i.e. "hip", "cuda", "local").
//   Level 1: Device instance ordinal.
//   Level 2: Instance topology vector representing the logical organization
//     of the queues on the device instance.
//
// Concretely, this means that each leaf LocalSystemDevice instance consists
// of an iree_hal_device_t (as managed by an iree_hal_driver_t) and a
// single bit position within an iree_hal_queue_affinity_t. The total number
// of devices of a class is thus equal to the product of the device instance
// ordinal and every entry of the instance topology vector. There can be at
// most 64 queues on a device instance.
//
// How the topology is laid out is system and use case specific, with multiple
// valid arrangements which may be useful for different kinds of workloads.
// There are some general guidelines:
//
//   * All components of a device with peered memory should share the same
//     Level 1 / device instance.
//   * Whether cross-bus devices should share an instance is use case
//     specific, effectively dictated by the nature of the bus connection
//     and intended use. While an instance can be shared across a lower speed
//     link, it may be advantageous to split it and treat the corresponding
//     device leaves as independent from a memory and scheduling perspective.
//   * The instance topology should generally reflect some notion of locality
//     within the physical architecture of some hardware such that co scheduling
//     at leaf nodes of the vector may have some benefit.
//
// Examples:
//   * Large CPU system with two NUMA nodes:
//     - Split instances on NUMA node: local/2/8
//     - Unified instances for an entire chip: local/1/2,8
//     - Different exotic topologies can be represented with a longer topology
//       vector with machine specific communication cost.
//   * Machine with 8 MI300 GPUs (each with 4 memory controllers and 8
//     partitions):
//     - Split instances per host NUMA node: hip/2/4,4,2
//     - Unified instances: hip/1/8,4,2
//     - Simple 8x partition (ignore memory controller): hip/1/8,8,1
//  * Machine with 8 MI300 GPUs operating as one large GPU each: hip/1/8,1,1
//
// Generally, there is a tension that must be negotiated between how much an
// application cares about the hierarchy vs benefiting from tighter coordination
// of locality. The shape of the instance topology must match among all devices
// attached to a driver.
struct SHORTFIN_API LocalDeviceAddress {
 public:
  // Note that all string_views passed should be literals or have a lifetime
  // that exceeds the instance.
  LocalDeviceAddress(std::string_view system_device_class,
                     std::string_view logical_device_class,
                     std::string_view hal_driver_prefix,
                     iree_host_size_t instance_ordinal,
                     iree_host_size_t queue_ordinal,
                     std::vector<iree_host_size_t> instance_topology_address);
  std::string to_s() const;

  // User driver name (i.e. 'amdgpu'). In user visible names/messages, this
  // is preferred over hal_driver_prefix, but must be 1:1 with it.
  std::string_view system_device_class;
  // User device class (i.e. 'gpu', 'cpu').
  std::string_view logical_device_class;
  // System HAL driver prefix (i.e. 'hip', 'cuda', 'local').
  std::string_view hal_driver_prefix;
  iree_host_size_t instance_ordinal;
  iree_host_size_t queue_ordinal;
  std::vector<iree_host_size_t> instance_topology_address;
  // A system-unique device name:
  //   {system_device_class}:{instance_ordinal}:{queue_ordinal}@{instance_topology_address}
  std::string device_name;
};

// A device attached to the LocalSystem.
class SHORTFIN_API LocalDevice {
 public:
  LocalDevice(LocalDeviceAddress address, iree_hal_device_ptr hal_device,
              int node_affinity, bool node_locked);
  virtual ~LocalDevice();

  const LocalDeviceAddress &address() const { return address_; }
  std::string_view name() const { return address_.device_name; }
  int node_affinity() const { return node_affinity_; }
  bool node_locked() const { return node_locked_; }

  std::string to_s() const;

 private:
  LocalDeviceAddress address_;
  iree_hal_device_ptr hal_device_;
  int node_affinity_;
  bool node_locked_;
};

// Holds a reference to a LocalDevice* and a bitmask of queues that are being
// targeted.
class SHORTFIN_API DeviceAffinity {
 public:
  DeviceAffinity() : device_(nullptr), queue_affinity_(0) {}
  DeviceAffinity(LocalDevice *device)
      : device_(device),
        queue_affinity_(static_cast<iree_hal_queue_affinity_t>(1)
                        << device->address().queue_ordinal) {}

  // Adds a device to the set. It is only legal to do this with devices that
  // share the same instance_ordinal (i.e. have the same underlying HAL device).
  // In that case, the queue affinity bitmask will have a bit set for the
  // added device and true will be returned.
  // If the device is not compatible in this way, then false will be returned.
  [[nodiscard]] bool AddDevice(LocalDevice *new_device) {
    if (!new_device) return true;
    if (!device_) {
      // Empty to one device is always allowed.
      device_ = new_device;
    } else if (new_device->address().instance_ordinal !=
               device_->address().instance_ordinal) {
      // Different HAL device is disallowed.
      return false;
    }
    queue_affinity_ |= static_cast<iree_hal_queue_affinity_t>(1)
                       << new_device->address().queue_ordinal;
    return true;
  }

  DeviceAffinity &operator|=(LocalDevice *other) {
    if (!AddDevice(other)) {
      ThrowIllegalDeviceAffinity(device(), other);
    }
    return *this;
  }

  LocalDevice *device() const { return device_; }
  iree_hal_queue_affinity_t queue_affinity() const { return queue_affinity_; }

  std::string to_s() const;

 private:
  static void ThrowIllegalDeviceAffinity(LocalDevice *first,
                                         LocalDevice *second);

  LocalDevice *device_;
  iree_hal_queue_affinity_t queue_affinity_;
};

}  // namespace shortfin

#endif  // SHORTFIN_LOCAL_DEVICE_H
