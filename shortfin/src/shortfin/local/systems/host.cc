// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local/systems/host.h"

#include <bit>

#include "iree/hal/local/loaders/registration/init.h"
#include "shortfin/support/config.h"
#include "shortfin/support/iree_helpers.h"
#include "shortfin/support/logging.h"
#include "shortfin/support/sysconfig.h"

namespace shortfin::local::systems {

namespace {
const std::string_view SYSTEM_DEVICE_CLASS = "hostcpu";
const std::string_view LOGICAL_DEVICE_CLASS = "cpu";
const std::string_view HAL_DRIVER_PREFIX = "local";

struct TopologyHolder {
  TopologyHolder() { iree_task_topology_initialize(&topology); }
  ~TopologyHolder() { iree_task_topology_deinitialize(&topology); }

  iree_task_topology_t topology;
};

}  // namespace

// -------------------------------------------------------------------------- //
// HostCPUSystemBuilder
// -------------------------------------------------------------------------- //

HostCPUSystemBuilder::Deps::Deps(iree_allocator_t host_allocator) {
  iree_task_executor_options_initialize(&task_executor_options);
  iree_hal_task_device_params_initialize(&task_params);

#ifndef NDEBUG
  // TODO: In normal IREE programs, this is exposed as --task_abort_on_failure.
  // It is a critical debug feature as it forces an eager program crash at
  // the point encountered vs as a later, rolled up async status. Since it
  // guards things that are API usage bugs in how we are using the runtime,
  // from our perspective, it is assert like, and we treat it as such.
  // However, it would be best to be independently controllable.
  task_params.queue_scope_flags |= IREE_TASK_SCOPE_FLAG_ABORT_ON_FAILURE;
#endif
}

HostCPUSystemBuilder::Deps::~Deps() {
  for (iree_host_size_t i = 0; i < loader_count; ++i) {
    iree_hal_executable_loader_release(loaders[i]);
  }
  if (device_allocator) {
    iree_hal_allocator_release(device_allocator);
  }
  if (plugin_manager) {
    iree_hal_executable_plugin_manager_release(plugin_manager);
  }
}

HostCPUSystemBuilder::HostCPUSystemBuilder(iree_allocator_t host_allocator,
                                           ConfigOptions config_options)
    : HostSystemBuilder(host_allocator, std::move(config_options)),
      host_cpu_deps_(host_allocator) {
  hostcpu_allocator_specs_ = GetConfigAllocatorSpecs("hostcpu_allocators");
}

HostCPUSystemBuilder::~HostCPUSystemBuilder() = default;

void HostCPUSystemBuilder::InitializeHostCPUDefaults() {
  // Give it a default device allocator.
  if (!host_cpu_deps_.device_allocator) {
    SHORTFIN_THROW_IF_ERROR(iree_hal_allocator_create_heap(
        iree_make_cstring_view("local"), host_allocator(), host_allocator(),
        &host_cpu_deps_.device_allocator));
  }

  // And loaders.
  if (host_cpu_deps_.loader_count == 0) {
    SHORTFIN_THROW_IF_ERROR(iree_hal_create_all_available_executable_loaders(
        /*plugin_manager=*/nullptr, IREE_ARRAYSIZE(host_cpu_deps_.loaders),
        &host_cpu_deps_.loader_count, host_cpu_deps_.loaders,
        host_allocator()));
  }

  // Queue executors.
}

std::vector<iree_host_size_t>
HostCPUSystemBuilder::SelectHostCPUNodesFromOptions() {
  const unsigned MAX_NODE_COUNT = 64u;
  const iree_host_size_t available_node_count = std::max(
      1u, std::min(MAX_NODE_COUNT, static_cast<unsigned>(
                                       iree_task_topology_query_node_count())));
  auto topology_nodes = config_options().GetOption("hostcpu_topology_nodes");

  std::vector<iree_host_size_t> nodes;
  if (!topology_nodes || topology_nodes->empty() ||
      *topology_nodes == "current") {
    // If topology_nodes not specified or "current", use a single default node.
    nodes.push_back(iree_task_topology_query_current_node());
  } else if (*topology_nodes == "all") {
    // If topology_nodes == "all", create a mask of all available nodes.
    nodes.reserve(available_node_count);
    for (iree_host_size_t i = 0; i < available_node_count; ++i) {
      nodes.push_back(i);
    }
  } else {
    // Otherwise, parse it as an integer list.
    auto topology_node_ids =
        config_options().GetIntList("hostcpu_topology_nodes");
    assert(topology_node_ids);
    for (int64_t node_id : *topology_node_ids) {
      if (node_id < 0 || (iree_host_size_t)node_id >= available_node_count) {
        throw std::invalid_argument(fmt::format(
            "Illegal value {} in hostcpu_topology_nodes: Expected [0..{}]",
            node_id, available_node_count - 1));
      }
      nodes.push_back(node_id);
    }
  }
  return nodes;
}

SystemPtr HostCPUSystemBuilder::CreateSystem() {
  SHORTFIN_TRACE_SCOPE_NAMED("HostCPUSystemBuilder::CreateSystem");
  auto lsys = std::make_shared<System>(host_allocator());
  // TODO: Real NUMA awareness.
  lsys->InitializeNodes(1);
  InitializeHostCPUDefaults();
  auto *driver = InitializeHostCPUDriver(*lsys);
  InitializeHostCPUDevices(*lsys, driver);
  lsys->FinishInitialization();
  return lsys;
}

iree_hal_driver_t *HostCPUSystemBuilder::InitializeHostCPUDriver(System &lsys) {
  SHORTFIN_TRACE_SCOPE_NAMED("HostCPUSystemBuilder::InitializeHostCPUDriver");
  // TODO: Kill these flag variants in favor of settings on the config
  // object.
  SHORTFIN_THROW_IF_ERROR(iree_task_executor_options_initialize_from_flags(
      &host_cpu_deps_.task_executor_options));

  // Determine NUMA nodes to use.
  auto selected_nodes = SelectHostCPUNodesFromOptions();
  auto max_group_count = config_options().GetInt(
      "hostcpu_topology_max_group_count", /*non_negative=*/true);
  if (!max_group_count) {
    max_group_count = 64;
  }

  auto distribution_option = config_options().GetBool(
      "SHORTFIN_HOSTCPU_TOPOLOGY_DISTRIBUTION_SCATTER", true);
  iree_task_topology_distribution_t distribution =
      static_cast<iree_task_topology_distribution_t>(distribution_option);

  // Create one queue executor per node.
  unsigned total_needed_file_handles = 512;
  bool has_issued_limit_error = false;
  std::vector<iree::task_executor_ptr> queue_executors;
  queue_executors.reserve(selected_nodes.size());
  queue_node_ids_.reserve(selected_nodes.size());
  for (auto node_id : selected_nodes) {
    TopologyHolder topology;
    iree_task_topology_performance_level_t performance_level =
        IREE_TASK_TOPOLOGY_PERFORMANCE_LEVEL_ANY;
    SHORTFIN_THROW_IF_ERROR(iree_task_topology_initialize_from_physical_cores(
        node_id, performance_level, distribution, *max_group_count,
        &topology.topology));
    logging::debug("Creating hostcpu queue for NUMA node {} with {} groups",
                   node_id, iree_task_topology_group_count(&topology.topology));
    queue_executors.push_back({});
    auto &executor = queue_executors.back();
    // As of 2024-11-8, it took approximately 32 file handles per node-group.
    // To be conservative because file handle limits are basically free, we
    // round up to 64 and assume a floor of 512. This allows small, default
    // 8 group, single node configs to require no limit increase for Linux
    // 1024 default cases.
    total_needed_file_handles += 64 * topology.topology.group_count;
    if (!sysconfig::EnsureFileLimit(total_needed_file_handles) &&
        !has_issued_limit_error) {
      logging::error(
          "Could not ensure sufficient file handles for minimum operations: "
          "Suggest setting explicit limits with `ulimit -n` and system "
          "settings");
      has_issued_limit_error = true;
    }

    SHORTFIN_THROW_IF_ERROR(iree_task_executor_create(
        host_cpu_deps_.task_executor_options, &topology.topology,
        host_allocator(), executor.for_output()));
    queue_node_ids_.push_back(node_id);
  }

  // Create the driver and save it in the System.
  iree::hal_driver_ptr driver;
  iree_hal_driver_t *unowned_driver;
  SHORTFIN_THROW_IF_ERROR(iree_hal_task_driver_create(
      /*identifier=*/
      {
          .data = HAL_DRIVER_PREFIX.data(),
          .size = HAL_DRIVER_PREFIX.size(),
      },
      &host_cpu_deps_.task_params, /*queue_count=*/queue_executors.size(),
      reinterpret_cast<iree_task_executor_t **>(queue_executors.data()),
      host_cpu_deps_.loader_count, host_cpu_deps_.loaders,
      host_cpu_deps_.device_allocator, host_allocator(), driver.for_output()));
  unowned_driver = driver.get();
  lsys.InitializeHalDriver(SYSTEM_DEVICE_CLASS, std::move(driver));
  return unowned_driver;
}

void HostCPUSystemBuilder::InitializeHostCPUDevices(System &lsys,
                                                    iree_hal_driver_t *driver) {
  SHORTFIN_TRACE_SCOPE_NAMED("HostCPUSystemBuilder::InitializeHostCPUDevices");
  iree_host_size_t device_info_count = 0;
  iree::allocated_ptr<iree_hal_device_info_t> device_infos(host_allocator());
  SHORTFIN_THROW_IF_ERROR(iree_hal_driver_query_available_devices(
      driver, host_allocator(), &device_info_count, &device_infos));
  if (device_info_count != 1) {
    throw std::logic_error("Expected a single CPU device");
  }

  iree::hal_device_ptr device;
  iree_hal_device_info_t *it = &device_infos.get()[0];
  SHORTFIN_THROW_IF_ERROR(iree_hal_driver_create_device_by_id(
      driver, it->device_id, 0, nullptr, host_allocator(),
      device.for_output()));
  ConfigureAllocators(hostcpu_allocator_specs_, device, "hostcpu");

  iree_host_size_t queue_index = 0;
  for (auto node_id : queue_node_ids_) {
    DeviceAddress address(
        /*system_device_class=*/SYSTEM_DEVICE_CLASS,
        /*logical_device_class=*/LOGICAL_DEVICE_CLASS,
        /*hal_driver_prefix=*/HAL_DRIVER_PREFIX,
        /*instance_ordinal=*/0,
        /*queue_ordinal=*/queue_index,
        /*instance_topology_address=*/{queue_index});
    lsys.InitializeHalDevice(std::make_unique<HostCPUDevice>(
        address,
        /*hal_device=*/device,
        /*node_affinity=*/node_id,
        /*capabilities=*/
        static_cast<int32_t>(
            Device::Capabilities::PREFER_HOST_UNIFIED_MEMORY)));
    queue_index += 1;
  }
}

}  // namespace shortfin::local::systems
