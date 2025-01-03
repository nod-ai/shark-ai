// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fmt/xchar.h>

#include "shortfin/local/system.h"
#include "shortfin/support/logging.h"

#ifdef SHORTFIN_HAVE_HOSTCPU
#include "shortfin/local/systems/host.h"
#endif

#ifdef SHORTFIN_HAVE_AMDGPU
#include "shortfin/local/systems/amdgpu.h"
#endif

namespace shortfin::local {

SystemPtr System::Create(iree_allocator_t host_allocator,
                         std::string_view system_type,
                         ConfigOptions config_options) {
  auto builder = SystemBuilder::ForSystem(host_allocator, system_type,
                                          std::move(config_options));
  auto system = builder->CreateSystem();
  try {
    builder->config_options().ValidateUndef();
  } catch (...) {
    system->Shutdown();
    throw;
  }
  return system;
}

std::unique_ptr<SystemBuilder> SystemBuilder::ForSystem(
    iree_allocator_t host_allocator, std::string_view system_type,
    ConfigOptions config_options) {
  using Factory = std::unique_ptr<SystemBuilder> (*)(
      iree_allocator_t host_allocator, ConfigOptions);
  static const std::vector<std::pair<std::string_view, Factory>> factories{
#ifdef SHORTFIN_HAVE_HOSTCPU
      std::make_pair(
          "hostcpu",
          +[](iree_allocator_t host_allocator,
              ConfigOptions options) -> std::unique_ptr<SystemBuilder> {
            return std::make_unique<systems::HostCPUSystemBuilder>(
                host_allocator, std::move(options));
          }),
#endif
#ifdef SHORTFIN_HAVE_AMDGPU
      std::make_pair(
          "amdgpu",
          +[](iree_allocator_t host_allocator,
              ConfigOptions options) -> std::unique_ptr<SystemBuilder> {
            return std::make_unique<systems::AMDGPUSystemBuilder>(
                host_allocator, std::move(options));
          }),
#endif
  };

  for (auto &it : factories) {
    if (system_type == it.first) {
      return it.second(host_allocator, std::move(config_options));
    }
  }

  // Not found.
  std::vector<std::string_view> available;
  available.reserve(factories.size());
  for (auto &it : factories) {
    available.push_back(it.first);
  }

  throw std::invalid_argument(
      fmt::format("System type '{}' not known (available: {})", system_type,
                  fmt::join(available, ", ")));
}

}  // namespace shortfin::local
