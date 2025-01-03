// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local/fiber.h"

#include <fmt/core.h>
#include <fmt/ranges.h>
#include <fmt/xchar.h>

#include "shortfin/local/system.h"
#include "shortfin/support/logging.h"

namespace shortfin::local {

// -------------------------------------------------------------------------- //
// Fiber
// -------------------------------------------------------------------------- //

Fiber::Fiber(std::shared_ptr<System> system, Worker &worker,
             std::span<const std::pair<std::string_view, Device *>> devices)
    : system_(std::move(system)),
      host_allocator_(system_->host_allocator()),
      scheduler_(*system_),
      worker_(worker) {
  logging::construct("Fiber", this);
  for (auto &it : devices) {
    AddDevice(it.first, it.second);
  }
  Initialize();
}

Fiber::Fiber(std::shared_ptr<System> system, Worker &worker,
             std::span<Device *const> devices)
    : system_(std::move(system)),
      host_allocator_(system_->host_allocator()),
      scheduler_(*system_),
      worker_(worker) {
  logging::construct("Fiber", this);
  for (auto *device : devices) {
    AddDevice(device->address().logical_device_class, device);
  }
  Initialize();
}

Fiber::~Fiber() { logging::destruct("Fiber", this); }

std::string Fiber::to_s() const {
  return fmt::format("Fiber(worker='{}', devices=[{}])", worker_.name(),
                     fmt::join(device_names(), ", "));
}

void Fiber::Initialize() { scheduler_.Initialize(devices_); }

void Fiber::AddDevice(std::string_view device_class, Device *device) {
  device_class = interner_.intern(device_class);
  auto &count = device_class_count_[device_class];
  std::string_view device_name =
      interner_.intern(fmt::format("{}{}", device_class, count++));
  devices_.push_back(std::make_pair(device_name, device));
}

Device *Fiber::raw_device(std::string_view name) const {
  for (auto &it : devices_) {
    if (it.first == name) return it.second;
  }
  throw std::invalid_argument(
      fmt::format("Device '{}' not found (available: {})", name,
                  fmt::join(device_names(), ", ")));
}

Device *Fiber::raw_device(std::size_t ordinal) const {
  if (ordinal >= devices_.size()) {
    throw std::invalid_argument(
        fmt::format("Device ordinal ({}) out of bounds", ordinal));
  }
  return devices_[ordinal].second;
}

std::vector<std::string_view> Fiber::device_names() const {
  std::vector<std::string_view> names;
  names.reserve(devices_.size());
  for (auto &it : devices_) {
    names.push_back(it.first);
  }
  return names;
}

// -------------------------------------------------------------------------- //
// ScopedDevice
// -------------------------------------------------------------------------- //

VoidFuture ScopedDevice::OnSync(bool flush) {
  if (flush) {
    fiber().scheduler().Flush();
  }
  auto &default_account = fiber().scheduler().GetDefaultAccount(*this);
  return default_account.OnSync();
}

}  // namespace shortfin::local
