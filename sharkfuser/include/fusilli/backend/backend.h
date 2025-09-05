// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains backend specific code like the `Backend` type, code to
// map from Backend to `iree-compile` flags, IREE runtime types and deleters.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_BACKEND_BACKEND_H
#define FUSILLI_BACKEND_BACKEND_H

#include <iree/runtime/api.h>

#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fusilli {

// Target backend to run the generated kernels on
enum class Backend {
  CPU,
  GFX942,
};

static const std::unordered_map<Backend, std::string> BackendToStr = {
    {Backend::CPU, "CPU"},
    {Backend::GFX942, "GFX942"},
};

// Stream operator for Backend
inline std::ostream &operator<<(std::ostream &os, const Backend &backend) {
  auto it = BackendToStr.find(backend);
  if (it != BackendToStr.end())
    os << it->second;
  else
    os << "UNKNOWN_BACKEND";
  return os;
}

// Map from backend to IREE HAL driver name
static const std::unordered_map<Backend, const char *> halDriver = {
    {Backend::CPU, "local-task"},
    {Backend::GFX942, "hip"},
};

// Map from backend to IREE compile flags
static const std::unordered_map<Backend, std::vector<std::string>>
    backendFlags = {
        {
            Backend::CPU,
            {
                "--iree-hal-target-backends=llvm-cpu",
                "--iree-llvmcpu-target-cpu=host",
            },
        },
        {
            Backend::GFX942,
            {
                "--iree-hal-target-backends=rocm",
                "--iree-hip-target=gfx942",
                "--iree-opt-level=O3",
            },
        },
};

// Custom deleter for IREE runtime instance
struct IreeRuntimeInstanceDeleter {
  void operator()(iree_runtime_instance_t *instance) const {
    if (instance)
      iree_runtime_instance_release(instance);
  }
};

// Custom deleter for IREE HAL device
struct IreeHalDeviceDeleter {
  void operator()(iree_hal_device_t *device) const {
    if (device)
      iree_hal_device_release(device);
  }
};

// Custom deleter for IREE runtime session
struct IreeRuntimeSessionDeleter {
  void operator()(iree_runtime_session_t *session) const {
    if (session)
      iree_runtime_session_release(session);
  }
};

// Aliases for IREE runtime types with custom deleters
using IreeRuntimeInstanceSharedPtrType =
    std::shared_ptr<iree_runtime_instance_t>;
using IreeHalDeviceUniquePtrType =
    std::unique_ptr<iree_hal_device_t, IreeHalDeviceDeleter>;
using IreeRuntimeSessionUniquePtrType =
    std::unique_ptr<iree_runtime_session_t, IreeRuntimeSessionDeleter>;

} // namespace fusilli

#endif // FUSILLI_BACKEND_BACKEND_H
