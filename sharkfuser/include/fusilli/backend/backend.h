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

#include "fusilli/attributes/types.h"

#include <iree/runtime/api.h>

#include <memory>
#include <ostream>
#include <string>
#include <unordered_map>
#include <vector>

namespace fusilli {

// Target backend to run the generated kernels on.
enum class Backend {
  CPU,
  AMDGPU,
};

static const std::unordered_map<Backend, std::string> BackendToStr = {
    {Backend::CPU, "CPU"},
    {Backend::AMDGPU, "AMDGPU"},
};

// Stream operator for Backend.
inline std::ostream &operator<<(std::ostream &os, const Backend &backend) {
  if (BackendToStr.contains(backend)) // C++20
    os << BackendToStr.at(backend);
  else
    os << "UNKNOWN_BACKEND";
  return os;
}

// Map from backend to IREE HAL driver name.
static const std::unordered_map<Backend, const char *> halDriver = {
    {Backend::CPU, "local-task"},
    {Backend::AMDGPU, "hip"},
};

// Map from backend to IREE compile flags.
static const std::unordered_map<Backend, std::vector<std::string>> backendFlags = {
    {
        Backend::CPU,
        {
            "--iree-hal-target-backends=llvm-cpu",
            "--iree-llvmcpu-target-cpu=host",
        },
    },
    {
        // Specify a HIP target for AMD GPU by extracting the architecture
        // name for the first device using `rocm_agent_enumerator`.
        // See this page for a full list of supported architectures:
        // https://iree.dev/guides/deployment-configurations/gpu-rocm/#choosing-hip-targets
        Backend::AMDGPU,
        {
            // clang-format off
            "--iree-hal-target-backends=rocm",
            "--iree-hip-target=$(rocm_agent_enumerator | sed -n '1 p')",
            "--iree-opt-level=O3",
            "--iree-preprocessing-pass-pipeline=\"builtin.module(util.func(iree-preprocessing-sink-transpose-through-pad))\"",
            "--iree-dispatch-creation-enable-fuse-padding-into-linalg-consumer-ops",
            // clang-format on
        },
    },
};

// Template specializations to map from primitive types
// to IREE HAL element type.
template <typename T> struct IreeHalElementType;
//
// float -> IREE_HAL_ELEMENT_TYPE_FLOAT_32:
template <> struct IreeHalElementType<float> {
  static constexpr iree_hal_element_type_t kType =
      IREE_HAL_ELEMENT_TYPE_FLOAT_32;
};
//
// half -> IREE_HAL_ELEMENT_TYPE_FLOAT_16:
template <> struct IreeHalElementType<half> {
  static constexpr iree_hal_element_type_t kType =
      IREE_HAL_ELEMENT_TYPE_FLOAT_16;
};
//
// bf16 -> IREE_HAL_ELEMENT_TYPE_BFLOAT_16:
template <> struct IreeHalElementType<bf16> {
  static constexpr iree_hal_element_type_t kType =
      IREE_HAL_ELEMENT_TYPE_BFLOAT_16;
};
//
// int -> IREE_HAL_ELEMENT_TYPE_INT_32:
template <> struct IreeHalElementType<int> {
  static constexpr iree_hal_element_type_t kType = IREE_HAL_ELEMENT_TYPE_INT_32;
};
//
// int16 -> IREE_HAL_ELEMENT_TYPE_INT_16:
template <> struct IreeHalElementType<int16_t> {
  static constexpr iree_hal_element_type_t kType = IREE_HAL_ELEMENT_TYPE_INT_16;
};
//
// int8 -> IREE_HAL_ELEMENT_TYPE_INT_8:
template <> struct IreeHalElementType<int8_t> {
  static constexpr iree_hal_element_type_t kType = IREE_HAL_ELEMENT_TYPE_INT_8;
};
//
// Assert for unsupported types:
template <typename T> struct IreeHalElementType {
  static_assert(sizeof(T) == 0, "Unsupported type for IREE_HAL_ELEMENT_TYPE");
};
//
// Getter:
template <typename T> iree_hal_element_type_t getIreeHalElementTypeForT() {
  return IreeHalElementType<T>::kType;
}

// Custom deleter for IREE runtime instance.
struct IreeRuntimeInstanceDeleter {
  void operator()(iree_runtime_instance_t *instance) const {
    if (instance)
      iree_runtime_instance_release(instance);
  }
};

// Custom deleter for IREE HAL device.
struct IreeHalDeviceDeleter {
  void operator()(iree_hal_device_t *device) const {
    if (device)
      iree_hal_device_release(device);
  }
};

// Custom deleter for IREE runtime session.
struct IreeRuntimeSessionDeleter {
  void operator()(iree_runtime_session_t *session) const {
    if (session)
      iree_runtime_session_release(session);
  }
};

// Custom deleter for IREE HAL buffer view.
struct IreeHalBufferViewDeleter {
  void operator()(iree_hal_buffer_view_t *bufferView) const {
    if (bufferView)
      iree_hal_buffer_view_release(bufferView);
  }
};

// Aliases for IREE runtime types with custom deleters.
using IreeRuntimeInstanceSharedPtrType =
    std::shared_ptr<iree_runtime_instance_t>;
using IreeHalDeviceUniquePtrType =
    std::unique_ptr<iree_hal_device_t, IreeHalDeviceDeleter>;
using IreeRuntimeSessionUniquePtrType =
    std::unique_ptr<iree_runtime_session_t, IreeRuntimeSessionDeleter>;
using IreeHalBufferViewUniquePtrType =
    std::unique_ptr<iree_hal_buffer_view_t, IreeHalBufferViewDeleter>;

} // namespace fusilli

#endif // FUSILLI_BACKEND_BACKEND_H
