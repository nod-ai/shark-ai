// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <hip/hip_runtime.h>
#include <iree/base/status.h>
#include <iree/hal/api.h>
#include <iree/runtime/api.h>

#include <catch2/catch_test_macros.hpp>
#include <cstdint>
#include <cstdio>

#include "hip_kernels.h"

// Utility macro to check status of HIP functions that are set with nodiscard.
#define HIP_REQUIRE_SUCCESS(expr)                                              \
  ({                                                                           \
    auto err = (expr);                                                         \
    if (err != hipSuccess) {                                                   \
      fprintf(stderr, "Error: %s\n", hipGetErrorString(err));                  \
    }                                                                          \
    REQUIRE(err == hipSuccess);                                                \
  })

TEST_CASE("proof of life for HIP", "[hip_tests]") {
  // ----------------------------------------------------------------------
  //  proof of life for GPU connection
  // ----------------------------------------------------------------------

  int dev = 0;
  HIP_REQUIRE_SUCCESS(hipGetDevice(&dev));

  hipDeviceProp_t prop{};
  HIP_REQUIRE_SUCCESS(hipGetDeviceProperties(&prop, dev));

  void *ptr;
  HIP_REQUIRE_SUCCESS(hipMalloc(&ptr, sizeof(float) * 64));

  // Launch kernel (1 block, 4 threads).
  launchHelloKernel(dim3(1), dim3(4));

  HIP_REQUIRE_SUCCESS(hipDeviceSynchronize());
}

TEST_CASE("Buffer import", "[hip_tests]") {
  // --------------------------------------------------------------------------
  //  Test for the externally managed buffers imported as IREE HAL Buffer View
  //
  //  `hipMalloc`'ed `void*` -> `iree_hal_buffer_view` (via `import_buffer`)
  // --------------------------------------------------------------------------

  // Pointer to hipMalloc'ed buffer on device.
  void *ptr;
  HIP_REQUIRE_SUCCESS(hipMalloc(&ptr, sizeof(float) * 64));

  // Get runtime instance.
  iree_runtime_instance_options_t instance_options;
  iree_runtime_instance_options_initialize(&instance_options);
  iree_runtime_instance_options_use_all_available_drivers(&instance_options);
  iree_runtime_instance_t *instance = nullptr;
  REQUIRE(iree_status_is_ok(iree_runtime_instance_create(
      &instance_options, iree_allocator_system(), &instance)));

  // Get runtime device.
  iree_hal_device_t *device = nullptr;
  REQUIRE(iree_status_is_ok(iree_runtime_instance_try_create_default_device(
      instance, iree_make_cstring_view("hip"), &device)));

  // Get device allocator.
  iree_hal_allocator_t *allocator = iree_hal_device_allocator(device);

  // Create external buffer wrapping the void*.
  iree_hal_external_buffer_t external_buffer = {
      .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
      .flags = 0,
      .size = sizeof(float) * 64,
      .handle =
          {
              .device_allocation =
                  {
                      .ptr = (uint64_t)ptr,
                  },
          },
  };

  // Set buffer parameters.
  iree_hal_buffer_params_t params = {0};
  // Where to allocate (host or device).
  params.type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL;
  // Access to allow to this memory.
  params.access = IREE_HAL_MEMORY_ACCESS_ALL;
  // Intended usage of the buffer (transfers, dispatches, etc).
  params.usage = IREE_HAL_BUFFER_USAGE_DEFAULT;

  // Create imported buffer to be populated with externally managed buffer.
  iree_hal_buffer_t *out_buffer = nullptr;

  iree_hal_buffer_release_callback_t release_callback =
      iree_hal_buffer_release_callback_null();

  // Import buffer.
  REQUIRE(iree_status_is_ok(iree_hal_allocator_import_buffer(
      allocator, params, &external_buffer, release_callback, &out_buffer)));

  // Cleanup.
  iree_hal_device_release(device);
  iree_runtime_instance_release(instance);
}
