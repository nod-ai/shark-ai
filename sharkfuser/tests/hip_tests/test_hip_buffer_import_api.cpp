// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>
#include <utils.h>

#include <catch2/catch_test_macros.hpp>
#include <hip/hip_runtime.h>
#include <iree/base/status.h>
#include <iree/hal/api.h>
#include <iree/runtime/api.h>

#include <cstdint>
#include <cstdio>

#include "hip_kernels.h"

using namespace fusilli;

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

  HIP_REQUIRE_SUCCESS(hipFree(ptr));
}

static void testBufferReleaseCallback(void *didCleanupBuffer,
                                      iree_hal_buffer_t *) {
  *(bool *)didCleanupBuffer = true;
}

TEST_CASE("Buffer import", "[hip_tests]") {
  // --------------------------------------------------------------------------
  //  Test that externally managed buffers, imported as IREE HAL Buffer View,
  //  and finally as a fusilli Buffer, are created successfully and cleaned up
  //  correctly.
  // --------------------------------------------------------------------------

  Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::AMDGPU));

  // IREE allocators.
  iree_hal_allocator_t *deviceAllocator = iree_hal_device_allocator(handle);
  iree_allocator_t hostAllocator = iree_allocator_system();

  // Pointer to hipMalloc'ed buffer on device.
  void *ptr;
  const size_t elementCount = 64;
  const size_t bufferSize = elementCount * sizeof(float);
  HIP_REQUIRE_SUCCESS(hipMalloc(&ptr, sizeof(float) * elementCount));

  // Callback set on IREE buffer and called in `iree_hal_hip_buffer_destroy`.
  bool didCleanupBuffer = false;
  iree_hal_buffer_release_callback_t callback = {
      .fn = testBufferReleaseCallback, .user_data = &didCleanupBuffer};

  // Import external buffer into IREE runtime.
  iree_hal_buffer_params_t bufferParams = {
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      .access = IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE,
      .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
  };
  iree_hal_external_buffer_t externalBuffer = {
      .type = IREE_HAL_EXTERNAL_BUFFER_TYPE_DEVICE_ALLOCATION,
      .flags = 0,
      .size = static_cast<iree_device_size_t>(bufferSize),
      .handle =
          {
              .device_allocation =
                  {
                      .ptr = (uint64_t)ptr,
                  },
          },
  };
  iree_hal_buffer_t *importedBuffer = nullptr;
  REQUIRE(iree_status_is_ok(iree_hal_allocator_import_buffer(
      /*allocator=*/deviceAllocator, /*params=*/bufferParams,
      /*external_buffer=*/&externalBuffer, /*release_callback=*/callback,
      /*out_buffer=*/&importedBuffer)));

  // Create an IREE buffer view for external buffer.
  const iree_hal_dim_t shape[] = {elementCount};
  const size_t shapeRank = 1;
  iree_hal_buffer_view_t *outBufferView = nullptr;
  REQUIRE(iree_status_is_ok(iree_hal_buffer_view_create(
      /*buffer=*/importedBuffer,
      /*shape_rank=*/shapeRank,
      /*shape=*/shape,
      /*element_type=*/IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      /*encoding_type=*/IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      /*host_allocator=*/hostAllocator,
      /*out_buffer_view=*/&outBufferView)));

  // The buffer view holds a reference to buffer and will handle release.
  iree_hal_buffer_release(importedBuffer);

  {
    // Buffer is a RAII type that retains the buffer view.
    Buffer fusilliBufferResult =
        FUSILLI_REQUIRE_UNWRAP(fusilli::Buffer::import(outBufferView));

    // The IREE buffer view and IREE buffer will now be tied to fusilli Buffer's
    // lifetime.
    iree_hal_buffer_view_release(outBufferView);

    // Ensure that IREE buffer has not been released.
    REQUIRE_FALSE(didCleanupBuffer);
  }

  // Ensure that fusilli Buffer released the IREE buffer view which in turn
  // should release the buffer.
  REQUIRE(didCleanupBuffer);

  HIP_REQUIRE_SUCCESS(hipFree(ptr));
}
