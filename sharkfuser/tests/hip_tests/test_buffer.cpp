// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>
#include <hip_utils.h>
#include <utils.h>

#include <catch2/catch_test_macros.hpp>

#include <hip/hip_runtime.h>

#include <iree/base/status.h>
#include <iree/hal/api.h>
#include <iree/runtime/api.h>

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace fusilli;

// Utility function for to import a HIP-allocated buffer into a 1D IREE buffer
// view.
static fusilli::ErrorOr<iree_hal_buffer_view_t *>
importTo1DBufferView(fusilli::Handle &handle, void *devicePtr,
                     size_t bufferSize, size_t elementCount,
                     iree_hal_buffer_release_callback_t releaseCallback =
                         iree_hal_buffer_release_callback_null()) {
  // Alocators.
  iree_hal_allocator_t *deviceAllocator = iree_hal_device_allocator(handle);
  iree_allocator_t hostAllocator = iree_allocator_system();

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
                      .ptr = reinterpret_cast<uint64_t>(devicePtr),
                  },
          },
  };
  iree_hal_buffer_t *importedBuffer = nullptr;
  FUSILLI_CHECK_ERROR(iree_hal_allocator_import_buffer(
      /*allocator=*/deviceAllocator, /*params=*/bufferParams,
      /*external_buffer=*/&externalBuffer, /*release_callback=*/releaseCallback,
      /*out_buffer=*/&importedBuffer));

  // Create an IREE buffer view for external buffer.
  const iree_hal_dim_t shape[] = {elementCount};
  const size_t shapeRank = 1;
  iree_hal_buffer_view_t *outBufferView = nullptr;
  FUSILLI_CHECK_ERROR(iree_hal_buffer_view_create(
      /*buffer=*/importedBuffer,
      /*shape_rank=*/shapeRank,
      /*shape=*/shape,
      /*element_type=*/IREE_HAL_ELEMENT_TYPE_FLOAT_32,
      /*encoding_type=*/IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      /*host_allocator=*/hostAllocator,
      /*out_buffer_view=*/&outBufferView));

  // The buffer view holds a reference to buffer and will handle release.
  iree_hal_buffer_release(importedBuffer);

  return fusilli::ok(outBufferView);
}

static void testBufferReleaseCallback(void *didCleanupBuffer,
                                      iree_hal_buffer_t *) {
  *(bool *)didCleanupBuffer = true;
}

TEST_CASE("Buffer import", "[buffer][hip_tests]") {
  // --------------------------------------------------------------------------
  //  Test that externally managed buffers, imported as IREE HAL Buffer View,
  //  and finally as a fusilli Buffer, are created successfully and cleaned up
  //  correctly.
  // --------------------------------------------------------------------------

  Handle handle = FUSILLI_REQUIRE_UNWRAP(Handle::create(Backend::AMDGPU));

  // Pointer to hipMalloc'ed buffer on device.
  void *ptr;
  const size_t elementCount = 64;
  const size_t bufferSize = elementCount * sizeof(float);
  HIP_REQUIRE_SUCCESS(hipMalloc(&ptr, sizeof(float) * elementCount));

  // Callback set on IREE buffer and called in `iree_hal_hip_buffer_destroy`.
  bool didCleanupBuffer = false;
  iree_hal_buffer_release_callback_t releaseCallback = {
      .fn = testBufferReleaseCallback, .user_data = &didCleanupBuffer};

  // Import external buffer using the utility function
  iree_hal_buffer_view_t *outBufferView =
      FUSILLI_REQUIRE_UNWRAP(importTo1DBufferView(
          handle, ptr, bufferSize, elementCount, releaseCallback));

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

TEST_CASE("Buffer read async allocated and populated buffer",
          "[buffer][hip_tests]") {
  // --------------------------------------------------------------------------
  //  Test that async allocations + writes to externally managed buffers are
  //  correctly stream ordered + read through fusilli::buffer::read + async hip
  //  APIs.
  // --------------------------------------------------------------------------

  // Get number of devices.
  int deviceCount;
  HIP_REQUIRE_SUCCESS(hipGetDeviceCount(&deviceCount));

  // Parameterize test by device id.
  int deviceId = 0;
  if (deviceCount > 1) {
    SECTION("non default device") {
      deviceId = deviceCount - 1;
      HIP_REQUIRE_SUCCESS(hipSetDevice(deviceId));
    }
  };

  // Create HIP stream.
  hipStream_t stream;
  HIP_REQUIRE_SUCCESS(hipStreamCreate(&stream));

  // Create handle.
  Handle handle = FUSILLI_REQUIRE_UNWRAP(
      Handle::create(Backend::AMDGPU, /*deviceId=*/deviceId,
                     /*stream=*/reinterpret_cast<uintptr_t>(stream)));

  // Allocate buffer asynchronously on the stream. `hipMallocAsync` returns
  // immediately with a pointer to where buffer will be allocated.
  const size_t elementCount = 64;
  const size_t bufferSize = elementCount * sizeof(float);
  void *devicePtr;
  HIP_REQUIRE_SUCCESS(hipMallocAsync(&devicePtr, bufferSize, stream));

  // Verify the buffer is on the correct device.
  hipPointerAttribute_t attributes;
  HIP_REQUIRE_SUCCESS(hipPointerGetAttributes(&attributes, devicePtr));
  REQUIRE(attributes.device == deviceId);

  // Import as IREE hal buffer view.
  iree_hal_buffer_view_t *bufferView = FUSILLI_REQUIRE_UNWRAP(
      importTo1DBufferView(handle, devicePtr, bufferSize, elementCount));

  // Import as fusilli Buffer
  Buffer fusilliBuffer =
      FUSILLI_REQUIRE_UNWRAP(fusilli::Buffer::import(bufferView));
  iree_hal_buffer_view_release(bufferView);

  // Write test data to buffer asynchronously.
  std::vector<float> hostData(elementCount);
  for (size_t i = 0; i < elementCount; ++i) {
    hostData[i] = static_cast<float>(i);
  }
  // NOLINTBEGIN(misc-include-cleaner)
  // The header usage here is correct but this hip headers don't follow
  // misc-include-cleaner.
  HIP_REQUIRE_SUCCESS(hipMemcpyAsync(devicePtr, hostData.data(), bufferSize,
                                     hipMemcpyHostToDevice, stream));
  // NOLINTEND(misc-include-cleaner)

  // Read buffer through fusilli::Buffer::read, read should be stream ordered
  // with buffer initialization, i.e. the async allocation and memCopy should
  // happen before data is read.
  std::vector<float> readData;
  FUSILLI_REQUIRE_OK(fusilliBuffer.read(handle, readData));

  // Verify the data.
  REQUIRE(readData == hostData);

  // Clean up.
  HIP_REQUIRE_SUCCESS(hipFree(devicePtr));
  HIP_REQUIRE_SUCCESS(hipStreamDestroy(stream));
}
