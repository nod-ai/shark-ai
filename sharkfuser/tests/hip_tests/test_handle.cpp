// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>
#include <hip_utils.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>
#include <hip/hip_runtime.h>
#include <iree/runtime/api.h>

#include <cstdint>
#include <cstdio>
#include <vector>

using namespace fusilli;

TEST_CASE("Handle creation with deviceId", "[handle][hip_tests]") {
  SECTION("Create handle", "[handle][hip_tests]") {
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

    // Create handle.
    Handle handle = FUSILLI_REQUIRE_UNWRAP(
        Handle::create(Backend::AMDGPU, /*deviceId=*/deviceId));
    REQUIRE(static_cast<iree_hal_device_t *>(handle) != nullptr);

    // Verify we can allocate, write, and read buffers.
    const std::vector<iree_hal_dim_t> bufferShape = {32};
    std::vector<float> hostData(32, 3.14f);
    Buffer deviceBuffer =
        FUSILLI_REQUIRE_UNWRAP(Buffer::allocate(handle, bufferShape, hostData));
    std::vector<float> readData;
    FUSILLI_REQUIRE_OK(deviceBuffer.read(handle, readData));
    REQUIRE(readData == hostData);
  }
}

TEST_CASE("Handle creation with stream and deviceId", "[handle][hip_tests]") {
  SECTION("Create handle") {
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

    // Create a HIP stream.
    hipStream_t stream;
    HIP_REQUIRE_SUCCESS(hipStreamCreate(&stream));

    // Create a handle.
    Handle handle = FUSILLI_REQUIRE_UNWRAP(
        Handle::create(Backend::AMDGPU, /*deviceId=*/deviceId,
                       /*stream=*/reinterpret_cast<uintptr_t>(stream)));
    REQUIRE(static_cast<iree_hal_device_t *>(handle) != nullptr);

    // Verify we can allocate, write, and read buffers.
    const std::vector<iree_hal_dim_t> bufferShape = {32};
    std::vector<float> hostData(32, 3.14f);
    Buffer deviceBuffer =
        FUSILLI_REQUIRE_UNWRAP(Buffer::allocate(handle, bufferShape, hostData));
    std::vector<float> readData;
    FUSILLI_REQUIRE_OK(deviceBuffer.read(handle, readData));
    REQUIRE(readData == hostData);

    // Clean up.
    HIP_REQUIRE_SUCCESS(hipStreamDestroy(stream));
  }

  SECTION("Create multiple handles with different HIP streams") {
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

    // Create multiple HIP streams.
    hipStream_t stream1, stream2;
    HIP_REQUIRE_SUCCESS(hipStreamCreate(&stream1));
    HIP_REQUIRE_SUCCESS(hipStreamCreate(&stream2));

    // Create handles.
    Handle handle1 = FUSILLI_REQUIRE_UNWRAP(
        Handle::create(Backend::AMDGPU, /*deviceId=*/deviceId,
                       /*stream=*/reinterpret_cast<uintptr_t>(stream1)));
    Handle handle2 = FUSILLI_REQUIRE_UNWRAP(
        Handle::create(Backend::AMDGPU, /*deviceId=*/deviceId,
                       /*stream=*/reinterpret_cast<uintptr_t>(stream2)));
    REQUIRE(static_cast<iree_hal_device_t *>(handle1) != nullptr);
    REQUIRE(static_cast<iree_hal_device_t *>(handle2) != nullptr);

    // Verify we can allocate, write, and read buffers on handle1.
    const std::vector<iree_hal_dim_t> bufferShape = {32};
    std::vector<float> hostData(32, 3.14f);
    Buffer deviceBuffer1 = FUSILLI_REQUIRE_UNWRAP(
        Buffer::allocate(handle1, bufferShape, hostData));
    std::vector<float> readData;
    FUSILLI_REQUIRE_OK(deviceBuffer1.read(handle1, readData));
    REQUIRE(readData == hostData);

    // Verify we can allocate, write, and read buffers on handle2
    Buffer deviceBuffer2 = FUSILLI_REQUIRE_UNWRAP(
        Buffer::allocate(handle1, bufferShape, hostData));
    readData.clear();
    FUSILLI_REQUIRE_OK(deviceBuffer2.read(handle1, readData));
    REQUIRE(readData == hostData);

    // Clean up.
    HIP_REQUIRE_SUCCESS(hipStreamDestroy(stream1));
    HIP_REQUIRE_SUCCESS(hipStreamDestroy(stream2));
  }
}
