// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the code to create and manage a Fusilli handle
// which is an RAII wrapper around shared IREE runtime resources
// (instances and devices) for proper initialization, cleanup and
// lifetime management.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_BACKEND_HANDLE_H
#define FUSILLI_BACKEND_HANDLE_H

#include "fusilli/backend/backend.h"
#include "fusilli/support/logging.h"

#include <iree/runtime/api.h>

#include <cstdint>
#include <utility>

namespace fusilli {

// An application using Fusilli to run operations on a given device
// must first initialize a handle on that device by calling
// `Handle::create()`. This allocates the necessary resources
// (runtime instance, HAL device) whose lifetimes are managed / owned
// by the handle(s).
class Handle {
public:
  // Creates a Handle for the specified backend. For AMDGPU backend, created
  // handle will use device 0 with the default (null) stream. Other create
  // overloads offer more specificity when setting device and stream.
  static ErrorOr<Handle> create(Backend backend) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Creating handle for backend: " << backend);

    // Create a shared IREE runtime instance (thread-safe) and use it
    // along with the backend to construct a handle (without initializing
    // the device yet).
    auto handle = Handle(backend, FUSILLI_TRY(Handle::createSharedInstance()));

    // Lazy create handle-specific IREE HAL device and populate the handle.
    switch (backend) {
    case Backend::CPU:
      FUSILLI_CHECK_ERROR(handle.createCPUDevice());
      break;
    case Backend::AMDGPU:
      FUSILLI_CHECK_ERROR(
          handle.createAMDGPUDevice(/*deviceId=*/0, /*stream=*/0));
      break;
    default:
      return ErrorObject(ErrorCode::InternalError,
                         "Handle::create got an unknown backend");
    }

    return ok(std::move(handle));
  }

  // Creates a Handle on the specified device. Currently device selection
  // supported only for AMDGPU backend. Created handle will use the default
  // (null) stream on device.
  //
  // NOTE: setting the device may set the active hip device read with
  // `hipGetDevice`.
  static ErrorOr<Handle> create(Backend backend, int deviceId) {
    return create(backend, deviceId, 0);
  }

  // Creates a Handle using the specified device and stream. Any executions will
  // be launched in stream order on passed in stream. Currently device and
  // stream selection supported only for AMDGPU backend.
  //
  // NOTE: `0` is valid hip stream, it's the null or default stream. The default
  // stream has implicit synchronization with all other streams and will
  // therefore limit concurrency with other streams.
  //
  // WARNING: the stream must be attached to device identified with deviceId.
  // One can check what device a stream is talking to with
  // `hipStreamGetDevice(stream, ...)`.
  static ErrorOr<Handle> create(Backend backend, int deviceId,
                                uintptr_t stream) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Creating handle for backend: "
                           << backend << " on device: " << deviceId
                           << " and stream: "
                           << reinterpret_cast<void *>(stream));

    FUSILLI_RETURN_ERROR_IF(backend != Backend::AMDGPU,
                            ErrorCode::InvalidArgument,
                            "Stream can only be set on AMDGPU backend");

    // Create a shared IREE runtime instance (thread-safe) and use it
    // along with the backend to construct a handle (without initializing
    // the device yet).
    auto handle = Handle(backend, FUSILLI_TRY(Handle::createSharedInstance()));

    // Lazy create handle-specific IREE HAL device and populate the handle.
    FUSILLI_CHECK_ERROR(
        handle.createAMDGPUDevice(/*deviceId=*/deviceId, /*stream=*/stream));

    return ok(std::move(handle));
  }

  // Automatic (implicit) conversion operator for
  // `Handle` -> `iree_hal_device_t *`.
  operator iree_hal_device_t *() const { return getDevice(); }

  // Delete copy constructors, keep default move constructor and destructor.
  Handle(const Handle &) = delete;
  Handle &operator=(const Handle &) = delete;
  Handle(Handle &&) noexcept = default;
  Handle &operator=(Handle &&) noexcept = default;
  ~Handle() = default;

  // Allow Graph and Buffer objects to access private Handle methods
  // namely `getDevice()` and `getInstance()`.
  friend class Graph;
  friend class Buffer;

private:
  // Creates static singleton IREE runtime instance shared across
  // handles/threads. Definition in `fusilli/backend/runtime.h`.
  static ErrorOr<IreeRuntimeInstanceSharedPtrType> createSharedInstance();

  // Creates IREE HAL CPU device for this handle. Definition in
  // `fusilli/backend/runtime.h`.
  ErrorObject createCPUDevice();

  // Creates a IREE HAL HIP device for this handle around the provided stream.
  // Definition in `fusilli/backend/runtime.h`.
  ErrorObject createAMDGPUDevice(int deviceId, uintptr_t stream);

  // Private constructor (use factory `create` method for handle creation).
  Handle(Backend backend, IreeRuntimeInstanceSharedPtrType instance)
      : backend_(backend), instance_(std::move(instance)) {}

  Backend getBackend() const { return backend_; }

  // Returns a raw pointer to the underlying IREE HAL device.
  // WARNING: The returned raw pointer is not safe to store since
  // its lifetime is tied to the `Handle` object and only
  // valid as long as this handle exists.
  iree_hal_device_t *getDevice() const { return device_.get(); }

  // Returns a raw pointer to the underlying IREE runtime instance.
  // WARNING: The returned raw pointer is not safe to store since
  // its lifetime is tied to the `Handle` objects and only
  // valid as long as at least one handle exists.
  iree_runtime_instance_t *getInstance() const { return instance_.get(); }

  // Order of initialization matters here.
  // `device_` depends on `backend_` and `instance_`.
  Backend backend_;
  IreeRuntimeInstanceSharedPtrType instance_;
  IreeHalDeviceUniquePtrType device_;
};

} // namespace fusilli

#endif // FUSILLI_BACKEND_HANDLE_H
