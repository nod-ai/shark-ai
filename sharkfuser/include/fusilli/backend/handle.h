// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the code to create and manage Fusilli handles
// which wrap around shared IREE runtime resources (instances and devices).
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_BACKEND_HANDLE_H
#define FUSILLI_BACKEND_HANDLE_H

#include "fusilli/backend/backend.h"
#include "fusilli/support/logging.h"

#include <iree/runtime/api.h>

namespace fusilli {

class FusilliHandle {
public:
  static ErrorOr<FusilliHandle> create(Backend backend) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Creating handle for backend: " << backend);
    // Create a shared IREE runtime instance (thread-safe) and use it
    // along with the backend to construct a handle (without
    // initializing the device yet)
    auto handle = FusilliHandle(backend, FUSILLI_TRY(createSharedInstance()));
    // Lazy create handle-specific IREE HAL device and populate the handle
    FUSILLI_CHECK_ERROR(handle.createPerHandleDevice());
    return ok(std::move(handle));
  }

  // Delete copy constructors, keep default move constructor and destructor
  FusilliHandle(const FusilliHandle &) = delete;
  FusilliHandle &operator=(const FusilliHandle &) = delete;
  FusilliHandle(FusilliHandle &&) = default;
  FusilliHandle &operator=(FusilliHandle &&) = default;
  ~FusilliHandle() = default;

  Backend getBackend() const { return backend_; }
  iree_hal_device_t *getDevice() const { return device_.get(); }
  iree_runtime_instance_t *getInstance() const { return instance_.get(); }

private:
  // Creates static singleton IREE runtime instance shared across
  // handles/threads
  static ErrorOr<IreeRuntimeInstanceSharedPtrType> createSharedInstance();

  // Creates IREE HAL device for this handle
  ErrorObject createPerHandleDevice();

  // Private constructor (use factory `create` method for handle creation)
  FusilliHandle(Backend backend, IreeRuntimeInstanceSharedPtrType instance)
      : backend_(backend), instance_(instance) {}

  // Order of initialization matters here.
  // `device_` depends on `backend_` and `instance_`.
  Backend backend_;
  IreeRuntimeInstanceSharedPtrType instance_;
  IreeHalDeviceUniquePtrType device_;
};

} // namespace fusilli

#endif // FUSILLI_BACKEND_HANDLE_H
