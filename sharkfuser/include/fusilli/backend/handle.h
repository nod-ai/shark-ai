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

#include <memory>
#include <mutex>

namespace fusilli {

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

using IreeRuntimeInstanceSharedPtrType =
    std::shared_ptr<iree_runtime_instance_t>;
using IreeHalDeviceUniquePtrType =
    std::unique_ptr<iree_hal_device_t, IreeHalDeviceDeleter>;

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
  // Create static singleton IREE runtime instance shared across handles/threads
  // TODO(sjain-stanford): Consider moving to `std::call_once` to avoid
  // paying the cost of acquiring/releasing the mutex lock on every call
  // to `FusilliHandle::createSharedInstance()`. The only minor issue is the
  // lambda for `call_once` expects a void callable but in our case we
  // return ErrorObject inside `FUSILLI_CHECK_ERROR` so it might need
  // some restructuring to properly capture/propagate the error state.
  static ErrorOr<IreeRuntimeInstanceSharedPtrType> createSharedInstance() {
    // Mutex for thread-safe initialization of sharedInstance
    static std::mutex instanceMutex;
    static IreeRuntimeInstanceSharedPtrType sharedInstance;

    std::lock_guard<std::mutex> lock(instanceMutex);
    if (sharedInstance == nullptr) {
      FUSILLI_LOG_LABEL_ENDL("INFO: Creating shared IREE runtime instance");
      iree_runtime_instance_options_t opts;
      iree_runtime_instance_options_initialize(&opts);
      iree_runtime_instance_options_use_all_available_drivers(&opts);
      iree_runtime_instance_t *rawInstance = nullptr;

      FUSILLI_CHECK_ERROR(iree_runtime_instance_create(
          &opts, iree_allocator_system(), &rawInstance));

      sharedInstance = IreeRuntimeInstanceSharedPtrType(
          rawInstance, IreeRuntimeInstanceDeleter());
    }

    return ok(sharedInstance);
  }

  // Create IREE HAL device for this handle
  ErrorObject createPerHandleDevice() {
    FUSILLI_LOG_LABEL_ENDL("INFO: Creating per-handle IREE HAL device");
    iree_hal_device_t *rawDevice = nullptr;

    FUSILLI_CHECK_ERROR(iree_runtime_instance_try_create_default_device(
        instance_.get(), iree_make_cstring_view(halDriver.at(backend_)),
        &rawDevice));

    device_ = IreeHalDeviceUniquePtrType(rawDevice);
    return ok();
  }

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
