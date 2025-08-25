// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains all the wrapper code around IREE runtime C-APIs to create
// and manage instances, devices, sessions and calls.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_BACKEND_RUNTIME_H
#define FUSILLI_BACKEND_RUNTIME_H

#include "fusilli/backend/backend.h"
#include "fusilli/backend/handle.h"
#include "fusilli/graph/graph.h"
#include "fusilli/support/logging.h"

#include <iree/runtime/api.h>

#include <mutex>
#include <string>

namespace fusilli {

// Create static singleton IREE runtime instance shared across handles/threads
// TODO(sjain-stanford): Consider moving to `std::call_once` to avoid
// paying the cost of acquiring/releasing the mutex lock on every call
// to `FusilliHandle::createSharedInstance()`. The only minor issue is the
// lambda for `call_once` expects a void callable but in our case we
// return ErrorObject inside `FUSILLI_CHECK_ERROR` so it might need
// some restructuring to properly capture/propagate the error state.
inline ErrorOr<IreeRuntimeInstanceSharedPtrType>
FusilliHandle::createSharedInstance() {
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
inline ErrorObject FusilliHandle::createPerHandleDevice() {
  FUSILLI_LOG_LABEL_ENDL("INFO: Creating per-handle IREE HAL device");
  iree_hal_device_t *rawDevice = nullptr;

  FUSILLI_CHECK_ERROR(iree_runtime_instance_try_create_default_device(
      instance_.get(), iree_make_cstring_view(halDriver.at(backend_)),
      &rawDevice));

  device_ = IreeHalDeviceUniquePtrType(rawDevice);
  return ok();
}

// Create IREE runtime session for this graph (if not available already)
// and load the compiled artifact into it
inline ErrorObject Graph::createPerGraphSession(const FusilliHandle &handle,
                                                const std::string &vmfbPath) {
  // Skip to loading if session is already created
  if (session_ == nullptr) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Creating per-graph IREE runtime session");
    iree_runtime_session_options_t opts;
    iree_runtime_session_options_initialize(&opts);
    iree_runtime_session_t *rawSession = nullptr;

    FUSILLI_CHECK_ERROR(iree_runtime_session_create_with_device(
        handle.getInstance(), &opts, handle.getDevice(),
        iree_runtime_instance_host_allocator(handle.getInstance()),
        &rawSession));

    session_ = IreeRuntimeSessionUniquePtrType(rawSession);
  }

  FUSILLI_LOG_LABEL_ENDL("INFO: Loading module in IREE runtime session");
  FUSILLI_CHECK_ERROR(iree_runtime_session_append_bytecode_module_from_file(
      session_.get(), vmfbPath.c_str()));

  return ok();
}

} // namespace fusilli

#endif // FUSILLI_BACKEND_RUNTIME_H
