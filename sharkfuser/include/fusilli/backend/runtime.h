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
inline ErrorOr<IreeRuntimeInstanceSharedPtrType>
FusilliHandle::createSharedInstance() {
  // Mutex for thread-safe initialization of weakInstance
  static std::mutex instanceMutex;

  // Static weak_ptr to the IREE runtime instance ensures that the
  // instance is only created once and shared across all handles
  // without prolonging its lifetime till program termination. This
  // allows the instance to be released when the last handle owning
  // it goes out of scope, as opposed to hogging on to it until the
  // static variable goes out of scope upon program termination.
  static std::weak_ptr<iree_runtime_instance_t> weakInstance;

  // If multiple threads simultaneously request a handle, they will
  // race into `createSharedInstance()` but only one will succeed in
  // creating the instance, and others will use it.
  std::lock_guard<std::mutex> lock(instanceMutex);

  // Try to lock the weak_ptr to get the shared_ptr.
  IreeRuntimeInstanceSharedPtrType sharedInstance = weakInstance.lock();

  // If weak_ptr expired, it means no handles are alive (and holding the
  // instance), create a new one.
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

    weakInstance = sharedInstance;
  }

  return ok(sharedInstance);
}

// Create IREE HAL device for this handle
// TODO: This just creates the default device for now and needs
// work to allow configuring the specific device based on path /
// ID / ordinal / URI.
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
// and load the compiled artifact into it.
inline ErrorObject Graph::createPerGraphSession(const FusilliHandle &handle,
                                                const std::string &vmfbPath) {
  // Skip to loading if session is already created.
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
