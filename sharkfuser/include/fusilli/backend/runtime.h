// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the inline definitions for all the wrapper code around
// IREE runtime C-APIs to create and manage instances, devices, sessions and
// calls.
//
// Here's a rough mapping of Fusilli constructs to IREE runtime constructs
// (based on scope and lifetime):
//
//  - Group of `Handle`s manage the IREE runtime instance lifetime.
//    An instance is shared across handles/threads/sessions and released
//    when the last handle goes out of scope.
//  - `Handle` manages IREE HAL device lifetime. Handles may be shared
//    by multiple graphs (as long as they intend to run on the same device).
//    Separate physical devices should have their own handles (hence logical
//    HAL device) created. Graphs running on the same physical devices should
//    reuse the same handle (hence logical HAL device). The device is released
//    when the handle holding it goes out of scope.
//  - `Graph` manages IREE runtime session lifetime. A session holds state on
//    the HAL device and the loaded VM modules.
//  - `Buffer` manages IREE HAL buffer view lifetime. The buffer view is
//    released when the `Buffer` object holding it goes out of scope.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_BACKEND_RUNTIME_H
#define FUSILLI_BACKEND_RUNTIME_H

#include "fusilli/backend/backend.h"
#include "fusilli/backend/buffer.h"
#include "fusilli/backend/handle.h"
#include "fusilli/graph/graph.h"
#include "fusilli/support/logging.h"

#include <iree/runtime/api.h>

#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace fusilli {

//===----------------------------------------------------------------------===//
//
// Handle Runtime API Methods
//
//===----------------------------------------------------------------------===//

// Create static singleton IREE runtime instance shared across handles/threads.
inline ErrorOr<IreeRuntimeInstanceSharedPtrType>
Handle::createSharedInstance() {
  // Mutex for thread-safe initialization of weakInstance.
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

  // Try to get the shared_ptr from the weak_ptr (if it exists).
  IreeRuntimeInstanceSharedPtrType sharedInstance = weakInstance.lock();

  // If weak_ptr expired, it means no handles are alive and holding the
  // instance, so create a new instance.
  if (sharedInstance == nullptr) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Creating shared IREE runtime instance");
    iree_runtime_instance_options_t opts;
    iree_runtime_instance_options_initialize(&opts);
    iree_runtime_instance_options_use_all_available_drivers(&opts);

    iree_runtime_instance_t *rawInstance = nullptr;
    FUSILLI_CHECK_ERROR(iree_runtime_instance_create(
        &opts, iree_allocator_system(), &rawInstance));

    // Wrap the raw instance ptr with a shared_ptr and custom deleter
    // for lifetime management.
    sharedInstance = IreeRuntimeInstanceSharedPtrType(
        rawInstance, IreeRuntimeInstanceDeleter());

    weakInstance = sharedInstance;
  }

  return ok(sharedInstance);
}

// Create IREE HAL device for this handle.
// TODO(#2151): This just creates the default device for now (which is like
// a die roll when multiple GPUs are available). In the future we need to
// allow specifying the exact device based on path or ID.
inline ErrorObject Handle::createPerHandleDevice() {
  FUSILLI_LOG_LABEL_ENDL("INFO: Creating per-handle IREE HAL device");

  iree_hal_device_t *rawDevice = nullptr;
  FUSILLI_CHECK_ERROR(iree_runtime_instance_try_create_default_device(
      instance_.get(), iree_make_cstring_view(halDriver.at(backend_)),
      &rawDevice));

  // Wrap the raw device ptr with a unique_ptr and custom deleter
  // for lifetime management.
  device_ = IreeHalDeviceUniquePtrType(rawDevice);

  return ok();
}

//===----------------------------------------------------------------------===//
//
// Graph Runtime API Methods
//
//===----------------------------------------------------------------------===//

// Create IREE runtime session for this graph and load the compiled artifact.
inline ErrorObject Graph::createPerGraphSession(const Handle &handle,
                                                const std::string &vmfbPath) {
  // Create a session even if one was created earlier, since the handle
  // (hence device) might have changed and we might be re-compiling the graph
  // for the new device.
  FUSILLI_LOG_LABEL_ENDL("INFO: Creating per-graph IREE runtime session");
  iree_runtime_session_options_t opts;
  iree_runtime_session_options_initialize(&opts);

  iree_runtime_session_t *rawSession = nullptr;
  FUSILLI_CHECK_ERROR(iree_runtime_session_create_with_device(
      handle.getInstance(), &opts, handle.getDevice(),
      iree_runtime_instance_host_allocator(handle.getInstance()), &rawSession));

  // Wrap the raw session ptr with a unique_ptr and custom deleter
  // for lifetime management.
  session_ = IreeRuntimeSessionUniquePtrType(rawSession);

  // Load the vmfb into the session.
  FUSILLI_LOG_LABEL_ENDL("INFO: Loading module in IREE runtime session");
  FUSILLI_CHECK_ERROR(iree_runtime_session_append_bytecode_module_from_file(
      session_.get(), vmfbPath.c_str()));

  return ok();
}

// Executes the graph using IREE runtime. Requires a `variantPack` which is a
// map from `TensorAttr` to `Buffer` wrapping the `iree_hal_buffer_view_t *`.
//
// TODO(#2232): Memoize `iree_runtime_call_t` initialization and populate buffer
// views at setup to avoid paying the penalty for every `Graph::execute`
// invocation. Use `iree_runtime_call_reset` to reset the call inputs/outputs
// if needed.
inline ErrorObject Graph::execute(
    const std::unordered_map<std::shared_ptr<TensorAttr>,
                             std::shared_ptr<Buffer>> &variantPack) const {
  FUSILLI_LOG_LABEL_ENDL("INFO: Executing Graph");
  FUSILLI_RETURN_ERROR_IF(session_ == nullptr, ErrorCode::NotCompiled,
                          "Graph must be compiled before being executed");

  iree_runtime_call_t call;
  FUSILLI_CHECK_ERROR(iree_runtime_call_initialize_by_name(
      session_.get(), iree_make_cstring_view("module.main"), &call));

  // Populate output buffers.
  for (const auto &output : fullGraphOutputsSorted_) {
    FUSILLI_RETURN_ERROR_IF(!variantPack.contains(output), // C++20
                            ErrorCode::TensorNotFound,
                            "Output tensor missing from variantPack");
    FUSILLI_CHECK_ERROR(iree_runtime_call_inputs_push_back_buffer_view(
        &call, *(variantPack.at(output))));
  }

  // Populate input buffers.
  for (const auto &input : fullGraphInputsSorted_) {
    FUSILLI_RETURN_ERROR_IF(!variantPack.contains(input), // C++20
                            ErrorCode::TensorNotFound,
                            "Input tensor missing from variantPack");
    FUSILLI_CHECK_ERROR(iree_runtime_call_inputs_push_back_buffer_view(
        &call, *(variantPack.at(input))));
  }

  // Synchronously perform the call.
  FUSILLI_CHECK_ERROR(iree_runtime_call_invoke(&call, /*flags=*/0));

  iree_runtime_call_deinitialize(&call);
  return ok();
}

// Factory: Allocates a new buffer view and takes ownership.
template <typename T>
inline ErrorOr<Buffer>
Buffer::allocate(const Handle &handle,
                 const std::vector<iree_hal_dim_t> &bufferShape,
                 const std::vector<T> &bufferData) {
  FUSILLI_LOG_LABEL_ENDL("INFO: Allocating new device buffer");

  iree_hal_buffer_view_t *rawBufferView = nullptr;
  FUSILLI_CHECK_ERROR(iree_hal_buffer_view_allocate_buffer_copy(
      // IREE HAL device and allocator:
      handle.getDevice(), iree_hal_device_allocator(handle.getDevice()),
      // Shape rank and dimensions:
      bufferShape.size(), bufferShape.data(),
      // Element type:
      getIreeHalElementTypeForT<T>(),
      // Encoding type:
      IREE_HAL_ENCODING_TYPE_DENSE_ROW_MAJOR,
      (iree_hal_buffer_params_t){
          // Intended usage of this buffer (transfers, dispatches, etc):
          .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
          // Access to allow to this memory:
          .access = IREE_HAL_MEMORY_ACCESS_ALL,
          // Where to allocate (host or device):
          .type = IREE_HAL_MEMORY_TYPE_DEVICE_LOCAL,
      },
      // The actual heap buffer to wrap or clone and its allocator:
      iree_make_const_byte_span(bufferData.data(),
                                bufferData.size() * sizeof(T)),
      // Buffer view + storage are returned and owned by the caller
      // (this Buffer object in this case):
      &rawBufferView));

  return ok(Buffer(IreeHalBufferViewUniquePtrType(rawBufferView)));
}

//===----------------------------------------------------------------------===//
//
// Buffer Runtime API Methods
//
//===----------------------------------------------------------------------===//

// Factory: Imports an existing buffer view and retains ownership.
inline ErrorOr<Buffer>
Buffer::import(iree_hal_buffer_view_t *externalBufferView) {
  FUSILLI_LOG_LABEL_ENDL("INFO: Importing pre-allocated device buffer");
  FUSILLI_RETURN_ERROR_IF(
      externalBufferView == nullptr, ErrorCode::RuntimeFailure,
      "Buffer::import failed as externalBufferView* is NULL");
  iree_hal_buffer_view_retain(externalBufferView);
  return ok(Buffer(IreeHalBufferViewUniquePtrType(externalBufferView)));
}

// Reads device buffer by initiating a device-to-host transfer and
// populating `outData`.
template <typename T>
inline ErrorObject Buffer::read(const Handle &handle, std::vector<T> &outData) {
  FUSILLI_LOG_LABEL_ENDL("INFO: Reading device buffer through D2H transfer");
  FUSILLI_RETURN_ERROR_IF(outData.size() != 0, ErrorCode::RuntimeFailure,
                          "Buffer::read failed as outData is NOT empty");

  // Get the underlying buffer from the buffer view.
  iree_hal_buffer_t *buffer = iree_hal_buffer_view_buffer(getBufferView());

  // Resize output vector `outData` based on buffer size.
  iree_device_size_t byte_length =
      iree_hal_buffer_view_byte_length(getBufferView());
  outData.resize(byte_length / sizeof(T));

  // Copy results back from device.
  FUSILLI_CHECK_ERROR(iree_hal_device_transfer_d2h(
      handle.getDevice(), buffer, 0, outData.data(), byte_length,
      IREE_HAL_TRANSFER_BUFFER_FLAG_DEFAULT, iree_infinite_timeout()));

  return ok();
}

} // namespace fusilli

#endif // FUSILLI_BACKEND_RUNTIME_H
