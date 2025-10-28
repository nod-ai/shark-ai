// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains the code to create and manage a Fusilli buffer
// which is an RAII wrapper around IREE HAL buffer view for proper
// initialization, cleanup and lifetime management.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_BACKEND_BUFFER_H
#define FUSILLI_BACKEND_BUFFER_H

#include "fusilli/backend/backend.h"
#include "fusilli/support/logging.h"

#include <iree/runtime/api.h>

#include <utility>
#include <vector>

namespace fusilli {

// Forward declaration of Handle class.
class Handle;

class Buffer {
public:
  // Factory: Allocates a new buffer view and takes ownership.
  // Definition in `fusilli/backend/runtime.h`.
  template <typename T>
  static ErrorOr<Buffer>
  allocate(const Handle &handle, const std::vector<iree_hal_dim_t> &bufferShape,
           const std::vector<T> &bufferData);

  // Factory: Imports an existing buffer view and retains ownership.
  // Definition in `fusilli/backend/runtime.h`.
  static ErrorOr<Buffer> import(iree_hal_buffer_view_t *externalBufferView);

  // Reads device buffer by initiating a device-to-host transfer then
  // populating `outData`. Definition in `fusilli/backend/runtime.h`.
  template <typename T>
  ErrorObject read(const Handle &handle, std::vector<T> &outData);

  // Automatic (implicit) conversion operator for
  // `Buffer` -> `iree_hal_buffer_view_t *`.
  operator iree_hal_buffer_view_t *() const { return getBufferView(); }

  // Delete copy constructors, keep default move constructor and destructor.
  Buffer(const Buffer &) = delete;
  Buffer &operator=(const Buffer &) = delete;
  Buffer(Buffer &&) noexcept = default;
  Buffer &operator=(Buffer &&) noexcept = default;
  ~Buffer() = default;

private:
  // Returns a raw pointer to the underlying IREE HAL buffer view.
  // WARNING: The returned raw pointer is not safe to store since
  // its lifetime is tied to the `Buffer` object and only valid
  // as long as this buffer exists.
  iree_hal_buffer_view_t *getBufferView() const { return bufferView_.get(); }

  // Explicit constructor is private. Create `Buffer` using one of the
  // factory methods above - `Buffer::import` or `Buffer::allocate`.
  explicit Buffer(IreeHalBufferViewUniquePtrType bufferView)
      : bufferView_(std::move(bufferView)) {}

  IreeHalBufferViewUniquePtrType bufferView_;
};

} // namespace fusilli

#endif // FUSILLI_BACKEND_BUFFER_H
