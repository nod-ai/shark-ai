// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_SUPPORT_IREE_HELPERS_H
#define SHORTFIN_SUPPORT_IREE_HELPERS_H

#include <memory>
#include <stdexcept>

#include "iree/base/api.h"
#include "iree/base/internal/file_io.h"
#include "iree/hal/api.h"
#include "iree/io/parameter_index_provider.h"
#include "iree/modules/hal/types.h"
#include "iree/task/api.h"
#include "iree/vm/api.h"
#include "iree/vm/ref_cc.h"
#include "shortfin/support/api.h"

#if !defined(SHORTFIN_IREE_LOG_RC)
#define SHORTFIN_IREE_LOG_RC 0
#endif

namespace shortfin {

// -------------------------------------------------------------------------- //
// Type conversion from C types.
// These are in the global shortfin namespace.
// -------------------------------------------------------------------------- //

inline std::string_view to_string_view(iree_string_view_t isv) {
  return std::string_view(isv.data, isv.size);
}

inline iree_string_view_t to_iree_string_view(std::string_view sv) {
  return iree_make_string_view(sv.data(), sv.size());
}

namespace iree {

// -------------------------------------------------------------------------- //
// RAII wrappers
// -------------------------------------------------------------------------- //

namespace detail {

#if SHORTFIN_IREE_LOG_RC
void SHORTFIN_API LogIREERetain(const char *type_name, void *ptr);
void SHORTFIN_API LogIREERelease(const char *type_name, void *ptr);
void SHORTFIN_API LogIREESteal(const char *type_name, void *ptr);
void SHORTFIN_API LogLiveRefs();
#else
inline void LogIREERetain(const char *type_name, void *ptr) {}
inline void LogIREERelease(const char *type_name, void *ptr) {}
inline void LogIREESteal(const char *type_name, void *ptr) {}
inline void LogLiveRefs() {}
#endif

};  // namespace detail

// Wraps an IREE retain/release style object pointer in a smart-pointer
// like wrapper.
template <typename T, typename Helper>
class object_ptr {
 public:
  object_ptr() : ptr(nullptr) {}
  object_ptr(const object_ptr &other) : ptr(other.ptr) {
    if (ptr) {
      Helper::retain(ptr);
    }
  }
  object_ptr(object_ptr &&other) : ptr(other.ptr) { other.ptr = nullptr; }
  object_ptr &operator=(const object_ptr &other) = delete;
  object_ptr &operator=(object_ptr &&other) {
    reset();
    ptr = other.ptr;
    other.ptr = nullptr;
    return *this;
  }
  ~object_ptr() { reset(); }

  // Constructs a new object_ptr by transferring ownership of a raw
  // pointer.
  static object_ptr steal_reference(T *owned) {
    Helper::steal(owned);
    return object_ptr(owned);
  }
  // Constructs a new object_ptr by retaining a raw pointer.
  static object_ptr borrow_reference(T *owned) {
    Helper::retain(owned);
    return object_ptr(owned);
  }
  operator T *() const noexcept { return ptr; }

  class Assignment {
   public:
    explicit Assignment(object_ptr *assign) : assign(assign) {}
    ~Assignment() {
      if (assign->ptr) {
        Helper::steal(assign->ptr);
      }
    }

    constexpr operator T **() noexcept {
      return reinterpret_cast<T **>(&assign->ptr);
    }

   private:
    object_ptr *assign = nullptr;
  };

  // Releases any current reference held by this instance and returns a
  // pointer to the raw backing pointer. This is typically used for passing
  // to out parameters which are expected to store a new owned pointer directly.
  constexpr Assignment for_output() noexcept {
    reset();
    return Assignment(this);
  }

  operator bool() const { return ptr != nullptr; }
  T *get() const { return ptr; }
  void reset() {
    if (ptr) {
      Helper::release(ptr);
    }
    ptr = nullptr;
  }
  T *release() {
    T *ret = ptr;
    ptr = nullptr;
    return ret;
  }

 private:
  // Assumes the reference count for owned_ptr.
  object_ptr(T *owned_ptr) : ptr(owned_ptr) {}
  T *ptr = nullptr;

  friend class Assignment;
};

// Defines a reference counting helper struct named like
// iree_hal_buffer_ptr_helper (for type_stem == hal_buffer).
// These must be defined in the shortfin::iree::detail namespace.
#define SHORTFIN_IREE_DEF_PTR(type_stem)                                \
  namespace detail {                                                    \
  struct type_stem##_ptr_helper {                                       \
    static void steal(iree_##type_stem##_t *obj) {                      \
      LogIREESteal(#type_stem "_t", obj);                               \
    }                                                                   \
    static void retain(iree_##type_stem##_t *obj) {                     \
      LogIREERetain(#type_stem "_t", obj);                              \
      iree_##type_stem##_retain(obj);                                   \
    }                                                                   \
    static void release(iree_##type_stem##_t *obj) {                    \
      LogIREERelease(#type_stem "_t", obj);                             \
      iree_##type_stem##_release(obj);                                  \
    }                                                                   \
  };                                                                    \
  }                                                                     \
  using type_stem##_ptr =                                               \
      object_ptr<iree_##type_stem##_t, detail::type_stem##_ptr_helper>; \
  static_assert(sizeof(type_stem##_ptr) == sizeof(iree_##type_stem##_t *))

SHORTFIN_IREE_DEF_PTR(hal_command_buffer);
SHORTFIN_IREE_DEF_PTR(hal_buffer);
SHORTFIN_IREE_DEF_PTR(hal_buffer_view);
SHORTFIN_IREE_DEF_PTR(hal_device);
SHORTFIN_IREE_DEF_PTR(hal_driver);
SHORTFIN_IREE_DEF_PTR(hal_fence);
SHORTFIN_IREE_DEF_PTR(hal_semaphore);
SHORTFIN_IREE_DEF_PTR(io_file_handle);
SHORTFIN_IREE_DEF_PTR(io_parameter_index);
SHORTFIN_IREE_DEF_PTR(io_parameter_provider);
SHORTFIN_IREE_DEF_PTR(task_executor);
SHORTFIN_IREE_DEF_PTR(vm_context);
SHORTFIN_IREE_DEF_PTR(vm_instance);
SHORTFIN_IREE_DEF_PTR(vm_list);
SHORTFIN_IREE_DEF_PTR(vm_module);

// Holds a pointer allocated by some allocator, deleting it if still owned
// at destruction time.
template <typename T>
struct allocated_ptr {
  iree_allocator_t allocator;

  allocated_ptr(iree_allocator_t allocator) : allocator(allocator) {}
  ~allocated_ptr() { reset(); }

  void reset(T *other = nullptr) {
    iree_allocator_free(allocator, ptr);
    ptr = other;
  }

  T *release() {
    T *ret = ptr;
    ptr = nullptr;
    return ret;
  }

  T *operator->() noexcept { return ptr; }
  T **operator&() noexcept { return &ptr; }
  operator T *() noexcept { return ptr; }
  T *get() noexcept { return ptr; }

  // Releases any current reference held by this instance and returns a
  // pointer to the raw backing pointer. This is typically used for passing
  // to out parameters which are expected to store a new owned pointer directly.
  T **for_output() {
    reset();
    return &ptr;
  }

 private:
  T *ptr = nullptr;
};

// Wraps an iree_file_contents_t*, freeing it when it goes out of scope.
// The contents can be released as an iree_allocator_t which transfers
// ownership to some consumer.
class file_contents_ptr {
 public:
  file_contents_ptr() {}

  // Frees any contained contents.
  void reset() noexcept {
    if (contents_) {
      iree_file_contents_free(contents_);
      contents_ = nullptr;
    }
  }

  // Frees any contained contents and returns a pointer to the pointer that
  // can be passed as an out parameter, causing this instance to take ownership
  // of anything set on it.
  iree_file_contents_t **for_output() noexcept {
    reset();
    return &contents_;
  }

  operator iree_file_contents_t *() noexcept { return contents_; }

  // Access the raw contents.
  iree_const_byte_span_t const_buffer() const noexcept {
    return contents_->const_buffer;
  }

  // Returns a deallocator that can be used to free the contents. Note that
  // this method alone does not release ownership of the contents. Typically
  // that is done once the consumer of this allocator returns successfully.
  iree_allocator_t deallocator() {
    return iree_file_contents_deallocator(contents_);
  }

  // Releases ownership of the contained contents.
  iree_file_contents_t *release() {
    iree_file_contents_t *p = contents_;
    contents_ = nullptr;
    return p;
  }

 private:
  iree_file_contents_t *contents_ = nullptr;
};

// -------------------------------------------------------------------------- //
// error and status handling
// -------------------------------------------------------------------------- //

// Captures an iree_status_t as an exception. The intent is that this is
// only used for failing statuses and the error instance will be
// immediately thrown.
class SHORTFIN_API error : public std::exception {
 public:
  error(std::string message, iree_status_t failing_status);
  error(iree_status_t failing_status);
  error(const error &other)
      : code_(other.code_),
        message_(other.message_),
        failing_status_(iree_status_clone(other.failing_status_)) {}
  error &operator=(const error &) = delete;
  ~error() { iree_status_ignore(failing_status_); }
  const char *what() const noexcept override { return message_.c_str(); };

  iree_status_code_t code() const { return code_; }

 private:
  void AppendStatusMessage();
  iree_status_code_t code_;
  std::string message_;
  mutable iree_status_t failing_status_;
};

#define SHORTFIN_IMPL_HANDLE_IF_API_ERROR(var, ...)                          \
  iree_status_t var = (IREE_STATUS_IMPL_IDENTITY_(                           \
      IREE_STATUS_IMPL_IDENTITY_(IREE_STATUS_IMPL_GET_EXPR_)(__VA_ARGS__))); \
  if (IREE_UNLIKELY(var)) {                                                  \
    throw ::shortfin::iree::error(                                           \
        IREE_STATUS_IMPL_ANNOTATE_SWITCH_(var, __VA_ARGS__));                \
  }

// Similar to IREE_RETURN_IF_ERROR but throws an error exception instead.
#define SHORTFIN_THROW_IF_ERROR(...)                    \
  SHORTFIN_IMPL_HANDLE_IF_API_ERROR(                    \
      IREE_STATUS_IMPL_CONCAT_(__status_, __COUNTER__), \
      IREE_STATUS_IMPL_IDENTITY_(IREE_STATUS_IMPL_IDENTITY_(__VA_ARGS__)))

// Convert an arbitrary C++ exception to an iree_status_t.
// WARNING: This is not to be used as flow control as it is expensive,
// perfoming type probing and comparisons in some cases!
inline iree_status_t exception_to_status(std::exception &e) {
  return iree_make_status(IREE_STATUS_UNKNOWN, "Unhandled exception: %s",
                          e.what());
}

// RAII wrapper around an iree_status_t that will ignore it when going out
// of scope. This is needed to avoid resource leaks when statuses are being
// used to signal a failure which may not be harvested.
class ignorable_status {
 public:
  ignorable_status() : status_(iree_ok_status()) {}
  ignorable_status(iree_status_t status) : status_(status) {}
  ignorable_status(const ignorable_status &) = delete;
  ignorable_status &operator=(iree_status_t status) {
    iree_status_ignore(status_);
    status_ = status;
    return *this;
  }
  ignorable_status &operator=(const ignorable_status &) = delete;
  ignorable_status(ignorable_status &&other) = delete;
  ~ignorable_status() { iree_status_ignore(status_); }

  // Consumes that status. Only the first consumer will receive all payloads.
  // Others will just get the cloned basic status.
  iree_status_t ConsumeStatus() {
    iree_status_t local_status = status_;
    status_ = iree_status_clone(status_);
    return local_status;
  }
  iree_status_t status() const { return status_; }

 private:
  mutable iree_status_t status_;
};

// -------------------------------------------------------------------------- //
// VM Ref and Variant Interop
// -------------------------------------------------------------------------- //

using vm_opaque_ref = ::iree::vm::opaque_ref;
template <typename T>
using vm_ref = ::iree::vm::ref<T>;

// -------------------------------------------------------------------------- //
// Debugging
// -------------------------------------------------------------------------- //

std::string DebugPrintSemaphoreList(iree_hal_semaphore_list_t &sl);

}  // namespace iree
}  // namespace shortfin

#endif  // SHORTFIN_SUPPORT_IREE_HELPERS_H
