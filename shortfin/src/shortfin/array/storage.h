// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_ARRAY_STORAGE_H
#define SHORTFIN_ARRAY_STORAGE_H

#include <string_view>

#include "shortfin/local/fiber.h"
#include "shortfin/local/program_interfaces.h"
#include "shortfin/support/api.h"

namespace shortfin::array {

// Access to mapped memory.
// Mappings are moveable but not copyable. When default constructed or moved
// from, they will not be valid and have nullptr semantics.
class SHORTFIN_API mapping {
 public:
  mapping();
  mapping(const mapping &) = delete;
  mapping &operator=(const mapping &) = delete;
  mapping &operator=(mapping &&other) {
    timeline_resource_ = std::move(other.timeline_resource_);
    access_ = other.access_;
    mapping_ = other.mapping_;
    other.access_ = IREE_HAL_MEMORY_ACCESS_NONE;
    std::memset(&other.mapping_, 0, sizeof(other.mapping_));
    return *this;
  }
  mapping(mapping &&other)
      : timeline_resource_(std::move(other.timeline_resource_)),
        access_(other.access_),
        mapping_(other.mapping_) {
    other.access_ = IREE_HAL_MEMORY_ACCESS_NONE;
    std::memset(&other.mapping_, 0, sizeof(other.mapping_));
  }
  ~mapping() noexcept;

  // Whether the mapping is valid.
  operator bool() const { return access_ != IREE_HAL_MEMORY_ACCESS_NONE; }

  // Resets the mapping, making it invalid (if not already so);
  void reset() noexcept;

  // Access the mapped data. The mapping must be valid or else it is UB.
  const uint8_t *data() const {
    assert(*this && "mapping is not valid");
    return mapping_.contents.data;
  }
  uint8_t *data() {
    assert(*this && "mapping is not valid");
    return mapping_.contents.data;
  }

  // The size of the mapped data. Will return 0 if the mapping is not valid.
  iree_device_size_t size() const { return mapping_.contents.data_length; }

  bool readable() const { return access_ & IREE_HAL_MEMORY_ACCESS_READ; }
  bool writable() const { return access_ & IREE_HAL_MEMORY_ACCESS_WRITE; }

 private:
  // See note on storage::timeline_resource_. Must be declared first.
  local::detail::TimelineResource::Ref timeline_resource_;
  iree_hal_memory_access_t access_ = IREE_HAL_MEMORY_ACCESS_NONE;
  iree_hal_buffer_mapping_t mapping_;
  friend class storage;
};

// Array storage backed by an IREE buffer of some form.
class SHORTFIN_API storage : public local::ProgramInvocationMarshalable {
 public:
  ~storage();
  local::ScopedDevice &device() { return device_; }
  local::Fiber &fiber() { return device_.fiber(); }
  const local::ScopedDevice &device() const { return device_; }
  local::Fiber &fiber() const { return device_.fiber(); }

  // Allocates device storage, compatible with the given device affinity.
  // By default, this will be IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE.
  static storage allocate_device(local::ScopedDevice &device,
                                 iree_device_size_t allocation_size);

  // Allocates host storage, compatible with the given device affinity.
  // By default, if there are any affinity bits set in the device, then
  // the storage will be device visible and have permitted usage for
  // transfers. This default policy can be overriden based on device defaults
  // or explicit options. Pass `device_visible=false` to create a pure host
  // heap buffer.
  static storage allocate_host(local::ScopedDevice &device,
                               iree_device_size_t allocation_size,
                               bool device_visible = true);

  // Creates a subspan view of the current storage given a byte offset and
  // length. The returned storage shares the underlying allocation and
  // scheduling control block.
  storage subspan(iree_device_size_t byte_offset,
                  iree_device_size_t byte_length);

  // Enqueues a fill of the storage with an arbitrary pattern of the given
  // size. The pattern size must be 1, 2, or 4.
  void fill(const void *pattern, iree_host_size_t pattern_length);

  // Performs either a d2h, h2d or d2d transfer from a source storage to this
  // storage.
  void copy_from(storage &source_storage);

  iree_device_size_t byte_length() const {
    return iree_hal_buffer_byte_length(buffer_.get());
  }

  // Memory type and access.
  iree_hal_memory_type_t memory_type() const;
  iree_hal_memory_access_t memory_access() const;
  iree_hal_buffer_usage_t buffer_usage() const;

  // Formatted type and access.
  std::string formatted_memory_type() const;
  std::string formatted_memory_access() const;
  std::string formatted_buffer_usage() const;

  // Whether the buffer supports host mappable memory.
  bool is_mappable_for_read() const;
  bool is_mappable_for_read_write() const;

  // Maps the memory for access from a host pointer using a scoped mapping.
  void map_explicit(mapping &mapping, iree_hal_memory_access_t access);

  // Maps the memory for read/write access, preserving any contents.
  mapping map_read_write() {
    mapping m;
    map_explicit(m, IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE);
    return m;
  }

  // Maps the memory for discard write. This is used if populating an initial
  // buffer.
  mapping map_write_discard() {
    mapping m;
    map_explicit(m, IREE_HAL_MEMORY_ACCESS_DISCARD_WRITE);
    return m;
  }

  // Maps the memory for read-only access.
  mapping map_read() {
    mapping m;
    map_explicit(m, IREE_HAL_MEMORY_ACCESS_READ);
    return m;
  }

  const mapping map_read() const {
    mapping m;
    const_cast<storage *>(this)->map_explicit(m, IREE_HAL_MEMORY_ACCESS_READ);
    return m;
  }

  std::string to_s() const;

  bool operator==(const storage &other) const {
    return other.buffer_.get() == buffer_.get();
  }

  // Access raw buffer. This must not be retained apart from the storage for
  // any length of time that may extend its lifetime (as the storage keeps
  // underlying device references alive as needed).
  operator iree_hal_buffer_t *() { return buffer_; }

  iree_allocator_t host_allocator() {
    return timeline_resource_->host_allocator();
  }

 private:
  storage(local::ScopedDevice device, iree::hal_buffer_ptr buffer,
          local::detail::TimelineResource::Ref timeline_resource);
  void AsyncDeallocate();
  // ProgramInvocationMarshalable implementation.
  void AddAsInvocationArgument(local::ProgramInvocation *inv,
                               local::ProgramResourceBarrier barrier) override;
  static storage CreateFromInvocationResultRef(
      local::ProgramInvocation *inv,
      local::CoarseInvocationTimelineImporter *timeline_importer,
      iree::vm_opaque_ref ref);
  static iree_vm_ref_type_t invocation_marshalable_type();

  // Adds any necessary wait barriers to the invocation on behalf of this
  // storage.
  void AddInvocationArgBarrier(local::ProgramInvocation *inv,
                               local::ProgramResourceBarrier barrier);

  // Imports a raw hal buffer from an invocation as a storage, attaching any
  // needed barriers.
  static storage ImportInvocationResultStorage(
      local::ProgramInvocation *inv,
      local::CoarseInvocationTimelineImporter *timeline_importer,
      iree::hal_buffer_ptr buffer);

  local::detail::TimelineResource::Ref timeline_resource_;
  iree::hal_buffer_ptr buffer_;
  local::ScopedDevice device_;

  friend class shortfin::local::ProgramInvocationMarshalableFactory;
  friend class device_array;
};

// Wraps an untyped mapping, providing typed access.
template <typename EltTy>
class typed_mapping {
 public:
  using span_type = std::span<EltTy>;
  using const_span_type = std::span<const EltTy>;

  typed_mapping(mapping untyped_mapping)
      : untyped_mapping_(std::move(untyped_mapping)) {}
  typed_mapping(const typed_mapping &) = delete;
  typed_mapping &operator=(const typed_mapping &) = delete;

  iree_device_size_t size() const noexcept {
    return untyped_mapping_.size() / sizeof(EltTy);
  }
  bool empty() const noexcept { return size() == 0; }
  EltTy *data() noexcept {
    return reinterpret_cast<EltTy *>(untyped_mapping_.data());
  }
  EltTy *data() const noexcept {
    return reinterpret_cast<const EltTy *>(untyped_mapping_.data());
  }

  span_type span() { return span_type(data(), size()); }
  const_span_type span() const { return const_span_type(data(), size()); }

  typename span_type::iterator begin() { return span().begin(); }
  typename span_type::iterator end() { return span().end(); }

  typename const_span_type::iterator begin() const { return span().begin(); }
  typename const_span_type::iterator end() const { return span().end(); }

  typename const_span_type::iterator cbegin() const { return span().begin(); }
  typename const_span_type::iterator cend() const { return span().end(); }

 private:
  mapping untyped_mapping_;
};

}  // namespace shortfin::array

#endif  // SHORTFIN_ARRAY_STORAGE_H
