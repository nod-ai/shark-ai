// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/array/storage.h"

#include "fmt/core.h"
#include "shortfin/support/logging.h"

namespace shortfin::array {

using namespace local;
using namespace local::detail;

// -------------------------------------------------------------------------- //
// storage
// -------------------------------------------------------------------------- //

namespace detail {
void ThrowIllegalDeviceAffinity(Device *first, Device *second) {
  throw std::invalid_argument(fmt::format(
      "Cannot combine unrelated devices into a DeviceAffinity: {} vs {}",
      first->name(), second->name()));
}
}  // namespace detail

storage::storage(local::ScopedDevice device, iree::hal_buffer_ptr buffer,
                 local::detail::TimelineResource::Ref timeline_resource)
    : timeline_resource_(std::move(timeline_resource)),
      buffer_(std::move(buffer)),
      device_(device) {
  logging::construct("array::storage", this);
}
storage::~storage() {
  logging::destruct("array::storage", this);
  SHORTFIN_TRACE_SCOPE_NAMED("storage::~storage");
  // The timeline resource holds the back reference to the owning fiber,
  // which keeps all devices alive. Buffers must be destroyed before devices,
  // so destruction sequencing matters and we make it explicit.
  buffer_.reset();
  timeline_resource_.reset();
}

storage storage::allocate_device(ScopedDevice &device,
                                 iree_device_size_t allocation_size) {
  SHORTFIN_TRACE_SCOPE_NAMED("storage::allocate_device");
  if (!device.raw_device()) {
    throw std::invalid_argument("Cannot allocate with a null device affinity");
  }
  iree::hal_buffer_ptr buffer;
  iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_DEFAULT,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_DEVICE,
      .queue_affinity = device.affinity().queue_affinity(),
  };
  Account &account = device.fiber().scheduler().GetDefaultAccount(device);
  iree_hal_semaphore_t *timeline_sem = account.timeline_sem();
  uint64_t current_timepoint = account.timeline_idle_timepoint();
  uint64_t signal_timepoint = account.timeline_acquire_timepoint();
  iree_hal_semaphore_list_t wait_semaphore_list{
      .count = 1,
      .semaphores = &timeline_sem,
      .payload_values = &current_timepoint,
  };
  iree_hal_semaphore_list_t signal_semaphore_list{
      .count = 1,
      .semaphores = &timeline_sem,
      .payload_values = &signal_timepoint,
  };
  // Async allocate.
  SHORTFIN_THROW_IF_ERROR(iree_hal_device_queue_alloca(
      device.raw_device()->hal_device(), device.affinity().queue_affinity(),
      wait_semaphore_list, signal_semaphore_list,
      IREE_HAL_ALLOCATOR_POOL_DEFAULT, params, allocation_size,
      IREE_HAL_ALLOCA_FLAG_NONE, buffer.for_output()));
  SHORTFIN_SCHED_LOG(
      "storage::allocate_device(device={}, affinity={:x}):[{}, Wait@{}->"
      "Signal:@{}] -> buffer={}",
      static_cast<void *>(device.raw_device()->hal_device()),
      device.affinity().queue_affinity(), static_cast<void *>(timeline_sem),
      current_timepoint, signal_timepoint, static_cast<void *>(buffer.get()));

  // Device allocations are always async.
  TimelineResourceDestructor dtor =
      TimelineResource::CreateAsyncBufferDestructor(device, buffer);
  auto resource = device.fiber().NewTimelineResource(std::move(dtor));
  resource->set_mutation_barrier(timeline_sem, signal_timepoint);
  resource->use_barrier_insert(timeline_sem, signal_timepoint);
  return storage(device, std::move(buffer), std::move(resource));
}

storage storage::allocate_host(ScopedDevice &device,
                               iree_device_size_t allocation_size,
                               bool device_visible) {
  SHORTFIN_TRACE_SCOPE_NAMED("storage::allocate_host");
  if (!device.raw_device()) {
    throw std::invalid_argument("Cannot allocate with a null device affinity");
  }
  auto allocator = iree_hal_device_allocator(device.raw_device()->hal_device());
  iree::hal_buffer_ptr buffer;
  iree_hal_buffer_params_t params = {
      .usage = IREE_HAL_BUFFER_USAGE_MAPPING,
      .access = IREE_HAL_MEMORY_ACCESS_ALL,
      .type = IREE_HAL_MEMORY_TYPE_OPTIMAL_FOR_HOST,
      .queue_affinity = device.affinity().queue_affinity(),
  };
  if (device_visible) {
    params.type |= IREE_HAL_MEMORY_TYPE_DEVICE_VISIBLE;
    if (device.affinity().queue_affinity() != 0) {
      params.usage |= IREE_HAL_BUFFER_USAGE_TRANSFER;
    }
  }
  SHORTFIN_THROW_IF_ERROR(iree_hal_allocator_allocate_buffer(
      allocator, params, allocation_size, buffer.for_output()));
  return storage(device, std::move(buffer),
                 device.fiber().NewTimelineResource());
}

storage storage::subspan(iree_device_size_t byte_offset,
                         iree_device_size_t byte_length) {
  storage new_storage(device_, {}, timeline_resource_);
  SHORTFIN_THROW_IF_ERROR(iree_hal_buffer_subspan(
      buffer_, byte_offset, byte_length, host_allocator(),
      new_storage.buffer_.for_output()));
  return new_storage;
}

void storage::fill(const void *pattern, iree_host_size_t pattern_length) {
  device_.fiber().scheduler().AppendCommandBuffer(
      device_, TransactionType::TRANSFER, [&](Account &account) {
        // Must depend on all of this buffer's use dependencies to avoid
        // write-after-read hazard (which implicitly includes
        // write-after-write).
        account.active_deps_extend(timeline_resource_->use_barrier());

        SHORTFIN_SCHED_LOG("  : FillBuffer({})",
                           static_cast<void *>(buffer_.get()));
        SHORTFIN_THROW_IF_ERROR(iree_hal_command_buffer_fill_buffer(
            account.active_command_buffer(),
            iree_hal_make_buffer_ref(
                buffer_, /*offset=*/0,
                /*length=*/iree_hal_buffer_byte_length(buffer_)),
            pattern, pattern_length, IREE_HAL_FILL_FLAG_NONE));

        // And move our own use and mutation barrier to the current pending
        // timeline value.
        timeline_resource_->set_mutation_barrier(
            account.timeline_sem(), account.timeline_idle_timepoint());
        timeline_resource_->use_barrier_insert(
            account.timeline_sem(), account.timeline_idle_timepoint());
      });
}

void storage::copy_from(storage &source_storage) {
  device_.fiber().scheduler().AppendCommandBuffer(
      device_, TransactionType::TRANSFER, [&](Account &account) {
        // Must depend on the source's mutation dependencies to avoid
        // read-before-write hazard.
        account.active_deps_extend(
            source_storage.timeline_resource_->mutation_barrier());
        // And depend on our own use and mutations dependencies.
        account.active_deps_extend(timeline_resource_->use_barrier());

        SHORTFIN_SCHED_LOG("  : CopyBuffer({} -> {})",
                           static_cast<void *>(source_storage.buffer_.get()),
                           static_cast<void *>(buffer_.get()));
        SHORTFIN_THROW_IF_ERROR(iree_hal_command_buffer_copy_buffer(
            account.active_command_buffer(),
            /*source_ref=*/
            iree_hal_make_buffer_ref(source_storage.buffer_, 0, byte_length()),
            /*target_ref=*/
            iree_hal_make_buffer_ref(buffer_, 0, byte_length()),
            IREE_HAL_COPY_FLAG_NONE));

        // Move our own use and mutation barrier to the current pending timeline
        // value.
        timeline_resource_->set_mutation_barrier(
            account.timeline_sem(), account.timeline_idle_timepoint());
        timeline_resource_->use_barrier_insert(
            account.timeline_sem(), account.timeline_idle_timepoint());
        // And extend the source use barrier.
        source_storage.timeline_resource_->use_barrier_insert(
            account.timeline_sem(), account.timeline_idle_timepoint());
      });
}

bool storage::is_mappable_for_read() const {
  return (iree_hal_buffer_allowed_usage(buffer_) &
          IREE_HAL_MEMORY_TYPE_HOST_VISIBLE) &&
         (iree_hal_buffer_allowed_access(buffer_) &
          IREE_HAL_MEMORY_ACCESS_READ);
}

bool storage::is_mappable_for_read_write() const {
  return (iree_hal_buffer_allowed_usage(buffer_) &
          IREE_HAL_MEMORY_TYPE_HOST_VISIBLE) &&
         (iree_hal_buffer_allowed_access(buffer_) &
          (IREE_HAL_MEMORY_ACCESS_READ | IREE_HAL_MEMORY_ACCESS_WRITE));
}

void storage::map_explicit(mapping &mapping, iree_hal_memory_access_t access) {
  assert(access != IREE_HAL_MEMORY_ACCESS_NONE);
  mapping.reset();
  SHORTFIN_THROW_IF_ERROR(iree_hal_buffer_map_range(
      buffer_, IREE_HAL_MAPPING_MODE_SCOPED, access,
      /*byte_offset=*/0, byte_length(), &mapping.mapping_));
  mapping.access_ = access;
  mapping.timeline_resource_ = timeline_resource_;
}

iree_hal_memory_type_t storage::memory_type() const {
  return iree_hal_buffer_memory_type(buffer_);
}
iree_hal_memory_access_t storage::memory_access() const {
  return iree_hal_buffer_allowed_access(buffer_);
}
iree_hal_buffer_usage_t storage::buffer_usage() const {
  return iree_hal_buffer_allowed_usage(buffer_);
}

// Formatted type and access.
std::string storage::formatted_memory_type() const {
  iree_bitfield_string_temp_t temp;
  auto sv = iree_hal_memory_type_format(memory_type(), &temp);
  return std::string(sv.data, sv.size);
}

std::string storage::formatted_memory_access() const {
  iree_bitfield_string_temp_t temp;
  auto sv = iree_hal_memory_access_format(memory_access(), &temp);
  return std::string(sv.data, sv.size);
}

std::string storage::formatted_buffer_usage() const {
  iree_bitfield_string_temp_t temp;
  auto sv = iree_hal_buffer_usage_format(buffer_usage(), &temp);
  return std::string(sv.data, sv.size);
}

void storage::AddAsInvocationArgument(local::ProgramInvocation *inv,
                                      local::ProgramResourceBarrier barrier) {
  SHORTFIN_TRACE_SCOPE_NAMED("storage::AddAsInvocationArgument");
  iree::vm_opaque_ref ref;
  *(&ref) = iree_hal_buffer_retain_ref(buffer_);
  inv->AddArg(std::move(ref), timeline_resource_.get());

  AddInvocationArgBarrier(inv, barrier);
}

iree_vm_ref_type_t storage::invocation_marshalable_type() {
  return iree_hal_buffer_type();
}

storage storage::CreateFromInvocationResultRef(
    local::ProgramInvocation *inv,
    local::CoarseInvocationTimelineImporter *timeline_importer,
    iree::vm_opaque_ref ref) {
  SHORTFIN_TRACE_SCOPE_NAMED("storage::CreateFromInvocationResultRef");
  // Steal the ref to one of our smart pointers.
  // TODO: Should have an opaque_ref::release().
  iree::hal_buffer_ptr buffer =
      iree::hal_buffer_ptr::steal_reference(iree_hal_buffer_deref(*ref.get()));
  (&ref)->ptr = nullptr;
  return ImportInvocationResultStorage(inv, timeline_importer,
                                       std::move(buffer));
}

storage storage::ImportInvocationResultStorage(
    local::ProgramInvocation *inv,
    local::CoarseInvocationTimelineImporter *timeline_importer,
    iree::hal_buffer_ptr buffer) {
  SHORTFIN_TRACE_SCOPE_NAMED("storage::ImportInvocationResultStorage");
  local::ScopedDevice device =
      local::ScopedDevice(*inv->fiber(), inv->device_selection());
  iree_hal_buffer_t *raw_buffer = buffer.get();
  storage imported_storage(
      device, std::move(buffer),
      timeline_importer->ImportTimelineResource(raw_buffer));

  auto coarse_signal = inv->coarse_signal();
  if (coarse_signal.first) {
    SHORTFIN_SCHED_LOG("Storage buffer {}: Ready barrier {}@{}",
                       static_cast<void *>(imported_storage.buffer_.get()),
                       static_cast<void *>(coarse_signal.first),
                       coarse_signal.second);
    imported_storage.timeline_resource_->set_mutation_barrier(
        coarse_signal.first, coarse_signal.second);
    imported_storage.timeline_resource_->use_barrier_insert(
        coarse_signal.first, coarse_signal.second);
  }

  return imported_storage;
}

void storage::AddInvocationArgBarrier(local::ProgramInvocation *inv,
                                      local::ProgramResourceBarrier barrier) {
  SHORTFIN_TRACE_SCOPE_NAMED("storage::AddInvocationArgBarrier");
  switch (barrier) {
    case ProgramResourceBarrier::DEFAULT:
    case ProgramResourceBarrier::READ:
      inv->wait_insert(timeline_resource_->mutation_barrier());
      inv->DeviceSelect(device_.affinity());
      break;
    case ProgramResourceBarrier::WRITE:
      inv->wait_insert(timeline_resource_->use_barrier());
      inv->DeviceSelect(device_.affinity());
      break;
    case ProgramResourceBarrier::NONE:
      break;
  }
}

std::string storage::to_s() const {
  return fmt::format("<storage {} size {}>", static_cast<void *>(buffer_.get()),
                     byte_length());
}

// -------------------------------------------------------------------------- //
// mapping
// -------------------------------------------------------------------------- //

mapping::mapping() {
  logging::construct("array::mapping", this);
  std::memset(&mapping_, 0, sizeof(mapping_));
}

mapping::~mapping() noexcept {
  logging::destruct("array::mapping", this);
  reset();
}

void mapping::reset() noexcept {
  if (*this) {
    // Crash the process on failure to unmap. We don't have a good mitigation,
    IREE_CHECK_OK(iree_hal_buffer_unmap_range(&mapping_));
    access_ = IREE_HAL_MEMORY_ACCESS_NONE;
    timeline_resource_.reset();
  }
}

}  // namespace shortfin::array
