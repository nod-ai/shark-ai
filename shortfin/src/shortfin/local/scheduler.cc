// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local/scheduler.h"

#include "shortfin/local/fiber.h"
#include "shortfin/local/system.h"
#include "shortfin/support/logging.h"

namespace shortfin::local::detail {

namespace {

[[maybe_unused]] std::string SummarizeFence(iree_hal_fence_t *fence) {
  if (!SHORTFIN_SCHED_LOG_ENABLED) {
    return std::string();
  }
  std::string result("fence(");
  iree_hal_semaphore_list_t list = iree_hal_fence_semaphore_list(fence);
  for (iree_host_size_t i = 0; i < list.count; ++i) {
    if (i > 0) result.append(", ");
    result.append(fmt::format("[{}@{}]",
                              static_cast<void *>(list.semaphores[i]),
                              list.payload_values[i]));
  }
  result.append(")");
  return result;
}

}  // namespace

// -------------------------------------------------------------------------- //
// Account
// -------------------------------------------------------------------------- //

Account::Account(Scheduler &scheduler, Device *device)
    : scheduler_(scheduler),
      device_(device),
      hal_device_(device->hal_device()) {}

void Account::Initialize() {
  SHORTFIN_THROW_IF_ERROR(iree_hal_semaphore_create(
      hal_device(), /*initial_value=*/idle_timepoint_,
      /*flags=*/IREE_HAL_SEMAPHORE_FLAG_NONE, sem_.for_output()));
  Reset();
}

void Account::Reset() {
  active_tx_type_ = TransactionType::NONE;
  active_command_buffer_.reset();
}

void Account::active_deps_extend(iree_hal_semaphore_list_t sem_list) {
  for (iree_host_size_t i = 0; i < sem_list.count; ++i) {
    SHORTFIN_THROW_IF_ERROR(iree_hal_fence_insert(
        active_deps_, sem_list.semaphores[i], sem_list.payload_values[i]));
  }
}

VoidFuture Account::OnSync() {
  SHORTFIN_TRACE_SCOPE_NAMED("Account::OnSync");
  // TODO: Burn this path with fire! No attempt has been made to make this
  // particularly good: the backend is being implemented now to export
  // HAL semaphores via iree_hal_semaphore_await, and that should be used
  // when supported. This is merely here so as to unblock local progress.
  // This should be something like:
  // return CompletionEvent(sem_, idle_timepoint_);
  iree::hal_semaphore_ptr sem = sem_;
  auto idle_timepoint = idle_timepoint_;
  SHORTFIN_SCHED_LOG("OnSync::Wait({}@{})", static_cast<void *>(sem.get()),
                     idle_timepoint);
  VoidFuture future;
  scheduler_.system().blocking_executor().Schedule([sem = std::move(sem),
                                                    idle_timepoint, future]() {
    iree_status_t status =
        iree_hal_semaphore_wait(sem, idle_timepoint, iree_infinite_timeout());
    if (!iree_status_is_ok(status)) {
      const_cast<VoidFuture &>(future).set_failure(status);
    } else {
      SHORTFIN_SCHED_LOG("OnSync::Complete({}@{})",
                         static_cast<void *>(sem.get()), idle_timepoint);
      const_cast<VoidFuture &>(future).set_success();
    }
  });
  return future;
}

// -------------------------------------------------------------------------- //
// TimelineResource
// -------------------------------------------------------------------------- //

TimelineResource::TimelineResource(std::shared_ptr<Fiber> fiber,
                                   size_t semaphore_capacity,
                                   TimelineResourceDestructor destructor)
    : fiber_(std::move(fiber)), destructor_(std::move(destructor)) {
  logging::construct("TimelineResource", this);
  SHORTFIN_THROW_IF_ERROR(
      iree_hal_fence_create(semaphore_capacity, fiber_->host_allocator(),
                            use_barrier_fence_.for_output()));
}

TimelineResource::~TimelineResource() {
  logging::destruct("TimelineResource", this);
  if (destructor_) {
    destructor_(*this);
  }
}

TimelineResourceDestructor TimelineResource::CreateAsyncBufferDestructor(
    ScopedDevice &scoped_device, iree::hal_buffer_ptr buffer) {
  // The ScopedDevice doesn't lifetime extend the underlying hal device, so
  // we must do that manually across the callback (and then release at the end).
  iree_hal_device_retain(scoped_device.raw_device()->hal_device());
  return [device_affinity = scoped_device.affinity(),
          buffer = std::move(buffer)](TimelineResource &res) {
    ScopedDevice scoped_device(*res.fiber(), device_affinity);
    iree_hal_device_t *hal_device = scoped_device.raw_device()->hal_device();
    auto queue_affinity = scoped_device.affinity().queue_affinity();
    SHORTFIN_TRACE_SCOPE_NAMED("TimelineResource::AsyncBufferDestructor");

    // The dealloca needs to wait on the current timepoint+all uses, and it
    // needs to signal the next timepoint. Since this is a destructor, we
    // now have exclusive control of the TimelineResource and simply record
    // the current timepoint in its uses as a (small) optimization for merging
    // the wait semaphore list, prior to acquiring a new timepoint
    // (to signal).
    auto fiber = res.fiber();
    auto &account = fiber->scheduler().GetDefaultAccount(scoped_device);
    iree_hal_semaphore_t *timeline_sem = account.timeline_sem();
    res.use_barrier_insert(timeline_sem, account.timeline_idle_timepoint());
    uint64_t signal_timepoint = account.timeline_acquire_timepoint();
    iree_hal_semaphore_list_t wait_semaphore_list = res.use_barrier();
    iree_hal_semaphore_list_t signal_semaphore_list{
        .count = 1,
        .semaphores = &timeline_sem,
        .payload_values = &signal_timepoint,
    };

    if (SHORTFIN_SCHED_LOG_ENABLED) {
      auto wait_sum = iree::DebugPrintSemaphoreList(wait_semaphore_list);
      auto signal_sum = iree::DebugPrintSemaphoreList(signal_semaphore_list);
      SHORTFIN_SCHED_LOG(
          "async dealloca(device={}, affinity={:x}, buffer={}):[Wait:{}, "
          "Signal:{}]",
          static_cast<void *>(hal_device), queue_affinity,
          static_cast<void *>(buffer.get()), wait_sum, signal_sum);
    }
    SHORTFIN_THROW_IF_ERROR(iree_hal_device_queue_dealloca(
        hal_device, queue_affinity, wait_semaphore_list, signal_semaphore_list,
        buffer));
    iree_hal_device_release(hal_device);
  };
}

void TimelineResource::use_barrier_insert(iree_hal_semaphore_t *sem,
                                          uint64_t timepoint) {
  SHORTFIN_THROW_IF_ERROR(
      iree_hal_fence_insert(use_barrier_fence_, sem, timepoint));
}

iree_allocator_t TimelineResource::host_allocator() {
  return fiber_->host_allocator();
}

// -------------------------------------------------------------------------- //
// Scheduler
// -------------------------------------------------------------------------- //

Scheduler::Scheduler(System &system) : system_(system) {
  logging::construct("Scheduler", this);
}

Scheduler::~Scheduler() {
  logging::destruct("Scheduler", this);

  // Explicitly reset account state prior to implicit destruction.
  for (auto &account : accounts_) {
    account.Reset();
  }
}

void Scheduler::Initialize(
    std::span<const std::pair<std::string_view, Device *>> devices) {
  SHORTFIN_TRACE_SCOPE_NAMED("Scheduler::Initialize");
  for (auto &it : devices) {
    accounts_.emplace_back(*this, it.second);
  }

  for (Account &account : accounts_) {
    auto [it, inserted] = accounts_by_device_id_.emplace(
        std::make_pair(account.device()->address().device_id(), &account));
    if (!inserted) {
      throw std::logic_error("Duplicate device in Scheduler");
    }
    account.Initialize();
    semaphore_count_ += account.semaphore_count() + 8;
  }
}

Account &Scheduler::GetDefaultAccount(ScopedDevice &device) {
  auto queue_ordinal = device.affinity().lowest_queue_ordinal();
  auto device_id =
      device.raw_device()->address().device_id_for_queue(queue_ordinal);
  auto it = accounts_by_device_id_.find(device_id);
  if (it == accounts_by_device_id_.end()) [[unlikely]] {
    throw std::logic_error(
        fmt::format("Topology check failed: could not find scheduling account "
                    "for device id {:x}",
                    device_id));
  }
  return *it->second;
}

void Scheduler::AppendCommandBuffer(ScopedDevice &device,
                                    TransactionType tx_type,
                                    std::function<void(Account &)> callback) {
  SHORTFIN_TRACE_SCOPE_NAMED("Scheduler::AppendCommandBuffer");
  Account &account = GetDefaultAccount(device);
  auto needed_affinity_bits = device.affinity().queue_affinity();
  SHORTFIN_SCHED_LOG(
      "AppendCommandBuffer(account=0x{:x}, tx_type={}, queue_affinity={}):",
      account.id(), static_cast<int>(tx_type), needed_affinity_bits);

  // Initialize a fresh command buffer if needed.
  if (!account.active_command_buffer_) {
    // Map to a command buffer category.
    iree_hal_command_category_t category;
    switch (tx_type) {
      case TransactionType::TRANSFER:
        category = IREE_HAL_COMMAND_CATEGORY_TRANSFER;
        break;
      case TransactionType::SEQUENTIAL_DISPATCH:
      case TransactionType::PARALLEL_DISPATCH:
        category = IREE_HAL_COMMAND_CATEGORY_DISPATCH;
        break;
      default:
        throw std::logic_error(fmt::format("Unsupported transaction type {}",
                                           static_cast<int>(tx_type)));
    }

    // Set up the command buffer.
    iree::hal_command_buffer_ptr new_cb;
    iree::hal_fence_ptr new_active_deps;
    SHORTFIN_THROW_IF_ERROR(
        iree_hal_fence_create(semaphore_count_, system_.host_allocator(),
                              new_active_deps.for_output()));
    SHORTFIN_THROW_IF_ERROR(iree_hal_command_buffer_create(
        account.hal_device(),
        /*mode=*/IREE_HAL_COMMAND_BUFFER_MODE_ONE_SHOT,
        /*command_categories=*/category,
        /*queue_affinity=*/IREE_HAL_QUEUE_AFFINITY_ANY,
        /*binding_capacity=*/0,  // Never indirect so no bindings needed.
        new_cb.for_output()));
    SHORTFIN_THROW_IF_ERROR(iree_hal_command_buffer_begin(new_cb));

    // Memoize what mode we are in now.
    account.active_tx_type_ = tx_type;
    account.active_queue_affinity_bits_ = needed_affinity_bits;
    account.active_deps_ = std::move(new_active_deps);
    account.active_command_buffer_ = std::move(new_cb);

    // Sence the command buffer will be submitted to signal the next
    // timepoint on the main timeline, we must depend on its current value
    // to be value (semaphores must strictly advance). This has the effect of
    // serializing all submissions, which while correct, is not a particularly
    // enlightened scheduling policy.
    // TODO: Revisit this when scheduling is generalized and consider that such
    // serialization be retained only as a debug feature.
    iree_hal_semaphore_t *main_timeline_sem = account.sem_.get();
    account.active_deps_extend(iree_hal_semaphore_list_t{
        .count = 1,
        .semaphores = &main_timeline_sem,
        .payload_values = &account.idle_timepoint_,
    });

    // Signal an advance of the main timeline.
    account.idle_timepoint_ += 1;
    SHORTFIN_SCHED_LOG(
        "  : New command buffer (category={}, idle_timepoint={})", category,
        account.idle_timepoint_);
  } else {
    SHORTFIN_SCHED_LOG("  : Continue active command buffer");
  }

  // Perform the mutation.
  callback(account);

  // Flush.
  if (tx_mode_ == TransactionMode::EAGER) {
    Flush();
  }
}

iree_status_t Scheduler::FlushWithStatus() noexcept {
  SHORTFIN_TRACE_SCOPE_NAMED("Scheduler::FlushWithStatus");
  // This loop is optimized for a small number of accounts, where it is
  // fine to just linearly probe. If this ever becomes cumbersome, we can
  // maintain a dirty list which is appended to when an account transitions
  // from idle to active.
  for (Account &account : accounts_) {
    if (!account.active_command_buffer_) continue;
    iree_hal_semaphore_t *signal_sem = account.sem_;
    uint64_t signal_timepoint = account.idle_timepoint_;
    iree_hal_command_buffer_t *active_command_buffer =
        account.active_command_buffer_;
    iree_hal_buffer_binding_table_t binding_tables =
        iree_hal_buffer_binding_table_empty();

    SHORTFIN_SCHED_LOG(
        "Flush command buffer (account=0x{:x}, queue_affinity={}, "
        "signal_timepoint={}, deps={})",
        account.id(), account.active_queue_affinity_bits_, signal_timepoint,
        SummarizeFence(account.active_deps_));

    // End recording and submit.
    IREE_RETURN_IF_ERROR(iree_hal_command_buffer_end(active_command_buffer));
    IREE_RETURN_IF_ERROR(iree_hal_device_queue_execute(
        account.hal_device(),
        /*queue_affinity=*/account.active_queue_affinity_bits_,
        /*wait_sempahore_list=*/account.active_deps_
            ? iree_hal_fence_semaphore_list(account.active_deps_)
            : iree_hal_semaphore_list_empty(),
        /*signal_semaphore_list=*/
        iree_hal_semaphore_list_t{
            .count = 1,
            .semaphores = &signal_sem,
            .payload_values = &signal_timepoint,
        },
        /*command_buffers=*/active_command_buffer,
        /*binding_tables=*/binding_tables));
    account.Reset();
  }
  return iree_ok_status();
}

iree::hal_fence_ptr Scheduler::NewFence() {
  iree::hal_fence_ptr fence;
  iree_hal_fence_create(semaphore_count_, system_.host_allocator(),
                        fence.for_output());
  return fence;
}

}  // namespace shortfin::local::detail
