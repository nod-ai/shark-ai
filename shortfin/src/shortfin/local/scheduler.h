// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_SCHEDULER_H
#define SHORTFIN_LOCAL_SCHEDULER_H

#include <functional>
#include <span>

#include "shortfin/local/async.h"
#include "shortfin/local/device.h"
#include "shortfin/support/iree_helpers.h"

namespace shortfin::local {

class Fiber;
class ScopedDevice;
class System;

namespace detail {

class Account;
class Scheduler;
class TimelineResource;

// Transactions are accumulated into a command buffer by type and in
// auto-flush mode, the command buffer is submitted upon a change of type.
enum class TransactionType {
  // Non-aliasing transfer. All transfers submitted in a sequence must not
  // alias each other. If there is a dependency, then they must be flushed
  // before adding an alising transfer.
  TRANSFER = 0,
  // Standalone dispatch that will be enqueued in an individual dispatch.
  SEQUENTIAL_DISPATCH = 1,
  // Parallelizable dispatch that can be enqueued with other parallel
  // dispatches.
  PARALLEL_DISPATCH = 2,
  // Special case: no active transaction type sentinel. Highest number
  // so it doesn't clutter switch-jumps.
  NONE = 3,
};

enum class TransactionMode {
  // All transactions are eagerly flushed in their own command buffers.
  // While potentially not as efficient as possible, this mode is
  // deterministic with respect to typical programming models. The submission
  // logic may apply "as if eager" optimizations in narrow cases where it
  // is obvious that batching is well defined.
  EAGER = 0,
  // Pending command buffers are not flushed until explicitly set to do so.
  EXPLICIT = 1,
};

// Destructor callback to be invoked just before the timeline resource is
// destroyed.
using TimelineResourceDestructor = std::function<void(TimelineResource &)>;

// Control object for a resource that is tracked on the timeline. Each such
// resource is associated with a single Account. In the case of a resource
// that is shared across device queues, a consistent account will be chosen
// as primary (typically corresponding to the lowest numbered queue). This
// object tracks two things about the resource:
//   1. Mutation Barrier: Idle timepoint of the most recent affecting
//      mutation. Since any such mutation mutation must have been enqueued
//      on a single Account, we only need to track the timepoint here
//      (against the account's sem).
//   2. Use Barrier: Fence joined to all timepoints accessing the resource
//      (read and write). At any given time, this represents the point at
//      which the resource has no further pending uses.
// Since TimelineResources are shared (i.e. across subspan storage, etc),
// they are modeled as reference counted (using non atomics, since this is
// "scoped" same thread access). They must only be held in a context that
// is keeping the containing Fiber alive.
//
// Note to the future: in discussing the above, many cases were noted where
// a more advanced programming model would be desirable in order to exercise
// more concurrency. However, the conservative default behavior of
// (effectively) reader/writer locks on resources gives solid, understandable
// behavior for a default programming model. It is not meant to be valid
// for everything over time and should not be considered sacred.
class SHORTFIN_API TimelineResource {
 public:
  class SHORTFIN_API Ref {
   public:
    Ref() : res_(nullptr) {}
    explicit Ref(TimelineResource *res) : res_(res) {
      if (res_) res_->Retain();
    }
    Ref(const Ref &other) : res_(other.res_) {
      if (res_) res_->Retain();
    }
    operator bool() const { return res_ != nullptr; }
    Ref &operator=(const Ref &other) {
      if (other.res_ != res_) {
        reset();
        if (other.res_) {
          other.res_->Retain();
          res_ = other.res_;
        }
      }
      return *this;
    }
    Ref &operator=(Ref &&other) {
      if (other.res_ != res_) {
        reset();
        res_ = other.res_;
        other.res_ = nullptr;
      }
      return *this;
    }
    Ref(Ref &&other) : res_(other.res_) { other.res_ = nullptr; }
    ~Ref() { reset(); }
    TimelineResource *operator->() { return res_; }
    TimelineResource *get() { return res_; }

    void reset() {
      if (res_) {
        res_->Release();
        res_ = nullptr;
      }
    }

   private:
    TimelineResource *res_;
  };
  TimelineResource(TimelineResource &other) = delete;

  // Creates an asynchronous buffer destructor for this resource. When the
  // resource is about to be destroyed, an async dealloca will be issued at
  // the use barrier.
  static TimelineResourceDestructor CreateAsyncBufferDestructor(
      ScopedDevice &scoped_device, iree::hal_buffer_ptr buffer);

  // Sets the mutation barrier. The mutation barrier is the point on a timeline
  // beyond which there are no further writes.
  // Note that the semaphore set in this way is not retained as it is
  // assumed to be part of the local scheduler.
  void set_mutation_barrier(iree_hal_semaphore_t *sem, uint64_t timepoint) {
    mutation_barrier_sem_ = sem;
    mutation_barrier_timepoint_ = timepoint;
  }
  iree_hal_semaphore_list_t mutation_barrier() {
    if (!mutation_barrier_sem_) {
      return iree_hal_semaphore_list_empty();
    } else {
      return iree_hal_semaphore_list_t{
          .count = 1,
          .semaphores = &mutation_barrier_sem_,
          .payload_values = &mutation_barrier_timepoint_};
    }
  }

  // Barrier beyond which there are no further uses of the resource, including
  // both reads and writes.
  // Use barrier can have new timepoints inserted or converted to a
  // semaphore list.
  void use_barrier_insert(iree_hal_semaphore_t *sem, uint64_t timepoint);
  iree_hal_semaphore_list_t use_barrier() {
    return iree_hal_fence_semaphore_list(use_barrier_fence_);
  }

  iree_allocator_t host_allocator();

  void Retain() { refcnt_++; }
  void Release() {
    if (--refcnt_ == 0) delete this;
  }

  Fiber *fiber() { return fiber_.get(); }

 private:
  TimelineResource(std::shared_ptr<Fiber> fiber, size_t semaphore_capacity,
                   TimelineResourceDestructor destructor);
  ~TimelineResource();
  int refcnt_ = 0;

  // Back reference to the owning fiber.
  std::shared_ptr<Fiber> fiber_;

  // Non-owning mutation barrier semaphore and timepoint. The fact that this
  // is a single semaphore is an implementation detail that may be generalized
  // in the future should it be necessary to track multiple write sources.
  iree_hal_semaphore_t *mutation_barrier_sem_ = nullptr;
  uint64_t mutation_barrier_timepoint_ = 0;

  // Use barrier fence. The fact that this is a fence object with a fixed
  // capacity is an implementation detail.
  iree::hal_fence_ptr use_barrier_fence_;

  // Destructor to be called just prior to the TimelineResource being destroyed.
  TimelineResourceDestructor destructor_;
  friend class Scheduler;
};

// Accounting structure for a single logical device (Device*), which
// means that each addressable queue gets its own Account.
class SHORTFIN_API Account {
 public:
  Account(Scheduler &scheduler, Device *device);
  Device *device() const { return device_; }
  iree_hal_device_t *hal_device() { return hal_device_; }

  size_t semaphore_count() const { return 1; }
  // Gets a unique integer id for this account. Currently just the address of
  // the sem, but can be derived from any owned entity.
  uintptr_t id() const { return reinterpret_cast<uintptr_t>(sem_.get()); }

  // Accesses the active command buffer. This will only be non-null if a
  // pending transaction has been set up (i.e. via AppendCommandBuffer).
  iree_hal_command_buffer_t *active_command_buffer() {
    return active_command_buffer_.get();
  }

  // Extend the current command buffer active deps to join over sem_list.
  void active_deps_extend(iree_hal_semaphore_list_t sem_list);

  // Queue timeline.
  iree_hal_semaphore_t *timeline_sem() { return sem_; }
  uint64_t timeline_idle_timepoint() { return idle_timepoint_; }
  uint64_t timeline_acquire_timepoint() { return ++idle_timepoint_; }

  // Returns a future that is satisfied when the timeline of this account
  // reaches its current idle timepoint (i.e. all currently pending work
  // is complete).
  VoidFuture OnSync();

 private:
  void Initialize();
  void Reset();
  Scheduler &scheduler_;
  iree::hal_semaphore_ptr sem_;
  iree::hal_fence_ptr active_deps_;
  iree::hal_command_buffer_ptr active_command_buffer_;

  Device *device_;
  iree_hal_device_t *hal_device_;
  TransactionType active_tx_type_ = TransactionType::NONE;
  iree_hal_queue_affinity_t active_queue_affinity_bits_;

  // Timepoint at which this device is considered idle, inclusive of any
  // active_command_buffer that has not yet been submitted. This means
  // that at any time, a wait on this timepoint will produce the expected
  // temporal sequencing, but it may represent a point in the future where
  // the work needed to reach it has not been submitted yet.
  // This means that it has two interpretations based on the state of
  // active_command_buffer_:
  //   - nullptr:  All work has been flushed to reach this timepoint.
  //   - !nullptr: When flushed, the active command buffer will be submitted
  //     to signal this timepoint, fulfilling any waiters.
  // Whenever we transition active_command_buffer_ from nullptr,
  // idle_timepoint_ *must* be incremented by at least 1 (or otherwise
  // an eventual submission would submit a duplicate timepoint). This
  // timepoint is only valid for the local sem_.
  uint64_t idle_timepoint_ = 0;
  friend class Scheduler;
};

// Handles scheduling state for a fiber.
class SHORTFIN_API Scheduler {
 public:
  Scheduler(System &system);
  ~Scheduler();

  TransactionMode transaction_mode() const { return tx_mode_; }

  // Given a ScopedDevice (which may logically bind to multiple queues),
  // returns a deterministic Account associated with the device that can be
  // used for accounting and scheduling.
  Account &GetDefaultAccount(ScopedDevice &device);

  // Sets up |device| for appending commands to a command buffer, invoking
  // callback to complete the mutation. Depending on the current transaction
  // mode and tx_type, this may involve flushing the current command buffer.
  // This will always allocate a new timepoint from the global timeline and
  // increment the elected device account's idle timepoint. It will also
  // set any affinity bits on the pending submission.
  void AppendCommandBuffer(ScopedDevice &device, TransactionType tx_type,
                           std::function<void(Account &)> callback);

  // Flushes any pending accounts that have accumulated commands.
  iree_status_t FlushWithStatus() noexcept;
  void Flush() { SHORTFIN_THROW_IF_ERROR(FlushWithStatus()); }

  // Gets a fresh TimelineResource which can be used for tracking resource
  // read/write and setting barriers. Note that these are all allocated fresh
  // on each call today but may be pooled in the future.
  TimelineResource::Ref NewTimelineResource(
      std::shared_ptr<Fiber> fiber,
      TimelineResourceDestructor destructor = nullptr) {
    return TimelineResource::Ref(new TimelineResource(
        std::move(fiber), semaphore_count_, std::move(destructor)));
  }

  // Creates a new fence with capacity for all semaphores that are extant at
  // the point of the call.
  iree::hal_fence_ptr NewFence();

  System &system() { return system_; }

 private:
  void Initialize(
      std::span<const std::pair<std::string_view, Device *>> devices);
  System &system_;

  // Each distinct hal device gets an account.
  std::vector<Account> accounts_;
  // Accounts indexed by DeviceAddress::device_id().
  std::unordered_map<uint64_t, Account *> accounts_by_device_id_;
  size_t semaphore_count_ = 0;

  // Transaction management.
  TransactionMode tx_mode_ = TransactionMode::EAGER;
  TransactionType current_tx_type_ = TransactionType::NONE;

  friend class local::Fiber;
};

}  // namespace detail
}  // namespace shortfin::local

#endif  // SHORTFIN_LOCAL_SCHEDULER_H
