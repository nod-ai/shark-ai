// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/support/blocking_executor.h"

#include "fmt/core.h"
#include "shortfin/support/logging.h"

namespace shortfin {

BlockingExecutor::BlockingExecutor(iree_allocator_t allocator)
    : allocator_(allocator) {}
BlockingExecutor::~BlockingExecutor() { Kill(/*wait=*/true); }

bool BlockingExecutor::has_free_threads() {
  iree::slim_mutex_lock_guard g(control_mu_);
  return free_threads_ != nullptr;
}

void BlockingExecutor::Kill(bool wait, iree_timeout_t warn_timeout) {
  {
    iree::slim_mutex_lock_guard g(control_mu_);
    kill_ = true;
  }

  // Because there can be some asynchronicity in terms of the kill signal and
  // additional scheduling, we loop on a short delay, sending kill signals,
  // issuing a warning if waiting for a long time. Once the number of
  // live threads drops to zero, we set the inhibit flag, causing no more
  // work to be accepted and exit.
  iree_time_t report_deadline = iree_timeout_as_deadline_ns(warn_timeout);
  for (;;) {
    // Wake up all free threads, leaving their task null. This will cause them
    // to loop once and ask the executor to add them back to the free list
    // (at which point, they will be told to shutdown).
    {
      iree::slim_mutex_lock_guard g(control_mu_);
      for (;;) {
        ThreadInstance *current = free_threads_;
        if (!current) break;
        current->signal_transact.set();
        free_threads_ = current->next;
        current->next = nullptr;
      }
    }

    // If not asked to wait, just exit.
    if (!wait) {
      break;
    }

    // Check if still alive.
    int last_live_thread_count;
    int total_thread_count;
    {
      iree::slim_mutex_lock_guard g(control_mu_);
      last_live_thread_count = live_thread_count_;
      total_thread_count = created_thread_count_;
      // If transitioned to 0 live threads, there is a short period of time
      // that can exist between the scan of the free list above and a task
      // getting scheduled. Therefore, the first time we hit this condition,
      // enter the inhibited state, which denies further scheduling. Then
      // the next time we encounter no live threads, that will be a true
      // count.
      if (live_thread_count_ == 0) {
        if (inhibit_) {
          break;
        } else {
          inhibit_ = true;
        }
      }
    }

    // Report if over the reporting deadling.
    if (iree_time_now() > report_deadline) {
      logging::warn(
          "Still waiting for blocking executor threads to terminate (live "
          "threads = {}, total created = {}).",
          last_live_thread_count, total_thread_count);
      report_deadline = iree_timeout_as_deadline_ns(warn_timeout);
    }

    // Short spin delay to signal/wait.
    // TODO: Introduce iree_thread_try_join and refactor the shutdown sequence
    // to be based on thread joining with a warn deadline.
    iree_wait_until(iree_timeout_as_deadline_ns(iree_make_timeout_ms(50)));
  }
}

void BlockingExecutor::Schedule(Task task) {
  ThreadInstance *target;
  bool target_is_new = false;
  {
    iree::slim_mutex_lock_guard g(control_mu_);
    if (inhibit_) {
      throw std::logic_error(
          "BlockingExecutor has shut down and new work cannot be scheduled");
    }
    target = free_threads_;
    if (target) {
      // Edit out of free list.
      free_threads_ = target->next;
      target->next = nullptr;
    } else {
      // Create new.
      std::string name = fmt::format("blocking-{}", created_thread_count_++);
      iree_thread_create_params_t params = {
          .name = {name.data(), name.size()},
          .create_suspended = true,
      };
      target = new ThreadInstance(this);
      auto EntryFunction = +[](void *self) noexcept {
        return static_cast<ThreadInstance *>(self)->RunOnThread();
      };
      auto status = iree_thread_create(EntryFunction, target, params,
                                       allocator_, target->thread.for_output());
      if (!iree_status_is_ok(status)) {
        delete target;
        SHORTFIN_THROW_IF_ERROR(status);
      }
      target_is_new = true;
      live_thread_count_ += 1;
    }
    target->current_task = std::move(task);
  }

  // Out of lock, continue dispatch.
  target->signal_transact.set();
  if (target_is_new) {
    iree_thread_resume(target->thread);
  }
}

bool BlockingExecutor::LockAndMoveToFreeList(ThreadInstance *inst) {
  iree::slim_mutex_lock_guard g(control_mu_);
  // If still running, add to free list. Else delete.
  if (kill_) {
    inst->executor = nullptr;
    live_thread_count_ -= 1;
    return false;
  } else {
    inst->next = free_threads_;
    free_threads_ = inst;
    inst->signal_transact.reset();
    return true;
  }
}

BlockingExecutor::ThreadInstance::ThreadInstance(BlockingExecutor *executor)
    : executor(executor), signal_transact(false), next(nullptr) {}

int BlockingExecutor::ThreadInstance::RunOnThread() noexcept {
  for (;;) {
    auto status = iree_wait_source_wait_one(signal_transact.await(),
                                            iree_infinite_timeout());
    IREE_CHECK_OK(status);
    Task exec_task;
    {
      iree::slim_mutex_lock_guard g(executor->control_mu_);
      exec_task = std::move(current_task);
    }
    if (exec_task) {
      exec_task();
    }
    if (!executor->LockAndMoveToFreeList(this)) {
      break;
    }
  }

  delete this;
  return 0;
}

}  // namespace shortfin
