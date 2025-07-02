// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_CORO_TASK_H
#define SHORTFIN_CORO_TASK_H

#include <coroutine>
#include <exception>
#include <future>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <variant>

namespace shortfin::local {
namespace coro {
namespace detail {
template <typename ReturnTy>
class Promise;
}

template <typename ReturnTy = void>
class Task {
 public:
  using TaskTy = Task<ReturnTy>;
  // To make this class a coroutine, the compiler will look for a
  // promise_type declared in this class. This is specified by the
  // standard and we do not have any control over the naming.
  using promise_type = detail::Promise<ReturnTy>;
  using HandleTy = std::coroutine_handle<promise_type>;

  struct BaseAwaitable {
    BaseAwaitable(std::coroutine_handle<promise_type> handle)
        : handle_(handle) {}

    bool await_ready() const { return !handle_ || handle_.done(); }
    std::coroutine_handle<> await_suspend(
        std::coroutine_handle<> waiting_handle) {
      handle_.promise().set_continuation(waiting_handle);
      return handle_;
    }

   private:
    HandleTy handle_ = nullptr;
  };

  Task() noexcept : handle_(nullptr) {}
  Task(HandleTy handle) : handle_(handle) {}
  Task(const Task &) = delete;
  Task(Task &&other) : handle_(std::exchange(other.handle_, nullptr)) {}
  Task &operator=(const Task &) = delete;

  Task &operator=(Task &&other) {
    // TODO (vinayakdsci): It might be required to check if the address of other
    // is the same as this.
    if (handle_) {
      handle_.destroy();
    }
    std::exchange(other.handle_, nullptr);
    return *this;
  }

  ~Task() {
    if (handle_) {
      handle_.destroy();
    }
  }

  bool is_ready() const noexcept { return !handle_ || handle_.done(); }
  bool resume() {
    if (!handle_.done()) {
      handle_.resume();
    }
    return !handle_.done();
  }
  bool destroy() {
    if (!handle_) {
      return false;
    }

    handle_.destroy();
    handle_ = nullptr;
    return true;
  }

  // Implement the operator co_await for Task, which is what actually
  // makes a task awaitable. Again, we define the operator once by const
  // reference, and once by moveable reference.
  auto operator co_await() const & noexcept {
    struct Awaitable : public BaseAwaitable {
      // Let the compiler decide the return type.
      auto await_resume() -> decltype(auto) {
        return handle_.promise().result();
      }
    };
    // When we hit a co_await on a Task, it returns a struct Awaitable,
    // and we actually await on the Awaitable instead of the Task.
    return Awaitable{handle_};
  }

  auto operator co_await() const && noexcept {
    struct Awaitable : public BaseAwaitable {
      // Same thing here, except the promise is moved.
      auto await_resume() -> decltype(auto) {
        return std::move(handle_.promise()).result();
      }
    };
    return Awaitable{handle_};
  }

  // Returns a non-const lvalue ref, callable on lvalue refs.
  // coro.promise();
  promise_type &promise() & { return handle_.promise(); }

  // Returns a const lvalue ref, callable on const lvalue refs.
  // const_coro.promise();
  const promise_type &promise() const & { return handle_.promise(); }

  // Returns an rvalue ref, callable on rvalue refs (values being moved).
  // std::move(coro).promise();
  promise_type &&promise() && { return std::move(handle_.promise()); }

  HandleTy handle() { return handle_; }

  // Returns the future from the handle's promise. This is not much of use
  // from a user's perspective, and can possibly be removed.
  std::future<void> getFuture() { return handle_.promise().getFuture(); }

  // Meant to be called from a non-async function to block the main
  // thread of execution and wait for this Task to return.
  void BlockingWait() { handle_.promise().blockingWait(); }

  void OnCompletion(std::function<void()> callback) {
    handle_.promise().onCompletion(callback);
  }

 private:
  HandleTy handle_;
};

namespace detail {
class BasePromise {
 public:
  struct final_awaitable {
    constexpr bool await_ready() const noexcept { return false; }

    template <typename PromiseTy>
    std::coroutine_handle<> await_suspend(
        std::coroutine_handle<PromiseTy> handle) noexcept {
      auto &promise = handle.promise();

      promise.invoke_completion_callback();

      if (promise.continuation_) {
        return promise.continuation_;
      }
      return std::noop_coroutine();
    }

    constexpr void await_resume() noexcept {}
  };

  BasePromise() = default;
  ~BasePromise() = default;

  void onCompletion(std::function<void()> callback) {
    completion_callback_ = callback;
  }

  void invoke_completion_callback() {
    if (completion_callback_) {
      completion_callback_();
    }
  }

  std::suspend_always initial_suspend() noexcept { return {}; }
  final_awaitable final_suspend() noexcept { return final_awaitable{}; }

  void set_continuation(std::coroutine_handle<> handle) noexcept {
    continuation_ = handle;
  }

 protected:
  std::coroutine_handle<> continuation_;
  std::function<void()> completion_callback_;
};

/// Represents a basic Promise type. We do not allow any other
/// class to inherit from Promise.
/// A Promise can only be moved once constructed, which prevents
/// leakage of coroutine memory in unintended places.
template <typename ReturnTy = void>
class Promise final : public BasePromise {
  struct UnsetReturnValue;

 public:
  using TaskTy = Task<ReturnTy>;
  using HandleTy = std::coroutine_handle<Promise<ReturnTy>>;
  static constexpr bool returnTyIsRef = std::is_reference_v<ReturnTy>;
  // In case the return type is a reference, we store a pointer to the
  // underlying type referred to by the ref. Otherwise, treat it as const and
  // then get the underlying type by removing the TOPMOST const qualifier.
  using StorageTy =
      std::conditional_t<returnTyIsRef, std::remove_reference_t<ReturnTy> *,
                         std::remove_const_t<ReturnTy>>;

  // A Promise can either be unitialized, holding a typed value, or an
  // exception.
  using VariantTy =
      std::variant<UnsetReturnValue, StorageTy, std::exception_ptr>;

  // Constructors. As pointed above, once constructed, we do NOT allow any kind
  // of constructor to be called on a Promise. This helps prevent memory safety
  // issues and resource leakages,
  Promise() noexcept {}
  Promise(const auto &Promise) = delete;
  Promise(Promise &&) = delete;
  Promise &operator=(const Promise &) = delete;
  Promise &operator=(Promise &&) = delete;
  ~Promise() = default;

  TaskTy get_return_object() { return HandleTy::from_promise(*this); }

  // C++ Concepts here place constraints on the template typename.
  template <typename ValueTy>
    requires(returnTyIsRef && std::is_constructible_v<ReturnTy, ValueTy &&>) ||
            (!returnTyIsRef && std::is_constructible_v<StorageTy, ValueTy &&>)
  void return_value(ValueTy &&value) {
    if constexpr (returnTyIsRef) {
      auto ref = static_cast<ValueTy &&>(value);
      // std::variant::emplace constructs an std::variant in-place.
      // Because the type of the variant is dependent on a template,
      // we need to use the template keyword before the call to emplace.
      // We convert the reference (auto ref) here to a pointer to whatever
      // memory the variable ref was referencing and store it in the storage
      // variant.
      storage_variant_.template emplace<StorageTy>(std::addressof(ref));
    } else {
      // In case this is not a reference, store it directly.
      storage_variant_.template emplace<StorageTy>(value);
    }

    sync_future_.set_value();
  }

  void unhandled_exception() noexcept {
    storage_variant_ = VariantTy(std::current_exception());
  }

  std::future<void> getFuture() { return sync_future_.get_future(); }

  void blockingWait() { getFuture().wait(); }

  // We return the result three ways below. By const reference, by non-const
  // reference, and by moveable reference. The return type is variadic, so a
  // decltype on the returned result is used to pass the typechecker.
  auto result() & -> decltype(auto) {
    if (std::holds_alternative<StorageTy>(storage_variant_)) {
      // The storage_variant_ holds a value of type StorageTy.
      if (returnTyIsRef) {
        // We are holding a pointer in the variant, so dereference
        // and cast it.
        return static_cast<ReturnTy>(*std::get<StorageTy>(storage_variant_));
      }
      return static_cast<const ReturnTy &>(
          std::get<StorageTy>(storage_variant_));
    }

    // The execution hit an exception, and storage_variant_ holds an
    // and exception_ptr at this point.
    if (std::holds_alternative<std::exception_ptr>(storage_variant_)) {
      std::rethrow_exception(std::get<std::exception_ptr>(storage_variant_));
    }

    // The coroutine probably did not execute.
    throw std::runtime_error(
        "Return value was unset. Coroutine must be run to set a return value");
  }

  auto result() const & -> decltype(auto) {
    if (std::holds_alternative<StorageTy>(storage_variant_)) {
      // The storage_variant_ holds a value of type StorageTy.
      if (returnTyIsRef) {
        // We are holding a pointer in the variant, so dereference
        // and cast it.
        return static_cast<std::add_const_t<ReturnTy>>(
            *std::get<StorageTy>(storage_variant_));
      }
      return static_cast<const ReturnTy &>(
          std::get<StorageTy>(storage_variant_));
    }

    // The execution hit an exception, and storage_variant_ holds an
    // and exception_ptr at this point.
    if (std::holds_alternative<std::exception_ptr>(storage_variant_)) {
      std::rethrow_exception(std::get<std::exception_ptr>(storage_variant_));
    }

    // The coroutine probably did not execute.
    throw std::runtime_error(
        "Return value was unset. Coroutine must be run to set a return value");
  }

  auto result() && -> decltype(auto) {
    if (std::holds_alternative<StorageTy>(storage_variant_)) {
      // The storage_variant_ holds a value of type StorageTy.
      if (returnTyIsRef) {
        // We are holding a pointer in the variant, so dereference
        // and cast it.
        return static_cast<ReturnTy>(*std::get<StorageTy>(storage_variant_));
      }

      if (std::is_move_constructible_v<ReturnTy>) {
        return static_cast<ReturnTy &&>(std::get<StorageTy>(storage_variant_));
      }

      return static_cast<ReturnTy &&>(std::get<StorageTy>(storage_variant_));
    }

    // The execution hit an exception, and storage_variant_ holds an
    // and exception_ptr at this point.
    if (std::holds_alternative<std::exception_ptr>(storage_variant_)) {
      std::rethrow_exception(std::get<std::exception_ptr>(storage_variant_));
    }

    // The coroutine probably did not execute.
    throw std::runtime_error(
        "Return value was unset. Coroutine must be run to set a return value");
  }

 private:
  /// Serves as sentinel return value (i.e. an uninitialized return value).
  struct UnsetReturnValue {
    UnsetReturnValue() {}
    UnsetReturnValue(UnsetReturnValue &&) = delete;
    UnsetReturnValue(const UnsetReturnValue &) = delete;
    UnsetReturnValue operator=(const UnsetReturnValue &) = delete;
    UnsetReturnValue operator=(UnsetReturnValue &&) = delete;
  };
  VariantTy storage_variant_;
  // This future is ONLY used as a method of blocking wait on the same thread.
  // TODO: This can be better generalized to multiple threads using a mutex and
  // a condition_variable.
  std::promise<void> sync_future_;
};

// Specialize Promise for void type.
template <>
class Promise<void> : public BasePromise {
 public:
  using TaskTy = Task<void>;
  using HandleTy = std::coroutine_handle<Promise<void>>;
  Promise() noexcept {}
  Promise(const auto &Promise) = delete;
  Promise(Promise &&) = delete;
  Promise &operator=(const Promise &) = delete;
  Promise &operator=(Promise &&) = delete;
  ~Promise() = default;

  Task<void> get_return_object() { return HandleTy::from_promise(*this); }
  void return_void() noexcept { sync_future_.set_value(); }
  void unhandled_exception() noexcept {
    exception_ptr_ = std::current_exception();
  }

  void result() {
    if (exception_ptr_) {
      std::rethrow_exception(exception_ptr_);
    }
  }

  std::future<void> getFuture() { return sync_future_.get_future(); }

  void blockingWait() { getFuture().wait(); }

 private:
  std::exception_ptr exception_ptr_;
  std::promise<void> sync_future_;
};

}  // namespace detail

}  // namespace coro
}  // namespace shortfin::local

#endif
