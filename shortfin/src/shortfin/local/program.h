// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_LOCAL_PROGRAM_H
#define SHORTFIN_LOCAL_PROGRAM_H

#include <filesystem>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "shortfin/local/async.h"
#include "shortfin/local/device.h"
#include "shortfin/local/program_interfaces.h"
#include "shortfin/local/scheduler.h"
#include "shortfin/local/worker.h"
#include "shortfin/support/api.h"
#include "shortfin/support/iree_helpers.h"

namespace shortfin::local {

class BaseProgramParameters;
class Fiber;
class System;

namespace detail {
struct ProgramIsolate;
}  // namespace detail

enum class ProgramInvocationModel {
  // Uses the coarse-fences invocation model. In this model, the last two
  // arguments are a wait and signal fence, which are used for function-level
  // scheduling.
  COARSE_FENCES,
  // The function was not annotated with an invocation model.
  NONE,
  // The function is not annotated or is simple/synchronous.
  UNKNOWN,
};

// The level of isolation that a program has with respect to concurrent use.
enum class ProgramIsolation {
  // There is no isolation: Callers are completely on their own to only issue
  // concurrent invocations if supported.
  NONE = 0,

  // Each fiber in the system that makes calls into the program will have its
  // own shallow fork of the module. This is done on-demand and the root
  // program is retained for the lifetime of any referencing fibers.
  // Concurrent calls on the same fiber are considered programming errors and
  // will be flagged as such at an appropriate debug level.
  PER_FIBER = 1,

  // Each call triggers a shallow fork of the module. This is the most expensive
  // but safest way to ensure complete isolation of stateless invocations.
  PER_CALL = 2,
};

// State related to making an invocation of a function on a program.
//
// Since ownership of this object is transferred to the loop/callback and
// internal pointers into it must remain stable, it is only valid to heap
// allocate it.
class SHORTFIN_API ProgramInvocation {
  struct Deleter {
    void operator()(ProgramInvocation *);
  };

 public:
  // The fact that we traffic in invocation pointers based on unique_ptr
  // is incidental. By cloaking its public interface this way, we use the
  // unique_ptr machinery but template meta-programming that is specialized
  // for unique_ptr sees this as a bespoke class (which is what we want because
  // ownership semantics are special).
  class Ptr : private std::unique_ptr<ProgramInvocation, Deleter> {
   public:
    using unique_ptr::unique_ptr;
    using unique_ptr::operator=;
    using unique_ptr::operator->;
    using unique_ptr::operator bool;
    using unique_ptr::get;
    using unique_ptr::release;
  };
  static_assert(sizeof(Ptr) == sizeof(void *));
  using Future = TypedFuture<ProgramInvocation::Ptr>;

  static Ptr New(std::shared_ptr<Fiber> fiber, iree::vm_context_ptr vm_context,
                 iree_vm_function_t &vm_function,
                 ProgramInvocationModel invocation_model,
                 detail::ProgramIsolate *isolate);
  ProgramInvocation(const ProgramInvocation &) = delete;
  ProgramInvocation &operator=(const ProgramInvocation &) = delete;
  ProgramInvocation &operator=(ProgramInvocation &&) = delete;
  ProgramInvocation(ProgramInvocation &&inv) = delete;
  ~ProgramInvocation();

  // Whether the ProgramInvocation has entered the scheduled state. Once
  // scheduled, arguments and initialization parameters can no longer be
  // accessed.
  bool scheduled() const { return scheduled_; }

  // The fiber this invocation was scheduled against.
  Fiber *fiber() const { return fiber_.get(); }

  // Adds wait barriers to the invocation. For coarse fences invocations, these
  // will cause execution of the function to wait until all sempahores added
  // thusly are satisfied.
  void wait_insert(iree_hal_semaphore_list_t sem_list);

  // Adds a ref object argument. This low level interface directly adds a
  // reference object and does not manipulate any execution barriers.
  void AddArg(iree::vm_opaque_ref ref,
              detail::TimelineResource *resource);  // Moves a reference in.
  void AddArg(iree_vm_ref_t *ref,
              detail::TimelineResource *resource);  // Borrows the reference.

  // Transfers ownership of an invocation and schedules it on worker, returning
  // a future that will resolve to the owned invocation upon completion.
  static ProgramInvocation::Future Invoke(ProgramInvocation::Ptr invocation);

  // Gets the number of outputs.
  iree_host_size_t results_size();

  // Gets the i'th result as an opaque ref object. Returns a null ref if the
  // result is a primitive. Outputs accessed in this way are not marshaled
  // nor do they have concurrency barriers applied.
  iree::vm_opaque_ref result_ref(iree_host_size_t i);

  // As arguments are processed, the device they are associated with should be
  // passed here. The accumulation of these will drive the selection of the
  // scheduling account used for the invocation timeline. In the absence of
  // a specific directive, all arguments implicated in scheduling (i.e.
  // excepting those with ProgramResourceBarrier::NONE) must be on the same
  // logical device and only differ by queue affinity.
  // This method will raise an exception if the implied semantics are violated.
  void DeviceSelect(DeviceAffinity device_affinity);

  // Selected device affinity used for scheduling.
  const DeviceAffinity &device_selection() { return device_selection_; }

  // If this invocation provides coarse signaling of result availability,
  // the semaphore and timepoint are returned here. If the semaphore is null,
  // then coarse signaling is not available.
  // Valid after invocation has been scheduled.
  std::pair<iree_hal_semaphore_t *, uint64_t> coarse_signal() {
    return std::make_pair(signal_sem_, signal_timepoint_);
  }

  std::string to_s();

 private:
  ProgramInvocation();
  void CheckNotScheduled();
  // Eagerly releases context when it is known that no further use of it can
  // be made (allowing it to be returned to a pool prior to the invocation
  // actually being recycled). Object destruction also does this, but possibly
  // extending the context lifetime.
  void ReleaseContext();

  // Returns a pointer to the trailing arg list.
  iree_vm_list_t *arg_list();

  // Accesses the invocation owned wait fence, creating it if needed.
  iree_hal_fence_t *wait_fence();

  // Called as part of scheduling to finalize the calling convention and
  // invocation model after user arguments have been added. Because this is
  // potentially run in a foreign callback context, it uses iree_status_t
  // error reporting vs exceptions.
  iree_status_t FinalizeCallingConvention(
      iree_vm_list_t *arg_list, iree_vm_function_t &function,
      ProgramInvocationModel invocation_model);

  // Parameters needed to make the async call are stored at construction time
  // up until the point the call is made in the params union. When invoking,
  // these will be copied to the stack and passed to the async invocation,
  // which initializes the async_invoke_state. Phasing it like this saves
  // memory that would otherwise be retained for the life of the invocation.
  // This must not contain entities that require destruction or cannot be
  // trivially copied.
  struct Params {
    iree_vm_function_t function;
    ProgramInvocationModel invocation_model;
  };
  union State {
    State() { new (&params) Params(); }
    ~State() {}
    Params params;
    iree_vm_async_invoke_state_t async_invoke_state;
  } state;

  std::shared_ptr<Fiber> fiber_;
  iree::vm_context_ptr vm_context_;
  detail::ProgramIsolate *isolate_;
  iree_vm_list_t *result_list_ = nullptr;
  detail::TimelineResource **arg_resources_ = nullptr;
  std::optional<Future> future_;
  iree::hal_fence_ptr wait_fence_;
  iree_hal_semaphore_t *signal_sem_ = nullptr;
  uint64_t signal_timepoint_ = 0;
  DeviceAffinity device_selection_;
  bool scheduled_ = false;
};

// References a function in a Program.
class SHORTFIN_API ProgramFunction {
 public:
  operator bool() const { return vm_context_; }

  std::string_view name() const;
  std::string_view calling_convention() const;
  ProgramInvocationModel invocation_model() const { return invocation_model_; }
  // Gets the default isolation level for this function.
  ProgramIsolation isolation() const { return isolation_; }
  ProgramInvocation::Ptr CreateInvocation(
      std::shared_ptr<Fiber> fiber,
      std::optional<ProgramIsolation> isolation = std::nullopt);

  std::string to_s() const;

  operator iree_vm_context_t *() { return vm_context_.get(); }
  operator iree_vm_function_t &() { return vm_function_; }

 private:
  ProgramFunction(iree::vm_context_ptr vm_context,
                  iree_vm_function_t vm_function, ProgramIsolation isolation,
                  std::optional<ProgramInvocationModel> invocation_model = {});

  static ProgramInvocationModel GetInvocationModelFromFunction(
      iree_vm_function_t &f);

  iree::vm_context_ptr vm_context_;
  iree_vm_function_t vm_function_;
  ProgramIsolation isolation_;
  ProgramInvocationModel invocation_model_;
  friend class Program;
};

// High level API for working with program modules. Think of a module as
// a shared library in a traditional Unix system:
//
//   * Has a name and access to a certain amount of metadata.
//   * Exports functions which can be resolved and invoked.
//   * Imports functions that must be resolved by previously loaded modules.
//   * Can perform arbitrary initialization activity.
//   * Are loaded into an overall ProgramContext.
//
// Modules are thread-safe and typically loaded globally (think of them as
// files on disk, although they can be composed in multiple ways), while
// loading them into a ProgramContext causes them to be linked and made
// available for specific invocations.
//
// Under the hood, these are implemented in terms of iree_vm_module_t, which
// can be either custom, builtin or loaded from bytecode.
class SHORTFIN_API ProgramModule {
 public:
  std::string to_s() const;
  iree_vm_module_t *vm_module() const { return vm_module_; }
  std::string_view name() const;
  System &system() const { return *system_; }

  // Loads a dynamic bytecode module (VMFB) from a path on the file system.
  static ProgramModule Load(System &system, const std::filesystem::path &path,
                            bool mmap = false);

  // Creates a ProgramModule that will provide the given list of parameters
  // to modules loaded after it. In IREE parlance, this produces an
  // 'io_parameters' VM module.
  static ProgramModule ParameterProvider(
      System &system, std::span<BaseProgramParameters *> params);

  // Gets the name of all exported functions.
  std::vector<std::string> exports() const;

 protected:
  explicit ProgramModule(std::shared_ptr<System> system,
                         iree::vm_module_ptr vm_module)
      : system_(std::move(system)), vm_module_(std::move(vm_module)) {}

 private:
  std::shared_ptr<System> system_;
  iree::vm_module_ptr vm_module_;
};

// Programs consist of ProgramModules instantiated together and capable of
// having functions invoked on them. While the underlying programming model
// is a bit broader and can be exploited in various advanced way, generally,
// a program should be thought of as a fiber, and it is therefore bound to
// a Fiber, which provides a logical thread of execution. By default, all
// invocations will take place in logical order (there are certain ways to
// violate this constraint safely that are provided for separately).
//
// The program will source any needed parameters from the System and it will
// make an effort to cache them for proper locality on individual devices
// (TODO: make this actually true).
class SHORTFIN_API Program {
 public:
  struct Options {
    Options() {}

    // Ordered list of devices to bind this program to.
    std::span<const Device *> devices;

    // The isolation level to apply to program invocation.
    ProgramIsolation isolation = ProgramIsolation::PER_FIBER;

    // Enables program-wide execution tracing (to stderr).
    bool trace_execution = false;
  };

  // Load a program from a list of modules and options.
  static Program Load(std::span<const ProgramModule> modules,
                      Options &&options);

  // Looks up a public function by fully qualified name (i.e. module.function).
  // Returns nothing if not found.
  std::optional<ProgramFunction> LookupFunction(std::string_view name);

  // Looks up a public function by fully qualified name, throwing an
  // invalid_argument exception on failure to find.
  ProgramFunction LookupRequiredFunction(std::string_view name);

  // Gets the name of all exported functions.
  std::vector<std::string> exports() const;

  // Gets the default isolation level for all functions in this program.
  ProgramIsolation isolation() const { return isolation_; }

  // Eagerly does any per-fiber isolation preparation for the program at a
  // convenient point (usually init time) to avoid first-invocation overhead.
  void PrepareIsolate(Fiber &fiber);

 private:
  explicit Program(iree::vm_context_ptr vm_context, ProgramIsolation isolation)
      : vm_context_(std::move(vm_context)), isolation_(isolation) {}

  iree::vm_context_ptr vm_context_;
  ProgramIsolation isolation_;
  friend class Fiber;
};

// Base class for classes that can be interpreted as a provider of program
// parameters.
class SHORTFIN_API BaseProgramParameters {
 public:
  BaseProgramParameters() = default;
  BaseProgramParameters(const BaseProgramParameters &) = delete;
  BaseProgramParameters &operator=(const BaseProgramParameters &) = delete;
  virtual ~BaseProgramParameters();

  operator iree_io_parameter_provider_t *() { return provider_.get(); }

 protected:
  iree::io_parameter_provider_ptr provider_;
};

// Pool of parameters that can be made available to ProgramModules. Each
// instance represents a unique "parameter scope" name which corresponds to
// some set of parameters that one or more ProgramModules were compiled to
// depend on.
//
// This class wraps the lower level iree_io_parameter_provider_t and a single
// iree_io_parameter_index_t. While the underlying APIs have many ways that
// they can be composed, populated and manipulated, this facility presumes
// that has been done elsewhere and primarily targets referencing them from
// somewhere statically known. More advanced use cases will be served by
// additional APIs.
class SHORTFIN_API StaticProgramParameters : public BaseProgramParameters {
 public:
  StaticProgramParameters(
      System &system, std::string_view parameter_scope,
      iree_host_size_t max_concurrent_operations =
          IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS);

  struct LoadOptions {
    // File format. If empty, then it is inferred from the file name or
    // contents. Can be one of "irpa", "gguf", "safetensors", etc.
    std::string format;

    // Whether the backing file can be read.
    bool readable = true;
    // Whether the backing file can be written.
    bool writable = false;
    // Whether to mmap the file.
    bool mmap = false;
  };
  // Load parameters from a supported file format, applying no name
  // transformation.
  void Load(std::filesystem::path file_path, LoadOptions options);
  void Load(std::filesystem::path file_path) { Load(file_path, LoadOptions()); }

 private:
  iree_allocator_t host_allocator_;
  iree::io_parameter_index_ptr index_;

  void LoadMmap(std::filesystem::path file_path, LoadOptions options);
};

// Handles importing a batch of VM reference types and creating a
// TimelineResource that can be used to anchor their use and deallocation.
// This is intended to be a stack allocated instance used as part of a loop for
// transforming things like a list of function results.
//
// The transformation must be performed in two steps, first by visiting every
// ref and unwinding it to its root iree_hal_buffer_t allocation. Then every
// direct child allocation of this buffer is counted. If the reference count
// of the root buffer == the ref count of all children we can observe, then
// we can claim that the root allocation is uniquely owned by us (the receiver).
// This allows us to make certain claims/optimizations against its timeline.
// Otherwise, if the providence cannot be established, we must treat its
// lifetime conservatively.
//
// Every uniquely owned root buffer will have a new timeline resource created
// for it created with an async destructor, and all of its child buffers will
// share this resource.
//
// Note that this class could be optimized with a small object optimization and
// avoid accounting allocations, but this is left for a future upgrade.
class SHORTFIN_API CoarseInvocationTimelineImporter {
 public:
  struct Options {
    // Assumes that the invocation results cannot alias buffers that are
    // otherwise owned and can be assumed to be uniquely allocated for the
    // caller. It is possible to compile programs that violate this, and this
    // makes certain runtime optimizations impossible. We will restrict this
    // in the compiler's calling convention at a certain point so that someone
    // cannot accidentally do this.
    bool assume_no_alias = true;
  };
  CoarseInvocationTimelineImporter(ProgramInvocation *inv, Options &options);

  // Access timeline resources for a buffer.
  detail::TimelineResource::Ref ImportTimelineResource(
      iree_hal_buffer_t *user_buffer);

 private:
  void CatalogInternal(iree_hal_buffer_t *user_buffer, bool from_buffer_view);
  void FinalizeCatalog();
  struct allocated_buffer_record {
    size_t primordial_ref_count = 0;
    size_t observed_ref_count = 0;
    // For root buffers, the timeline resource.
    detail::TimelineResource::Ref timeline_resource;
  };

  ProgramInvocation *inv_;
  Options options_;
  std::unordered_map<iree_hal_buffer_t *, allocated_buffer_record>
      allocated_buffers_;
};

namespace detail {
// See Fiber::program_isolates_.
struct ProgramIsolate {
  ProgramIsolate(iree::vm_context_ptr parent_context)
      : parent_context(std::move(parent_context)) {}
  iree::vm_context_ptr parent_context;
  std::vector<iree::vm_context_ptr> fork_contexts;

  // Acquires an isolate for the given fiber. This will return a context which
  // may be the original program context or may be a forked child that is
  // available for use. It is only valid to call this when isolation != NONE.
  static std::pair<iree::vm_context_ptr, detail::ProgramIsolate *>
  AcquireIsolate(Fiber &fiber, iree::vm_context_ptr root_context,
                 ProgramIsolation isolation);

  // Releases an isolate obtained from a fiber in AcquireIsolate.
  static void ReleaseIsolate(Fiber &fiber, iree::vm_context_ptr context,
                             ProgramIsolate *isolate);
};
};  // namespace detail

}  // namespace shortfin::local

#endif  // SHORTFIN_LOCAL_PROGRAM_H
