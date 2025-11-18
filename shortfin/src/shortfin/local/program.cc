// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/local/program.h"

#include "fmt/core.h"
#include "fmt/std.h"
#include "fmt/xchar.h"
#include "iree/io/formats/parser_registry.h"
#include "iree/modules/hal/module.h"
#include "iree/modules/io/parameters/module.h"
#include "iree/vm/bytecode/module.h"
#include "shortfin/local/fiber.h"
#include "shortfin/local/system.h"
#include "shortfin/support/logging.h"

namespace shortfin::local {

namespace {
void GetVmModuleExports(iree_vm_module_t *vm_module,
                        std::vector<std::string> &exports) {
  auto sig = iree_vm_module_signature(vm_module);
  for (iree_host_size_t i = 0; i < sig.export_function_count; ++i) {
    iree_vm_function_t f;
    SHORTFIN_THROW_IF_ERROR(iree_vm_module_lookup_function_by_ordinal(
        vm_module, IREE_VM_FUNCTION_LINKAGE_EXPORT, i, &f));
    exports.emplace_back(to_string_view(iree_vm_function_name(&f)));
  }
}
}  // namespace

// -------------------------------------------------------------------------- //
// ProgramFunction
// -------------------------------------------------------------------------- //

ProgramFunction::ProgramFunction(
    iree::vm_context_ptr vm_context, iree_vm_function_t vm_function,
    ProgramIsolation isolation,
    std::optional<ProgramInvocationModel> invocation_model)
    : vm_context_(std::move(vm_context)),
      vm_function_(vm_function),
      isolation_(isolation),
      invocation_model_(invocation_model
                            ? *invocation_model
                            : GetInvocationModelFromFunction(vm_function)) {}

ProgramInvocationModel ProgramFunction::GetInvocationModelFromFunction(
    iree_vm_function_t &f) {
  iree_string_view_t invocation_model_sv =
      iree_vm_function_lookup_attr_by_name(&f, IREE_SV("iree.abi.model"));
  if (iree_string_view_equal(invocation_model_sv, IREE_SV("coarse-fences"))) {
    return ProgramInvocationModel::COARSE_FENCES;
  } else if (invocation_model_sv.size == 0) {
    return ProgramInvocationModel::NONE;
  } else {
    logging::warn("Unknown function invocation model '{}': '{}'",
                  to_string_view(iree_vm_function_name(&f)),
                  to_string_view(invocation_model_sv));
    return ProgramInvocationModel::UNKNOWN;
  }
}

std::string_view ProgramFunction::name() const {
  if (!*this) return {};
  return to_string_view(iree_vm_function_name(&vm_function_));
}

std::string_view ProgramFunction::calling_convention() const {
  if (!*this) return {};
  return to_string_view(
      iree_vm_function_signature(&vm_function_).calling_convention);
}

ProgramInvocation::Ptr ProgramFunction::CreateInvocation(
    std::shared_ptr<Fiber> fiber, std::optional<ProgramIsolation> isolation) {
  SHORTFIN_TRACE_SCOPE_NAMED("ProgramFunction::CreateInvocation");
  ProgramIsolation actual_isolation = isolation ? *isolation : isolation_;
  // Low-overhead NONE isolation handling (saves some ref-count twiddling).
  if (actual_isolation == ProgramIsolation::NONE) {
    return ProgramInvocation::New(std::move(fiber), vm_context_, vm_function_,
                                  invocation_model_, /*isolate=*/nullptr);
  }

  // Create an isolated invocation.
  auto [isolated_context, isolate] = detail::ProgramIsolate::AcquireIsolate(
      *fiber, vm_context_, actual_isolation);
  return ProgramInvocation::New(std::move(fiber), std::move(isolated_context),
                                vm_function_, invocation_model_, isolate);
}

std::string ProgramFunction::to_s() const {
  if (!*this) return std::string("ProgramFunction(NULL)");
  return fmt::format("ProgramFunction({}: {})", name(), calling_convention());
}

// -------------------------------------------------------------------------- //
// ProgramModule
// -------------------------------------------------------------------------- //

ProgramModule ProgramModule::Load(System &system,
                                  const std::filesystem::path &path,
                                  bool mmap) {
  SHORTFIN_TRACE_SCOPE_NAMED("ProgramModule::Load");
  iree::file_contents_ptr contents;
  if (mmap) {
    SHORTFIN_THROW_IF_ERROR(iree_io_file_contents_map(
        iree_make_string_view(path.string().c_str(), path.string().size()),
        IREE_IO_FILE_ACCESS_READ, system.host_allocator(),
        contents.for_output()));
  } else {
    SHORTFIN_THROW_IF_ERROR(iree_io_file_contents_read(
        iree_make_string_view(path.string().c_str(), path.string().size()),
        system.host_allocator(), contents.for_output()));
  }

  // Ownership hazard: iree_vm_bytecode_module_create only assumes ownership
  // of the contents when it returns *sucessfully*. In the exceptional case,
  // ownership remains with the caller, so we let the RAII wrapper hold on to
  // it until after success.
  iree::vm_module_ptr module;
  SHORTFIN_THROW_IF_ERROR(iree_vm_bytecode_module_create(
      system.vm_instance(), contents.const_buffer(), contents.deallocator(),
      system.host_allocator(), module.for_output()));
  contents.release();  // Must be invoked on success path only.
  return ProgramModule(system.shared_from_this(), std::move(module));
}

ProgramModule ProgramModule::ParameterProvider(
    System &system, std::span<BaseProgramParameters *> params) {
  std::vector<iree_io_parameter_provider_t *> providers;
  providers.reserve(params.size());
  for (auto *param : params) {
    iree_io_parameter_provider_t *provider = *param;
    if (!provider) {
      throw std::logic_error(
          "Cannot pass uninitialized parameters to ParameterProvider");
    }
    providers.push_back(provider);
  }

  iree::vm_module_ptr module;
  SHORTFIN_THROW_IF_ERROR(iree_io_parameters_module_create(
      system.vm_instance(), providers.size(), providers.data(),
      system.host_allocator(), module.for_output()));
  return ProgramModule(system.shared_from_this(), std::move(module));
}

std::string_view ProgramModule::name() const {
  return to_string_view(iree_vm_module_name(vm_module_));
}

std::string ProgramModule::to_s() const {
  auto sig = iree_vm_module_signature(vm_module_);
  std::vector<std::string> exports;
  for (iree_host_size_t i = 0; i < sig.export_function_count; ++i) {
    iree_vm_function_t f;
    SHORTFIN_THROW_IF_ERROR(iree_vm_module_lookup_function_by_ordinal(
        vm_module_, IREE_VM_FUNCTION_LINKAGE_EXPORT, i, &f));
    exports.push_back(fmt::format(
        "{}({})", to_string_view(iree_vm_function_name(&f)),
        to_string_view(iree_vm_function_signature(&f).calling_convention)));
  }
  return fmt::format("ProgramModule('{}', version={}, exports=[{}])", name(),
                     sig.version, fmt::join(exports, ", "));
}

std::vector<std::string> ProgramModule::exports() const {
  std::vector<std::string> exports;
  GetVmModuleExports(vm_module_, exports);
  return exports;
}

// -------------------------------------------------------------------------- //
// Program
// -------------------------------------------------------------------------- //

Program Program::Load(std::span<const ProgramModule> modules,
                      Options &&options) {
  SHORTFIN_TRACE_SCOPE_NAMED("Program::Load");
  std::vector<iree_vm_module_t *> all_modules;
  std::vector<iree_hal_device_t *> raw_devices;

  System *system = nullptr;
  // By default, bind all devices in the fiber in order to the program.
  for (auto &it : options.devices) {
    raw_devices.push_back(it->hal_device());
  }

  for (auto &mod : modules) {
    if (system && &mod.system() != system) {
      throw std::invalid_argument(
          "Cannot create Program from modules loaded from multiple system "
          "instances");
    }
    system = &mod.system();
  }
  if (!system) {
    throw std::invalid_argument("Cannot create Program with no modules");
  }

  // Add a HAL module.
  // TODO: at some point may want to change this to something similar to
  // what the tooling does in iree_tooling_resolve_modules - it uses
  // iree_vm_module_enumerate_dependencies to walk the dependencies and add the
  // required modules only as needed. to start you could use it just to see if
  // the hal is used, but as you add other module types for exposing amdsharkfin
  // functionality (or module versions; iree_vm_module_dependency_t has the
  // minimum version required so you can switch between them, and whether they
  // are optional/required).
  iree::vm_module_ptr hal_module;
  SHORTFIN_THROW_IF_ERROR(                           //
      iree_hal_module_create(                        //
          system->vm_instance(),                     //
          iree_hal_module_device_policy_default(),   //
          raw_devices.size(),                        //
          raw_devices.data(),                        //
          IREE_HAL_MODULE_FLAG_NONE,                 //
          iree_hal_module_debug_sink_stdio(stderr),  //
          system->host_allocator(),                  //
          hal_module.for_output()                    //
          )                                          //
  );
  all_modules.push_back(hal_module);

  // Add explicit modules.
  for (auto &pm : modules) {
    all_modules.push_back(pm.vm_module());
  }

  // Create the context.
  iree::vm_context_ptr context;
  iree_vm_context_flags_t flags = IREE_VM_CONTEXT_FLAG_CONCURRENT;
  if (options.trace_execution) flags |= IREE_VM_CONTEXT_FLAG_TRACE_EXECUTION;
  SHORTFIN_THROW_IF_ERROR(iree_vm_context_create_with_modules(
      system->vm_instance(), flags, all_modules.size(), all_modules.data(),
      system->host_allocator(), context.for_output()));

  return Program(std::move(context), options.isolation);
}

std::optional<ProgramFunction> Program::LookupFunction(std::string_view name) {
  // By convention, we currently name our coarse-fences function variants
  // as ending in "$async". These are the ones we want but it is inconvenient.
  // Therefore, we probe for that first.
  // TODO: We should add attributes to the function that better describe this
  // relationship.
  iree_vm_function_t f;
  if (!name.ends_with("$async")) {
    std::string async_name(name);
    async_name.append("$async");
    iree_status_t status = iree_vm_context_resolve_function(
        vm_context_, to_iree_string_view(async_name), &f);
    if (iree_status_is_ok(status)) {
      // TODO: Torch import is not setting the coarse-fences abi.model on
      // its functions. Get it from there instead of just assuming based on
      // name.
      return ProgramFunction(vm_context_, f, isolation_,
                             ProgramInvocationModel::COARSE_FENCES);
    } else if (!iree_status_is_not_found(status)) {
      SHORTFIN_THROW_IF_ERROR(status);
    }
  }

  // Resolve the exactly named function.
  iree_status_t status = iree_vm_context_resolve_function(
      vm_context_, to_iree_string_view(name), &f);
  if (iree_status_is_not_found(status)) return {};
  SHORTFIN_THROW_IF_ERROR(status);
  return ProgramFunction(vm_context_, f, isolation_);
}

ProgramFunction Program::LookupRequiredFunction(std::string_view name) {
  auto f = LookupFunction(name);
  if (!f) {
    throw std::invalid_argument(
        fmt::format("Function '{}' not found in program. Available exports: {}",
                    name, fmt::join(exports(), ", ")));
  }
  return std::move(*f);
}

std::vector<std::string> Program::exports() const {
  std::vector<std::string> results;

  // Iterate in reverse since "user modules" are typically last.
  int module_count = iree_vm_context_module_count(vm_context_);
  for (int i = module_count - 1; i >= 0; --i) {
    auto vm_module = iree_vm_context_module_at(vm_context_, i);
    std::string_view module_name =
        to_string_view(iree_vm_module_name(vm_module));
    std::vector<std::string> names;
    GetVmModuleExports(vm_module, names);
    for (auto &name : names) {
      results.push_back(fmt::format("{}.{}", module_name, name));
    }
  }
  return results;
}

void Program::PrepareIsolate(Fiber &fiber) {
  if (isolation_ == ProgramIsolation::NONE) return;
  auto [context, isolate] =
      detail::ProgramIsolate::AcquireIsolate(fiber, vm_context_, isolation_);
  if (isolate) {
    detail::ProgramIsolate::ReleaseIsolate(fiber, std::move(context), isolate);
  }
}

// -------------------------------------------------------------------------- //
// ProgramInvocation
// -------------------------------------------------------------------------- //

iree_vm_list_t *ProgramInvocation::arg_list() {
  // The arg list is located immediately after this, allocated as a trailing
  // data structure.
  return reinterpret_cast<iree_vm_list_t *>(reinterpret_cast<uint8_t *>(this) +
                                            sizeof(*this));
}

void ProgramInvocation::Deleter::operator()(ProgramInvocation *inst) {
  inst->~ProgramInvocation();
  uint8_t *memory = static_cast<uint8_t *>(static_cast<void *>(inst));

  // Trailing arg list and result list. The arg list pointer is only available
  // at construction, so we use the knowledge that it is stored right after
  // the object. The result_list_ is available for the life of the invocation.
  auto *arg_list = static_cast<iree_vm_list_t *>(
      static_cast<void *>(memory + sizeof(ProgramInvocation)));
  iree_host_size_t arg_size = iree_vm_list_size(arg_list);

  iree_vm_list_deinitialize(arg_list);
  iree_vm_list_deinitialize(inst->result_list_);

  // Release any arg resource references.
  for (iree_host_size_t i = 0; i < arg_size; ++i) {
    if (inst->arg_resources_[i]) inst->arg_resources_[i]->Release();
  }

  // Was allocated in New as a uint8_t[] so delete it by whence it came.
  delete[] memory;
}

ProgramInvocation::ProgramInvocation() = default;
ProgramInvocation::~ProgramInvocation() { ReleaseContext(); }

void ProgramInvocation::ReleaseContext() {
  if (vm_context_) {
    if (isolate_) {
      detail::ProgramIsolate::ReleaseIsolate(*fiber_, std::move(vm_context_),
                                             isolate_);
    } else {
      vm_context_.reset();
    }
  }
}

ProgramInvocation::Ptr ProgramInvocation::New(
    std::shared_ptr<Fiber> fiber, iree::vm_context_ptr vm_context,
    iree_vm_function_t &vm_function, ProgramInvocationModel invocation_model,
    detail::ProgramIsolate *isolate) {
  auto sig = iree_vm_function_signature(&vm_function);
  iree_host_size_t arg_count;
  iree_host_size_t result_count;
  SHORTFIN_THROW_IF_ERROR(iree_vm_function_call_count_arguments_and_results(
      &sig, &arg_count, &result_count));

  // Compute size of trailing arg/result storage.
  auto variant_type_def = iree_vm_make_undefined_type_def();
  iree_host_size_t arg_storage_size =
      iree_vm_list_storage_size(&variant_type_def, arg_count);
  iree_host_size_t result_storage_size =
      iree_vm_list_storage_size(&variant_type_def, result_count);
  iree_host_size_t arg_resource_size =
      sizeof(detail::TimelineResource *) * arg_count;

  // Allocate storage for the ProgramInvocation, arg, result list and placement
  // new the ProgramInvocation into the storage area.
  std::unique_ptr<uint8_t[]> inst_storage(
      new uint8_t[sizeof(ProgramInvocation) + arg_storage_size +
                  result_storage_size + arg_resource_size]);
  new (inst_storage.get()) ProgramInvocation();
  // Initialize trailing lists. Abort on failure since this is a bug and we
  // would otherwise leak.
  iree_vm_list_t *arg_list;
  iree_vm_list_t *result_list;
  IREE_CHECK_OK(iree_vm_list_initialize(
      {.data = inst_storage.get() + sizeof(ProgramInvocation),
       .data_length = arg_storage_size},
      &variant_type_def, arg_count, &arg_list));
  IREE_CHECK_OK(iree_vm_list_initialize(
      {.data =
           inst_storage.get() + sizeof(ProgramInvocation) + arg_storage_size,
       .data_length = result_storage_size},
      &variant_type_def, result_count, &result_list));

  uint8_t *arg_resource_ptr = inst_storage.get() + sizeof(ProgramInvocation) +
                              arg_storage_size + result_storage_size;
  std::memset(arg_resource_ptr, 0, arg_resource_size);

  Ptr inst(static_cast<ProgramInvocation *>(
               static_cast<void *>(inst_storage.release())),
           Deleter());
  inst->fiber_ = std::move(fiber);
  inst->vm_context_ = std::move(vm_context);
  inst->isolate_ = isolate;
  inst->state.params.function = vm_function;
  inst->state.params.invocation_model = invocation_model;
  inst->result_list_ = result_list;
  inst->arg_resources_ = static_cast<detail::TimelineResource **>(
      static_cast<void *>(arg_resource_ptr));
  return inst;
}

void ProgramInvocation::CheckNotScheduled() {
  if (scheduled()) {
    throw std::logic_error("Cannot mutate an invocation once scheduled.");
  }
}

void ProgramInvocation::AddArg(iree::vm_opaque_ref ref,
                               detail::TimelineResource *resource) {
  CheckNotScheduled();
  iree_host_size_t arg_index = iree_vm_list_size(arg_list());
  SHORTFIN_THROW_IF_ERROR(iree_vm_list_push_ref_move(arg_list(), &ref));
  if (resource) {
    arg_resources_[arg_index] = resource;
    resource->Retain();
  }
}

void ProgramInvocation::AddArg(iree_vm_ref_t *ref,
                               detail::TimelineResource *resource) {
  CheckNotScheduled();
  iree_host_size_t arg_index = iree_vm_list_size(arg_list());
  SHORTFIN_THROW_IF_ERROR(iree_vm_list_push_ref_retain(arg_list(), ref));
  if (resource) {
    arg_resources_[arg_index] = resource;
    resource->Retain();
  }
}

iree_status_t ProgramInvocation::FinalizeCallingConvention(
    iree_vm_list_t *arg_list, iree_vm_function_t &function,
    ProgramInvocationModel invocation_model) {
  // Handle post-processing invocation model setup.
  if (invocation_model == ProgramInvocationModel::COARSE_FENCES) {
    // If we have a device_selection, set up to signal the leader account.
    iree_hal_fence_t *maybe_wait_fence = nullptr;
    if (device_selection_) {
      ScopedDevice scoped_device(*fiber(), device_selection_);
      auto &sched_account =
          fiber()->scheduler().GetDefaultAccount(scoped_device);
      maybe_wait_fence = this->wait_fence();
      iree_hal_semaphore_t *timeline_sem = sched_account.timeline_sem();
      uint64_t timeline_now = sched_account.timeline_idle_timepoint();
      SHORTFIN_SCHED_LOG("Invocation {}: Wait on account timeline {}@{}",
                         static_cast<void *>(this),
                         static_cast<void *>(timeline_sem), timeline_now);
      IREE_RETURN_IF_ERROR(
          iree_hal_fence_insert(maybe_wait_fence, timeline_sem, timeline_now));
      signal_sem_ = sched_account.timeline_sem();
      signal_timepoint_ = sched_account.timeline_acquire_timepoint();

      // Extend any arg resources to our signal timepoint.
      iree_host_size_t arg_count = iree_vm_list_size(arg_list);
      for (iree_host_size_t i = 0; i < arg_count; ++i) {
        detail::TimelineResource *resource = arg_resources_[i];
        if (resource) {
          resource->use_barrier_insert(signal_sem_, signal_timepoint_);
        }
      }
    }

    // Push wait fence (or null if no wait needed).
    ::iree::vm::ref<iree_hal_fence_t> wait_ref;
    if (maybe_wait_fence) {
      wait_ref = ::iree::vm::retain_ref(maybe_wait_fence);
    }
    IREE_RETURN_IF_ERROR(iree_vm_list_push_ref_move(arg_list, wait_ref));

    // Create and push signal fence (or null if no signal needed).
    ::iree::vm::ref<iree_hal_fence_t> signal_ref;
    if (signal_sem_) {
      SHORTFIN_SCHED_LOG("Invocation {}: Set signal {}@{}",
                         static_cast<void *>(this),
                         static_cast<void *>(signal_sem_), signal_timepoint_);
      IREE_RETURN_IF_ERROR(
          iree_hal_fence_create_at(signal_sem_, signal_timepoint_,
                                   fiber()->host_allocator(), &signal_ref));
    }
    IREE_RETURN_IF_ERROR(iree_vm_list_push_ref_move(arg_list, signal_ref));
  } else {
    logging::warn(
        "Invoking function '{}' with unknown or synchronous invocation model "
        "is not fully supported",
        to_string_view(iree_vm_function_name(&function)));
  }

  return iree_ok_status();
}

ProgramInvocation::Future ProgramInvocation::Invoke(
    ProgramInvocation::Ptr invocation) {
  SHORTFIN_TRACE_SCOPE_NAMED("ProgramInvocation::Invoke");
  invocation->CheckNotScheduled();

  Worker &worker = invocation->fiber_->worker();
  // We're about to overwrite the instance level storage for params, so move
  // it to the stack and access there.
  Params params = invocation->state.params;

  auto schedule = [](ProgramInvocation *raw_invocation, Worker *worker,
                     iree_vm_function_t function,
                     ProgramInvocationModel invocation_model,
                     std::optional<ProgramInvocation::Future> failure_future) {
    SHORTFIN_TRACE_SCOPE_NAMED("ProgramInvocation::InvokeAsync");
    auto complete_callback =
        [](void *user_data, iree_loop_t loop, iree_status_t status,
           iree_vm_list_t *outputs) noexcept -> iree_status_t {
      SHORTFIN_TRACE_SCOPE_NAMED("ProgramInvocation::Complete");
      // Async invocation helpfully gives us a retained reference to the
      // outputs, but we already have one statically on the
      // ProgramInvocation. So release this one, which makes it safe to
      // deallocate the ProgramInvocation at any point after this (there
      // must be no live references to inputs/outputs when the
      // ProgramInvocation::Ptr deleter is invoked).
      iree::vm_list_ptr::steal_reference(outputs);

      // Repatriate the ProgramInvocation.
      ProgramInvocation::Ptr invocation(
          static_cast<ProgramInvocation *>(user_data));
      ProgramInvocation *raw_invocation = invocation.get();
      raw_invocation->ReleaseContext();
      if (iree_status_is_ok(status)) {
        raw_invocation->future_->set_result(std::move(invocation));
      } else {
        raw_invocation->future_->set_failure(status);
      }

      // Must release the future from the invocation to break the
      // circular reference (we are setting the invocation as the result
      // of the future).
      raw_invocation->future_.reset();

      return iree_ok_status();
    };

    ProgramInvocation::Ptr invocation(raw_invocation);
    iree_status_t status = iree_ok_status();

    // Multiple steps needed to schedule need to all exit via the same
    // path.
    if (iree_status_is_ok(status)) {
      status = invocation->fiber()->scheduler().FlushWithStatus();
    }
    if (iree_status_is_ok(status)) {
      status = invocation->FinalizeCallingConvention(
          invocation->arg_list(), function, invocation_model);
    }
    if (iree_status_is_ok(status)) {
      status = iree_vm_async_invoke(worker->loop(),
                                    &invocation->state.async_invoke_state,
                                    invocation->vm_context_.get(), function,
                                    /*flags=*/IREE_VM_INVOCATION_FLAG_NONE,
                                    /*policy=*/nullptr,
                                    /*inputs=*/invocation->arg_list(),
                                    /*outputs=*/invocation->result_list_,
                                    iree_allocator_system(), +complete_callback,
                                    /*user_data=*/invocation.get());
    }

    // On success, then the complete callback takes ownership of the
    // invocation, so we release it here and return. We have to treat
    // the invocation as possibly deallocated at this point, since the
    // async invocation may have finished already.
    if (iree_status_is_ok(status)) {
      invocation.release();
    } else if (failure_future) {
      // Requested to set any failure on the future.
      invocation->ReleaseContext();
      failure_future->set_failure(status);
    } else {
      // Synchronous: just throw.
      invocation->ReleaseContext();
      SHORTFIN_THROW_IF_ERROR(status);
    }
  };

  // Transition to the scheduled state.
  invocation->future_.emplace(&worker);
  auto fork_future = *invocation->future_;
  invocation->scheduled_ = true;

  if (&worker == Worker::GetCurrent()) {
    // On the same worker: fast-path directly to the loop.
    schedule(invocation.release(), &worker, params.function,
             params.invocation_model, /*failure_future=*/{});
  } else {
    // Cross worker coordination: submit an external task to bootstrap.
    auto bound_schedule = std::bind(schedule, invocation.release(), &worker,
                                    params.function, params.invocation_model,
                                    /*failure_future=*/fork_future);
    worker.CallThreadsafe(bound_schedule);
  }

  return fork_future;
}

iree_host_size_t ProgramInvocation::results_size() {
  return iree_vm_list_size(result_list_);
}

iree::vm_opaque_ref ProgramInvocation::result_ref(iree_host_size_t i) {
  iree::vm_opaque_ref out_value;
  auto status = iree_vm_list_get_ref_retain(result_list_, i, &out_value);
  if (iree_status_is_failed_precondition(status)) return {};
  SHORTFIN_THROW_IF_ERROR(status, "accessing invocation result");
  return out_value;
}

iree_hal_fence_t *ProgramInvocation::wait_fence() {
  if (!wait_fence_) {
    wait_fence_ = fiber_->scheduler().NewFence();
  }
  return wait_fence_.get();
}

void ProgramInvocation::wait_insert(iree_hal_semaphore_list_t sem_list) {
  iree_hal_fence_t *f = wait_fence();
  for (iree_host_size_t i = 0; i < sem_list.count; ++i) {
    SHORTFIN_SCHED_LOG("Invocation {}: Wait on {}@{}",
                       static_cast<void *>(this),
                       static_cast<void *>(sem_list.semaphores[i]),
                       sem_list.payload_values[i]);
    SHORTFIN_THROW_IF_ERROR(iree_hal_fence_insert(f, sem_list.semaphores[i],
                                                  sem_list.payload_values[i]));
  }
}

void ProgramInvocation::DeviceSelect(DeviceAffinity device_affinity) {
  CheckNotScheduled();
  SHORTFIN_SCHED_LOG("Invocation {}: DeviceSelect {}",
                     static_cast<void *>(this), device_affinity.to_s());
  device_selection_ |= device_affinity;
}

std::string ProgramInvocation::to_s() {
  return fmt::format("ProgramInvocation({}: result_size={})",
                     (scheduled_ ? "SCHEDULED" : "NOT_SCHEDULED"),
                     results_size());
}

// -------------------------------------------------------------------------- //
// BaseProgramParameters
// -------------------------------------------------------------------------- //

BaseProgramParameters::~BaseProgramParameters() = default;

// -------------------------------------------------------------------------- //
// StaticProgramParameters
// -------------------------------------------------------------------------- //

StaticProgramParameters::StaticProgramParameters(
    System &system, std::string_view parameter_scope,
    iree_host_size_t max_concurrent_operations)
    : host_allocator_(system.host_allocator()) {
  SHORTFIN_THROW_IF_ERROR(
      iree_io_parameter_index_create(host_allocator_, index_.for_output()));
  SHORTFIN_THROW_IF_ERROR(iree_io_parameter_index_provider_create(
      to_iree_string_view(parameter_scope), index_, max_concurrent_operations,
      host_allocator_, provider_.for_output()));
}

void StaticProgramParameters::Load(std::filesystem::path file_path,
                                   LoadOptions options) {
  SHORTFIN_TRACE_SCOPE_NAMED("StaticProgramParameters::Load");
  // Default format from extension.
  if (options.format.empty()) {
    options.format = file_path.extension().string();
  }

  if (options.mmap) {
    this->LoadMmap(file_path, options);
    return;
  }

  auto file_path_string = file_path.string();
  const iree_string_view_t path =
      iree_make_cstring_view(file_path_string.c_str());
  iree_io_file_handle_t *file_handle = NULL;
  SHORTFIN_THROW_IF_ERROR(iree_io_file_handle_open(
      IREE_IO_FILE_MODE_READ, path, host_allocator_, &file_handle));

  // Parse.
  SHORTFIN_THROW_IF_ERROR(
      iree_io_parse_file_index(to_iree_string_view(options.format), file_handle,
                               index_.get(), host_allocator_));
}

void StaticProgramParameters::LoadMmap(std::filesystem::path file_path,
                                       LoadOptions options) {
  SHORTFIN_TRACE_SCOPE_NAMED("StaticProgramParameters::LoadMmap");

  iree_io_file_contents_t *file_contents = nullptr;
  SHORTFIN_THROW_IF_ERROR(iree_io_file_contents_map(
      iree_make_string_view(file_path.string().c_str(),
                            file_path.string().size()),
      IREE_IO_FILE_ACCESS_READ, host_allocator_, &file_contents));
  iree_io_file_handle_release_callback_t release_callback = {
      +[](void *user_data, iree_io_file_handle_primitive_t handle_primitive) {
        iree_io_file_contents_t *file_contents =
            (iree_io_file_contents_t *)user_data;
        iree_io_file_contents_free(file_contents);
      },
      file_contents,
  };

  // Wrap contents.
  iree::io_file_handle_ptr file_handle;
  iree_status_t status = iree_io_file_handle_wrap_host_allocation(
      IREE_IO_FILE_ACCESS_READ, file_contents->buffer, release_callback,
      host_allocator_, file_handle.for_output());
  if (!iree_status_is_ok(status)) {
    iree_io_file_contents_free(file_contents);
    SHORTFIN_THROW_IF_ERROR(status);
  }

  // Parse.
  SHORTFIN_THROW_IF_ERROR(iree_io_parse_file_index(
      to_iree_string_view(options.format), file_handle.get(), index_.get(),
      host_allocator_));
}

// -------------------------------------------------------------------------- //
// CoarseInvocationTimelineImporter
// -------------------------------------------------------------------------- //

CoarseInvocationTimelineImporter::CoarseInvocationTimelineImporter(
    ProgramInvocation *inv, Options &options)
    : inv_(inv), options_(options) {
  SHORTFIN_TRACE_SCOPE_NAMED("CoarseInvocationTimelineImporter");
  auto buffer_type = iree_hal_buffer_type();
  auto buffer_view_type = iree_hal_buffer_view_type();
  for (iree_host_size_t i = 0; i < inv->results_size(); ++i) {
    auto ref = inv->result_ref(i);
    auto type = ref.get()->type;
    if (type == buffer_type) {
      iree_hal_buffer_t *buffer = iree_hal_buffer_deref(*ref.get());
      CatalogInternal(buffer, false);
    } else if (type == buffer_view_type) {
      iree_hal_buffer_view_t *bv = iree_hal_buffer_view_deref(*ref.get());
      iree_hal_buffer_t *buffer = iree_hal_buffer_view_buffer(bv);
      CatalogInternal(buffer, true);
    }
  }
  FinalizeCatalog();
}

void CoarseInvocationTimelineImporter::CatalogInternal(
    iree_hal_buffer_t *user_buffer, bool from_buffer_view) {
  SHORTFIN_TRACE_SCOPE_NAMED("CoarseInvocationTimelineImporter::Catalog");
  iree_hal_buffer_t *allocated_buffer =
      iree_hal_buffer_allocated_buffer(user_buffer);
  allocated_buffer_record &record = allocated_buffers_[allocated_buffer];
  // NOTE: Refcnt snooping is a sordid affair, but we believe is ok in this
  // very specific case. This is because we are holding the DAG of live uses
  // that we care about and this entire mechanism activates an optimization if
  // the number of references in our DAG accounts for *all* outstanding
  // references to the backing allocation. If this allocation was off somewhere
  // else having its reference count bumped, that could only invalidate the
  // optimization vs cause a logic issue.
  if (record.primordial_ref_count == 0) {
    // Only do the atomic read if not yet initialized.
    record.primordial_ref_count =
        iree_atomic_ref_count_load(&allocated_buffer->resource.ref_count);
    // The reference we are holding.
    record.observed_ref_count = 1;
  }
  if (allocated_buffer != user_buffer) {
    // It is a child buffer that contributes to the observed number of
    // references.
    record.observed_ref_count += from_buffer_view ? 2 : 1;
  } else if (from_buffer_view) {
    // It is a reference to the root allocation, which is from a buffer view.
    record.observed_ref_count += 1;
  }
}

void CoarseInvocationTimelineImporter::FinalizeCatalog() {
  for (auto &it : allocated_buffers_) {
    iree_hal_buffer_t *allocated_buffer = it.first;
    allocated_buffer_record &record = it.second;

    // We can only track for asynchronous dealloc if uniquely owned and
    // transferred to us.
    if (record.primordial_ref_count != record.observed_ref_count) {
      if (options_.assume_no_alias) {
        SHORTFIN_SCHED_LOG(
            "  : Async dealloc override: not unique but assume_no_alias ({} vs "
            "{})",
            record.primordial_ref_count, record.observed_ref_count);
      } else {
        SHORTFIN_SCHED_LOG(
            "  : Synchronous dealloc: Not uniquely owned ({} vs {})",
            record.primordial_ref_count, record.observed_ref_count);
        continue;
      }
    }

    // See iree_hal_buffer_placement_t documentation for criteria to satisfy
    // tracking for async deallocation.
    auto placement = iree_hal_buffer_allocation_placement(allocated_buffer);
    iree_hal_device_t *schedule_device =
        inv_->device_selection().device()->hal_device();
    auto schedule_affinity = inv_->device_selection().queue_affinity();
    if (!(placement.flags & IREE_HAL_BUFFER_PLACEMENT_FLAG_ASYNCHRONOUS)) {
      // Not an async allocation: disable optimization by ignoring this
      // entirely.
      SHORTFIN_SCHED_LOG(
          "  : Synchronous dealloc: Buffer not async allocated ({:x})",
          placement.flags);
      continue;
    }
    if (placement.device != schedule_device) {
      // Placed on a device other than our scheduling timeline.
      SHORTFIN_SCHED_LOG(
          "  : Synchronous dealloc: Schedule vs placement device mismatch ({} "
          "vs "
          "{})",
          static_cast<void *>(placement.device),
          static_cast<void *>(schedule_device));
      continue;
    }
    if ((placement.queue_affinity & schedule_affinity) != schedule_affinity) {
      SHORTFIN_SCHED_LOG(
          "  : Synchronous dealloc: Schedule affinity not a subset of "
          "placement "
          "affinity ({:x} vs {:x})",
          placement.queue_affinity, schedule_affinity);
      continue;
    }

    // All checks passed: Set it up for async deallocation.
    SHORTFIN_SCHED_LOG(
        "  : Asynchronous dealloc: Unique, compatible placement");
    ScopedDevice scoped_device(*inv_->fiber(), inv_->device_selection());
    auto dtor = detail::TimelineResource::CreateAsyncBufferDestructor(
        scoped_device,
        iree::hal_buffer_ptr::borrow_reference(allocated_buffer));
    record.timeline_resource =
        inv_->fiber()->NewTimelineResource(std::move(dtor));
  }
}

detail::TimelineResource::Ref
CoarseInvocationTimelineImporter::ImportTimelineResource(
    iree_hal_buffer_t *user_buffer) {
  iree_hal_buffer_t *allocated_buffer =
      iree_hal_buffer_allocated_buffer(user_buffer);
  auto found_it = allocated_buffers_.find(allocated_buffer);
  // Fork sync/async to different functions so that the distinction shows up
  // vividly in traces.
  if (found_it != allocated_buffers_.end() &&
      found_it->second.timeline_resource) {
    SHORTFIN_SCHED_LOG("  : Import async TimelineResource for buffer {}",
                       static_cast<void *>(user_buffer));
    return found_it->second.timeline_resource;
  } else {
    SHORTFIN_TRACE_SCOPE_NAMED("SyncImportTimelineResource");
    SHORTFIN_SCHED_LOG("  : Import default/sync TimelineResource for buffer {}",
                       static_cast<void *>(user_buffer));
    return inv_->fiber()->NewTimelineResource();
  }
}

// -------------------------------------------------------------------------- //
// ProgramIsolate
// -------------------------------------------------------------------------- //

std::pair<iree::vm_context_ptr, detail::ProgramIsolate *>
detail::ProgramIsolate::AcquireIsolate(Fiber &fiber,
                                       iree::vm_context_ptr root_context,
                                       ProgramIsolation isolation) {
  assert(isolation != ProgramIsolation::NONE &&
         "cannot AcquireIsolate when isolation == NONE");
  // Some isolation required.
  detail::ProgramIsolate *isolate = nullptr;
  {
    iree::slim_mutex_lock_guard lock(fiber.program_isolate_mu_);
    auto found_it = fiber.program_isolates_.find(root_context.get());
    if (found_it != fiber.program_isolates_.end()) {
      isolate = found_it->second.get();
    }
    if (isolate && !isolate->fork_contexts.empty()) {
      // Fast path: there is an existing isolate and a context avaialable.
      auto isolated_context = std::move(isolate->fork_contexts.back());
      isolate->fork_contexts.pop_back();
      return std::make_pair(std::move(isolated_context), isolate);
    } else if (!isolate) {
      // Initialize a new isolate accounting struct while in the lock.
      // Note that this can cause a fault for PER_FIBER mode if the call
      // to fork fails below as it will leave the isolate with no available
      // context and every future call will raise an exception indicating that
      // the context is busy (vs trying to create a new one). This is deemed
      // an acceptable situation for a system fault (which is the only reason
      // a fork will fail).
      auto [inserted_it, inserted] =
          fiber.program_isolates_.insert(std::make_pair(
              root_context.get(),
              std::make_unique<detail::ProgramIsolate>(root_context)));
      isolate = inserted_it->second.get();
    } else if (isolation == ProgramIsolation::PER_FIBER) {
      throw std::logic_error(
          "Cannot make concurrent invocations of a PER_FIBER program from "
          "the same Fiber. This typically means that two invocations were "
          "attempted on the same program on the same fiber without an "
          "await. Consider fixing adding appropriate sequencing or switching "
          "to either PER_CALL or NONE isolation if appropriate for the use "
          "case. This exception can also occur if the first invocation to this "
          "Program failed, leaving no initialized Program for this fiber.");
    }
  }

  // Slow-path: fork needed (and possibly new isolate registration needed).
  iree::vm_context_ptr new_context;
  SHORTFIN_THROW_IF_ERROR(iree_vm_context_fork(
      root_context.get(), fiber.host_allocator(), new_context.for_output()));
  return std::make_pair(std::move(new_context), isolate);
}

void detail::ProgramIsolate::ReleaseIsolate(Fiber &fiber,
                                            iree::vm_context_ptr context,
                                            detail::ProgramIsolate *isolate) {
  assert(isolate && "attempt to release null isolate");
  {
    iree::slim_mutex_lock_guard lock(fiber.program_isolate_mu_);
    isolate->fork_contexts.push_back(std::move(context));
  }
}

}  // namespace shortfin::local
