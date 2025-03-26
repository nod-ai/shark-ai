// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./lib_ext.h"

#include "./utils.h"
#include "shortfin/array/array.h"
#include "shortfin/array/storage.h"
#include "shortfin/local/async.h"
#include "shortfin/local/fiber.h"
#include "shortfin/local/messaging.h"
#include "shortfin/local/process.h"
#include "shortfin/local/program.h"
#include "shortfin/local/system.h"
#if defined(SHORTFIN_HAVE_AMDGPU)
#include "shortfin/local/systems/amdgpu.h"
#endif  // SHORTFIN_HAVE_AMDGPU
#include "shortfin/local/systems/host.h"
#include "shortfin/support/globals.h"
#include "shortfin/support/logging.h"

namespace shortfin::python {

namespace {

static const char DOCSTRING_SYSTEM_CTOR[] =
    R"(Constructs a System based on system type and kwargs.

System types depend on how the library was compiled and correspond to
SystemBuilder classes. This API is a shorthand for creating a SystemBuilder
and calling create_system() on it.
)";

static const char DOCSTRING_SYSTEM_BUILDER_CTOR[] =
    R"(Constructs a system with environment dependent configuration.

Most configuration is done by way of key/value arguments. Arguments are meant
to be derived from flags or config files and are expected to be simple strings
or integer values:

  * "system_type": A supported system type, corresponding to a subclass of
    SystemBuilder. Typically available values include "hostcpu", "amdgpu", etc.
  * Other keywords are passed to the concrete SystemBuilder subclass.

Resolution for the `system_type` is special and is checked in three steps, with
the first defined winning:

1. `kwargs["system_type"]`
2. Environment `SHORTFIN_SYSTEM_TYPE` variable (or as controlled by `env_prefix`)
3. `SystemBuilder.default_system_type` as a process-wide default.

Args:
  env_prefix: Controls how options are looked up in the environment. By default,
    the prefix is "SHORTFIN_" and upper-cased options are appended to it. Any
    option not explicitly specified but in the environment will be used. Pass
    None to disable environment lookup.
  **kwargs: Key/value arguments for controlling setup of the system.
)";

static const char DOCSTRING_HOSTCPU_SYSTEM_BUILDER_CTOR[] =
    R"(Constructs a system with CPU based devices.

Most configuration is done by way of key/value arguments. Arguments are meant
to be derived from flags or config files and are expected to be simple strings
or integer values:

  * "hostcpu_topology_nodes": Takes one of the special values "current" (default)
    or "all". If not one of those, this should be a comma-delimited list of
    NUMA node ids. Each NUMA node will be modeled as one device queue and will
    show up on the system as a device.
  * "hostcpu_topology_max_group_count": Maximum number of groups to create per
    node. The actual number of groups is derived by a heuristic (which can be
    influenced by other options) such that there will not be more groups than
    eligible physical cores on the node.

Args:
  env_prefix: Controls how options are looked up in the environment. By default,
    the prefix is "SHORTFIN_" and upper-cased options are appended to it. Any
    option not explicitly specified but in the environment will be used. Pass
    None to disable environment lookup.
  **kwargs: Key/value arguments for controlling setup of the system.
)";

static const char DOCSTRING_HOSTCPU_SYSTEM_BUILDER_HOSTCPU_ALLOCATOR_SPECS[] =
    R"(Allocator specs to apply to HOSTCPU devices configured by this builder.

This uses syntax like::

  some_allocator
  some_allocator:key=value
  some_allocator:key=value,key=value
  some_allocator:key=value,key=value;other_allocator:key=value

Typical values for `some_allocator` include `caching` and `debug`.

This can be set via a keyword of `amdgpu_allocators`, which will only apply to
HOSTCPU devices or `allocators` which will apply to all contained devices.
Similarly, it is available on a `SHORTFIN_` prefixed env variable if environment
lookup is not disabled.
)";

static const char DOCSTRING_PROGRAM_FUNCTION_INVOCATION[] =
    R"(Creates an invocation object targeting the function.

This is a low-level interface for performing an invocation, and it should be
used when precise, non-default control is needed.
)";

class Refs {
 public:
  py::object asyncio_create_task =
      py::module_::import_("asyncio").attr("create_task");
  py::object asyncio_set_event_loop =
      py::module_::import_("asyncio").attr("set_event_loop");
  py::object asyncio_get_running_loop =
      py::module_::import_("asyncio.events").attr("get_running_loop");
  py::object asyncio_set_running_loop =
      py::module_::import_("asyncio.events").attr("_set_running_loop");
  py::object threading_Thread =
      py::module_::import_("threading").attr("Thread");
  py::object threading_current_thread =
      py::module_::import_("threading").attr("current_thread");
  py::object threading_main_thread =
      py::module_::import_("threading").attr("main_thread");

  py::handle lazy_PyWorkerEventLoop() {
    if (!lazy_PyWorkerEventLoop_.is_valid()) {
      lazy_PyWorkerEventLoop_ = py::module_::import_("_shortfin.asyncio_bridge")
                                    .attr("PyWorkerEventLoop");
    }
    return lazy_PyWorkerEventLoop_;
  }

  std::optional<std::string> default_system_type() {
    iree::slim_mutex_lock_guard g(mu_);
    return default_system_type_;
  }
  void set_default_system_type(std::optional<std::string> st) {
    iree::slim_mutex_lock_guard g(mu_);
    default_system_type_ = st;
  }

 private:
  iree::slim_mutex mu_;
  py::object lazy_PyWorkerEventLoop_;
  std::optional<std::string> default_system_type_;
};

// We need a fair bit of accounting additions in order to make Workers usable
// as asyncio loops. This extension holds that.
class PyWorkerExtension : public local::Worker::Extension {
 public:
  PyWorkerExtension(local::Worker &worker, PyInterpreterState *interp,
                    std::shared_ptr<Refs> refs)
      : Extension(worker), interp_(interp), refs_(std::move(refs)) {
    py::object worker_obj = py::cast(worker, py::rv_policy::reference);
    loop_ = refs_->lazy_PyWorkerEventLoop()(worker_obj);
  }

  static PyWorkerExtension &GetCurrent() {
    PyWorkerExtension *ext =
        local::Worker::GetCurrentExtension<PyWorkerExtension>();
    if (!ext) {
      throw std::logic_error(
          "There is no shortfin worker associated with this thread.");
    }
    return *ext;
  }

  py::handle loop() { return loop_; }

  void OnThreadStart() noexcept override {
    SHORTFIN_TRACE_SCOPE_NAMED("PyWorker::OnThreadStart");
    // Python threading initialization.
    // If our own thread, teach Python about it. Not done for donated.
    if (worker().options().owned_thread) {
      PyThreadState_New(interp_);
    }

    py::gil_scoped_acquire g;
    // Aside from set_event_loop being old and _set_running_loop being new
    // it isn't clear to me that either can be left off.
    refs_->asyncio_set_event_loop(loop_);
    refs_->asyncio_set_running_loop(loop_);
  }

  void OnThreadStop() noexcept override {
    SHORTFIN_TRACE_SCOPE_NAMED("PyWorker::OnThreadStop");
    {
      // Do Python level thread cleanup.
      py::gil_scoped_acquire g;
      loop_.reset();

      // reset the event loop.
      refs_->asyncio_set_event_loop(py::none());
      refs_->asyncio_set_running_loop(py::none());
    }

    // And destroy our thread state (if not donated).
    if (worker().options().owned_thread) {
      // Ordinarily PyGILState_Ensure must be balanced with PyGILState_Release,
      // by PyThreadState_DeleteCurrent() implicitly releases it as part of
      // its cleanup process.
      PyGILState_STATE gil_state = PyGILState_Ensure();
      PyThreadState_Clear(PyThreadState_Get());
      PyThreadState_DeleteCurrent();
    }
  }

  // Because shutdown happens on the main thread under the purview of the
  // GIL, we must make arrangements to release it. Otherwise shutdown can
  // deadlock with processing.
  void OnBeforeShutdownWait() noexcept override {
    shutdown_wait_gil_state = PyEval_SaveThread();
  }
  void OnAfterShutdownWait() noexcept override {
    PyEval_RestoreThread(shutdown_wait_gil_state);
    shutdown_wait_gil_state = nullptr;
  }

 private:
  py::object loop_;
  PyInterpreterState *interp_ = nullptr;
  std::shared_ptr<Refs> refs_;

  PyThreadState *shutdown_wait_gil_state = nullptr;
};

class PyProcess : public local::detail::BaseProcess {
 public:
  PyProcess(std::shared_ptr<Refs> refs)
      : BaseProcess(), refs_(std::move(refs)) {}
  using BaseProcess::Initialize;
  using BaseProcess::is_initialized;
  using BaseProcess::Launch;

  void AssertInitialized() {
    if (!is_initialized()) {
      throw std::logic_error("Process.__init__ not called in constructor");
    }
  }

  void ScheduleOnWorker() override {
    // This is tricky: We need to retain the object reference across the
    // thread transition, but on the receiving side, the GIL will not be
    // held initially, so we must avoid any refcount maintenance until it
    // is acquired. Therefore, we manually borrow a reference and steal it in
    // the callback.
    py::handle self_object = py::cast(this, py::rv_policy::none);
    self_object.inc_ref();
    fiber()->worker().CallThreadsafe(
        std::bind(&PyProcess::RunOnWorker, self_object));
  }
  static void RunOnWorker(py::handle self_handle) {
    SHORTFIN_TRACE_SCOPE_NAMED("PyProcess:RunOnWorker");
    py::gil_scoped_acquire g;
    // Steal the reference back from ScheduleOnWorker. Important: this is
    // very likely the last reference to the process. So self must not be
    // touched after self_object goes out of fiber.
    py::object self_object = py::steal(self_handle);
    PyProcess *self = py::cast<PyProcess *>(self_handle);
    // We assume that the run method either returns None (def) or a coroutine
    // (async def).
    auto coro = self_object.attr("run")();
    if (!coro.is_none()) {
      auto task = self->refs_->asyncio_create_task(coro);
      // Capture the self object to avoid lifetime hazzard with PyProcess
      // going away before done.
      task.attr("add_done_callback")(
          py::cpp_function([self_object](py::handle future) {
            PyProcess *done_self = py::cast<PyProcess *>(self_object);
            done_self->Terminate();
            // The result of the process future doesn't matter to us, but it
            // may be carrying an exception and this is our only chance to
            // bubble it. If it is, this will throw and be handled by the
            // last chance exception handler in the worker.
            // TODO: Route process termination and exceptions to a supervisor.
            future.attr("result")();
          }));
    } else {
      // Synchronous termination.
      self->Terminate();
    }
  }

  std::shared_ptr<Refs> refs_;
};

void PyAddProgramInvocationArg(py::capsule &inv_capsule, py::handle arg) {
  // See if the object implements our marshaling protocol. If it does, then
  // We invoke the marshaling method with the Invocation wrapped as a capsule
  // and the ProgramResourceBarrier.
  py::object marshaler = py::getattr(arg, "__sfinv_marshal__", py::none());
  if (!marshaler.is_none()) {
    marshaler(inv_capsule,
              static_cast<int>(local::ProgramResourceBarrier::DEFAULT));
    return;
  }

  throw std::invalid_argument(
      fmt::format("Unsupported argument type {} in call to ProgramFunction",
                  py::cast<std::string>(py::repr(arg.type()))));
}

local::ProgramInvocation::Future PyFunctionCall(
    local::ProgramFunction &self, py::args args, local::Fiber &fiber,
    std::optional<local::ProgramIsolation> isolation) {
  auto inv = self.CreateInvocation(fiber.shared_from_this(), isolation);
  py::capsule inv_capsule(inv.get());
  for (py::handle arg : args) {
    PyAddProgramInvocationArg(inv_capsule, arg);
  }
  return local::ProgramInvocation::Invoke(std::move(inv));
}

// Wraps a ProgramInvocation::Ptr representing a completed (awaited) invocation.
// Holds some additional accounting for marshaling results back to Python.
class PyProgramInvocation {
 public:
  PyProgramInvocation(local::ProgramInvocation::Ptr inv)
      : inv_(std::move(inv)) {}
  PyProgramInvocation(const PyProgramInvocation &) = delete;
  PyProgramInvocation(PyProgramInvocation &&other)
      : inv_(std::move(other.inv_)),
        cached_results_(std::move(other.cached_results_)),
        results_failure_(other.results_failure_) {}

  // Fields that can be bound.
  bool assume_no_alias = true;
  static std::optional<bool> global_assume_no_alias;

  void CheckValid() {
    if (!inv_) throw std::invalid_argument("Deallocated invocation");
  }
  local::ProgramInvocation::Ptr &inv() { return inv_; }

  py::object results() {
    if (results_failure_) {
      throw std::logic_error("Prior attempt to marshal IREE results failed");
    }
    if (cached_results_) {
      return cached_results_;
    }

    // Cache results.
    CheckValid();
    results_failure_ = true;

    local::CoarseInvocationTimelineImporter::Options options;
    options.assume_no_alias = assume_no_alias;
    if (global_assume_no_alias) {
      options.assume_no_alias = *global_assume_no_alias;
    }
    local::CoarseInvocationTimelineImporter timeline_importer(inv().get(),
                                                              options);
    size_t size = inv_->results_size();
    py::object tp = py::steal(PyTuple_New(size));
    for (size_t i = 0; i < size; ++i) {
      iree::vm_opaque_ref ref = inv_->result_ref(i);
      if (!ref) {
        throw new std::logic_error("Program returned unsupported Python type");
      }
      py::object item = RehydrateRef(std::move(ref), &timeline_importer);
      PyTuple_SET_ITEM(tp.ptr(), i, item.release().ptr());
    }

    cached_results_ = std::move(tp);
    results_failure_ = false;
    return cached_results_;
  }

 private:
  py::object RehydrateRef(
      iree::vm_opaque_ref ref,
      local::CoarseInvocationTimelineImporter *timeline_importer) {
    auto type = ref.get()->type;
    // Note that these accessors are dangerous as they assert/abort if
    // process-wide registration is not done properly. We assume here that
    // since we got a ref out that the basics are set up soundly, but if
    // actually doing this on user/dynamic types, we would want to be more
    // defensive.
    // TODO: Don't just do a linear scan if we have more than a couple.
    // TODO: Find a reliable way to statically cache the type id.
    if (local::ProgramInvocationMarshalableFactory::invocation_marshalable_type<
            array::device_array>() == type) {
      // device_array
      return py::cast(local::ProgramInvocationMarshalableFactory::
                          CreateFromInvocationResultRef<array::device_array>(
                              inv().get(), timeline_importer, std::move(ref)));
    } else if (local::ProgramInvocationMarshalableFactory::
                   invocation_marshalable_type<array::storage>() == type) {
      // storage
      return py::cast(local::ProgramInvocationMarshalableFactory::
                          CreateFromInvocationResultRef<array::storage>(
                              inv().get(), timeline_importer, std::move(ref)));
    }
    throw std::invalid_argument(
        fmt::format("Cannot marshal ref type {} to Python",
                    to_string_view(iree_vm_ref_type_name(type))));
  }

  local::ProgramInvocation::Ptr inv_;
  py::object cached_results_;
  bool results_failure_ = false;
};
std::optional<bool> PyProgramInvocation::global_assume_no_alias;

py::object RunInForeground(std::shared_ptr<Refs> refs, local::System &self,
                           py::object coro) {
  SHORTFIN_TRACE_SCOPE_NAMED("CoroRunInForeground");
  bool is_main_thread =
      refs->threading_current_thread().is(refs->threading_main_thread());

  local::Worker &worker = self.init_worker();
  py::object result = py::none();
  py::object py_exception = py::none();
  auto done_callback = [&](py::handle future) {
    worker.Kill();
    py_exception = future.attr("exception")();
    if (py_exception.is_none()) {
      result = future.attr("result")();
    }
  };
  worker.CallThreadsafe([&]() {
    // Run within the worker we are about to donate to.
    py::gil_scoped_acquire g;
    auto task = refs->asyncio_create_task(coro);
    task.attr("add_done_callback")(py::cpp_function(done_callback));
  });

  auto run = py::cpp_function([&]() {
    // Release GIL and run until the worker exits.
    {
      py::gil_scoped_release g;
      worker.RunOnCurrentThread();
    }
  });

  // If running on the main thread, we spawn a background thread and join
  // it because that shields it from receiving spurious KeyboardInterrupt
  // exceptions at inopportune points.
  if (is_main_thread) {
    auto thread = refs->threading_Thread(/*group=*/py::none(), /*target=*/run);
    thread.attr("start")();
    try {
      thread.attr("join")();
    } catch (...) {
      logging::warn("Exception caught in run(). Shutting down.");
      // Leak warnings are hopeless in exceptional termination.
      py::set_leak_warnings(false);
      // Give it a go waiting for the worker thread to exit.
      worker.Kill();
      thread.attr("join")();
      self.Shutdown();
      throw;
    }
  } else {
    run();
  }

  self.Shutdown();

  if (!py_exception.is_none()) {
    // We got this exception from a future/user code, which could have done
    // something nefarious. So type check it.
    if (PyObject_IsInstance(py_exception.ptr(), PyExc_Exception)) {
      PyErr_SetObject(py_exception.type().ptr(), py_exception.ptr());
    } else {
      PyErr_SetObject(PyExc_RuntimeError, py_exception.ptr());
    }
    throw py::python_error();
  }
  return result;
}

ConfigOptions CreateConfigOptions(std::optional<std::string> &env_prefix,
                                  py::kwargs &kwargs, bool validate_undef) {
  ConfigOptions options(std::move(env_prefix),
                        validate_undef
                            ? ConfigOptions::ValidationLevel::UNDEF_ERROR
                            : ConfigOptions::ValidationLevel::UNDEF_WARN);
  for (auto it = kwargs.begin(); it != kwargs.end(); ++it) {
    std::string key = py::cast<std::string>((*it).first);
    std::string value = py::cast<std::string>(py::str((*it).second));
    options.SetOption(std::move(key), std::move(value));
  }
  return options;
}

}  // namespace

NB_MODULE(lib, m) {
  // Tragically, debug builds of Python do the right thing and don't immortalize
  // many identifiers and such. This makes the last chance leak checking that
  // nanobind does somewhat unreliable since the reports it prints may be
  // to identifiers that are no longer live (at a time in process shutdown
  // where it is expected that everything left just gets dropped on the floor).
  // This causes segfaults or ASAN violations in the leak checker on exit in
  // certain scenarios where we have spurious "leaks" of global objects.

  py::set_leak_warnings(false);

  logging::InitializeFromEnv();

  py::register_exception_translator(
      [](const std::exception_ptr &p, void * /*unused*/) {
        try {
          std::rethrow_exception(p);
        } catch (shortfin::iree::error &e) {
          PyObject *exc_type;
          switch (e.code()) {
            case IREE_STATUS_INVALID_ARGUMENT:
            case IREE_STATUS_OUT_OF_RANGE:
              exc_type = PyExc_ValueError;
              break;
            case IREE_STATUS_FAILED_PRECONDITION:
              exc_type = PyExc_AssertionError;
              break;
            case IREE_STATUS_UNIMPLEMENTED:
              exc_type = PyExc_NotImplementedError;
              break;
            default:
              exc_type = PyExc_RuntimeError;
          }
          PyErr_SetString(PyExc_ValueError, e.what());
        }
      });

  py::class_<iree::vm_opaque_ref>(m, "_OpaqueVmRef");

  // Logging entrypoints.
  m.def("log_debug", [](std::string_view sv) { logging::debug("{}", sv); });
  m.def("log_info", [](std::string_view sv) { logging::info("{}", sv); });
  m.def("log_warn", [](std::string_view sv) { logging::warn("{}", sv); });
  m.def("log_error", [](std::string_view sv) { logging::error("{}", sv); });

  auto local_m = m.def_submodule("local");
  BindLocal(local_m);
  BindHostSystem(local_m);
#if defined(SHORTFIN_HAVE_AMDGPU)
  BindAMDGPUSystem(local_m);
#endif  // SHORTFIN_HAVE_AMDGPU

  auto array_m = m.def_submodule("array");
  BindArray(array_m);
}

void BindLocal(py::module_ &m) {
  // Keep weak refs to key objects that need explicit atexit shutdown.
  auto weakref = py::module_::import_("weakref");
  py::object live_system_refs = weakref.attr("WeakSet")();
  auto atexit = py::module_::import_("atexit");
  // Manually shutdown all System instances atexit if still alive (it is
  // not reliable to shutdown during interpreter finalization).
  atexit.attr("register")(py::cpp_function([](py::handle live_system_refs) {
                            for (auto it = live_system_refs.begin();
                                 it != live_system_refs.end(); ++it) {
                              (*it).attr("shutdown")();
                            }
                          }),
                          live_system_refs);
  auto refs = std::make_shared<Refs>();
  auto worker_initializer =
      [refs, interp_state = PyInterpreterState_Get()](local::Worker &worker) {
        worker.SetExtension(
            std::make_unique<PyWorkerExtension>(worker, interp_state, refs));
      };

  py::enum_<local::ProgramIsolation>(m, "ProgramIsolation")
      .value("NONE", local::ProgramIsolation::NONE)
      .value("PER_FIBER", local::ProgramIsolation::PER_FIBER)
      .value("PER_CALL", local::ProgramIsolation::PER_CALL)
      .export_values();

  py::class_<local::SystemBuilder>(m, "SystemBuilder")
      .def("__init__", [](py::args, py::kwargs) {})
      .def_static(
          "__new__",
          [refs](py::handle cls, std::optional<std::string> env_prefix,
                 bool validate_undef, py::kwargs kwargs) {
            auto options =
                CreateConfigOptions(env_prefix, kwargs, validate_undef);
            std::optional<std::string_view> system_type =
                options.GetOption("system_type");
            std::optional<std::string> default_system_type;
            if (!system_type) {
              default_system_type = refs->default_system_type();
              if (!default_system_type) {
                throw std::invalid_argument(
                    "In order to construct a generic SystemBuilder, a "
                    "`system_type=` keyword (or appropriate environment "
                    "variable) must be specified. Alternatively, a default can "
                    "be set process wide via "
                    "`SystemBuilder.default_system_type =`");
              }
              system_type = *default_system_type;
            }
            return local::SystemBuilder::ForSystem(
                iree_allocator_system(), *system_type, std::move(options));
          },
          // Note that for some reason, no-arg construction passes no arguments
          // to __new__. We allow the single positional argument to be none,
          // which satisfies this case in practice.
          py::arg("cls") = py::none(), py::kw_only(),
          py::arg("env_prefix").none() = "SHORTFIN_",
          py::arg("validate_undef") = true, py::arg("kwargs"),
          DOCSTRING_SYSTEM_BUILDER_CTOR)
      .def_prop_rw_static(
          "default_system_type",
          [refs](py::handle /*unused*/) { return refs->default_system_type(); },
          [refs](py::handle /*unused*/, std::optional<std::string> st) {
            refs->set_default_system_type(std::move(st));
          })
      .def("create_system", [live_system_refs,
                             worker_initializer](local::SystemBuilder &self) {
        auto system_ptr = self.CreateSystem();
        system_ptr->AddWorkerInitializer(worker_initializer);
        auto system_obj = py::cast(system_ptr, py::rv_policy::take_ownership);
        live_system_refs.attr("add")(system_obj);
        try {
          self.config_options().ValidateUndef();
        } catch (...) {
          system_obj.attr("shutdown")();
          throw;
        }
        return system_obj;
      });
  py::class_<local::System>(m, "System", py::is_weak_referenceable())
      .def(
          "__init__",
          [live_system_refs](py::object self_obj, py::args, py::kwargs) {
            live_system_refs.attr("add")(self_obj);
          },
          DOCSTRING_SYSTEM_CTOR)
      .def_static(
          "__new__",
          [worker_initializer](py::handle py_type, std::string_view system_type,
                               std::optional<std::string> env_prefix,
                               bool validate_undef, py::kwargs kwargs) {
            auto options =
                CreateConfigOptions(env_prefix, kwargs, validate_undef);
            auto system = local::System::Create(
                iree_allocator_system(), system_type, std::move(options));
            system->AddWorkerInitializer(worker_initializer);
            return system;
          },
          py::arg("type"), py::arg("system_type"), py::kw_only(),
          py::arg("env_prefix") = "SHORTFIN_", py::arg("validate_undef") = true,
          py::arg("kwargs"))
      .def("__enter__", [](py::object self_obj) { return self_obj; })
      .def(
          "__exit__",
          [](local::System &self, py::handle exc_type, py::handle exc_value,
             py::handle exc_tb) { self.Shutdown(); },
          py::arg("exc_type").none(), py::arg("exc_value").none(),
          py::arg("exc_tb").none())
      .def("shutdown", &local::System::Shutdown)
      // Access devices by list, name, or lookup.
      .def_prop_ro("device_names",
                   [](local::System &self) {
                     py::list names;
                     for (auto &it : self.named_devices()) {
                       names.append(it.first);
                     }
                     return names;
                   })
      .def_prop_ro("devices", &local::System::devices,
                   py::rv_policy::reference_internal)
      .def(
          "device",
          [](local::System &self, std::string_view key) {
            local::Device *device = self.FindDeviceByName(key);
            if (!device) {
              throw std::invalid_argument(fmt::format("No device '{}'", key));
            }
            return device;
          },
          py::rv_policy::reference_internal)
      .def(
          "create_queue",
          [](local::System &self,
             std::optional<std::string> name) -> std::shared_ptr<local::Queue> {
            local::Queue::Options options;
            if (name) {
              options.name = std::move(*name);
            }
            return self.CreateQueue(std::move(options));
          },
          py::arg("name") = py::none(), py::rv_policy::reference_internal)
      .def("named_queue", &local::System::named_queue, py::arg("name"),
           py::rv_policy::reference_internal)
      .def(
          "create_fiber",
          [](local::System &self, local::Worker *worker,
             py::handle raw_devices) {
            // TODO: I couldn't really figure out how to directly accept an
            // optional kw-only arg without it just being a raw object/handle.
            // If the passed devices is none, then we create the fiber with
            // all devices in the system. Otherwise, with those explicitly
            // given.
            std::vector<local::Device *> devices;
            if (raw_devices.is_none()) {
              devices.assign(self.devices().begin(), self.devices().end());
            } else {
              devices = py::cast<std::vector<local::Device *>>(raw_devices);
            }

            // If no worker, default to the init worker.
            if (!worker) {
              worker = dynamic_cast<local::Worker *>(&self.init_worker());
            }

            return self.CreateFiber(*worker, devices);
          },
          py::rv_policy::reference_internal,
          py::arg("worker").none() = py::none(), py::kw_only(),
          py::arg("devices") = py::none())
      .def(
          "create_worker",
          [refs](local::System &self, std::string name) -> local::Worker & {
            local::Worker::Options options(self.host_allocator(),
                                           std::move(name));
            return self.CreateWorker(options);
          },
          py::arg("name"), py::rv_policy::reference_internal)
      .def_prop_ro("init_worker", &local::System::init_worker,
                   py::rv_policy::reference_internal)
      .def(
          "run",
          [refs](local::System &self, py::object coro) {
            return RunInForeground(refs, self, std::move(coro));
          },
          py::arg("coro"))
      // Methods not on System but on child objects, taking System as an arg.
      // Emitted here for convenience.
      .def("load_module", &local::ProgramModule::Load, py::arg("path"),
           py::arg("mmap") = false);

  // Support classes.
  py::class_<local::Node>(m, "Node")
      .def_prop_ro("node_num", &local::Node::node_num)
      .def("__repr__", [](local::Node &self) {
        return fmt::format("local::Node({})", self.node_num());
      });
  py::class_<local::Device>(m, "Device")
      .def_prop_ro("name", &local::Device::name)
      .def_prop_ro("node_affinity", &local::Device::node_affinity)
      .def(py::self == py::self)
      .def("__repr__", &local::Device::to_s);
  py::class_<local::DeviceAffinity>(m, "DeviceAffinity")
      .def(py::init<>())
      .def(py::init<local::Device *>())
      .def(py::self == py::self)
      .def("add", &local::DeviceAffinity::AddDevice)
      .def("__add__", &local::DeviceAffinity::AddDevice)
      .def("__repr__", &local::DeviceAffinity::to_s);

  py::class_<local::Program>(m, "Program")
      .def(
          py::new_([](std::span<const local::ProgramModule> modules,
                      std::vector<const local::Device *> devices,
                      bool trace_execution, local::ProgramIsolation isolation) {
            local::Program::Options options;
            options.devices = devices;
            options.trace_execution = trace_execution;
            options.isolation = isolation;
            return local::Program::Load(modules, std::move(options));
          }),
          py::arg("modules"), py::kw_only(), py::arg("devices"),
          py::arg("trace_execution") = false,
          py::arg("isolation") = local::ProgramIsolation::PER_FIBER)
      .def_prop_ro("exports", &local::Program::exports)
      .def_prop_ro("isolation", &local::Program::isolation)
      .def("lookup_function", &local::Program::LookupRequiredFunction)
      .def("__getitem__", &local::Program::LookupRequiredFunction);
  py::class_<local::ProgramFunction>(m, "ProgramFunction")
      .def_prop_ro("name", &local::ProgramFunction::name)
      .def_prop_ro("calling_convention",
                   &local::ProgramFunction::calling_convention)
      .def(
          "invocation",
          [](local::ProgramFunction &self, local::Fiber &fiber,
             std::optional<local::ProgramIsolation> isolation) {
            return self.CreateInvocation(fiber.shared_from_this(), isolation);
          },
          py::arg("fiber"), py::arg("isolation") = py::none(),
          DOCSTRING_PROGRAM_FUNCTION_INVOCATION)
      .def_prop_ro("isolation", &local::ProgramFunction::isolation)
      .def("__call__", PyFunctionCall, py::arg("args"), py::kw_only(),
           py::arg("fiber"), py::arg("isolation") = py::none())
      .def("__repr__", &local::ProgramFunction::to_s);
  py::class_<local::ProgramModule>(m, "ProgramModule")
      .def_prop_ro("exports", &local::ProgramModule::exports)
      .def("__repr__", &local::ProgramModule::to_s)
      .def_static("load", &local::ProgramModule::Load, py::arg("system"),
                  py::arg("path"), py::arg("mmap") = false)
      .def_static(
          "parameter_provider",
          [](local::System &system, py::args params) {
            std::vector<local::BaseProgramParameters *> c_params;
            c_params.reserve(params.size());
            for (py::handle h : params) {
              c_params.push_back(py::cast<local::BaseProgramParameters *>(h));
            }
            return local::ProgramModule::ParameterProvider(system, c_params);
          },
          py::arg("system"), py::arg("params"));
  py::class_<PyProgramInvocation>(m, "ProgramInvocation")
      .def_rw("assume_no_alias", &PyProgramInvocation::assume_no_alias,
              "Assumes that no results alias inputs or other buffers")
      .def_rw_static(
          "global_assume_no_alias",
          &PyProgramInvocation::global_assume_no_alias,
          "Globally changes the assume_no_alias flag for all invocations")
      .def("invoke",
           [](PyProgramInvocation &self) {
             self.CheckValid();
             return local::ProgramInvocation::Invoke(std::move(self.inv()));
           })
      .def("add_arg",
           [](PyProgramInvocation &self, py::handle arg) {
             self.CheckValid();
             py::capsule inv_capsule(&self.inv());
             PyAddProgramInvocationArg(inv_capsule, arg);
           })
      .def("__iter__",
           [](PyProgramInvocation &self) {
             return self.results().attr("__iter__")();
           })
      .def(
          "__len__",
          [](PyProgramInvocation &self) {
            self.CheckValid();
            return self.inv()->results_size();
          },
          "The number of results in this invocation")
      .def(
          "__getitem__",
          [](PyProgramInvocation &self, iree_host_size_t i) {
            self.CheckValid();
            return self.results().attr("__getitem__")(i);
          },
          "Gets the i'th result")
      .def("__repr__", [](PyProgramInvocation &self) {
        if (!self.inv()) return std::string("ProgramInvocation(INVALID)");
        return self.inv()->to_s();
      });

  py::class_<local::BaseProgramParameters>(m, "BaseProgramParameters");
  py::class_<local::StaticProgramParameters, local::BaseProgramParameters>(
      m, "StaticProgramParameters")
      .def(
          py::init<local::System &, std::string_view, iree_host_size_t>(),
          py::arg("system"), py::arg("parameter_scope"),
          py::arg("max_concurrent_operations") =
              IREE_IO_PARAMETER_INDEX_PROVIDER_DEFAULT_MAX_CONCURRENT_OPERATIONS)
      .def(
          "load",
          [](local::StaticProgramParameters &self,
             std::filesystem::path file_path, std::string_view format,
             bool readable, bool writable, bool mmap) {
            local::StaticProgramParameters::LoadOptions options;
            options.format = format;
            options.readable = readable;
            options.writable = writable;
            options.mmap = mmap;
            self.Load(file_path, options);
          },
          py::arg("file_path"), py::arg("format") = std::string_view(),
          py::arg("readable") = true, py::arg("writable") = false,
          py::arg("mmap") = false);

  struct DevicesSet {
    DevicesSet(py::object fiber_obj, std::optional<size_t> index = {})
        : fiber_obj(std::move(fiber_obj)), index(index) {}
    py::object KeepAlive(local::ScopedDevice device) {
      py::object device_obj = py::cast(device);
      py::detail::keep_alive(/*nurse=*/device_obj.ptr(),
                             /*patient=*/fiber_obj.ptr());
      return device_obj;
    }
    local::Fiber &fiber() { return py::cast<local::Fiber &>(fiber_obj); }
    py::object fiber_obj;
    std::optional<size_t> index;
  };
  py::class_<local::Fiber>(m, "Fiber")
      .def("__repr__", &local::Fiber::to_s)
      .def_prop_ro(
          "raw_devices",
          [](local::Fiber &self) {
            std::vector<local::Device *> devices;
            devices.reserve(self.raw_devices().size());
            for (auto it : self.raw_devices()) {
              devices.push_back(it.second);
            }
            return devices;
          },
          py::rv_policy::reference_internal)
      .def(
          "raw_device",
          [](local::Fiber &self, int index) { return self.raw_device(index); },
          py::rv_policy::reference_internal)
      .def(
          "raw_device",
          [](local::Fiber &self, std::string_view name) {
            return self.raw_device(name);
          },
          py::rv_policy::reference_internal)
      .def_prop_ro("devices",
                   [](py::object self) { return DevicesSet(std::move(self)); })
      .def_prop_ro("devices_dict",
                   [](py::handle self_obj) {
                     local::Fiber &self = py::cast<local::Fiber &>(self_obj);
                     py::dict d;
                     for (auto &it : self.raw_devices()) {
                       py::object scoped_device =
                           py::cast(self.device(it.second));
                       py::detail::keep_alive(/*nurse=*/scoped_device.ptr(),
                                              /*patient=*/self_obj.ptr());
                       d[py::cast(it.first)] = scoped_device;
                     }
                     return d;
                   })
      .def_prop_ro("device_names", &local::Fiber::device_names)
      .def(
          "device",
          [](local::Fiber &self, py::args args) {
            return CastDeviceAffinity(self, args);
          },
          py::rv_policy::reference_internal);

  py::class_<local::ScopedDevice>(m, "ScopedDevice")
      .def_prop_ro("fiber", &local::ScopedDevice::fiber,
                   py::rv_policy::reference)
      .def_prop_ro("affinity", &local::ScopedDevice::affinity,
                   py::rv_policy::reference_internal)
      .def_prop_ro("raw_device", &local::ScopedDevice::raw_device,
                   py::rv_policy::reference_internal)
      .def(py::self == py::self)
      .def("__await__",
           [](local::ScopedDevice &self) {
             py::object future = py::cast(self.OnSync(), py::rv_policy::move);
             return future.attr("__await__")();
           })
      .def("__repr__", &local::ScopedDevice::to_s);

  py::class_<DevicesSet>(m, "_ScopeDevicesSet")
      .def("__iter__",
           [](DevicesSet &self) { return DevicesSet(self.fiber_obj, 0); })
      .def("__next__",
           [](DevicesSet &self) {
             auto &fiber = self.fiber();
             if (!self.index || *self.index >= fiber.raw_devices().size()) {
               // Blurgh: Exception as flow control is not cheap. There is a
               // very obnoxious way to make this not be exception based but
               // this is a minority path.
               throw py::stop_iteration();
             }
             return self.KeepAlive(fiber.device((*self.index)++));
           })
      .def("__len__",
           [](DevicesSet &self) { return self.fiber().raw_devices().size(); })
      .def("__getitem__",
           [](DevicesSet &self, size_t index) {
             return self.KeepAlive(self.fiber().device(index));
           })
      .def("__getitem__",
           [](DevicesSet &self, std::string_view name) {
             return self.KeepAlive(self.fiber().device(name));
           })
      .def("__getattr__",
           [](DevicesSet &self, std::string_view name) -> py::object {
             return self.KeepAlive(self.fiber().device(name));
           });

  ;
  py::class_<local::Worker>(m, "Worker", py::is_weak_referenceable())
      .def_prop_ro("loop",
                   [](local::Worker &self) {
                     auto *ext = self.GetExtension<PyWorkerExtension>();
                     if (!ext) {
                       throw std::logic_error(
                           "Worker was not initialized for access from Python");
                     }
                     return ext->loop();
                   })
      .def("call_threadsafe", &local::Worker::CallThreadsafe)
      .def("call",
           [](local::Worker &self, py::handle callable) {
             callable.inc_ref();  // Stolen within the callback.
             auto thunk = +[](void *user_data, iree_loop_t loop,
                              iree_status_t status) noexcept -> iree_status_t {
               SHORTFIN_TRACE_SCOPE_NAMED("PyWorker::Callback");
               py::gil_scoped_acquire g;
               py::object user_callable =
                   py::steal(static_cast<PyObject *>(user_data));
               IREE_RETURN_IF_ERROR(status);
               try {
                 user_callable();
               } catch (std::exception &e) {
                 return iree::exception_to_status(e);
               }
               return iree_ok_status();
             };
             SHORTFIN_THROW_IF_ERROR(self.CallLowLevel(thunk, callable.ptr()));
           })
      .def("delay_call",
           [](local::Worker &self, iree_time_t deadline_ns,
              py::handle callable) {
             callable.inc_ref();  // Stolen within the callback.
             auto thunk = +[](void *user_data, iree_loop_t loop,
                              iree_status_t status) noexcept -> iree_status_t {
               SHORTFIN_TRACE_SCOPE_NAMED("PyWorker::DelayCallback");
               py::gil_scoped_acquire g;
               py::object user_callable =
                   py::steal(static_cast<PyObject *>(user_data));
               IREE_RETURN_IF_ERROR(status);
               try {
                 user_callable();
               } catch (std::exception &e) {
                 return iree_make_status(
                     IREE_STATUS_UNKNOWN,
                     "Python exception raised from async callback: %s",
                     e.what());
               }
               return iree_ok_status();
             };
             SHORTFIN_THROW_IF_ERROR(self.WaitUntilLowLevel(
                 iree_make_deadline(deadline_ns), thunk, callable.ptr()));
           })
      .def("_delay_to_deadline_ns",
           [](local::Worker &self, double delay_seconds) {
             return self.ConvertRelativeTimeoutToDeadlineNs(
                 static_cast<iree_duration_t>(delay_seconds * 1e9));
           })
      .def("_now", [](local::Worker &self) { return self.now(); })
      .def("__repr__", &local::Worker::to_s);

  py::class_<PyProcess>(m, "Process")
      .def(
          "__init__",
          [](py::handle self_obj, std::shared_ptr<local::Fiber> fiber) {
            PyProcess &self = py::cast<PyProcess &>(self_obj);
            self.Initialize(std::move(fiber));
          },
          py::kw_only(), py::arg("fiber"))
      .def_static(
          "__new__",
          [refs](py::handle py_type, py::args, py::kwargs) {
            return custom_new<PyProcess>(py_type, refs);
          },
          py::arg("type"), py::arg("args"), py::arg("kwargs"))
      .def_prop_ro("pid", &PyProcess::pid)
      .def_prop_ro("fiber",
                   [](PyProcess &self) -> std::shared_ptr<local::Fiber> {
                     self.AssertInitialized();
                     return self.fiber();
                   })
      .def_prop_ro("system",
                   [](PyProcess &self) {
                     self.AssertInitialized();
                     return self.fiber()->system().shared_ptr();
                   })
      .def("launch",
           [](py::object self_obj) {
             PyProcess &self = py::cast<PyProcess &>(self_obj);
             self.AssertInitialized();
             self.Launch();
             return self_obj;
           })
      .def("__await__",
           [](PyProcess &self) {
             self.AssertInitialized();
             py::object future =
                 py::cast(local::CompletionEvent(self.OnTermination()),
                          py::rv_policy::move);
             return future.attr("__await__")();
           })
      .def("__repr__", [](PyProcess &self) {
        if (!self.is_initialized()) {
          return std::string("Process(UNINITIALIZED)");
        }
        return self.to_s();
      });

  py::class_<local::CompletionEvent>(m, "CompletionEvent")
      .def(py::init<>())
      .def("__await__", [](py::handle self_obj) {
        SHORTFIN_TRACE_SCOPE_NAMED("PyCompletionEvent::__await__");
        auto &worker_ext = PyWorkerExtension::GetCurrent();
        auto &self = py::cast<local::CompletionEvent &>(self_obj);
        py::object future = worker_ext.loop().attr("create_future")();
        // Stashing self as an attribute on the future, keeps us alive,
        // which transitively means that the wait source we get from self
        // stays alive. This can be done better later with a custom
        // Future.
        future.attr("_sf_event") = self_obj;
        py::object iter_ret = future.attr("__iter__")();

        // Pass the future object as void* user data to the C callback
        // interface. This works because we release the object pointer going
        // in and then steal it back to a py::object once the GIL has been
        // acquired (since dec_ref'ing it must happen within the GIL).
        // Because the self object was stashed on an attribute above, the
        // wait_source is valid for the entire sequence.
        SHORTFIN_THROW_IF_ERROR(worker_ext.worker().WaitOneLowLevel(
            /*wait_source=*/
            self, iree_infinite_timeout(),
            +[](void *future_vp, iree_loop_t loop,
                iree_status_t status) noexcept -> iree_status_t {
              SHORTFIN_TRACE_SCOPE_NAMED("PyCompletionEvent::OnComplete");
              py::gil_scoped_acquire g;
              py::object future = py::steal(static_cast<PyObject *>(future_vp));
              try {
                SHORTFIN_THROW_IF_ERROR(status);
                future.attr("set_result")(py::none());
              } catch (std::exception &e) {
                auto RuntimeError = py::handle(PyExc_RuntimeError);
                future.attr("set_exception")(RuntimeError(e.what()));
              }
              return iree_ok_status();
            },
            static_cast<void *>(future.release().ptr())));
        return iter_ret;
      });

  // ------------------------------------------------------------------------ //
  // Messaging
  // ------------------------------------------------------------------------ //
  py::class_<local::Message>(
      m, "Message",
      // Message is special in that it supports vague ownership and can be
      // transferred to the Python side, sharing one reference count and
      // lifetime. This is done the first time a Message is seen by the Python
      // side (either by in-place construction in a Python object or by
      // taking ownership of an object originating on the C++ side). When this
      // happens, the owner struct is replaced and any C++ side reference counts
      // are turned into Python reference counts. Once transferred, only Python
      // reference counting is used, even if referenced from the C++ side.
      py::intrusive_ptr<local::Message>([](local::Message *self,
                                           PyObject *self_py) noexcept {
        local::detail::MessageLifetimeController owner(
            +[](local::detail::MessageLifetimeController::Request req,
                const local::Message &msg) {
              py::gil_scoped_acquire g;
              PyObject *msg_object = reinterpret_cast<PyObject *>(
                  local::detail::MessageLifetimeController::AccessOwnedRefData(
                      msg));
              if (req ==
                  local::detail::MessageLifetimeController::Request::RETAIN) {
                py::handle(msg_object).inc_ref();
              } else {
                py::handle(msg_object).dec_ref();
              }
            });
        intptr_t orig_ref_data =
            owner.TakeOwnership(*self, reinterpret_cast<intptr_t>(self_py));
        // Transfer any prior C++ references to the Python side (less 1
        // since we start with a live reference).
        for (int i = 0; i < orig_ref_data - 1; ++i) {
          py::handle(self_py).inc_ref();
        }
      }))
      .def(py::init<>());

  py::class_<local::Queue>(m, "Queue")
      .def("__repr__", &local::Queue::to_s)
      .def("close", &local::Queue::Close)
      .def("writer",
           [](local::Queue &self) {
             return custom_new_keep_alive<local::QueueWriter>(
                 py::type<local::QueueWriter>(),
                 /*keep_alive=*/self, /*queue=*/self);
           })
      .def("reader",
           [](local::Queue &self) {
             return custom_new_keep_alive<local::QueueReader>(
                 py::type<local::QueueReader>(),
                 /*keep_alive=*/self, /*queue=*/self);
           })
      .def_prop_ro("closed", &local::Queue::is_closed)
      .def("write_nodelay", [](local::Queue &self, local::Message &message) {
        self.WriteNoDelay(local::Message::Ref(message));
      });
  py::class_<local::QueueWriter>(m, "QueueWriter")
      .def("__call__",
           [](local::QueueWriter &self, local::Message &message) {
             return self.Write(local::Message::Ref(message));
           })
      .def("close", &local::QueueWriter::Close);
  py::class_<local::QueueReader>(m, "QueueReader")
      .def("__call__", [](local::QueueReader &self) { return self.Read(); });

  // ------------------------------------------------------------------------ //
  // Futures
  // ------------------------------------------------------------------------ //
  py::class_<local::Future>(m, "Future")
      // The generic await support relies on the result override from a
      // subclass. Here and for VoidFuture, it is always none.
      .def("result",
           [](local::Future &self) {
             self.ThrowFailure();
             return py::none();
           })
      .def("__await__", [](py::handle self_obj) {
        SHORTFIN_TRACE_SCOPE_NAMED("PyFuture::__await__");
        // TODO: We should make our C++ future able to be used directly
        // vs needing to bridge it like this.
        auto &worker_ext = PyWorkerExtension::GetCurrent();
        auto &self = py::cast<local::Future &>(self_obj);
        py::object future = worker_ext.loop().attr("create_future")();
        // Stashing self as an attribute on the future, keeps us alive,
        // which transitively means that the wait source we get from self
        // stays alive. This can be done better later with a custom
        // Future.
        future.attr("_sf_future") = self_obj;
        py::object iter_ret = future.attr("__iter__")();

        // Pass the future object as void* user data to the C callback
        // interface. This works because we release the object pointer going
        // in and then steal it back to a py::object once the GIL has been
        // acquired (since dec_ref'ing it must happen within the GIL).
        // Because the self object was stashed on an attribute above, the
        // wait_source is valid for the entire sequence.
        self.AddCallback(
            [py_future_vp = static_cast<void *>(future.release().ptr())](
                local::Future &sf_future) {
              SHORTFIN_TRACE_SCOPE_NAMED("PyFuture::OnComplete");
              py::gil_scoped_acquire g;
              py::object py_future =
                  py::steal(static_cast<PyObject *>(py_future_vp));
              try {
                // Consult the Python level "result" on our local::Future
                // binding as it will override properly to get the right
                // Python casted type, allowing us to just have this one
                // __await__ implementation for every typed future.
                py_future.attr("set_result")(
                    py_future.attr("_sf_future").attr("result")());
              } catch (std::exception &e) {
                auto RuntimeError = py::handle(PyExc_RuntimeError);
                py_future.attr("set_exception")(RuntimeError(e.what()));
              }
            });
        return iter_ret;
      });
  py::class_<local::VoidFuture, local::Future>(m, "VoidFuture")
      .def(py::init<>())
      .def("set_success", [](local::VoidFuture &self) { self.set_success(); });
  py::class_<local::ProgramInvocation::Future, local::Future>(
      m, "ProgramInvocationFuture")
      .def("result", [](local::ProgramInvocation::Future &self) {
        local::ProgramInvocation::Ptr &result = self.result();
        if (!result) return py::none();
        // Sharp edge: ProgramInvocationFutures are read-once since we move the
        // ProgramInvocation::Ptr out of the future here and transfer ownership
        // to a Python object. There isn't a better way to do this without
        // increasing overhead on this hot path or doing something more
        // expensive in the C++ API: essentially, ProgramInvocations flow
        // through the system precisely one way. As a low level facility, this
        // is deemed acceptable.
        return py::cast(PyProgramInvocation(std::move(result)));
      });
  py::class_<local::MessageFuture, local::Future>(m, "MessageFuture")
      .def("result", [](local::MessageFuture &self) {
        // Get a raw backing msg (without an increased refcount). When cast
        // to a py::object, it will get a new reference.
        local::Message::Ref &result = self.result();
        if (!result) return py::none();
        return py::cast(result.get());
      });
}

void BindHostSystem(py::module_ &global_m) {
  auto m = global_m.def_submodule("host", "Host device management");
  py::class_<local::systems::HostSystemBuilder, local::SystemBuilder>(
      m, "SystemBuilder");
  py::class_<local::systems::HostCPUSystemBuilder,
             local::systems::HostSystemBuilder>(m, "CPUSystemBuilder")
      .def("__init__", [](py::args, py::kwargs) {})
      .def_static(
          "__new__",
          [](py::handle cls, std::optional<std::string> env_prefix,
             bool validate_undef, py::kwargs kwargs) {
            auto options =
                CreateConfigOptions(env_prefix, kwargs, validate_undef);
            return std::make_unique<local::systems::HostCPUSystemBuilder>(
                iree_allocator_system(), std::move(options));
          },
          // Note that for some reason, no-arg construction passes no arguments
          // to __new__. We allow the single positional argument to be none,
          // which satisfies this case in practice.
          py::arg("cls") = py::none(), py::kw_only(),
          py::arg("env_prefix").none() = "SHORTFIN_",
          py::arg("validate_undef") = true, py::arg("kwargs"),
          DOCSTRING_HOSTCPU_SYSTEM_BUILDER_CTOR)
      .def_prop_rw(
          "hostcpu_allocator_specs",
          [](local::systems::HostCPUSystemBuilder &self) {
            return self.hostcpu_allocator_specs();
          },
          [](local::systems::HostCPUSystemBuilder &self,
             std::vector<std::string> specs) {
            self.hostcpu_allocator_specs() = std::move(specs);
          },
          DOCSTRING_HOSTCPU_SYSTEM_BUILDER_HOSTCPU_ALLOCATOR_SPECS);
  py::class_<local::systems::HostCPUDevice, local::Device>(m, "HostCPUDevice");
}

#if defined(SHORTFIN_HAVE_AMDGPU)

namespace {
static const char DOCSTRING_AMDGPU_SYSTEM_BUILDER_CTOR[] =
    R"(Constructs a system with AMDGPU based devices.

Most configuration is done by way of key/value arguments. See the properties
of this class, which document the option keywords that can be passed to this
constructor.

Args:
  env_prefix: Controls how options are looked up in the environment. By default,
    the prefix is "SHORTFIN_" and upper-cased options are appended to it. Any
    option not explicitly specified but in the environment will be used. Pass
    None to disable environment lookup.
  **kwargs: Key/value arguments for controlling setup of the system.
)";

static const char DOCSTRING_AMDGPU_SYSTEM_BUILDER_AMDGPU_ALLOCATOR_SPECS[] =
    R"(Allocator specs to apply to AMDGPU devices configured by this builder.

This uses syntax like::

  some_allocator
  some_allocator:key=value
  some_allocator:key=value,key=value
  some_allocator:key=value,key=value;other_allocator:key=value

Typical values for `some_allocator` include `caching` and `debug`.

This can be set via a keyword of `amdgpu_allocators`, which will only apply to
AMDGPU devices or `allocators` which will apply to all contained devices.
Similarly, it is available on a `SHORTFIN_` prefixed env variable if environment
lookup is not disabled.
)";

static const char DOCSTRING_AMDGPU_SYSTEM_BUILDER_AMDGPU_ASYNC_ALLOCATIONS[] =
    R"(Whether to use async allocations if supported (default true).)";

static const char DOCSTRING_AMDGPU_SYSTEM_BUILDER_CPU_DEVICES_ENABLED[] =
    R"(Whether to create a heterogenous system with hostcpu and amdgpu devices.

Defaults to false. If enabled, the resulting system will contain both device
types and it is up to application code to differentiate between them. All
options for the hostcpu system builder are applicable in this case.

This option can be set as an option keyword with the name
"amdgpu_cpu_devices_enabled" or the environment variable
"SHORTFIN_AMDGPU_CPU_DEVICES_ENABLED=true" (if `env_prefix` was not changed
at construction).
)";

static const char DOCSTRING_AMDGPU_SYSTEM_BUILDER_HIP_LIB_SEARCH_PATHS[] =
    R"(List of directories to search for libamdhip64.so (or amdhip64.dll).

If empty, then `dlopen` will be used without a path, meaning that the library
must be on the default search path or already loaded in the process (i.e.
if running within an overall framework).

Each entry should be a directory, but a full path to a file can be given by
prefixing with "file:".

This option can be set as an option keyword with the name
"amdgpu_hip_lib_search_path" or the environment variable
"SHORTFIN_AMDGPU_HIP_LIB_SEARCH_PATH" (if `env_prefix` was not changed at
construction). For compatibility with IREE tools, the "IREE_HIP_DYLIB_PATH"
environment variable is searched as a fallback in all cases. Multiple paths
can be separated by semicolons on all platforms.
)";

static const char
    DOCSTRING_AMDGPU_SYSTEM_BUILDER_LOGICAL_DEVICES_PER_PHYSICAL_DEVICE[] =
        R"(Number of logical devices to open per physical, visible device.

This option can be set as an option keyword with the name
"amdgpu_logical_devices_per_physical_device" or the environment variable
"SHORTFIN_AMDGPU_LOGICAL_DEVICES_PER_PHYSICAL_DEVICE" (if `env_prefix` was not
changed at construction).
)";

static const char DOCSTRING_AMDGPU_SYSTEM_BUILDER_TRACING_LEVEL[] =
    R"(Tracing level for AMDGPU device behavior.

Controls the verbosity of tracing when Tracy instrumentation is enabled.
The impact to benchmark timing becomes more severe as the verbosity
increases, and thus should be only enabled when needed.

This is the equivalent of the `--hip_tracing` IREE tools flag.
Permissible values are:
  * 0 : stream tracing disabled.
  * 1 : coarse command buffer level tracing enabled.
  * 2 : (default) fine-grained kernel level tracing enabled.

The setting only has an effect if using a tracing enabled runtime (i.e.
by running with `SHORTFIN_PY_RUNTIME=tracy` or equiv).

The default value for this setting is available as a
`amdgpu.SystemBuilder(amdgpu_tracing_level=2)` or (by default) from an
environment variable `SHORTFIN_AMDGPU_TRACING_LEVEL`.
)";

static const char DOCSTRING_AMDGPU_SYSTEM_BUILDER_AVAILABLE_DEVICES[] =
    R"(List of available device ids on the system.

Accessing this property triggers enumeration, so configuration needed to load
libraries and perform basic system setup must be set first.
)";

static const char DOCSTRING_AMDGPU_SYSTEM_BUILDER_VISIBLE_DEVICES[] =
    R"(Get or set the list of visible device ids.

If not set or None, then all available devices will be opened and added to
the system. See the property `available_devices` to access this list of ids.

If set, then each device with the given device id will be opened and added to
the system in the order listed. Note that in certain partitioned cases, multiple
devices may be available with the same device id. In this case, duplicates
in the visible devices list will cause instantiate a partition of the device
in enumeration order (so there can be as many duplicates as physical
partitions). This is an uncommon scenario and most users should not specify
duplicate device ids. Since there are several ways that partitioned devices
can be consumed, additional options will be available in the future for
controlling this behavior.

This property can be set as an option keyword with the name
"amdgpu_visible_devices" or the environment variable
"SHORTFIN_AMDGPU_VISIBLE_DEVICES" (if `env_prefix` was not changed at
construction). Multiples can be separated by a semicolon.
)";

}  // namespace

void BindAMDGPUSystem(py::module_ &global_m) {
  auto m = global_m.def_submodule("amdgpu", "AMDGPU system config");
  py::class_<local::systems::AMDGPUSystemBuilder,
             local::systems::HostCPUSystemBuilder>(m, "SystemBuilder")
      .def("__init__", [](py::args, py::kwargs) {})
      .def_static(
          "__new__",
          [](py::handle cls, std::optional<std::string> env_prefix,
             bool validate_undef, py::kwargs kwargs) {
            auto options =
                CreateConfigOptions(env_prefix, kwargs, validate_undef);
            return std::make_unique<local::systems::AMDGPUSystemBuilder>(
                iree_allocator_system(), std::move(options));
          },
          // Note that for some reason, no-arg construction passes no arguments
          // to __new__. We allow the single positional argument to be none,
          // which satisfies this case in practice.
          py::arg("cls") = py::none(), py::kw_only(),
          py::arg("env_prefix").none() = "SHORTFIN_",
          py::arg("validate_undef") = true, py::arg("kwargs"),
          DOCSTRING_AMDGPU_SYSTEM_BUILDER_CTOR)
      .def_prop_rw(
          "amdgpu_allocator_specs",
          [](local::systems::AMDGPUSystemBuilder &self) {
            return self.amdgpu_allocator_specs();
          },
          [](local::systems::AMDGPUSystemBuilder &self,
             std::vector<std::string> specs) {
            self.amdgpu_allocator_specs() = std::move(specs);
          },
          DOCSTRING_AMDGPU_SYSTEM_BUILDER_AMDGPU_ALLOCATOR_SPECS)
      .def_prop_ro(
          "available_devices",
          [](local::systems::AMDGPUSystemBuilder &self) {
            return self.GetAvailableDeviceIds();
          },
          DOCSTRING_AMDGPU_SYSTEM_BUILDER_AVAILABLE_DEVICES)
      .def_prop_rw(
          "async_allocations",
          [](local::systems::AMDGPUSystemBuilder &self) {
            return self.async_allocations();
          },
          [](local::systems::AMDGPUSystemBuilder &self, bool value) {
            self.async_allocations() = value;
          },
          DOCSTRING_AMDGPU_SYSTEM_BUILDER_AMDGPU_ASYNC_ALLOCATIONS)
      .def_prop_rw(
          "cpu_devices_enabled",
          [](local::systems::AMDGPUSystemBuilder &self) -> bool {
            return self.cpu_devices_enabled();
          },
          [](local::systems::AMDGPUSystemBuilder &self, bool en) {
            self.cpu_devices_enabled() = en;
          },
          DOCSTRING_AMDGPU_SYSTEM_BUILDER_CPU_DEVICES_ENABLED)
      .def_prop_rw(
          "hip_lib_search_paths",
          [](local::systems::AMDGPUSystemBuilder &self)
              -> std::vector<std::string> {
            return self.hip_lib_search_paths();
          },
          [](local::systems::AMDGPUSystemBuilder &self,
             std::vector<std::string> vs) { self.hip_lib_search_paths() = vs; },
          DOCSTRING_AMDGPU_SYSTEM_BUILDER_HIP_LIB_SEARCH_PATHS)
      .def_prop_rw(
          "tracing_level",
          [](local::systems::AMDGPUSystemBuilder &self) -> int {
            return self.tracing_level();
          },
          [](local::systems::AMDGPUSystemBuilder &self, int tracing_level) {
            self.tracing_level() = tracing_level;
          },
          DOCSTRING_AMDGPU_SYSTEM_BUILDER_TRACING_LEVEL)
      .def_prop_rw(
          "logical_devices_per_physical_device",
          [](local::systems::AMDGPUSystemBuilder &self) -> size_t {
            return self.logical_devices_per_physical_device();
          },
          [](local::systems::AMDGPUSystemBuilder &self, size_t value) {
            self.logical_devices_per_physical_device() = value;
          },
          DOCSTRING_AMDGPU_SYSTEM_BUILDER_LOGICAL_DEVICES_PER_PHYSICAL_DEVICE)
      .def_prop_rw(
          "visible_devices",
          [](local::systems::AMDGPUSystemBuilder &self)
              -> std::optional<std::vector<std::string>> {
            return self.visible_devices();
          },
          [](local::systems::AMDGPUSystemBuilder &self,
             std::optional<std::vector<std::string>> vs) {
            self.visible_devices() = std::move(vs);
          },
          DOCSTRING_AMDGPU_SYSTEM_BUILDER_VISIBLE_DEVICES);

  py::class_<local::systems::AMDGPUDevice, local::Device>(m, "AMDGPUDevice");
}
#endif  // SHORTFIN_HAVE_AMDGPU

}  // namespace shortfin::python
