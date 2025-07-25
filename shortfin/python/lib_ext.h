// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_BINDINGS_PYTHON_LIB_EXT_H
#define SHORTFIN_BINDINGS_PYTHON_LIB_EXT_H

#include <nanobind/nanobind.h>
#include <nanobind/operators.h>
#include <nanobind/stl/filesystem.h>
#include <nanobind/stl/function.h>
#include <nanobind/stl/optional.h>
#include <nanobind/stl/shared_ptr.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/string_view.h>
#include <nanobind/stl/unique_ptr.h>
#include <nanobind/stl/unordered_map.h>
#include <nanobind/stl/vector.h>

#include <span>

namespace shortfin::python {
namespace py = nanobind;

void BindArray(py::module_ &module);
void BindArrayHostOps(py::module_ &module);
void BindLocal(py::module_ &module);
void BindHostSystem(py::module_ &module);
void BindAMDGPUSystem(py::module_ &module);
void BindLLM(py::module_ &module);
// RAII wrapper for a Py_buffer which calls PyBuffer_Release when it goes
// out of scope.
class PyBufferReleaser {
 public:
  PyBufferReleaser(Py_buffer &b) : b_(b) {}
  ~PyBufferReleaser() { PyBuffer_Release(&b_); }

 private:
  Py_buffer &b_;
};

// Uses the low level object construction interface to do a custom, placement
// new based allocation:
// https://nanobind.readthedocs.io/en/latest/lowlevel.html#low-level-interface
// This is used in a __new__ method when you want to do something custom with
// the resulting py::object before returning it.
template <typename CppType, typename... Args>
inline py::object custom_new(py::handle py_type, Args &&...args) {
  py::object py_self = py::inst_alloc(py_type);
  CppType *self = py::inst_ptr<CppType>(py_self);
  new (self) CppType(std::forward<Args>(args)...);
  py::inst_mark_ready(py_self);
  return py_self;
}

// Extends custom_new to also keep a patient instance alive such that the
// keep_alive object is guaranteed to be kept alive for at least as long
// as the constructed object.
// This is used to perform a custom keep alive of some parent object vs
// just something accessible as arguments. The keep_alive instance must
// already be live python object (or a cast will fail).
template <typename CppType, typename KeepAlivePatient, typename... Args>
inline py::object custom_new_keep_alive(py::handle py_type,
                                        KeepAlivePatient &keep_alive,
                                        Args &&...args) {
  py::object self = custom_new<CppType>(py_type, std::forward<Args>(args)...);
  py::detail::keep_alive(
      self.ptr(),
      py::cast<KeepAlivePatient &>(keep_alive, py::rv_policy::none).ptr());
  return self;
}

}  // namespace shortfin::python

namespace nanobind::detail {

// Type-caster for a std::span<const T>.
// Code adapted from stl/detail/nb_list.h, stl/pair.h,
// Copyright Wenzel Jakob (BSD Licensed).
//
// Note that I couldn't figure out how to get the typing to work for one
// implementation of both mutable and const spans (since the intermediate
// backing vector needs to have a mutable type). If needing a mutable span
// caster, it would be easier to just duplicate this and remove the consts.
template <typename Type>
struct type_caster<std::span<const Type>> {
  static constexpr auto Name = io_name(NB_TYPING_SEQUENCE, NB_TYPING_LIST) +
                               const_name("[") + make_caster<Type>::Name +
                               const_name("]");
  using Value = std::vector<Type>;
  using Caster = make_caster<Type>;
  template <typename T>
  using Cast = std::span<const Type>;

  Value value;
  operator std::span<const Type>() { return {value}; }

  bool from_python(handle src, uint8_t flags, cleanup_list *cleanup) noexcept {
    size_t size;
    PyObject *temp;
    /* Will initialize 'size' and 'temp'. All return values and
       return parameters are zero/NULL in the case of a failure. */
    PyObject **o = seq_get(src.ptr(), &size, &temp);
    value.clear();
    value.reserve(size);
    Caster caster;
    bool success = o != nullptr;
    flags = flags_for_local_caster<Type>(flags);
    for (size_t i = 0; i < size; ++i) {
      if (!caster.from_python(o[i], flags, cleanup) ||
          !caster.template can_cast<Type>()) {
        success = false;
        break;
      }
      value.push_back(caster.operator cast_t<Type>());
    }
    Py_XDECREF(temp);
    return success;
  }

  static handle from_cpp(std::span<const Type> src, rv_policy policy,
                         cleanup_list *cleanup) {
    object ret = steal(PyList_New(src.size()));
    if (ret.is_valid()) {
      Py_ssize_t index = 0;
      for (auto &&value : src) {
        handle h =
            Caster::from_cpp(forward_like_<Type>(value), policy, cleanup);
        if (!h.is_valid()) {
          ret.reset();
          break;
        }
        NB_LIST_SET_ITEM(ret.ptr(), index++, h.ptr());
      }
    }
    return ret.release();
  }
};

}  // namespace nanobind::detail

#endif  // SHORTFIN_BINDINGS_PYTHON_LIB_EXT_H
