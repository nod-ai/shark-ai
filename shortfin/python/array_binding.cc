// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "./lib_ext.h"
#include "./utils.h"
#include "shortfin/array/api.h"
#include "shortfin/support/logging.h"

using namespace shortfin::array;

namespace shortfin::python {

namespace {
static const char DOCSTRING_ARRAY_COPY_FROM[] =
    R"(Copy contents from a source array to this array.

Equivalent to `dest_array.storage.copy_from(source_array.storage)`.
)";

static const char DOCSTRING_ARRAY_COPY_TO[] =
    R"(Copy contents this array to a destination array.

Equivalent to `dest_array.storage.copy_from(source_array.storage)`.
)";

static const char DOCSTRING_ARRAY_FILL[] = R"(Fill an array with a value.

Note that `fill` is asynchronous and may not be visible immediately. For immediate
manipulation of host visible arrays, assign to the `items` property or use the
`map(discard=True)` to get a mapping object which can be used to directly
update the contents.

Equivalent to `array.storage.fill(pattern)`.
)";

static const char DOCSTRING_ARRAY_ITEMS[] =
    R"(Convenience shorthand for map(...).items)";

static const char DOCSTRING_ARRAY_MAP[] =
    R"(Create a typed mapping of the buffer contents in host memory.

Support kwargs of:

| read: Enables read access to the mapped memory.
| write: Enables write access to the mapped memory and will flush upon close
  (for non-unified memory systems).
| discard: Indicates that the entire memory map should be treated as if it will
  be overwritten. Initial contents will be undefined. Implies `write=True`.

Mapping memory for access from the host requires a compatible buffer that has
been created with host visibility (which includes host buffers).

The returned mapping object is a context manager that will close/flush on
exit. Alternatively, the `close()` method can be invoked explicitly.

See also `storage.map()` which functions similarly but does not allow access
to dtype specific functionality.
)";

static const char DOCSTRING_ARRAY_VIEW[] =
    R"(Create a view of an array.

Either integer indices or slices can be passed to the view() method to create
an aliased device_array that shares a subset of the storage. Only view()
organizations that result in a row-major, dense array are currently supported.
)";

static const char DOCSTRING_MAPPING_FILL[] =
    R"(Fill the host mapping with a pattern.

The pattern can either be an object implementing the buffer protocol or a Python
int/float if the mapping has a dtype. In this case, the numeric value will be
converted to the appropriate typed pattern. Only dtypes supported by the
array.array class are supported in this fashion.

The pattern must evenly divide the mapping.

Note that like all methods on a mapping, any changes are immediately visible
(whereas the `fill` method on the array and storage are async operations).
)";

static const char DOCSTRING_MAPPING_ITEMS[] =
    R"(Access contents as a Python array.

When reading this attribute, an array.array will be constructed with the
contents of the mapping. This supports a subset of element types (byte aligned
integers, floats and doubles) corresponding to Python types.

On write, the mapping will be written with arbitrary Python types marshaled
via array.array into its contents.
)";

static const char DOCSTRING_STORAGE_COPY_FROM[] =
    R"(Copy contents from a source storage to this array.

This operation executes asynchronously and the effect will only be visible
once the execution fiber has been synced to the point of mutation.
)";

static const char DOCSTRING_STORAGE_FILL[] = R"(Fill a storage with a value.

Takes as argument any value that can be interpreted as a buffer with the Python
buffer protocol of size 1, 2, or 4 bytes. The storage will be filled uniformly
with the pattern.

This operation executes asynchronously and the effect will only be visible
once the execution fiber has been synced to the point of mutation.
)";

static const char DOCSTRING_STORAGE_MAP[] =
    R"(Create a mapping of the buffer contents in host memory.

Support kwargs of:

| read: Enables read access to the mapped memory.
| write: Enables write access to the mapped memory and will flush upon close
  (for non-unified memory systems).
| discard: Indicates that the entire memory map should be treated as if it will
  be overwritten. Initial contents will be undefined. Implies `write=True`.

Mapping memory for access from the host requires a compatible buffer that has
been created with host visibility (which includes host buffers).

The returned mapping object is a context manager that will close/flush on
exit. Alternatively, the `close()` method can be invoked explicitly.

See also `device_array.map()` which functions similarly but allows some
additional dtype specific accessors.
)";

device_array PyDeviceArrayView(device_array &array, py::args keys) {
  size_t rank = array.shape().size();
  Dims c_offsets(rank, 0);
  Dims c_sizes(array.shape_container());

  if (keys.size() > rank) {
    throw std::invalid_argument(
        "Cannot create view into device_array greater than its rank");
  }

  for (size_t idx = 0; py::handle key : keys) {
    if (py::isinstance<py::slice>(key)) {
      // Slice key.
      auto slice = py::cast<py::slice>(key);
      auto [start, stop, step, length] = slice.compute(c_sizes[idx]);
      if (step != 1) {
        throw std::logic_error("view does not support strided slices");
      }
      c_offsets[idx] = start;
      c_sizes[idx] = length;
    } else if (py::isinstance<iree_device_size_t>(key)) {
      // Integer key.
      c_offsets[idx] = py::cast<iree_device_size_t>(key);
      c_sizes[idx] = 1;
    } else {
      throw std::invalid_argument(
          "Args to view must either be integer indices or slices");
    }
    idx += 1;
  }

  return array.view(c_offsets, c_sizes);
}

class Refs {
 public:
  std::unordered_map<iree_hal_element_type_t, py::object>
      element_type_array_type_code_table =
          CreateElementTypeArrayTypeCodeTable();
  py::object array_array_ctor = py::module_::import_("array").attr("array");

 private:
  static std::unordered_map<iree_hal_element_type_t, py::object>
  CreateElementTypeArrayTypeCodeTable() {
    std::unordered_map<iree_hal_element_type_t, py::object> table;
    // This is really gross. Python's array type codes are pegged to C types,
    // which do not have portable sizes. We pick portablish things here and
    // carp on mismatch.
    auto add_type = [&](DType dt, const char *code, size_t size) {
      if (dt.dense_byte_count() != size) {
        throw std::invalid_argument(
            fmt::format("Illegal native type size for dtype {}, type code {}. "
                        "Native size mismatch: {} vs {}",
                        dt.name(), code, dt.dense_byte_count(), size));
      }
      table[dt] = py::str(code);
    };

    // See table at https://docs.python.org/3/library/array.html
    add_type(DType::int8(), "b", sizeof(char));
    add_type(DType::sint8(), "b", sizeof(char));
    add_type(DType::uint8(), "B", sizeof(unsigned char));
    add_type(DType::int16(), "h", sizeof(signed short));
    add_type(DType::sint16(), "h", sizeof(signed short));
    add_type(DType::uint16(), "H", sizeof(unsigned short));
    add_type(DType::int32(), "i", sizeof(signed int));
    add_type(DType::sint32(), "i", sizeof(signed int));
    add_type(DType::uint32(), "I", sizeof(unsigned int));
    add_type(DType::int64(), "q", sizeof(signed long long));
    add_type(DType::sint64(), "q", sizeof(signed long long));
    add_type(DType::uint64(), "Q", sizeof(unsigned long long));
    add_type(DType::bfloat16(), "H", sizeof(unsigned short));
    add_type(DType::float16(), "H", sizeof(unsigned short));
    add_type(DType::float32(), "f", sizeof(float));
    add_type(DType::float64(), "d", sizeof(double));
    return table;
  }
};

// Holder for a `mapping` class. Also holds additional metadata.
class PyMapping {
 public:
  PyMapping() = default;
  class mapping &mapping() { return mapping_; }
  std::optional<DType> dtype() { return dtype_; }
  void set_dtype(DType dtype) { dtype_ = dtype; }

  void AssertValid() {
    if (!mapping_) {
      throw std::logic_error("Mapping has been closed");
    }
  }

  void FillFromScalar(Refs *refs, py::handle value) {
    SHORTFIN_TRACE_SCOPE_NAMED("PyMapping::FillFromScalar");
    if (!dtype()) {
      throw std::invalid_argument(
          "The `fill` method is only valid for typed mappings but "
          "this does not have a dtype");
    }
    auto &table = refs->element_type_array_type_code_table;
    auto it = table.find(*dtype());
    if (it == table.end()) {
      throw std::invalid_argument(
          fmt::format("Python array.array type code not known for dtype "
                      "{}: Cannot access items",
                      dtype()->name()));
    }
    py::object pattern =
        refs->array_array_ctor(it->second, py::make_tuple(value));
    FillFromBuffer(pattern);
  }

  void FillFromBuffer(py::handle buffer) {
    SHORTFIN_TRACE_SCOPE_NAMED("PyMapping::FillFromBuffer");
    Py_buffer py_view;
    int flags = PyBUF_FORMAT | PyBUF_ND;  // C-Contiguous ND.
    if (PyObject_GetBuffer(buffer.ptr(), &py_view, flags) != 0) {
      throw py::python_error();
    }
    PyBufferReleaser py_view_releaser(py_view);
    if (mapping().size() % py_view.len != 0) {
      throw std::invalid_argument(
          fmt::format("Cannot fill mapping of size {} with pattern of "
                      "size {} (it does not evenly divide)",
                      mapping().size(), py_view.len));
    }

    // Specialize by fundamental sizes for fast-path implementations.
    if (py_view.len == 1) {
      std::memset(mapping().data(), *static_cast<char *>(py_view.buf),
                  mapping().size());
    } else if (py_view.len == 2) {
      uint16_t v = *static_cast<uint16_t *>(py_view.buf);
      uint16_t *begin =
          static_cast<uint16_t *>(static_cast<void *>((mapping().data())));
      std::fill(begin, begin + mapping().size() / sizeof(v), v);
    } else if (py_view.len == 4) {
      uint32_t v = *static_cast<uint32_t *>(py_view.buf);
      uint32_t *begin =
          static_cast<uint32_t *>(static_cast<void *>((mapping().data())));
      std::fill(begin, begin + mapping().size() / sizeof(v), v);
    } else if (py_view.len == 8) {
      uint64_t v = *static_cast<uint64_t *>(py_view.buf);
      uint64_t *begin =
          static_cast<uint64_t *>(static_cast<void *>((mapping().data())));
      std::fill(begin, begin + mapping().size() / sizeof(v), v);
    } else {
      // Slow path.
      uint8_t *begin = mapping().data();
      uint8_t *end = begin + mapping().size();
      while (begin < end) {
        std::memcpy(begin, py_view.buf, py_view.len);
        begin += py_view.len;
      }
    }
  }

  py::object GetItems(py::handle self_obj, Refs *refs) {
    SHORTFIN_TRACE_SCOPE_NAMED("PyMapping::GetItems");
    if (!dtype()) {
      throw std::invalid_argument(
          "The `items` property is only valid for typed mappings but "
          "this does not have a dtype");
    }
    AssertValid();
    auto &table = refs->element_type_array_type_code_table;
    auto it = table.find(*dtype());
    if (it == table.end()) {
      throw std::invalid_argument(
          fmt::format("Python array.array type code not known for dtype "
                      "{}: Cannot access items",
                      dtype()->name()));
    }
    py::object py_bytes = py::steal(PyBytes_FromObject(self_obj.ptr()));
    py::object items = refs->array_array_ctor(it->second, py_bytes);
    return items;
  }

  void SetItems(Refs *refs, py::handle initializer) {
    SHORTFIN_TRACE_SCOPE_NAMED("PyMapping::SetItems");
    if (!dtype()) {
      throw std::invalid_argument(
          "The `items` property is only valid for typed mappings but "
          "this does not have a dtype");
    }
    AssertValid();
    auto &table = refs->element_type_array_type_code_table;
    auto it = table.find(*dtype());
    if (it == table.end()) {
      throw std::invalid_argument(
          fmt::format("Python array.array type code not known for dtype "
                      "{}: Cannot access items",
                      dtype()->name()));
    }

    py::object items = refs->array_array_ctor(it->second, initializer);
    PyBufferRequest src_info(items, PyBUF_SIMPLE);
    if (src_info.view().len > mapping().size()) {
      throw std::invalid_argument(
          fmt::format("Cannot write {} bytes into buffer of {} bytes",
                      src_info.view().len, mapping().size()));
    }
    std::memcpy(mapping().data(), src_info.view().buf, src_info.view().len);
  }

 private:
  class mapping mapping_;
  std::optional<DType> dtype_;
};

// Does in-place creation of a mapping object and stores a pointer to the
// contained array::mapping C++ object.
py::object CreateMappingObject(PyMapping **out_cpp_mapping) {
  py::object py_mapping = py::inst_alloc(py::type<PyMapping>());
  PyMapping *cpp_mapping = py::inst_ptr<PyMapping>(py_mapping);
  new (cpp_mapping) mapping();
  py::inst_mark_ready(py_mapping);
  *out_cpp_mapping = cpp_mapping;
  return py_mapping;
}

// Wraps a delegate argument with ProgramResourceBarrier::READ.
class PyReadBarrier {
 public:
  explicit PyReadBarrier(py::object delegate)
      : delegate_(std::move(delegate)) {}
  void __sfinv_marshal__(py::capsule inv_capsule,
                         py::handle /*ignored_resource_barrier*/) {
    delegate_.attr("__sfinv_marshal__")(
        inv_capsule, static_cast<int>(local::ProgramResourceBarrier::READ));
  }

  py::object &delegate() { return delegate_; }

 private:
  py::object delegate_;
};

// Wraps a delegate argument with ProgramResourceBarrier::WRITE.
class PyWriteBarrier {
 public:
  explicit PyWriteBarrier(py::object delegate)
      : delegate_(std::move(delegate)) {}
  void __sfinv_marshal__(py::capsule inv_capsule,
                         py::handle /*ignored_resource_barrier*/) {
    delegate_.attr("__sfinv_marshal__")(
        inv_capsule, static_cast<int>(local::ProgramResourceBarrier::WRITE));
  }

  py::object &delegate() { return delegate_; }

 private:
  py::object delegate_;
};

// Wraps a delegate argument with ProgramResourceBarrier::NONE.
class PyDisableBarrier {
 public:
  explicit PyDisableBarrier(py::object delegate)
      : delegate_(std::move(delegate)) {}
  void __sfinv_marshal__(py::capsule inv_capsule,
                         py::handle /*ignored_resource_barrier*/) {
    delegate_.attr("__sfinv_marshal__")(
        inv_capsule, static_cast<int>(local::ProgramResourceBarrier::NONE));
  }

  py::object &delegate() { return delegate_; }

 private:
  py::object delegate_;
};

}  // namespace

void BindArray(py::module_ &m) {
  auto refs = std::make_shared<Refs>();

  py::class_<DType>(m, "DType")
      .def_prop_ro("name", &DType::name)
      .def_prop_ro("is_boolean", &DType::is_boolean)
      .def_prop_ro("is_integer", &DType::is_integer)
      .def_prop_ro("is_float", &DType::is_float)
      .def_prop_ro("is_complex", &DType::is_complex)
      .def_prop_ro("bit_count", &DType::bit_count)
      .def_prop_ro("is_byte_aligned", &DType::is_byte_aligned)
      .def_prop_ro("dense_byte_count", &DType::dense_byte_count)
      .def("is_integer_bitwidth", &DType::is_integer_bitwidth)
      .def("compute_dense_nd_size", &DType::compute_dense_nd_size)
      .def(py::self == py::self)
      .def("__repr__", &DType::name);

#define SHORTFIN_DTYPE_HANDLE(et, ident) m.attr(#ident) = DType::ident();
#include "shortfin/array/dtypes.inl"
#undef SHORTFIN_DTYPE_HANDLE

  // storage
  py::class_<storage>(m, "storage")
      .def("__sfinv_marshal__",
           [](device_array *self, py::capsule inv_capsule, int barrier) {
             auto *inv =
                 static_cast<local::ProgramInvocation *>(inv_capsule.data());
             static_cast<local::ProgramInvocationMarshalable *>(self)
                 ->AddAsInvocationArgument(
                     inv, static_cast<local::ProgramResourceBarrier>(barrier));
           })
      .def_static(
          "allocate_host",
          [](local::ScopedDevice &device, iree_device_size_t allocation_size) {
            return storage::allocate_host(device, allocation_size);
          },
          py::arg("device"), py::arg("allocation_size"), py::keep_alive<0, 1>())
      .def_static(
          "allocate_device",
          [](local::ScopedDevice &device, iree_device_size_t allocation_size) {
            return storage::allocate_device(device, allocation_size);
          },
          py::arg("device"), py::arg("allocation_size"), py::keep_alive<0, 1>())
      .def(
          "fill",
          [](storage &self, py::handle buffer) {
            Py_buffer py_view;
            int flags = PyBUF_FORMAT | PyBUF_ND;  // C-Contiguous ND.
            if (PyObject_GetBuffer(buffer.ptr(), &py_view, flags) != 0) {
              throw py::python_error();
            }
            PyBufferReleaser py_view_releaser(py_view);
            self.fill(py_view.buf, py_view.len);
          },
          py::arg("pattern"), DOCSTRING_STORAGE_FILL)
      .def(
          "copy_from", [](storage &self, storage &src) { self.copy_from(src); },
          py::arg("source_storage"), DOCSTRING_STORAGE_COPY_FROM)
      .def(
          "map",
          [](storage &self, bool read, bool write, bool discard) {
            SHORTFIN_TRACE_SCOPE_NAMED("PyStorage::map");
            int access = 0;
            if (read) access |= IREE_HAL_MEMORY_ACCESS_READ;
            if (write || discard) access |= IREE_HAL_MEMORY_ACCESS_WRITE;
            if (discard) access |= IREE_HAL_MEMORY_ACCESS_DISCARD;
            if (!access) {
              throw std::invalid_argument(
                  "One of the access flags must be set");
            }
            PyMapping *cpp_mapping = nullptr;
            py::object py_mapping = CreateMappingObject(&cpp_mapping);
            self.map_explicit(
                cpp_mapping->mapping(),
                static_cast<iree_hal_memory_access_bits_t>(access));
            return py_mapping;
          },
          py::kw_only(), py::arg("read") = false, py::arg("write") = false,
          py::arg("discard") = false, DOCSTRING_STORAGE_MAP)
      .def(py::self == py::self)
      .def("__len__", &storage::byte_length)
      .def("__repr__", &storage::to_s);

  // mapping
  auto mapping_class = py::class_<PyMapping>(m, "mapping");
  mapping_class.def("close", [](PyMapping &self) { self.mapping().reset(); })
      .def_prop_ro("valid",
                   [](PyMapping &self) -> bool { return self.mapping(); })
      .def("__enter__", [](py::object self_obj) { return self_obj; })
      .def(
          "__exit__",
          [](PyMapping &self, py::handle exc_type, py::handle exc_value,
             py::handle exc_tb) { self.mapping().reset(); },
          py::arg("exc_type").none(), py::arg("exc_value").none(),
          py::arg("exc_tb").none())
      .def(
          "fill",
          [refs](PyMapping &self, py::int_ value) {
            self.AssertValid();
            self.FillFromScalar(refs.get(), value);
          },
          py::arg("value"), DOCSTRING_MAPPING_FILL)
      .def(
          "fill",
          [refs](PyMapping &self, py::float_ value) {
            self.AssertValid();
            self.FillFromScalar(refs.get(), value);
          },
          py::arg("value"), DOCSTRING_MAPPING_FILL)
      .def(
          "fill",
          [](PyMapping &self, py::handle buffer) {
            self.AssertValid();
            self.FillFromBuffer(buffer);
          },
          py::arg("buffer"), DOCSTRING_MAPPING_FILL)
      .def_prop_rw(
          "items",
          [refs](py::handle self_obj) {
            PyMapping &self = py::cast<PyMapping &>(self_obj);
            return self.GetItems(self_obj, refs.get());
          },
          [refs](PyMapping &self, py::handle initializer) {
            self.SetItems(refs.get(), initializer);
          },
          DOCSTRING_MAPPING_ITEMS);

  struct MappingBufferHandler {
    int operator()(PyMapping &self, Py_buffer *view, int flags) {
      view->buf = self.mapping().data();
      view->len = self.mapping().size();
      view->readonly = !self.mapping().writable();
      view->itemsize = 1;
      view->format = (char *)"B";  // Byte
      view->ndim = 1;
      view->shape = nullptr;
      view->strides = nullptr;
      view->suboffsets = nullptr;
      view->internal = nullptr;
      return 0;
    }
  };
  BindBufferProtocol<PyMapping, MappingBufferHandler>(mapping_class);

  // base_array and subclasses
  py::class_<base_array>(m, "base_array")
      .def_prop_ro("dtype", &base_array::dtype)
      .def_prop_ro("shape", &base_array::shape);

  py::class_<device_array, base_array>(m, "device_array")
      .def("__init__", [](py::args, py::kwargs) {})
      .def_static(
          "__new__",
          [](py::handle py_type, class storage storage,
             std::span<const size_t> shape, DType dtype) {
            return custom_new_keep_alive<device_array>(
                py_type, /*keep_alive=*/storage.fiber(), storage, shape, dtype);
          },
          py::arg("cls"), py::arg("storage"), py::arg("shape"),
          py::arg("dtype"))
      .def_static(
          "__new__",
          [](py::handle py_type, local::ScopedDevice &device,
             std::span<const size_t> shape, DType dtype) {
            return custom_new_keep_alive<device_array>(
                py_type, /*keep_alive=*/device.fiber(),
                device_array::for_device(device, shape, dtype));
          },
          py::arg("cls"), py::arg("device"), py::arg("shape"), py::arg("dtype"))
      .def("__sfinv_marshal__",
           [](device_array *self, py::capsule inv_capsule, int barrier) {
             auto *inv =
                 static_cast<local::ProgramInvocation *>(inv_capsule.data());
             static_cast<local::ProgramInvocationMarshalable *>(self)
                 ->AddAsInvocationArgument(
                     inv, static_cast<local::ProgramResourceBarrier>(barrier));
           })
      .def_static(
          "for_device",
          [](local::ScopedDevice &device, std::span<const size_t> shape,
             DType dtype) {
            return custom_new_keep_alive<device_array>(
                py::type<device_array>(),
                /*keep_alive=*/device.fiber(),
                device_array::for_device(device, shape, dtype));
          },
          py::arg("device"), py::arg("shape"), py::arg("dtype"))
      .def_static(
          "for_host",
          [](local::ScopedDevice &device, std::span<const size_t> shape,
             DType dtype) {
            return custom_new_keep_alive<device_array>(
                py::type<device_array>(),
                /*keep_alive=*/device.fiber(),
                device_array::for_host(device, shape, dtype));
          },
          py::arg("device"), py::arg("shape"), py::arg("dtype"))
      .def("for_transfer",
           [](device_array &self) {
             return custom_new_keep_alive<device_array>(
                 py::type<device_array>(),
                 /*keep_alive=*/self.device().fiber(), self.for_transfer());
           })
      .def_prop_ro("device", &device_array::device,
                   py::rv_policy::reference_internal)
      .def_prop_ro("storage", &device_array::storage,
                   py::rv_policy::reference_internal)
      .def(
          "fill",
          [](py::handle_t<device_array> self, py::handle buffer) {
            self.attr("storage").attr("fill")(buffer);
          },
          py::arg("pattern"), DOCSTRING_ARRAY_FILL)
      .def("copy_from", &device_array::copy_from, py::arg("source_array"),
           DOCSTRING_ARRAY_COPY_FROM)
      .def("copy_to", &device_array::copy_to, py::arg("dest_array"),
           DOCSTRING_ARRAY_COPY_TO)
      .def("view", PyDeviceArrayView, DOCSTRING_ARRAY_VIEW)
      .def(
          "map",
          [](device_array &self, bool read, bool write, bool discard) {
            SHORTFIN_TRACE_SCOPE_NAMED("PyArray::map");
            int access = 0;
            if (read) access |= IREE_HAL_MEMORY_ACCESS_READ;
            if (write || discard) access |= IREE_HAL_MEMORY_ACCESS_WRITE;
            if (discard) access |= IREE_HAL_MEMORY_ACCESS_DISCARD;
            if (!access) {
              throw std::invalid_argument(
                  "One of the access flags must be set");
            }
            PyMapping *cpp_mapping = nullptr;
            py::object py_mapping = CreateMappingObject(&cpp_mapping);
            cpp_mapping->set_dtype(self.dtype());
            self.storage().map_explicit(
                cpp_mapping->mapping(),
                static_cast<iree_hal_memory_access_bits_t>(access));
            return py_mapping;
          },
          py::kw_only(), py::arg("read") = false, py::arg("write") = false,
          py::arg("discard") = false, DOCSTRING_ARRAY_MAP)
      .def_prop_rw(
          "items",
          [refs](device_array &self) {
            SHORTFIN_TRACE_SCOPE_NAMED("PyArray::items");
            PyMapping *mapping;
            py::object mapping_obj = CreateMappingObject(&mapping);
            mapping->set_dtype(self.dtype());
            self.storage().map_explicit(
                mapping->mapping(), static_cast<iree_hal_memory_access_bits_t>(
                                        IREE_HAL_MEMORY_ACCESS_READ));
            return mapping->GetItems(mapping_obj, refs.get());
          },
          [refs](device_array &self, py::handle initializer) {
            PyMapping mapping;
            mapping.set_dtype(self.dtype());
            self.storage().map_explicit(
                mapping.mapping(), static_cast<iree_hal_memory_access_bits_t>(
                                       IREE_HAL_MEMORY_ACCESS_READ));
            return mapping.SetItems(refs.get(), initializer);
          },
          DOCSTRING_ARRAY_ITEMS)
      .def_prop_ro(
          "__array_interface__",
          [refs](device_array &self) {
            SHORTFIN_TRACE_SCOPE_NAMED("PyArray::__array_interface__");
            py::dict interface;
            interface["version"] = 3;
            interface["strides"] = py::none();

            auto shape = self.shape();
            py::list shapeList;
            for (size_t i = 0; i < shape.size(); ++i) {
              shapeList.append(shape[i]);
            }
            py::tuple shape_tuple(py::cast(shapeList));
            interface["shape"] = shape_tuple;

            auto &table = refs->element_type_array_type_code_table;
            auto it = table.find(self.dtype());
            if (it == table.end()) {
              throw std::invalid_argument(fmt::format(
                  "Python array.array type code not known for dtype "
                  "{}: Cannot access items",
                  self.dtype().name()));
            }

            auto typeString = py::str();
            if (it->first == DType::float16()) {
              typeString = py::str("float16");
            } else {
              typeString = py::str(it->second);
            }
            interface["typestr"] = typeString;

            PyMapping *mapping;
            py::object mapping_obj = CreateMappingObject(&mapping);
            mapping->set_dtype(self.dtype());
            self.storage().map_explicit(
                mapping->mapping(), static_cast<iree_hal_memory_access_bits_t>(
                                        IREE_HAL_MEMORY_ACCESS_READ));
            auto items = mapping->GetItems(mapping_obj, refs.get());

            // Obtain pointer to first element in items
            Py_buffer buffer;
            if (PyObject_GetBuffer(items.ptr(), &buffer, PyBUF_SIMPLE) != 0) {
              throw std::runtime_error("Failed to get buffer from items");
            }
            void *itemsPtr = buffer.buf;
            interface["data"] =
                py::make_tuple(reinterpret_cast<intptr_t>(itemsPtr), false);
            return interface;
          })
      .def("__repr__", &device_array::to_s)
      .def("__str__", [](device_array &self) -> std::string {
        auto contents = self.contents_to_s();
        if (!contents) return "<<unmappable>>";
        return *contents;
      });

  py::class_<PyReadBarrier>(m, "read_barrier")
      .def(py::init<py::object>(), py::arg("delegate"),
           R"(Construct a read_barrier by passing a Python delegate object.)")
      .def(
          "__sfinv_marshal__", &PyReadBarrier::__sfinv_marshal__,
          py::arg("inv_capsule"), py::arg("ignored_resource_barrier"),
          R"(Forward `__sfinv_marshal__` to `delegate`, forcing the barrier argument to ProgramResourceBarrier::READ.)")
      .def("delegate", &PyReadBarrier::delegate,
           R"(Returns the internal device_array delegate.)");

  py::class_<PyWriteBarrier>(m, "write_barrier")
      .def(py::init<py::object>(), py::arg("delegate"),
           R"(Construct a write_barrier by passing a Python delegate object.)")
      .def(
          "__sfinv_marshal__", &PyWriteBarrier::__sfinv_marshal__,
          py::arg("inv_capsule"), py::arg("ignored_resource_barrier"),
          R"(Forward `__sfinv_marshal__` to `delegate`, forcing the barrier argument to ProgramResourceBarrier::WRITE.)")
      .def("delegate", &PyWriteBarrier::delegate,
           R"(Returns the internal device_array delegate.)");

  py::class_<PyDisableBarrier>(m, "disable_barrier")
      .def(
          py::init<py::object>(), py::arg("delegate"),
          R"(Construct a disable_barrier by passing a Python delegate object.)")
      .def(
          "__sfinv_marshal__", &PyDisableBarrier::__sfinv_marshal__,
          py::arg("inv_capsule"), py::arg("ignored_resource_barrier"),
          R"(Forward `__sfinv_marshal__` to `delegate`, forcing the barrier argument to ProgramResourceBarrier::DISABLE.)")
      .def("delegate", &PyDisableBarrier::delegate,
           R"(Returns the internal device_array delegate.)");

  BindArrayHostOps(m);
}

}  // namespace shortfin::python
