// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef SHORTFIN_ARRAY_DIMS_H
#define SHORTFIN_ARRAY_DIMS_H

#include <array>
#include <memory>
#include <span>

#include "iree/hal/buffer_view.h"
#include "shortfin/support/api.h"

namespace shortfin::array {

// Vector-alike for storing inlined dims. Note that this has a template
// signature identical to std::vector because xtensor specializes on this
// exact signature. See the concrete size_t instantiation below.
template <typename T = std::size_t, typename Alloc = std::allocator<T>>
class SHORTFIN_API InlinedDims {
 public:
  using element_type = T;
  using value_type = T;
  using allocator_type = Alloc;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type &;
  using const_reference = const value_type &;
  using pointer = value_type *;
  using const_pointer = const value_type *;

  class iterator {
   public:
    using difference_type = std::ptrdiff_t;
    using value_type = T;
    using pointer = T *;
    using reference = T &;
    using iterator_category = std::random_access_iterator_tag;
    iterator(pointer p) : p(p) {}
    constexpr iterator &operator++() {
      p++;
      return *this;
    }
    constexpr iterator operator++(int) {
      auto tmp = *this;
      p++;
      return tmp;
    }
    constexpr iterator &operator--() {
      p--;
      return *this;
    }
    constexpr iterator operator--(int) {
      auto tmp = *this;
      p--;
      return tmp;
    }
    constexpr bool operator==(iterator other) const { return p == other.p; }
    constexpr bool operator!=(iterator other) const { return p != other.p; }
    constexpr reference operator*() { return *p; }
    constexpr iterator operator+(difference_type d) const {
      return iterator(p + d);
    }
    constexpr iterator operator-(difference_type d) const {
      return iterator(p - d);
    }
    constexpr difference_type operator-(iterator rhs) const {
      return static_cast<difference_type>(p - rhs.p);
    }

   private:
    pointer p;
  };
  class const_iterator {
   public:
    using difference_type = std::ptrdiff_t;
    using value_type = const T;
    using pointer = const T *;
    using reference = const T &;
    using iterator_category = std::random_access_iterator_tag;

    const_iterator(pointer p) : p(p) {}
    constexpr const_iterator &operator++() {
      p++;
      return *this;
    }
    constexpr const_iterator operator++(int) {
      auto tmp = *this;
      p++;
      return tmp;
    }
    constexpr const_iterator &operator--() {
      p--;
      return *this;
    }
    constexpr const_iterator operator--(int) {
      auto tmp = *this;
      p--;
      return tmp;
    }
    constexpr bool operator==(const_iterator other) const {
      return p == other.p;
    }
    constexpr bool operator!=(const_iterator other) const {
      return p != other.p;
    }
    constexpr reference operator*() { return *p; }
    constexpr reference operator[](difference_type d) const { return *(p + d); }
    constexpr const_iterator operator+(difference_type d) const {
      return const_iterator(p + d);
    }
    constexpr const_iterator operator-(difference_type d) const {
      return const_iterator(p - d);
    }
    constexpr difference_type operator-(const_iterator rhs) const {
      return static_cast<difference_type>(p - rhs.p);
    }

   private:
    pointer p;
  };
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  InlinedDims() { new (&dims_.inline_dims) InlineTy(); }
  InlinedDims(size_type count, T value = T()) : size_(count) {
    if (size_ > MAX_INLINE_RANK) {
      // Dynamic allocation.
      new (&dims_.dynamic_dims) DynamicTy(new element_type[size_]);
      std::fill(dims_.dynamic_dims.get(), dims_.dynamic_dims.get() + size_,
                value);
    } else {
      // Inline allocation.
      new (&dims_.inline_dims) InlineTy();
      std::fill(dims_.inline_dims.begin(), dims_.inline_dims.end(), value);
    }
  }
  template <std::contiguous_iterator BeginTy, typename EndTy>
  InlinedDims(BeginTy begin, EndTy end) {
    assert(end > begin);
    set(std::span<const size_type>(&(*begin), end - begin));
  }
  InlinedDims(const InlinedDims &other) {
    new (&dims_.inline_dims) InlineTy();
    set(other.span());
  }
  InlinedDims(InlinedDims &&other) : size_(other.size_) {
    // Custom move the dims to avoid an additional allocation. This could just
    // be a memcpy on most impls, but this is the "right way".
    if (size_ > MAX_INLINE_RANK) {
      // Dynamic allocation.
      new (&dims_.dynamic_dims) DynamicTy();
      dims_.dynamic_dims = std::move(other.dims_.dynamic_dims);
    } else {
      // Inline allocation.
      new (&dims_.inline_dims) InlineTy();
      dims_.inline_dims = other.dims_.inline_dims;
    }
    other.size_ = 0;
  }
  InlinedDims &operator=(const InlinedDims &other) {
    set(other.span());
    return *this;
  }
  ~InlinedDims() { clear(); }

  T *data() {
    if (size_ > MAX_INLINE_RANK) {
      return dims_.dynamic_dims.get();
    } else {
      return &dims_.inline_dims.front();
    }
  }
  const T *data() const {
    if (size_ > MAX_INLINE_RANK) {
      return dims_.dynamic_dims.get();
    } else {
      return &dims_.inline_dims.front();
    }
  }
  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  // Clears shape, setting the rank to zero and deleting any non-inline
  // dimension storage.
  void clear() {
    if (size_ > MAX_INLINE_RANK) {
      dims_.dynamic_dims.~unique_ptr();
    } else {
      dims_.inline_dims.~array();
    }
    size_ = 0;
  }

  void set(std::span<const T> dims) {
    clear();
    size_ = dims.size();
    if (size_ > MAX_INLINE_RANK) {
      // Dynamic allocation.
      new (&dims_.dynamic_dims) DynamicTy(new element_type[size_]);
      std::copy(dims.begin(), dims.end(), dims_.dynamic_dims.get());
    } else {
      // Inline allocation.
      new (&dims_.inline_dims) InlineTy();
      std::copy(dims.begin(), dims.end(), dims_.inline_dims.begin());
    }
  }

  // Container access.
  iterator begin() { return iterator(data()); }
  iterator end() { return iterator(data() + size()); }
  const_iterator begin() const { return const_iterator(data()); }
  const_iterator end() const { return const_iterator(data() + size()); }
  const_iterator cbegin() const { return const_iterator(data()); }
  const_iterator cend() const { return const_iterator(data() + size()); }
  reverse_iterator rbegin() { return reverse_iterator(begin()); }
  reverse_iterator rend() { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const {
    return const_reverse_iterator(begin());
  }
  const_reverse_iterator rend() const { return const_reverse_iterator(end()); }

  void resize(size_type count) { resize_impl(count, value_type()); }
  void resize(size_type count, value_type value) { resize_impl(count, value); }

  reference operator[](std::size_t idx) { return *(data() + idx); }
  const_reference operator[](std::size_t idx) const { return *(data() + idx); }

  reference front() { return *data(); }
  const_reference front() const { return *data(); }
  reference back() { return *(data() + size() - 1); }
  const_reference back() const { return *(data() + size() - 1); }

  // Access as a span.
  std::span<T> span() { return std::span<T>(data(), size_); }
  std::span<const T> span() const { return std::span<const T>(data(), size_); }

 private:
  void resize_impl(size_type count, value_type value) {
    if (count == size()) return;
    if (size() > MAX_INLINE_RANK) {
      // Currently dynamically allocated.
      if (count < size()) {
        // Truncate.
        if (count < MAX_INLINE_RANK) {
          // Switch to inlined.
          InlineTy new_array;
          for (std::size_t i = 0; i < count; ++i)
            new_array[i] = dims_.dynamic_dims[i];
          dims_.dynamic_dims.~unique_ptr();
          new (&dims_.inline_dims) InlineTy(new_array);
          size_ = count;
        } else {
          // Stay dynamic and just truncate.
          size_ = count;
        }
      } else {
        // Expand and stay dynamic.
        DynamicTy new_array(new element_type[count]);
        for (std::size_t i = 0; i < size_; ++i)
          new_array[i] = dims_.dynamic_dims[i];
        for (std::size_t i = size_; i < count; ++i) new_array[i] = value;
        dims_.dynamic_dims = std::move(new_array);
        size_ = count;
      }
    } else {
      // Currently inlined.
      if (count < size()) {
        // Truncate.
        size_ = count;
      } else if (count < MAX_INLINE_RANK) {
        // Stay inlined and initialize new items.
        for (std::size_t i = size_; i < count; ++i)
          dims_.inline_dims[i] = value;
        size_ = count;
      } else {
        // Need to switch to dynamic size.
        DynamicTy new_array(new element_type[count]);
        for (std::size_t i = 0; i < size_; ++i)
          new_array[i] = dims_.inline_dims[i];
        for (std::size_t i = size_; i < count; ++i) new_array[i] = value;
        dims_.inline_dims.~array();
        new (&dims_.dynamic_dims) DynamicTy(std::move(new_array));
        size_ = count;
      }
    }
  }

  static constexpr size_t MAX_INLINE_RANK = 6;
  using InlineTy = std::array<T, MAX_INLINE_RANK>;
  using DynamicTy = std::unique_ptr<T[]>;
  union _D {
    _D() {}
    ~_D() {}
    InlineTy inline_dims;
    DynamicTy dynamic_dims;
  };

  std::size_t size_ = 0;
  _D dims_;
};

using Dims = InlinedDims<iree_hal_dim_t>;

}  // namespace shortfin::array

#endif  // SHORTFIN_ARRAY_DIMS_H
