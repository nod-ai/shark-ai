// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains general support methods and classes.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_SUPPORT_EXTRAS_H
#define FUSILLI_SUPPORT_EXTRAS_H

namespace fusilli {

// An STL-style algorithm similar to std::for_each that applies a second
// functor between every pair of elements.
//
// This provides the control flow logic to, for example, print a
// comma-separated list:
//
//   interleave(names.begin(), names.end(),
//              [&](std::string name) { os << name; },
//              [&] { os << ", "; });
//
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
inline void interleave(ForwardIterator begin, ForwardIterator end,
                       UnaryFunctor eachFn, NullaryFunctor betweenFn) {
  if (begin == end)
    return;
  eachFn(*begin);
  ++begin;
  for (; begin != end; ++begin) {
    betweenFn();
    eachFn(*begin);
  }
}

// An overload of `interleave` which additionally accepts a SkipFunctor
// to skip certain elements based on a predicate.
//
// This provides the control flow logic to, for example, print a
// comma-separated list excluding "foo":
//
//   interleave(names.begin(), names.end(),
//              [&](std::string name) { os << name; },
//              [&] { os << ", "; },
//              [&](std::string name) { return name == "foo"; });
//
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor, typename SkipFunctor>
inline void interleave(ForwardIterator begin, ForwardIterator end,
                       UnaryFunctor eachFn, NullaryFunctor betweenFn,
                       SkipFunctor skipFn) {
  if (begin == end)
    return;
  bool first = true;
  for (; begin != end; ++begin) {
    if (!skipFn(*begin)) {
      if (!first)
        betweenFn();
      first = false;
      eachFn(*begin);
    }
  }
}

} // namespace fusilli

#endif // FUSILLI_SUPPORT_EXTRAS_H
