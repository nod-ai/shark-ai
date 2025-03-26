// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Standalone interfaces needed for marshaling as part of a ProgramInvocation.h.
// They are available in this dep-free header in order to ease the burden on
// types that would otherwise need to pull in all of the includes.

#ifndef SHORTFIN_LOCAL_PROGRAM_INTERFACES_H
#define SHORTFIN_LOCAL_PROGRAM_INTERFACES_H

#include "shortfin/support/api.h"
#include "shortfin/support/iree_helpers.h"

namespace shortfin::local {

class CoarseInvocationTimelineImporter;
class ProgramInvocation;

// The type of barrier that should be managed for a program resource.
enum class ProgramResourceBarrier {
  // The caller has explicitly not stated a preference.
  DEFAULT,

  // The argument will be used by the program for input and the program
  // must not perform operations on it until all pending mutations have
  // been completed. Concurrent reads/uses are permitted.
  // This is the default concurrency in most situations.
  READ,

  // The argument will be used for input/output and the program must not
  // perform operations on it until all prior mutations and uses have been
  // complete.
  WRITE,

  // No concurrency barriers will be emplaced on behalf of the argument,
  // explicitly allowing racy access. The program and the caller must
  // ensure that only valid accesses are made.
  NONE,
};

// Implemented by a class if it can marshal itself to an invocation as an
// argument.
class SHORTFIN_API ProgramInvocationMarshalable {
 public:
  // Adds this object as an invocation argument.
  virtual void AddAsInvocationArgument(ProgramInvocation *inv,
                                       ProgramResourceBarrier barrier) = 0;
};

// Trampoline class that has visibility into marshalable types and can be used
// to construct them from an invocation reference.
class SHORTFIN_API ProgramInvocationMarshalableFactory {
 public:
  // Instantiates a new `T` from an opaque reference retrieved from an
  // invocation result. This will call through to a factory on the type to
  // construct a new user-value and setup any needed barriers from the
  // invocation.
  //
  // In order for a type to be eligible for such usage, it must expose a
  // `T CreateFromInvocationResultRef(ProgramInvocation *inv,
  // iree::vm_opaque_ref)` static method. The type `T` must be friends with this
  // class.
  template <typename T>
  static T CreateFromInvocationResultRef(
      ProgramInvocation *inv,
      CoarseInvocationTimelineImporter *timeline_importer,
      iree::vm_opaque_ref ref) {
    return T::CreateFromInvocationResultRef(inv, timeline_importer,
                                            std::move(ref));
  }

  // Gets the type id that corresponds to this marshalable type.
  //
  // Marshalable types should define the same method.
  // It is recommended that these type methods are defined in shortfin
  // implementation files (not headers) since that ensures that no cross-DSO
  // symbol visibility issues can transpire.
  template <typename T>
  static iree_vm_ref_type_t invocation_marshalable_type() {
    return T::invocation_marshalable_type();
  }
};

}  // namespace shortfin::local

#endif  // SHORTFIN_LOCAL_PROGRAM_INTERFACES_H
