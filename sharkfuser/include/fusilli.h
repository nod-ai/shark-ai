// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main header file for Fusilli that includes all necessary headers.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_H
#define FUSILLI_H

// External:
#include "fusilli/external/torch_types.h" // IWYU pragma: export

// Support:
#include "fusilli/support/asm_emitter.h"    // IWYU pragma: export
#include "fusilli/support/cache.h"          // IWYU pragma: export
#include "fusilli/support/external_tools.h" // IWYU pragma: export
#include "fusilli/support/extras.h"         // IWYU pragma: export
#include "fusilli/support/logging.h"        // IWYU pragma: export

// Attributes / Types:
#include "fusilli/attributes/attributes.h"           // IWYU pragma: export
#include "fusilli/attributes/conv_attributes.h"      // IWYU pragma: export
#include "fusilli/attributes/matmul_attributes.h"    // IWYU pragma: export
#include "fusilli/attributes/pointwise_attributes.h" // IWYU pragma: export
#include "fusilli/attributes/tensor_attributes.h"    // IWYU pragma: export
#include "fusilli/attributes/types.h"                // IWYU pragma: export

// Nodes:
#include "fusilli/node/conv_node.h"      // IWYU pragma: export
#include "fusilli/node/node.h"           // IWYU pragma: export
#include "fusilli/node/pointwise_node.h" // IWYU pragma: export

// Backend:
#include "fusilli/backend/backend.h" // IWYU pragma: export
#include "fusilli/backend/buffer.h"  // IWYU pragma: export
#include "fusilli/backend/handle.h"  // IWYU pragma: export
#include "fusilli/backend/runtime.h" // IWYU pragma: export

// Graph:
#include "fusilli/graph/context.h" // IWYU pragma: export
#include "fusilli/graph/graph.h"   // IWYU pragma: export

#endif // FUSILLI_H
