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

// Support:
#include "fusilli/support/asm_emitter.h"
#include "fusilli/support/cache.h"
#include "fusilli/support/external_tools.h"
#include "fusilli/support/extras.h"
#include "fusilli/support/logging.h"

// Attributes / Types:
#include "fusilli/attributes/attributes.h"
#include "fusilli/attributes/conv_attributes.h"
#include "fusilli/attributes/pointwise_attributes.h"
#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/attributes/types.h"

// Nodes:
#include "fusilli/node/conv_node.h"
#include "fusilli/node/node.h"
#include "fusilli/node/pointwise_node.h"

// Backend:
#include "fusilli/backend/backend.h"
#include "fusilli/backend/buffer.h"
#include "fusilli/backend/handle.h"
#include "fusilli/backend/runtime.h"

// Graph:
#include "fusilli/graph/context.h"
#include "fusilli/graph/graph.h"

#endif // FUSILLI_H
