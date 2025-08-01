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

// Attributes
#include "fusilli/attributes/attributes.h"
#include "fusilli/attributes/conv_attributes.h"
#include "fusilli/attributes/tensor_attributes.h"

// Nodes
#include "fusilli/node/conv_node.h"
#include "fusilli/node/node.h"

// Utilities
#include "fusilli/asm_emitter.h"
#include "fusilli/external_tools.h"
#include "fusilli/logging.h"
#include "fusilli/types.h"

// Graph
#include "fusilli/context.h"
#include "fusilli/graph.h"

#endif // FUSILLI_H
