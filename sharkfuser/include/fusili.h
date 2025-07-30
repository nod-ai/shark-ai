// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This is the main header file for Fusili that includes all necessary headers.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILI_H
#define FUSILI_H

// Attributes
#include "fusili/attributes/attributes.h"
#include "fusili/attributes/conv_attributes.h"
#include "fusili/attributes/tensor_attributes.h"

// Nodes
#include "fusili/node/conv_node.h"
#include "fusili/node/node.h"

// Utilities
#include "fusili/asm_emitter.h"
#include "fusili/external_tools.h"
#include "fusili/logging.h"
#include "fusili/types.h"

// Graph
#include "fusili/context.h"
#include "fusili/graph.h"

#endif // FUSILI_H
