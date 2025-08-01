// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %test_exe | iree-opt --verify-roundtrip
// RUN: %test_exe | FileCheck %s

#include <fusilli.h>

#include <cassert>
#include <iostream>
#include <memory>

using namespace fusilli;

int main() {
  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;

  auto graph = std::make_shared<Graph>();
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto X = graph->tensor(TensorAttr()
                             .setName("arg0_image")
                             .setDim({n, c, h, w})
                             .setStride({c * h * w, h * w, w, 1}));

  auto W = graph->tensor(TensorAttr()
                             .setName("arg1_filter")
                             .setDim({k, c, r, s})
                             .setStride({c * r * s, r * s, s, 1}));

  auto conv_attr = ConvFPropAttr()
                       .setPadding({0, 0})
                       .setStride({1, 1})
                       .setDilation({1, 1})
                       .setName("conv_fprop");

  auto Y = graph->convFProp(X, W, conv_attr);

  Y->setName("result").setDim({n, k, h, w}).setStride({k * h * w, h * w, w, 1});
  Y->setOutput(true);

  // clang-format off
  // CHECK:   module @module {
  // CHECK:     func.func @main(%arg0_image: !torch.vtensor<[16,128,64,64],f32>, %arg1_filter: !torch.vtensor<[256,128,1,1],f32>) -> !torch.vtensor<[16,256,64,64],f32> attributes {torch.assume_strict_symbolic_shapes} {
  // CHECK:       %bias_conv_fprop = torch.constant.none
  // CHECK:       %stride_val_0_conv_fprop = torch.constant.int 1
  // CHECK:       %stride_val_1_conv_fprop = torch.constant.int 1
  // CHECK:       %stride_conv_fprop = torch.prim.ListConstruct %stride_val_0_conv_fprop, %stride_val_1_conv_fprop : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK:       %padding_val_0_conv_fprop = torch.constant.int 0
  // CHECK:       %padding_val_1_conv_fprop = torch.constant.int 0
  // CHECK:       %padding_conv_fprop = torch.prim.ListConstruct %padding_val_0_conv_fprop, %padding_val_1_conv_fprop : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK:       %dilation_val_0_conv_fprop = torch.constant.int 1
  // CHECK:       %dilation_val_1_conv_fprop = torch.constant.int 1
  // CHECK:       %dilation_conv_fprop = torch.prim.ListConstruct %dilation_val_0_conv_fprop, %dilation_val_1_conv_fprop : (!torch.int, !torch.int) -> !torch.list<int>
  // CHECK:       %transposed_conv_fprop = torch.constant.bool false
  // CHECK:       %output_padding_conv_fprop = torch.prim.ListConstruct  : () -> !torch.list<int>
  // CHECK:       %groups_conv_fprop = torch.constant.int 1
  // CHECK:       %result = torch.aten.convolution %arg0_image, %arg1_filter, %bias_conv_fprop, %stride_conv_fprop, %padding_conv_fprop, %dilation_conv_fprop, %transposed_conv_fprop, %output_padding_conv_fprop, %groups_conv_fprop : !torch.vtensor<[16,128,64,64],f32>, !torch.vtensor<[256,128,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[16,256,64,64],f32>
  // CHECK:       return %result : !torch.vtensor<[16,256,64,64],f32>
  // CHECK:     }
  // CHECK:   }
  // clang-format on

  assert(isOk(graph->validate()) && "Graph is invalid");
  ErrorOr<std::string> errorOrAsm = graph->emitAsm();
  assert(isOk(errorOrAsm) && "Graph ASM emission failed");
  std::cout << *errorOrAsm << std::endl;

  return 0;
}
