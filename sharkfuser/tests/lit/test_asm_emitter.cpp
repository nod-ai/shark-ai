// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %test_exe | filecheck %s

#include <fusilli.h>

#include <iostream>
#include <string>
#include <vector>

using namespace fusilli;

void test_getListOfIntOpsAsm() {
  std::vector<int64_t> vals{1, 2, 3};
  std::string prefix = "stride";
  std::string suffix = "conv";
  std::string asmStr = getListOfIntOpsAsm(vals, prefix, suffix);

  // clang-format off
  // CHECK:  %stride_val_0_conv = torch.constant.int 1
  // CHECK:  %stride_val_1_conv = torch.constant.int 2
  // CHECK:  %stride_val_2_conv = torch.constant.int 3
  // CHECK:  %stride_conv = torch.prim.ListConstruct %stride_val_0_conv, %stride_val_1_conv, %stride_val_2_conv : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
  // clang-format on
  std::cout << asmStr << std::endl;
}

void test_getValueTensorTypeAsm() {
  TensorAttr t;
  t.setName("tensor").setDataType(DataType::Float).setDim({2, 3});

  // CHECK:  !torch.vtensor<[2,3],f32>
  std::cout << t.getValueTensorTypeAsm() << std::endl;
}

void test_getMlirSSAValueNameAsm() {
  TensorAttr t;
  t.setName("foo_Bar::X0").setDataType(DataType::Float).setDim({1});

  // CHECK:  %foo_BarX0
  std::cout << t.getMlirSSAValueNameAsm() << std::endl;
}

int main() {
  test_getListOfIntOpsAsm();
  test_getValueTensorTypeAsm();
  test_getMlirSSAValueNameAsm();
  return 0;
}
