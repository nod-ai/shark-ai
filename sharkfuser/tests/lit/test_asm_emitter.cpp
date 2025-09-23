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

void test_getTensorTypeAsm() {
  TensorAttr t1;
  t1.setName("tensor1")
      .setDataType(DataType::Float)
      .setDim({2, 3})
      .setStride({3, 1});

  // CHECK:  !torch.vtensor<[2,3],f32>
  std::cout << t1.getTensorTypeAsm(/*isValueTensor=*/true,
                                   /*useLogicalDims=*/false)
            << std::endl;

  // CHECK:  !torch.tensor<[2,3],f32>
  std::cout << t1.getTensorTypeAsm(/*isValueTensor=*/false,
                                   /*useLogicalDims=*/false)
            << std::endl;

  // CHECK:  !torch.vtensor<[2,3],f32>
  std::cout << t1.getTensorTypeAsm(/*isValueTensor=*/true,
                                   /*useLogicalDims=*/true)
            << std::endl;

  // CHECK:  !torch.tensor<[2,3],f32>
  std::cout << t1.getTensorTypeAsm(/*isValueTensor=*/false,
                                   /*useLogicalDims=*/true)
            << std::endl;

  TensorAttr t2;
  t2.setName("tensor2")
      .setDataType(DataType::Float)
      .setDim({2, 3, 4})
      .setStride({12, 1, 3});

  // CHECK:  !torch.vtensor<[2,4,3],f32>
  std::cout << t2.getTensorTypeAsm(/*isValueTensor=*/true,
                                   /*useLogicalDims=*/false)
            << std::endl;

  // CHECK:  !torch.tensor<[2,4,3],f32>
  std::cout << t2.getTensorTypeAsm(/*isValueTensor=*/false,
                                   /*useLogicalDims=*/false)
            << std::endl;

  // CHECK:  !torch.vtensor<[2,3,4],f32>
  std::cout << t2.getTensorTypeAsm(/*isValueTensor=*/true,
                                   /*useLogicalDims=*/true)
            << std::endl;

  // CHECK:  !torch.tensor<[2,3,4],f32>
  std::cout << t2.getTensorTypeAsm(/*isValueTensor=*/false,
                                   /*useLogicalDims=*/true)
            << std::endl;
}

void test_getValueNameAsm() {
  TensorAttr t;
  t.setName("foo_Bar::X0").setDataType(DataType::Float).setDim({1});

  // CHECK:  %foo_BarX0
  std::cout << t.getValueNameAsm(/*isOutputAliased=*/false) << std::endl;

  // CHECK:  %foo_BarX0_
  std::cout << t.getValueNameAsm(/*isOutputAliased=*/true) << std::endl;
}

int main() {
  test_getListOfIntOpsAsm();
  test_getTensorTypeAsm();
  test_getValueNameAsm();
  return 0;
}
