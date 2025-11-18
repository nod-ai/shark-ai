// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | iree-opt --verify-roundtrip
// RUN: %{TEST_EXE} | FileCheck %s --check-prefix=TORCH-CHECK
// RUN: %{TEST_EXE} stats | FileCheck %s --check-prefix=%{BACKEND}-STATS-CHECK

// clang-format off
//
// TORCH-CHECK:   module @module {
// TORCH-CHECK:     func.func @main(%result_: !torch.tensor<[2,3,128,128],f32>, %arg0_input: !torch.vtensor<[2,3,128,128],f32>, %arg1_mul: !torch.vtensor<[128],f32>) attributes {torch.assume_strict_symbolic_shapes} {
// TORCH-CHECK:       %permute_IN_0_val_0_pointwise_mul = torch.constant.int 0
// TORCH-CHECK:       %permute_IN_0_val_1_pointwise_mul = torch.constant.int 1
// TORCH-CHECK:       %permute_IN_0_val_2_pointwise_mul = torch.constant.int 2
// TORCH-CHECK:       %permute_IN_0_val_3_pointwise_mul = torch.constant.int 3
// TORCH-CHECK:       %permute_IN_0_pointwise_mul = torch.prim.ListConstruct %permute_IN_0_val_0_pointwise_mul, %permute_IN_0_val_1_pointwise_mul, %permute_IN_0_val_2_pointwise_mul, %permute_IN_0_val_3_pointwise_mul : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg0_input_in0_pointwise_mul_perm = torch.aten.permute %arg0_input, %permute_IN_0_pointwise_mul : !torch.vtensor<[2,3,128,128],f32>, !torch.list<int> -> !torch.vtensor<[2,3,128,128],f32>
// TORCH-CHECK:       %permute_IN_1_val_0_pointwise_mul = torch.constant.int 0
// TORCH-CHECK:       %permute_IN_1_pointwise_mul = torch.prim.ListConstruct %permute_IN_1_val_0_pointwise_mul : (!torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg1_mul_in1_pointwise_mul_perm = torch.aten.permute %arg1_mul, %permute_IN_1_pointwise_mul : !torch.vtensor<[128],f32>, !torch.list<int> -> !torch.vtensor<[128],f32>
// TORCH-CHECK:       %result_perm = torch.aten.mul.Tensor %arg0_input_in0_pointwise_mul_perm, %arg1_mul_in1_pointwise_mul_perm : !torch.vtensor<[2,3,128,128],f32>, !torch.vtensor<[128],f32> -> !torch.vtensor<[2,3,128,128],f32>
// TORCH-CHECK:       %permute_OUT_0_val_0_pointwise_mul = torch.constant.int 0
// TORCH-CHECK:       %permute_OUT_0_val_1_pointwise_mul = torch.constant.int 1
// TORCH-CHECK:       %permute_OUT_0_val_2_pointwise_mul = torch.constant.int 2
// TORCH-CHECK:       %permute_OUT_0_val_3_pointwise_mul = torch.constant.int 3
// TORCH-CHECK:       %permute_OUT_0_pointwise_mul = torch.prim.ListConstruct %permute_OUT_0_val_0_pointwise_mul, %permute_OUT_0_val_1_pointwise_mul, %permute_OUT_0_val_2_pointwise_mul, %permute_OUT_0_val_3_pointwise_mul : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %result = torch.aten.permute %result_perm, %permute_OUT_0_pointwise_mul : !torch.vtensor<[2,3,128,128],f32>, !torch.list<int> -> !torch.vtensor<[2,3,128,128],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %result overwrites %result_ : !torch.vtensor<[2,3,128,128],f32>, !torch.tensor<[2,3,128,128],f32>
// TORCH-CHECK:       return
// TORCH-CHECK:     }
// TORCH-CHECK:   }
//
// AMDGPU-STATS-CHECK: "dispatch-count": 1
// CPU-STATS-CHECK: "dispatch-count": 1
//
// clang-format on

#include <fusilli.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

using namespace fusilli;

static ErrorObject testPointwiseAsmEmitterMul(const std::string &mode) {
  int64_t n = 2, c = 3, h = 128, w = 128;
  auto graph = std::make_shared<Graph>();
  graph->setName("pointwise_asm_emitter_mul");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto xT = graph->tensor(TensorAttr()
                              .setName("arg0_input")
                              .setDim({n, c, h, w})
                              .setStride({c * h * w, h * w, w, 1})); // NCHW

  auto bT = graph->tensor(
      TensorAttr().setName("arg1_mul").setDim({w}).setStride({1})); // 1D mul

  auto pointwiseAttr = PointwiseAttr()
                           .setMode(PointwiseAttr::Mode::MUL)
                           .setName("pointwise_mul");

  auto yT = graph->pointwise(xT, bT, pointwiseAttr);

  yT->setName("result").setOutput(true);

  FUSILLI_CHECK_ERROR(graph->validate());

  if (mode == "default") {
    std::cout << FUSILLI_TRY(graph->emitAsm()) << std::endl;
  }

  if (mode == "stats") {
#ifdef FUSILLI_ENABLE_AMDGPU
    Handle handle = FUSILLI_TRY(Handle::create(Backend::AMDGPU));
#else
    Handle handle = FUSILLI_TRY(Handle::create(Backend::CPU));
#endif
    FUSILLI_CHECK_ERROR(graph->compile(handle, /*remove=*/true));
    std::cout << FUSILLI_TRY(graph->readCompilationCacheFile(
                     CachedAssetsType::Statistics))
              << std::endl;
  }

  return ok();
}

int main(int argc, char **argv) {
  std::string mode = (argc > 1) ? argv[1] : "default";

  auto status = testPointwiseAsmEmitterMul(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
