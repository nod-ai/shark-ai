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
// TORCH-CHECK:     func.func @main(%result_: !torch.tensor<[128,256],f32>, %arg0_input: !torch.vtensor<[128,256],f32>, %arg1_add_transposed: !torch.vtensor<[256,128],f32>) attributes {torch.assume_strict_symbolic_shapes} {
// TORCH-CHECK:       %permute_IN_0_val_0_pointwise_add_transposed = torch.constant.int 0
// TORCH-CHECK:       %permute_IN_0_val_1_pointwise_add_transposed = torch.constant.int 1
// TORCH-CHECK:       %permute_IN_0_pointwise_add_transposed = torch.prim.ListConstruct %permute_IN_0_val_0_pointwise_add_transposed, %permute_IN_0_val_1_pointwise_add_transposed : (!torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg0_input_in0_pointwise_add_transposed_perm = torch.aten.permute %arg0_input, %permute_IN_0_pointwise_add_transposed : !torch.vtensor<[128,256],f32>, !torch.list<int> -> !torch.vtensor<[128,256],f32>
// TORCH-CHECK:       %permute_IN_1_val_0_pointwise_add_transposed = torch.constant.int 1
// TORCH-CHECK:       %permute_IN_1_val_1_pointwise_add_transposed = torch.constant.int 0
// TORCH-CHECK:       %permute_IN_1_pointwise_add_transposed = torch.prim.ListConstruct %permute_IN_1_val_0_pointwise_add_transposed, %permute_IN_1_val_1_pointwise_add_transposed : (!torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg1_add_transposed_in1_pointwise_add_transposed_perm = torch.aten.permute %arg1_add_transposed, %permute_IN_1_pointwise_add_transposed : !torch.vtensor<[256,128],f32>, !torch.list<int> -> !torch.vtensor<[128,256],f32>
// TORCH-CHECK:       %alpha_pointwise_add_transposed = torch.constant.int 1
// TORCH-CHECK:       %result_perm = torch.aten.add.Tensor %arg0_input_in0_pointwise_add_transposed_perm, %arg1_add_transposed_in1_pointwise_add_transposed_perm, %alpha_pointwise_add_transposed : !torch.vtensor<[128,256],f32>, !torch.vtensor<[128,256],f32>, !torch.int -> !torch.vtensor<[128,256],f32>
// TORCH-CHECK:       %permute_OUT_0_val_0_pointwise_add_transposed = torch.constant.int 0
// TORCH-CHECK:       %permute_OUT_0_val_1_pointwise_add_transposed = torch.constant.int 1
// TORCH-CHECK:       %permute_OUT_0_pointwise_add_transposed = torch.prim.ListConstruct %permute_OUT_0_val_0_pointwise_add_transposed, %permute_OUT_0_val_1_pointwise_add_transposed : (!torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %result = torch.aten.permute %result_perm, %permute_OUT_0_pointwise_add_transposed : !torch.vtensor<[128,256],f32>, !torch.list<int> -> !torch.vtensor<[128,256],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %result overwrites %result_ : !torch.vtensor<[128,256],f32>, !torch.tensor<[128,256],f32>
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

static ErrorObject
testPointwiseAsmEmitterAddTransposed(const std::string &mode) {
  int64_t n = 128, c = 256;
  auto graph = std::make_shared<Graph>();
  graph->setName("pointwise_asm_emitter_add_transposed");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto xT =
      graph->tensor(TensorAttr()
                        .setName("arg0_input")
                        .setDim({n, c})
                        .setStride({c, 1})); // Contiguous: stride={256, 1}

  auto bT =
      graph->tensor(TensorAttr()
                        .setName("arg1_add_transposed")
                        .setDim({n, c})
                        .setStride({1, n})); // Transposed: stride={1, 128}

  auto pointwiseAttr = PointwiseAttr()
                           .setMode(PointwiseAttr::Mode::ADD)
                           .setName("pointwise_add_transposed");

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

  auto status = testPointwiseAsmEmitterAddTransposed(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
