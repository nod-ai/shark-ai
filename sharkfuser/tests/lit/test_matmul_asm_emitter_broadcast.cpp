// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | iree-opt --verify-roundtrip
// RUN: %{TEST_EXE} | FileCheck %s --check-prefix=TORCH-CHECK
// RUN: %{TEST_EXE} | iree-compile - --compile-to=input | \
// RUN:             FileCheck %s --check-prefix=LINALG-CHECK
// RUN: %{TEST_EXE} stats | FileCheck %s --check-prefix=%{BACKEND}-STATS-CHECK

// clang-format off
//
// Test batched matmul with broadcasting: A (4, 2, 64, 128) x B (128, 256) -> C (4, 2, 64, 256)
// B is broadcast across the batch dimensions.
//
// TORCH-CHECK:   module @module {
// TORCH-CHECK:     func.func @main(%result_: !torch.tensor<[4,2,64,256],f32>, %arg0_matrix_a: !torch.vtensor<[4,2,64,128],f32>, %arg1_matrix_b: !torch.vtensor<[128,256],f32>) attributes {torch.assume_strict_symbolic_shapes} {
// TORCH-CHECK:       %permute_A_val_0_broadcast_matmul = torch.constant.int 0
// TORCH-CHECK:       %permute_A_val_1_broadcast_matmul = torch.constant.int 1
// TORCH-CHECK:       %permute_A_val_2_broadcast_matmul = torch.constant.int 2
// TORCH-CHECK:       %permute_A_val_3_broadcast_matmul = torch.constant.int 3
// TORCH-CHECK:       %permute_A_broadcast_matmul = torch.prim.ListConstruct %permute_A_val_0_broadcast_matmul, %permute_A_val_1_broadcast_matmul, %permute_A_val_2_broadcast_matmul, %permute_A_val_3_broadcast_matmul : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg0_matrix_a_perm = torch.aten.permute %arg0_matrix_a, %permute_A_broadcast_matmul : !torch.vtensor<[4,2,64,128],f32>, !torch.list<int> -> !torch.vtensor<[4,2,64,128],f32>
// TORCH-CHECK:       %permute_B_val_0_broadcast_matmul = torch.constant.int 0
// TORCH-CHECK:       %permute_B_val_1_broadcast_matmul = torch.constant.int 1
// TORCH-CHECK:       %permute_B_broadcast_matmul = torch.prim.ListConstruct %permute_B_val_0_broadcast_matmul, %permute_B_val_1_broadcast_matmul : (!torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg1_matrix_b_perm = torch.aten.permute %arg1_matrix_b, %permute_B_broadcast_matmul : !torch.vtensor<[128,256],f32>, !torch.list<int> -> !torch.vtensor<[128,256],f32>
// TORCH-CHECK:       %result_perm = torch.aten.matmul %arg0_matrix_a_perm, %arg1_matrix_b_perm : !torch.vtensor<[4,2,64,128],f32>, !torch.vtensor<[128,256],f32> -> !torch.vtensor<[4,2,64,256],f32>
// TORCH-CHECK:       %permute_C_val_0_broadcast_matmul = torch.constant.int 0
// TORCH-CHECK:       %permute_C_val_1_broadcast_matmul = torch.constant.int 1
// TORCH-CHECK:       %permute_C_val_2_broadcast_matmul = torch.constant.int 2
// TORCH-CHECK:       %permute_C_val_3_broadcast_matmul = torch.constant.int 3
// TORCH-CHECK:       %permute_C_broadcast_matmul = torch.prim.ListConstruct %permute_C_val_0_broadcast_matmul, %permute_C_val_1_broadcast_matmul, %permute_C_val_2_broadcast_matmul, %permute_C_val_3_broadcast_matmul : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %result = torch.aten.permute %result_perm, %permute_C_broadcast_matmul : !torch.vtensor<[4,2,64,256],f32>, !torch.list<int> -> !torch.vtensor<[4,2,64,256],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %result overwrites %result_ : !torch.vtensor<[4,2,64,256],f32>, !torch.tensor<[4,2,64,256],f32>
// TORCH-CHECK:       return
// TORCH-CHECK:     }
// TORCH-CHECK:   }
//
// LINALG-CHECK:    util.func public @main$async(%[[ARG0:.+]]: !hal.buffer_view, %[[ARG1:.+]]: !hal.buffer_view, %[[ARG2:.+]]: !hal.buffer_view, {{.+}}
// LINALG-CHECK:      %[[A:.+]] = hal.tensor.import wait(%{{.+}}) => %[[ARG1]] : !hal.buffer_view -> tensor<4x2x64x128xf32>
// LINALG-CHECK:      %[[B:.+]] = hal.tensor.import wait(%{{.+}}) => %[[ARG2]] : !hal.buffer_view -> tensor<128x256xf32>
// LINALG-CHECK:      %[[B_BROADCAST:.+]] = linalg.generic {{.*}} ins(%[[B]] : tensor<128x256xf32>) outs(%{{.+}} : tensor<4x2x128x256xf32>)
// LINALG-CHECK:      %[[A_COLLAPSE:.+]] = tensor.collapse_shape %[[A]]
// LINALG-CHECK:      %[[B_COLLAPSE:.+]] = tensor.collapse_shape %[[B_BROADCAST]]
// LINALG-CHECK:      %[[OUT:.+]] = linalg.batch_matmul ins(%[[A_COLLAPSE]], %[[B_COLLAPSE]] : tensor<8x64x128xf32>, tensor<8x128x256xf32>) outs(%{{.+}} : tensor<8x64x256xf32>) -> tensor<8x64x256xf32>
// LINALG-CHECK:      %[[OUT_EXPAND:.+]] = tensor.expand_shape %[[OUT]]
// LINALG-CHECK:      %{{.+}} = hal.tensor.alias wait(%{{.+}}) => %[[OUT_EXPAND]] : tensor<4x2x64x256xf32> to %[[ARG0]] : !hal.buffer_view
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

static ErrorObject testMatmulAsmEmitterBroadcast(const std::string &mode) {
  int64_t b1 = 4, b2 = 2, m = 64, k = 128, n = 256;
  auto graph = std::make_shared<Graph>();
  graph->setName("matmul_asm_emitter_broadcast");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  // A has batch dimensions [4, 2]
  auto aT = graph->tensor(TensorAttr()
                              .setName("arg0_matrix_a")
                              .setDim({b1, b2, m, k})
                              .setStride({b2 * m * k, m * k, k, 1}));

  // B has no batch dimensions - will be broadcast
  auto bT = graph->tensor(
      TensorAttr().setName("arg1_matrix_b").setDim({k, n}).setStride({n, 1}));

  auto matmulAttr = MatmulAttr().setName("broadcast_matmul");

  auto cT = graph->matmul(aT, bT, matmulAttr);

  cT->setName("result").setOutput(true);

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

  auto status = testMatmulAsmEmitterBroadcast(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
