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
// Test matmul with non-contiguous layout (arbitrary strides).
// A and C have non-contiguous layouts, requiring permutations.
// This tests the general permutation support for any stride pattern.
//
// TORCH-CHECK:   module @module {
// TORCH-CHECK:     func.func @main(%result_: !torch.tensor<[16,256,64],f32>, %arg0_matrix_a: !torch.vtensor<[16,128,64],f32>, %arg1_matrix_b: !torch.vtensor<[16,128,256],f32>) attributes {torch.assume_strict_symbolic_shapes} {
// TORCH-CHECK:       %permute_A_val_0_matmul_nc = torch.constant.int 0
// TORCH-CHECK:       %permute_A_val_1_matmul_nc = torch.constant.int 2
// TORCH-CHECK:       %permute_A_val_2_matmul_nc = torch.constant.int 1
// TORCH-CHECK:       %permute_A_matmul_nc = torch.prim.ListConstruct %permute_A_val_0_matmul_nc, %permute_A_val_1_matmul_nc, %permute_A_val_2_matmul_nc : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg0_matrix_a_perm = torch.aten.permute %arg0_matrix_a, %permute_A_matmul_nc : !torch.vtensor<[16,128,64],f32>, !torch.list<int> -> !torch.vtensor<[16,64,128],f32>
// TORCH-CHECK:       %permute_B_val_0_matmul_nc = torch.constant.int 0
// TORCH-CHECK:       %permute_B_val_1_matmul_nc = torch.constant.int 1
// TORCH-CHECK:       %permute_B_val_2_matmul_nc = torch.constant.int 2
// TORCH-CHECK:       %permute_B_matmul_nc = torch.prim.ListConstruct %permute_B_val_0_matmul_nc, %permute_B_val_1_matmul_nc, %permute_B_val_2_matmul_nc : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg1_matrix_b_perm = torch.aten.permute %arg1_matrix_b, %permute_B_matmul_nc : !torch.vtensor<[16,128,256],f32>, !torch.list<int> -> !torch.vtensor<[16,128,256],f32>
// TORCH-CHECK:       %result_perm = torch.aten.matmul %arg0_matrix_a_perm, %arg1_matrix_b_perm : !torch.vtensor<[16,64,128],f32>, !torch.vtensor<[16,128,256],f32> -> !torch.vtensor<[16,64,256],f32>
// TORCH-CHECK:       %permute_C_val_0_matmul_nc = torch.constant.int 0
// TORCH-CHECK:       %permute_C_val_1_matmul_nc = torch.constant.int 2
// TORCH-CHECK:       %permute_C_val_2_matmul_nc = torch.constant.int 1
// TORCH-CHECK:       %permute_C_matmul_nc = torch.prim.ListConstruct %permute_C_val_0_matmul_nc, %permute_C_val_1_matmul_nc, %permute_C_val_2_matmul_nc : (!torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %result = torch.aten.permute %result_perm, %permute_C_matmul_nc : !torch.vtensor<[16,64,256],f32>, !torch.list<int> -> !torch.vtensor<[16,256,64],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %result overwrites %result_ : !torch.vtensor<[16,256,64],f32>, !torch.tensor<[16,256,64],f32>
// TORCH-CHECK:       return
// TORCH-CHECK:     }
// TORCH-CHECK:   }
//
// LINALG-CHECK:    util.func public @main$async(%[[ARG0:.+]]: !hal.buffer_view, %[[ARG1:.+]]: !hal.buffer_view, %[[ARG2:.+]]: !hal.buffer_view, {{.+}}
// LINALG-CHECK:      %[[A:.+]] = hal.tensor.import wait(%{{.+}}) => %[[ARG1]] : !hal.buffer_view -> tensor<16x128x64xf32>
// LINALG-CHECK:      %[[B:.+]] = hal.tensor.import wait(%{{.+}}) => %[[ARG2]] : !hal.buffer_view -> tensor<16x128x256xf32>
// LINALG-CHECK:      %[[AT:.+]] = linalg.transpose ins(%[[A]] : tensor<16x128x64xf32>) outs(%{{.+}} : tensor<16x64x128xf32>) permutation = [0, 2, 1]
// LINALG-CHECK:      %[[OUT:.+]] = linalg.batch_matmul ins(%[[AT]], %[[B]] : tensor<16x64x128xf32>, tensor<16x128x256xf32>) outs(%{{.+}} : tensor<16x64x256xf32>) -> tensor<16x64x256xf32>
// LINALG-CHECK:      %[[OUTT:.+]] = linalg.transpose ins(%[[OUT]] : tensor<16x64x256xf32>) outs(%{{.+}} : tensor<16x256x64xf32>) permutation = [0, 2, 1]
// LINALG-CHECK:      %{{.+}} = hal.tensor.alias wait(%{{.+}}) => %[[OUTT]] : tensor<16x256x64xf32> to %[[ARG0]] : !hal.buffer_view
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

static ErrorObject testMatmulAsmEmitterNoncontiguous(const std::string &mode) {
  int64_t batch = 16, m = 64, k = 128, n = 256;
  auto graph = std::make_shared<Graph>();
  graph->setName("matmul_asm_emitter_noncontiguous");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  // A has non-contiguous layout with last dimension having stride 1
  auto aT = graph->tensor(TensorAttr()
                              .setName("arg0_matrix_a")
                              .setDim({batch, m, k})
                              .setStride({m * k, 1, m})); // Non-contiguous

  // B is contiguous
  auto bT = graph->tensor(TensorAttr()
                              .setName("arg1_matrix_b")
                              .setDim({batch, k, n})
                              .setStride({k * n, n, 1}));

  auto matmulAttr = MatmulAttr().setName("matmul_nc");

  auto cT = graph->matmul(aT, bT, matmulAttr);

  // Manually specify non-contiguous output stride
  // Output logical dims: [batch, m, n] = [16, 64, 256]
  // Non-contiguous stride: last dimension has stride 1, similar to A
  cT->setStride({n * m, 1, m}); // [16384, 1, 64]

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

  auto status = testMatmulAsmEmitterNoncontiguous(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
