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
// TORCH-CHECK:   module @module {
// TORCH-CHECK:     func.func @main(%result_: !torch.tensor<[16,256,64,32],f32>, %arg0_input: !torch.vtensor<[16,256,64,32],f32>, %arg1_add: !torch.vtensor<[1,256,1,1],f32>) attributes {torch.assume_strict_symbolic_shapes} {
// TORCH-CHECK:       %alpha_pointwise_add = torch.constant.int 1
// TORCH-CHECK:       %result = torch.aten.add.Tensor %arg0_input, %arg1_add, %alpha_pointwise_add : !torch.vtensor<[16,256,64,32],f32>, !torch.vtensor<[1,256,1,1],f32>, !torch.int -> !torch.vtensor<[16,256,64,32],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %result overwrites %result_ : !torch.vtensor<[16,256,64,32],f32>, !torch.tensor<[16,256,64,32],f32>
// TORCH-CHECK:       return
// TORCH-CHECK:     }
// TORCH-CHECK:   }
//
// LINALG-CHECK:    util.func public @main$async(%[[ARG0:.+]]: !hal.buffer_view, %[[ARG1:.+]]: !hal.buffer_view, %[[ARG2:.+]]: !hal.buffer_view, {{.+}}
// LINALG-CHECK:      %[[BUF1:.+]] = hal.tensor.import wait(%{{.+}}) => %[[ARG1]] : !hal.buffer_view -> tensor<16x256x64x32xf32>
// LINALG-CHECK:      %[[BUF2:.+]] = hal.tensor.import wait(%{{.+}}) => %[[ARG2]] : !hal.buffer_view -> tensor<1x256x1x1xf32>
// LINALG-CHECK:      %3 = linalg.generic {indexing_maps = [affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>, affine_map<(d0, d1, d2, d3) -> (0, d1, 0, 0)>, affine_map<(d0, d1, d2, d3) -> (d0, d1, d2, d3)>], iterator_types = ["parallel", "parallel", "parallel", "parallel"]} ins(%0, %1 : tensor<16x256x64x32xf32>, tensor<1x256x1x1xf32>) outs(%2 : tensor<16x256x64x32xf32>) {
// LINALG-CHECK:      ^bb0(%in: f32, %in_0: f32, %out: f32):
// LINALG-CHECK:        %6 = arith.addf %in, %in_0 : f32
// LINALG-CHECK:        linalg.yield %6 : f32
// LINALG-CHECK:      } -> tensor<16x256x64x32xf32>
// LINALG-CHECK:      %{{.+}} = hal.tensor.alias wait(%{{.+}}) => %3 : tensor<16x256x64x32xf32> to %[[ARG0]] : !hal.buffer_view
//
// AMDGPU-STATS-CHECK: "dispatch-count": 1
// CPU-STATS-CHECK: "dispatch-count": 1
//
// clang-format on

#include <fusilli.h>

#include <iostream>
#include <memory>

using namespace fusilli;

ErrorObject test_pointwise_asm_emitter_add(const std::string &mode) {
  int64_t n = 16, c = 256, h = 64, w = 32;
  auto graph = std::make_shared<Graph>();
  graph->setName("pointwise_asm_emitter_add");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto X = graph->tensor(TensorAttr()
                             .setName("arg0_input")
                             .setDim({n, c, h, w})
                             .setStride({c * h * w, h * w, w, 1})); // NCHW

  auto B = graph->tensor(TensorAttr()
                             .setName("arg1_add")
                             .setDim({1, c, 1, 1})
                             .setStride({c, 1, 1, 1})); // 1D add

  auto pointwise_attr = PointwiseAttr()
                            .setMode(PointwiseAttr::Mode::ADD)
                            .setName("pointwise_add");

  auto Y = graph->pointwise(X, B, pointwise_attr);

  Y->setName("result").setOutput(true);

  FUSILLI_CHECK_ERROR(graph->validate());

  if (mode == "default") {
    std::cout << FUSILLI_TRY(graph->emitAsm()) << std::endl;
  }

  if (mode == "stats") {
#ifdef FUSILLI_ENABLE_AMDGPU
    Handle handle = FUSILLI_TRY(Handle::create(Backend::GFX942));
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

  auto status = test_pointwise_asm_emitter_add(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
