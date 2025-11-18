// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// RUN: %{TEST_EXE} | iree-opt --verify-roundtrip
// RUN: %{TEST_EXE} | FileCheck %s --check-prefix=TORCH-CHECK
// RUN: %{TEST_EXE} | iree-compile - --compile-to=input | \
// RUN:             FileCheck %s --check-prefix=LINALG-CHECK
// RUN: %{TEST_EXE} stats | FileCheck %s --check-prefix=CPU-STATS-CHECK

// clang-format off
//
// TORCH-CHECK:   module @module {
// TORCH-CHECK:     func.func @main(%result_: !torch.tensor<[256,128,1,1],f32>, %arg0_dy: !torch.vtensor<[16,64,32,256],f32>, %arg1_x: !torch.vtensor<[16,64,32,128],f32>) attributes {torch.assume_strict_symbolic_shapes} {
// TORCH-CHECK:       %bias_conv_wgrad = torch.constant.none
// TORCH-CHECK:       %transposed_conv_wgrad = torch.constant.bool false
// TORCH-CHECK:       %output_padding_conv_wgrad = torch.prim.ListConstruct  : () -> !torch.list<int>
// TORCH-CHECK:       %groups_conv_wgrad = torch.constant.int 1
// TORCH-CHECK:       %stride_val_0_conv_wgrad = torch.constant.int 1
// TORCH-CHECK:       %stride_val_1_conv_wgrad = torch.constant.int 1
// TORCH-CHECK:       %stride_conv_wgrad = torch.prim.ListConstruct %stride_val_0_conv_wgrad, %stride_val_1_conv_wgrad : (!torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %padding_val_0_conv_wgrad = torch.constant.int 0
// TORCH-CHECK:       %padding_val_1_conv_wgrad = torch.constant.int 0
// TORCH-CHECK:       %padding_conv_wgrad = torch.prim.ListConstruct %padding_val_0_conv_wgrad, %padding_val_1_conv_wgrad : (!torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %dilation_val_0_conv_wgrad = torch.constant.int 1
// TORCH-CHECK:       %dilation_val_1_conv_wgrad = torch.constant.int 1
// TORCH-CHECK:       %dilation_conv_wgrad = torch.prim.ListConstruct %dilation_val_0_conv_wgrad, %dilation_val_1_conv_wgrad : (!torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %permute_DY_val_0_conv_wgrad = torch.constant.int 0
// TORCH-CHECK:       %permute_DY_val_1_conv_wgrad = torch.constant.int 3
// TORCH-CHECK:       %permute_DY_val_2_conv_wgrad = torch.constant.int 1
// TORCH-CHECK:       %permute_DY_val_3_conv_wgrad = torch.constant.int 2
// TORCH-CHECK:       %permute_DY_conv_wgrad = torch.prim.ListConstruct %permute_DY_val_0_conv_wgrad, %permute_DY_val_1_conv_wgrad, %permute_DY_val_2_conv_wgrad, %permute_DY_val_3_conv_wgrad : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg0_dy_perm = torch.aten.permute %arg0_dy, %permute_DY_conv_wgrad : !torch.vtensor<[16,64,32,256],f32>, !torch.list<int> -> !torch.vtensor<[16,256,64,32],f32>
// TORCH-CHECK:       %permute_X_val_0_conv_wgrad = torch.constant.int 0
// TORCH-CHECK:       %permute_X_val_1_conv_wgrad = torch.constant.int 3
// TORCH-CHECK:       %permute_X_val_2_conv_wgrad = torch.constant.int 1
// TORCH-CHECK:       %permute_X_val_3_conv_wgrad = torch.constant.int 2
// TORCH-CHECK:       %permute_X_conv_wgrad = torch.prim.ListConstruct %permute_X_val_0_conv_wgrad, %permute_X_val_1_conv_wgrad, %permute_X_val_2_conv_wgrad, %permute_X_val_3_conv_wgrad : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg1_x_perm = torch.aten.permute %arg1_x, %permute_X_conv_wgrad : !torch.vtensor<[16,64,32,128],f32>, !torch.list<int> -> !torch.vtensor<[16,128,64,32],f32>
// TORCH-CHECK:       %empty_DW_val_0_conv_wgrad = torch.constant.int 256
// TORCH-CHECK:       %empty_DW_val_1_conv_wgrad = torch.constant.int 128
// TORCH-CHECK:       %empty_DW_val_2_conv_wgrad = torch.constant.int 1
// TORCH-CHECK:       %empty_DW_val_3_conv_wgrad = torch.constant.int 1
// TORCH-CHECK:       %empty_DW_conv_wgrad = torch.prim.ListConstruct %empty_DW_val_0_conv_wgrad, %empty_DW_val_1_conv_wgrad, %empty_DW_val_2_conv_wgrad, %empty_DW_val_3_conv_wgrad : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %none_DW_conv_wgrad = torch.constant.none
// TORCH-CHECK:       %dtype_DW_conv_wgrad = torch.constant.int 6
// TORCH-CHECK:       %empty_w_conv_wgrad = torch.aten.empty.memory_format %empty_DW_conv_wgrad, %dtype_DW_conv_wgrad, %none_DW_conv_wgrad, %none_DW_conv_wgrad, %none_DW_conv_wgrad, %none_DW_conv_wgrad : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[256,128,1,1],f32>
// TORCH-CHECK:       %true_conv_wgrad = torch.constant.bool true
// TORCH-CHECK:       %false_conv_wgrad = torch.constant.bool false
// TORCH-CHECK:       %output_mask_conv_wgrad = torch.prim.ListConstruct %false_conv_wgrad, %true_conv_wgrad, %false_conv_wgrad : (!torch.bool, !torch.bool, !torch.bool) -> !torch.list<bool>
// TORCH-CHECK:       %grad_input_conv_wgrad, %result_perm, %grad_bias_conv_wgrad = torch.aten.convolution_backward %arg0_dy_perm, %arg1_x_perm, %empty_w_conv_wgrad, %bias_conv_wgrad, %stride_conv_wgrad, %padding_conv_wgrad, %dilation_conv_wgrad, %transposed_conv_wgrad, %output_padding_conv_wgrad, %groups_conv_wgrad, %output_mask_conv_wgrad : !torch.vtensor<[16,256,64,32],f32>, !torch.vtensor<[16,128,64,32],f32>, !torch.vtensor<[256,128,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int, !torch.list<bool> -> !torch.none, !torch.vtensor<[256,128,1,1],f32>, !torch.none
// TORCH-CHECK:       %permute_DW_val_0_conv_wgrad = torch.constant.int 0
// TORCH-CHECK:       %permute_DW_val_1_conv_wgrad = torch.constant.int 1
// TORCH-CHECK:       %permute_DW_val_2_conv_wgrad = torch.constant.int 2
// TORCH-CHECK:       %permute_DW_val_3_conv_wgrad = torch.constant.int 3
// TORCH-CHECK:       %permute_DW_conv_wgrad = torch.prim.ListConstruct %permute_DW_val_0_conv_wgrad, %permute_DW_val_1_conv_wgrad, %permute_DW_val_2_conv_wgrad, %permute_DW_val_3_conv_wgrad : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %result = torch.aten.permute %result_perm, %permute_DW_conv_wgrad : !torch.vtensor<[256,128,1,1],f32>, !torch.list<int> -> !torch.vtensor<[256,128,1,1],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %result overwrites %result_ : !torch.vtensor<[256,128,1,1],f32>, !torch.tensor<[256,128,1,1],f32>
// TORCH-CHECK:       return
// TORCH-CHECK:     }
// TORCH-CHECK:   }
//
// LINALG-CHECK:    util.func public @main$async(%[[ARG0:.+]]: !hal.buffer_view, %[[ARG1:.+]]: !hal.buffer_view, %[[ARG2:.+]]: !hal.buffer_view, {{.+}}
// LINALG-CHECK:      %[[BUF1:.+]] = hal.tensor.import wait(%{{.+}}) => %[[ARG1]] : !hal.buffer_view -> tensor<16x64x32x256xf32>
// LINALG-CHECK:      %[[BUF2:.+]] = hal.tensor.import wait(%{{.+}}) => %[[ARG2]] : !hal.buffer_view -> tensor<16x64x32x128xf32>
// LINALG-CHECK:      %[[E2:.+]] = tensor.empty() : tensor<128x16x64x32xf32>
// LINALG-CHECK:      %[[X_T:.+]] = linalg.transpose ins(%[[BUF2]] : tensor<16x64x32x128xf32>) outs(%[[E2]] : tensor<128x16x64x32xf32>) permutation = [3, 0, 1, 2]
// LINALG-CHECK:      %[[E1:.+]] = tensor.empty() : tensor<256x16x64x32xf32>
// LINALG-CHECK:      %[[DY_T:.+]] = linalg.transpose ins(%[[BUF1]] : tensor<16x64x32x256xf32>) outs(%[[E1]] : tensor<256x16x64x32xf32>) permutation = [3, 0, 1, 2]
// LINALG-CHECK:      %[[EOUT:.+]] = tensor.empty() : tensor<128x256x1x1xf32>
// LINALG-CHECK:      %[[OUT:.+]] = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%{{.+}}, %{{.+}} : tensor<128x16x64x32xf32>, tensor<256x16x64x32xf32>) outs(%{{.+}} : tensor<128x256x1x1xf32>) -> tensor<128x256x1x1xf32>
// LINALG-CHECK:      %[[OUTT:.+]] = linalg.transpose ins(%[[OUT]] : tensor<128x256x1x1xf32>) outs(%{{.+}} : tensor<256x128x1x1xf32>) permutation = [1, 0, 2, 3]
// LINALG-CHECK:      %{{.+}} = hal.tensor.alias wait(%{{.+}}) => %[[OUTT]] : tensor<256x128x1x1xf32> to %[[ARG0]] : !hal.buffer_view
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

static ErrorObject testConvWgradAsmEmitterDyNhwcXNhwc(const std::string &mode) {
  int64_t n = 16, c = 128, h = 64, w = 32, k = 256, r = 1, s = 1;
  auto graph = std::make_shared<Graph>();
  graph->setName("conv_wgrad_asm_emitter_dy_nhwc_x_nhwc");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto dyT = graph->tensor(TensorAttr()
                               .setName("arg0_dy")
                               .setDim({n, k, h, w})
                               .setStride({k * h * w, 1, k * w, k})); // NHWC

  auto xT = graph->tensor(TensorAttr()
                              .setName("arg1_x")
                              .setDim({n, c, h, w})
                              .setStride({c * h * w, 1, c * w, c})); // NHWC

  auto convWGradAttr = ConvWGradAttr()
                           .setPadding({0, 0})
                           .setStride({1, 1})
                           .setDilation({1, 1})
                           .setName("conv_wgrad");

  auto dwT = graph->convWGrad(dyT, xT, convWGradAttr);

  dwT->setName("result").setOutput(true).setDim({k, c, r, s});

  FUSILLI_CHECK_ERROR(graph->validate());

  if (mode == "default") {
    std::cout << FUSILLI_TRY(graph->emitAsm()) << std::endl;
  }

  if (mode == "stats") {
    Handle handle = FUSILLI_TRY(Handle::create(Backend::CPU));
    FUSILLI_CHECK_ERROR(graph->compile(handle, /*remove=*/true));
    std::cout << FUSILLI_TRY(graph->readCompilationCacheFile(
                     CachedAssetsType::Statistics))
              << std::endl;
  }

  return ok();
}

int main(int argc, char **argv) {
  std::string mode = (argc > 1) ? argv[1] : "default";

  auto status = testConvWgradAsmEmitterDyNhwcXNhwc(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
