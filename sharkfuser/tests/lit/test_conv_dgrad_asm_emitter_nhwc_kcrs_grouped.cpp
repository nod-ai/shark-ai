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
// TORCH-CHECK:     func.func @main(%result_: !torch.tensor<[16,64,32,128],f32>, %arg0_dy: !torch.vtensor<[16,64,32,256],f32>, %arg1_w: !torch.vtensor<[256,16,1,1],f32>) attributes {torch.assume_strict_symbolic_shapes} {
// TORCH-CHECK:       %bias_conv_dgrad = torch.constant.none
// TORCH-CHECK:       %transposed_conv_dgrad = torch.constant.bool false
// TORCH-CHECK:       %output_padding_conv_dgrad = torch.prim.ListConstruct  : () -> !torch.list<int>
// TORCH-CHECK:       %groups_conv_dgrad = torch.constant.int 8
// TORCH-CHECK:       %stride_val_0_conv_dgrad = torch.constant.int 1
// TORCH-CHECK:       %stride_val_1_conv_dgrad = torch.constant.int 1
// TORCH-CHECK:       %stride_conv_dgrad = torch.prim.ListConstruct %stride_val_0_conv_dgrad, %stride_val_1_conv_dgrad : (!torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %padding_val_0_conv_dgrad = torch.constant.int 0
// TORCH-CHECK:       %padding_val_1_conv_dgrad = torch.constant.int 0
// TORCH-CHECK:       %padding_conv_dgrad = torch.prim.ListConstruct %padding_val_0_conv_dgrad, %padding_val_1_conv_dgrad : (!torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %dilation_val_0_conv_dgrad = torch.constant.int 1
// TORCH-CHECK:       %dilation_val_1_conv_dgrad = torch.constant.int 1
// TORCH-CHECK:       %dilation_conv_dgrad = torch.prim.ListConstruct %dilation_val_0_conv_dgrad, %dilation_val_1_conv_dgrad : (!torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %permute_DY_val_0_conv_dgrad = torch.constant.int 0
// TORCH-CHECK:       %permute_DY_val_1_conv_dgrad = torch.constant.int 3
// TORCH-CHECK:       %permute_DY_val_2_conv_dgrad = torch.constant.int 1
// TORCH-CHECK:       %permute_DY_val_3_conv_dgrad = torch.constant.int 2
// TORCH-CHECK:       %permute_DY_conv_dgrad = torch.prim.ListConstruct %permute_DY_val_0_conv_dgrad, %permute_DY_val_1_conv_dgrad, %permute_DY_val_2_conv_dgrad, %permute_DY_val_3_conv_dgrad : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg0_dy_perm = torch.aten.permute %arg0_dy, %permute_DY_conv_dgrad : !torch.vtensor<[16,64,32,256],f32>, !torch.list<int> -> !torch.vtensor<[16,256,64,32],f32>
// TORCH-CHECK:       %permute_W_val_0_conv_dgrad = torch.constant.int 0
// TORCH-CHECK:       %permute_W_val_1_conv_dgrad = torch.constant.int 1
// TORCH-CHECK:       %permute_W_val_2_conv_dgrad = torch.constant.int 2
// TORCH-CHECK:       %permute_W_val_3_conv_dgrad = torch.constant.int 3
// TORCH-CHECK:       %permute_W_conv_dgrad = torch.prim.ListConstruct %permute_W_val_0_conv_dgrad, %permute_W_val_1_conv_dgrad, %permute_W_val_2_conv_dgrad, %permute_W_val_3_conv_dgrad : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg1_w_perm = torch.aten.permute %arg1_w, %permute_W_conv_dgrad : !torch.vtensor<[256,16,1,1],f32>, !torch.list<int> -> !torch.vtensor<[256,16,1,1],f32>
// TORCH-CHECK:       %empty_DX_val_0_conv_dgrad = torch.constant.int 16
// TORCH-CHECK:       %empty_DX_val_1_conv_dgrad = torch.constant.int 128
// TORCH-CHECK:       %empty_DX_val_2_conv_dgrad = torch.constant.int 64
// TORCH-CHECK:       %empty_DX_val_3_conv_dgrad = torch.constant.int 32
// TORCH-CHECK:       %empty_DX_conv_dgrad = torch.prim.ListConstruct %empty_DX_val_0_conv_dgrad, %empty_DX_val_1_conv_dgrad, %empty_DX_val_2_conv_dgrad, %empty_DX_val_3_conv_dgrad : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %none_DX_conv_dgrad = torch.constant.none
// TORCH-CHECK:       %dtype_DX_conv_dgrad = torch.constant.int 6
// TORCH-CHECK:       %empty_x_conv_dgrad = torch.aten.empty.memory_format %empty_DX_conv_dgrad, %dtype_DX_conv_dgrad, %none_DX_conv_dgrad, %none_DX_conv_dgrad, %none_DX_conv_dgrad, %none_DX_conv_dgrad : !torch.list<int>, !torch.int, !torch.none, !torch.none, !torch.none, !torch.none -> !torch.vtensor<[16,128,64,32],f32>
// TORCH-CHECK:       %true_conv_dgrad = torch.constant.bool true
// TORCH-CHECK:       %false_conv_dgrad = torch.constant.bool false
// TORCH-CHECK:       %output_mask_conv_dgrad = torch.prim.ListConstruct %true_conv_dgrad, %false_conv_dgrad, %false_conv_dgrad : (!torch.bool, !torch.bool, !torch.bool) -> !torch.list<bool>
// TORCH-CHECK:       %result_perm, %grad_weight_conv_dgrad, %grad_bias_conv_dgrad = torch.aten.convolution_backward %arg0_dy_perm, %empty_x_conv_dgrad, %arg1_w_perm, %bias_conv_dgrad, %stride_conv_dgrad, %padding_conv_dgrad, %dilation_conv_dgrad, %transposed_conv_dgrad, %output_padding_conv_dgrad, %groups_conv_dgrad, %output_mask_conv_dgrad : !torch.vtensor<[16,256,64,32],f32>, !torch.vtensor<[16,128,64,32],f32>, !torch.vtensor<[256,16,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int, !torch.list<bool> -> !torch.vtensor<[16,128,64,32],f32>, !torch.none, !torch.none
// TORCH-CHECK:       %permute_DX_val_0_conv_dgrad = torch.constant.int 0
// TORCH-CHECK:       %permute_DX_val_1_conv_dgrad = torch.constant.int 2
// TORCH-CHECK:       %permute_DX_val_2_conv_dgrad = torch.constant.int 3
// TORCH-CHECK:       %permute_DX_val_3_conv_dgrad = torch.constant.int 1
// TORCH-CHECK:       %permute_DX_conv_dgrad = torch.prim.ListConstruct %permute_DX_val_0_conv_dgrad, %permute_DX_val_1_conv_dgrad, %permute_DX_val_2_conv_dgrad, %permute_DX_val_3_conv_dgrad : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %result = torch.aten.permute %result_perm, %permute_DX_conv_dgrad : !torch.vtensor<[16,128,64,32],f32>, !torch.list<int> -> !torch.vtensor<[16,64,32,128],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %result overwrites %result_ : !torch.vtensor<[16,64,32,128],f32>, !torch.tensor<[16,64,32,128],f32>
// TORCH-CHECK:       return
// TORCH-CHECK:     }
// TORCH-CHECK:   }
//
// LINALG-CHECK:    util.func public @main$async(%[[ARG0:.+]]: !hal.buffer_view, %[[ARG1:.+]]: !hal.buffer_view, %[[ARG2:.+]]: !hal.buffer_view, {{.+}}
// LINALG-CHECK:      %[[BUF1:.+]] = hal.tensor.import wait(%{{.+}}) => %[[ARG1]] : !hal.buffer_view -> tensor<16x64x32x256xf32>
// LINALG-CHECK:      %[[BUF2:.+]] = hal.tensor.import wait(%{{.+}}) => %[[ARG2]] : !hal.buffer_view -> tensor<256x16x1x1xf32>
// LINALG-CHECK:      %[[E1:.+]] = tensor.empty() : tensor<16x256x64x32xf32>
// LINALG-CHECK:      %[[DY_T:.+]] = linalg.transpose ins(%[[BUF1]] : tensor<16x64x32x256xf32>) outs(%[[E1]] : tensor<16x256x64x32xf32>) permutation = [0, 3, 1, 2]
// LINALG-CHECK:      %[[W_E:.+]] = tensor.expand_shape %[[BUF2]] {{\[\[0, 1\], \[2\], \[3\], \[4\]\]}} output_shape [8, 32, 16, 1, 1] : tensor<256x16x1x1xf32> into tensor<8x32x16x1x1xf32>
// LINALG-CHECK:      %[[E2:.+]] = tensor.empty() : tensor<8x16x32x1x1xf32>
// LINALG-CHECK:      %[[FILL:.+]] = linalg.fill {{.*}} outs(%[[E2]]
// LINALG-CHECK:      %[[W_T:.+]] = linalg.generic {{.+}} outs(%[[FILL]] : tensor<8x16x32x1x1xf32>) {{.+}}
// LINALG-CHECK:      %[[E3:.+]] = tensor.empty() : tensor<16x128x64x32xf32>
// LINALG-CHECK:      %[[DY_E:.+]] = tensor.expand_shape %[[DY_T]] {{\[\[0\], \[1, 2\], \[3\], \[4\]\]}} output_shape [16, 8, 32, 64, 32] : tensor<16x256x64x32xf32> into tensor<16x8x32x64x32xf32>
// LINALG-CHECK:      %[[E3_E:.+]] = tensor.expand_shape %[[E3]] {{\[\[0\], \[1, 2\], \[3\], \[4\]\]}} output_shape [16, 8, 16, 64, 32] : tensor<16x128x64x32xf32> into tensor<16x8x16x64x32xf32>
// LINALG-CHECK:      %[[FILL1:.+]] = linalg.fill {{.*}} outs(%[[E3_E]]
// LINALG-CHECK:      %[[OUT:.+]] = linalg.conv_2d_ngchw_gfchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%[[DY_E]], %[[W_T]] : tensor<16x8x32x64x32xf32>, tensor<8x16x32x1x1xf32>) outs(%[[FILL1]] : tensor<16x8x16x64x32xf32>) -> tensor<16x8x16x64x32xf32>
// LINALG-CHECK:      %[[OUTC:.+]] = tensor.collapse_shape %[[OUT]] {{\[\[0\], \[1, 2\], \[3\], \[4\]\]}} : tensor<16x8x16x64x32xf32> into tensor<16x128x64x32xf32>
// LINALG-CHECK:      %[[OUTBUF:.+]] = tensor.empty() : tensor<16x64x32x128xf32>
// LINALG-CHECK:      %[[OUTT:.+]] = linalg.transpose ins(%[[OUTC]] : tensor<16x128x64x32xf32>) outs(%[[OUTBUF]] : tensor<16x64x32x128xf32>) permutation = [0, 2, 3, 1]
// LINALG-CHECK:      %{{.+}} = hal.tensor.alias wait(%{{.+}}) => %[[OUTT]] : tensor<16x64x32x128xf32> to %[[ARG0]] : !hal.buffer_view
//
// TODO(#2594): This should only require a single dispatch.
// AMDGPU-STATS-CHECK: "dispatch-count": 2
// CPU-STATS-CHECK: "dispatch-count": 2
//
// clang-format on

#include <fusilli.h>

#include <cstdint>
#include <iostream>
#include <memory>
#include <string>

using namespace fusilli;

static ErrorObject
testConvDgradAsmEmitterDyNhwcDxNhwcGrouped(const std::string &mode) {
  int64_t n = 16, c = 128, h = 64, w = 32, k = 256, fc = 16, r = 1, s = 1;
  auto graph = std::make_shared<Graph>();
  graph->setName("conv_dgrad_asm_emitter_dy_nhwc_w_kcrs_grouped");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto dyT = graph->tensor(TensorAttr()
                               .setName("arg0_dy")
                               .setDim({n, k, h, w})
                               .setStride({k * h * w, 1, k * w, k})); // NHWC

  auto wT = graph->tensor(TensorAttr()
                              .setName("arg1_w")
                              .setDim({k, fc, r, s})
                              .setStride({fc * r * s, r * s, s, 1})); // KCRS

  auto convDGradAttr = ConvDGradAttr()
                           .setPadding({0, 0})
                           .setStride({1, 1})
                           .setDilation({1, 1})
                           .setName("conv_dgrad");

  auto dxT = graph->convDGrad(dyT, wT, convDGradAttr);

  dxT->setName("result").setOutput(true).setDim({n, c, h, w});

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

  auto status = testConvDgradAsmEmitterDyNhwcDxNhwcGrouped(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
