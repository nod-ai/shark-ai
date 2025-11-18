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
// TORCH-CHECK:     func.func @main(%result_: !torch.tensor<[16,64,32,256],f32>, %arg0_image: !torch.vtensor<[16,64,32,128],f32>, %arg1_filter: !torch.vtensor<[256,128,1,1],f32>) attributes {torch.assume_strict_symbolic_shapes} {
// TORCH-CHECK:       %bias_conv_fprop = torch.constant.none
// TORCH-CHECK:       %transposed_conv_fprop = torch.constant.bool false
// TORCH-CHECK:       %output_padding_conv_fprop = torch.prim.ListConstruct  : () -> !torch.list<int>
// TORCH-CHECK:       %groups_conv_fprop = torch.constant.int 1
// TORCH-CHECK:       %stride_val_0_conv_fprop = torch.constant.int 1
// TORCH-CHECK:       %stride_val_1_conv_fprop = torch.constant.int 1
// TORCH-CHECK:       %stride_conv_fprop = torch.prim.ListConstruct %stride_val_0_conv_fprop, %stride_val_1_conv_fprop : (!torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %padding_val_0_conv_fprop = torch.constant.int 0
// TORCH-CHECK:       %padding_val_1_conv_fprop = torch.constant.int 0
// TORCH-CHECK:       %padding_conv_fprop = torch.prim.ListConstruct %padding_val_0_conv_fprop, %padding_val_1_conv_fprop : (!torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %dilation_val_0_conv_fprop = torch.constant.int 1
// TORCH-CHECK:       %dilation_val_1_conv_fprop = torch.constant.int 1
// TORCH-CHECK:       %dilation_conv_fprop = torch.prim.ListConstruct %dilation_val_0_conv_fprop, %dilation_val_1_conv_fprop : (!torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %permute_X_val_0_conv_fprop = torch.constant.int 0
// TORCH-CHECK:       %permute_X_val_1_conv_fprop = torch.constant.int 3
// TORCH-CHECK:       %permute_X_val_2_conv_fprop = torch.constant.int 1
// TORCH-CHECK:       %permute_X_val_3_conv_fprop = torch.constant.int 2
// TORCH-CHECK:       %permute_X_conv_fprop = torch.prim.ListConstruct %permute_X_val_0_conv_fprop, %permute_X_val_1_conv_fprop, %permute_X_val_2_conv_fprop, %permute_X_val_3_conv_fprop : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg0_image_perm = torch.aten.permute %arg0_image, %permute_X_conv_fprop : !torch.vtensor<[16,64,32,128],f32>, !torch.list<int> -> !torch.vtensor<[16,128,64,32],f32>
// TORCH-CHECK:       %permute_W_val_0_conv_fprop = torch.constant.int 0
// TORCH-CHECK:       %permute_W_val_1_conv_fprop = torch.constant.int 1
// TORCH-CHECK:       %permute_W_val_2_conv_fprop = torch.constant.int 2
// TORCH-CHECK:       %permute_W_val_3_conv_fprop = torch.constant.int 3
// TORCH-CHECK:       %permute_W_conv_fprop = torch.prim.ListConstruct %permute_W_val_0_conv_fprop, %permute_W_val_1_conv_fprop, %permute_W_val_2_conv_fprop, %permute_W_val_3_conv_fprop : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %arg1_filter_perm = torch.aten.permute %arg1_filter, %permute_W_conv_fprop : !torch.vtensor<[256,128,1,1],f32>, !torch.list<int> -> !torch.vtensor<[256,128,1,1],f32>
// TORCH-CHECK:       %result_perm = torch.aten.convolution %arg0_image_perm, %arg1_filter_perm, %bias_conv_fprop, %stride_conv_fprop, %padding_conv_fprop, %dilation_conv_fprop, %transposed_conv_fprop, %output_padding_conv_fprop, %groups_conv_fprop : !torch.vtensor<[16,128,64,32],f32>, !torch.vtensor<[256,128,1,1],f32>, !torch.none, !torch.list<int>, !torch.list<int>, !torch.list<int>, !torch.bool, !torch.list<int>, !torch.int -> !torch.vtensor<[16,256,64,32],f32>
// TORCH-CHECK:       %permute_Y_val_0_conv_fprop = torch.constant.int 0
// TORCH-CHECK:       %permute_Y_val_1_conv_fprop = torch.constant.int 2
// TORCH-CHECK:       %permute_Y_val_2_conv_fprop = torch.constant.int 3
// TORCH-CHECK:       %permute_Y_val_3_conv_fprop = torch.constant.int 1
// TORCH-CHECK:       %permute_Y_conv_fprop = torch.prim.ListConstruct %permute_Y_val_0_conv_fprop, %permute_Y_val_1_conv_fprop, %permute_Y_val_2_conv_fprop, %permute_Y_val_3_conv_fprop : (!torch.int, !torch.int, !torch.int, !torch.int) -> !torch.list<int>
// TORCH-CHECK:       %result = torch.aten.permute %result_perm, %permute_Y_conv_fprop : !torch.vtensor<[16,256,64,32],f32>, !torch.list<int> -> !torch.vtensor<[16,64,32,256],f32>
// TORCH-CHECK:       torch.overwrite.tensor.contents %result overwrites %result_ : !torch.vtensor<[16,64,32,256],f32>, !torch.tensor<[16,64,32,256],f32>
// TORCH-CHECK:       return
// TORCH-CHECK:     }
// TORCH-CHECK:   }
//
// LINALG-CHECK:    util.func public @main$async(%[[ARG0:.+]]: !hal.buffer_view, %[[ARG1:.+]]: !hal.buffer_view, %[[ARG2:.+]]: !hal.buffer_view, {{.+}}
// LINALG-CHECK:      %[[BUF1:.+]] = hal.tensor.import wait(%{{.+}}) => %[[ARG1]] : !hal.buffer_view -> tensor<16x64x32x128xf32>
// LINALG-CHECK:      %[[BUF2:.+]] = hal.tensor.import wait(%{{.+}}) => %[[ARG2]] : !hal.buffer_view -> tensor<256x128x1x1xf32>
// LINALG-CHECK:      %[[BUF1T:.+]] = linalg.transpose ins(%[[BUF1]] : tensor<16x64x32x128xf32>) outs(%{{.+}} : tensor<16x128x64x32xf32>) permutation = [0, 3, 1, 2]
// LINALG-CHECK:      %[[OUT:.+]] = linalg.conv_2d_nchw_fchw {dilations = dense<1> : vector<2xi64>, strides = dense<1> : vector<2xi64>} ins(%[[BUF1T]], %[[BUF2]] : tensor<16x128x64x32xf32>, tensor<256x128x1x1xf32>) outs(%{{.+}} : tensor<16x256x64x32xf32>) -> tensor<16x256x64x32xf32>
// LINALG-CHECK:      %[[OUTT:.+]] = linalg.transpose ins(%[[OUT]] : tensor<16x256x64x32xf32>) outs(%{{.+}} : tensor<16x64x32x256xf32>) permutation = [0, 2, 3, 1]
// LINALG-CHECK:      %{{.+}} = hal.tensor.alias wait(%{{.+}}) => %[[OUTT]] : tensor<16x64x32x256xf32> to %[[ARG0]] : !hal.buffer_view
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

static ErrorObject testConvAsmEmitterXNhwcWKrsc(const std::string &mode) {
  int64_t n = 16, c = 128, h = 64, w = 32, k = 256, r = 1, s = 1;
  auto graph = std::make_shared<Graph>();
  graph->setName("conv_asm_emitter_x_nhwc_w_krsc");
  graph->setIODataType(DataType::Float).setComputeDataType(DataType::Float);

  auto xT = graph->tensor(TensorAttr()
                              .setName("arg0_image")
                              .setDim({n, c, h, w})
                              .setStride({c * h * w, 1, c * w, c})); // NHWC

  auto wT = graph->tensor(TensorAttr()
                              .setName("arg1_filter")
                              .setDim({k, c, r, s})
                              .setStride({c * r * s, 1, c * s, c})); // KRSC

  auto convAttr = ConvFPropAttr()
                      .setPadding({0, 0})
                      .setStride({1, 1})
                      .setDilation({1, 1})
                      .setName("conv_fprop");

  auto yT = graph->convFProp(xT, wT, convAttr);

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

  auto status = testConvAsmEmitterXNhwcWKrsc(mode);
  if (isError(status)) {
    std::cerr << "Test failed: " << status << std::endl;
    return 1;
  }
  return 0;
}
