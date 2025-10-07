// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <CLI/CLI.hpp>
#include <cstdint>
#include <limits>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace fusilli;

// For CLI11 Range Validators
const auto NonNegativeInteger =
    CLI::Range(int64_t{0}, std::numeric_limits<int64_t>::max());
const auto PositiveInteger =
    CLI::Range(int64_t{1}, std::numeric_limits<int64_t>::max());
const auto ValidConvLayout = CLI::IsMember({"NCHW", "NHWC", "NCDHW", "NDHWC"});

ErrorObject benchmark_conv_fprop(int64_t n, int64_t c, int64_t d, int64_t h,
                                 int64_t w, int64_t k, int64_t z, int64_t y,
                                 int64_t x, int64_t t, int64_t u, int64_t v,
                                 int64_t o, int64_t p, int64_t q, int64_t m,
                                 int64_t l, int64_t j, std::string_view I,
                                 std::string_view O, std::string_view F,
                                 int64_t S, int64_t iter) {
#ifdef FUSILLI_ENABLE_AMDGPU
  Handle handle = FUSILLI_TRY(Handle::create(Backend::GFX942));
#else
  Handle handle = FUSILLI_TRY(Handle::create(Backend::CPU));
#endif

  // Build attributes based on 2D/3D conv and layouts.
  auto xDims = (S == 2) ? std::vector<int64_t>{n, c, h, w}
                        : std::vector<int64_t>{n, c, d, h, w};
  auto wDims = (S == 2) ? std::vector<int64_t>{k, c, y, x}
                        : std::vector<int64_t>{k, c, z, y, x};
  auto xStride =
      (S == 2)
          ? (I == "NCHW" ? std::vector<int64_t>{c * h * w, h * w, w, 1}
                         : std::vector<int64_t>{c * h * w, 1, c * w, c})
          : (I == "NCDHW"
                 ? std::vector<int64_t>{c * d * h * w, d * h * w, h * w, w, 1}
                 : std::vector<int64_t>{c * d * h * w, 1, c * h * w, w * c, c});
  auto wStride =
      (S == 2)
          ? (F == "NCHW" ? std::vector<int64_t>{c * y * x, y * x, x, 1}
                         : std::vector<int64_t>{c * y * x, 1, x * c, c})
          : (F == "NCDHW"
                 ? std::vector<int64_t>{c * z * y * x, z * y * x, y * x, x, 1}
                 : std::vector<int64_t>{c * z * y * x, 1, y * x * c, x * c, c});
  auto convStride =
      (S == 2) ? std::vector<int64_t>{u, v} : std::vector<int64_t>{t, u, v};
  auto convPadding =
      (S == 2) ? std::vector<int64_t>{p, q} : std::vector<int64_t>{o, p, q};
  auto convDilation =
      (S == 2) ? std::vector<int64_t>{l, j} : std::vector<int64_t>{m, l, j};

  // Build graph for the given handle (device), validate and compile it.
  auto graph = std::make_shared<Graph>();
  graph->setName("benchmark_conv_fprop");
  graph->setIODataType(DataType::Half).setComputeDataType(DataType::Float);

  auto X = graph->tensor(
      TensorAttr().setName("input").setDim(xDims).setStride(xStride));

  auto W = graph->tensor(
      TensorAttr().setName("filter").setDim(wDims).setStride(wStride));

  auto conv_attr = ConvFPropAttr()
                       .setStride(convStride)
                       .setPadding(convPadding)
                       .setDilation(convDilation)
                       .setName("conv_fprop");

  auto Y = graph->convFProp(X, W, conv_attr);
  Y->setOutput(true);

  // Validate, infer missing properties
  FUSILLI_CHECK_ERROR(graph->validate());

  // Compile
  FUSILLI_CHECK_ERROR(graph->compile(handle, /*remove=*/true));

  // Allocate input buffer.
  auto xBuf = std::make_shared<Buffer>(FUSILLI_TRY(Buffer::allocate(
      handle,
      /*shape=*/castToSizeT(X->getPhysicalDim()),
      /*data=*/std::vector<half>(X->getVolume(), half(1.0f)))));

  // Allocate weight buffer.
  auto wBuf = std::make_shared<Buffer>(FUSILLI_TRY(Buffer::allocate(
      handle,
      /*shape=*/castToSizeT(W->getPhysicalDim()),
      /*data=*/std::vector<half>(W->getVolume(), half(1.0f)))));

  // Allocate output buffer.
  auto yBuf = std::make_shared<Buffer>(FUSILLI_TRY(Buffer::allocate(
      handle,
      /*shape=*/castToSizeT(Y->getPhysicalDim()),
      /*data=*/std::vector<half>(Y->getVolume(), half(0.0f)))));

  // Create variant pack.
  const std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {X, xBuf},
          {W, wBuf},
          {Y, yBuf},
      };

  // Execute graph a few times.
  for (size_t i = 0; i < iter; i++)
    FUSILLI_CHECK_ERROR(graph->execute(variantPack));

  return ok();
}

int main(int argc, char **argv) {
  CLI::App mainApp{"Fusilli Benchmark Driver"};
  mainApp.require_subcommand(1);

  int64_t iter;
  mainApp.add_option("--iter,-i", iter, "Benchmark iterations")
      ->required()
      ->check(PositiveInteger);

  // Conv flags are kept in sync with MIOpen's ConvDriver:
  // https://github.com/ROCm/rocm-libraries/blob/db0544fb61f2c7bd5a86dce98d4963420c1c741a/projects/miopen/driver/conv_driver.hpp#L878
  CLI::App *convApp =
      mainApp.add_subcommand("conv", "Fusilli Benchmark Forward Convolution");
  int64_t n, c, d, h, w, k, z, y, x, t, u, v, o, p, q, m, l, j, S;
  std::string I, F, O;
  convApp->add_option("--batchsize,-n", n, "Input batch size")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--in_channels,-c", c, "Input channels")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--in_d", d, "Input depth")
      ->default_val("-1")
      ->check(PositiveInteger);
  convApp->add_option("--in_h,-H", h, "Input height")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--in_w,-W", w, "Input width")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--out_channels,-k", k, "Output channels")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--fil_d", z, "Filter depth")
      ->default_val("-1")
      ->check(PositiveInteger);
  convApp->add_option("--fil_h,-y", y, "Filter height")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--fil_w,-x", x, "Filter width")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--conv_stride_d", t, "Conv stride depth")
      ->default_val("-1")
      ->check(PositiveInteger);
  convApp->add_option("--conv_stride_h,-u", u, "Conv stride height")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--conv_stride_w,-v", v, "Conv stride width")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--pad_d", o, "Conv padding depth")
      ->default_val("-1")
      ->check(NonNegativeInteger);
  convApp->add_option("--pad_h,-p", p, "Conv padding height")
      ->required()
      ->check(NonNegativeInteger);
  convApp->add_option("--pad_w,-q", q, "Conv padding width")
      ->required()
      ->check(NonNegativeInteger);
  convApp->add_option("--dilation_d", m, "Conv dilation depth")
      ->default_val("-1")
      ->check(PositiveInteger);
  convApp->add_option("--dilation_h,-l", l, "Conv dilation height")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--dilation_w,-j", j, "Conv dilation width")
      ->required()
      ->check(PositiveInteger);
  convApp->add_option("--in_layout", I, "Input layout")
      ->required()
      ->check(ValidConvLayout);
  convApp->add_option("--fil_layout", F, "Filter layout")
      ->required()
      ->check(ValidConvLayout);
  convApp->add_option("--out_layout", O, "Output layout")
      ->required()
      ->check(ValidConvLayout);
  convApp
      ->add_option("--spatial_dim", S,
                   "Num spatial dimensions (2 for conv2d, 3 for conv3d)")
      ->required()
      ->check(CLI::IsMember({2, 3}));

  CLI11_PARSE(mainApp, argc, argv);

  // Additional validation of convApp options (apart from default CLI checks)
  if (S == 2) {
    // Reject 3D layouts for 2D conv
    if (I.size() != 4 || F.size() != 4 || O.size() != 4) {
      std::cerr << "Detected at least one invalid {input, filter, output} "
                   "layout for 2D convolution."
                << std::endl;
      return 1;
    }
  }
  if (S == 3) {
    // Reject 2D layouts for 3D conv
    if (I.size() != 5 || F.size() != 5 || O.size() != 5) {
      std::cerr << "Detected at least one invalid {input, filter, output} "
                   "layout for 3D convolution."
                << std::endl;
      return 1;
    }
    // Reject default (sentinel) values for optional args in 3D conv
    if (d == -1 || z == -1 || t == -1 || o == -1 || m == -1) {
      std::cerr << "Detected at least one of {in_d, fil_d, conv_stride_d, "
                   "pad_d, dilation_d} that was not set for 3D convolution."
                << std::endl;
      return 1;
    }
  }

  std::cout << "Fusilli Benchmark started..." << std::endl;

  if (convApp->parsed()) {
    auto status = benchmark_conv_fprop(n, c, d, h, w, k, z, y, x, t, u, v, o, p,
                                       q, m, l, j, I, O, F, S, iter);
    if (isError(status)) {
      std::cerr << "Fusilli Benchmark failed: " << status << std::endl;
      return 1;
    }
  }

  std::cout << "Fusilli Benchmark complete!" << std::endl;
  return 0;
}
