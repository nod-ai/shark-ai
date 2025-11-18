// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <CLI/CLI.hpp>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <format>
#include <iostream>
#include <limits>
#include <memory>
#include <string_view>
#include <unordered_map>
#include <vector>

using namespace fusilli;

// For CLI11 Option Validators
const auto kIsNonNegativeInteger =
    CLI::Range(int64_t{0}, std::numeric_limits<int64_t>::max());
const auto kIsPositiveInteger =
    CLI::Range(int64_t{1}, std::numeric_limits<int64_t>::max());
const auto kIsValidConvLayout =
    CLI::IsMember({"NCHW", "NHWC", "NCDHW", "NDHWC"});

static ErrorObject
benchmarkConvFprop(int64_t n, int64_t c, int64_t d, int64_t h, int64_t w,
                   int64_t g, int64_t k, int64_t z, int64_t y, int64_t x,
                   int64_t t, int64_t u, int64_t v, int64_t o, int64_t p,
                   int64_t q, int64_t m, int64_t l, int64_t j,
                   std::string_view imageLayout, std::string_view outputLayout,
                   std::string_view filterLayout, int64_t s, bool bias,
                   int64_t iter, DataType convIOType) {
#ifdef FUSILLI_ENABLE_AMDGPU
  Handle handle = FUSILLI_TRY(Handle::create(Backend::AMDGPU));
#else
  Handle handle = FUSILLI_TRY(Handle::create(Backend::CPU));
#endif

  // Calculate filter channels
  auto fc = c / g;

  // Build attributes based on 2D/3D conv and layouts.
  auto xDims = (s == 2) ? std::vector<int64_t>{n, c, h, w}
                        : std::vector<int64_t>{n, c, d, h, w};
  auto wDims = (s == 2) ? std::vector<int64_t>{k, fc, y, x}
                        : std::vector<int64_t>{k, fc, z, y, x};
  auto xStride =
      (imageLayout == "NCHW" || imageLayout == "NCDHW")
          ? generateStrideFromDim(xDims, getContiguousStrideOrder(xDims.size()))
          : generateStrideFromDim(xDims,
                                  getChannelsLastStrideOrder(xDims.size()));
  auto wStride =
      (filterLayout == "NCHW" || filterLayout == "NCDHW")
          ? generateStrideFromDim(wDims, getContiguousStrideOrder(wDims.size()))
          : generateStrideFromDim(wDims,
                                  getChannelsLastStrideOrder(wDims.size()));
  auto convStride =
      (s == 2) ? std::vector<int64_t>{u, v} : std::vector<int64_t>{t, u, v};
  auto convPadding =
      (s == 2) ? std::vector<int64_t>{p, q} : std::vector<int64_t>{o, p, q};
  auto convDilation =
      (s == 2) ? std::vector<int64_t>{l, j} : std::vector<int64_t>{m, l, j};
  auto biasDims = (s == 2) ? std::vector<int64_t>{1, k, 1, 1}
                           : std::vector<int64_t>{1, k, 1, 1, 1};
  auto biasStride =
      (imageLayout == "NCHW" || imageLayout == "NCDHW")
          ? generateStrideFromDim(biasDims,
                                  getContiguousStrideOrder(biasDims.size()))
          : generateStrideFromDim(biasDims,
                                  getChannelsLastStrideOrder(biasDims.size()));

  // Build graph for the given handle (device), validate and compile it.
  Graph graph;

  // Set unique name to prevent concurrent invocations of the benchmark driver
  // from polluting the same cache files leading to race conditions.
  auto graphName =
      std::format("benchmark_conv_fprop_n{}_c{}_d{}_h{}_w{}_g{}_k{"
                  "}_z{}_y{}_x{}_t{}_u{}_v{}_o{}"
                  "_p{}_q{}_m{}_l{}_j{}_S{}_I{}_O{}_F{}_bias{}",
                  n, c, d, h, w, g, k, z, y, x, t, u, v, o, p, q, m, l, j, s,
                  imageLayout, outputLayout, filterLayout, bias);
  graph.setName(graphName);

  // Types on the graph are kept at fp32 but we explicitly set
  // individual tensor types below based on configuration. These
  // types hence don't matter much and are used only to infer
  // missing type annotations on tensors.
  graph.setIODataType(DataType::Float)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  auto xT = graph.tensor(TensorAttr()
                             .setName("input")
                             .setDim(xDims)
                             .setStride(xStride)
                             .setDataType(convIOType));

  auto wT = graph.tensor(TensorAttr()
                             .setName("filter")
                             .setDim(wDims)
                             .setStride(wStride)
                             .setDataType(convIOType));

  auto convAttr = ConvFPropAttr()
                      .setStride(convStride)
                      .setPadding(convPadding)
                      .setDilation(convDilation)
                      .setName("conv_fprop");

  auto yT = graph.convFProp(xT, wT, convAttr);
  yT->setDataType(convIOType);

  std::shared_ptr<TensorAttr> bT;
  if (bias) {
    bT = graph.tensor(TensorAttr()
                          .setName("bias")
                          .setDim(biasDims)
                          .setStride(biasStride)
                          .setDataType(convIOType));
    auto biasAttr = PointwiseAttr().setMode(PointwiseAttr::Mode::ADD);
    yT = graph.pointwise(yT, bT, biasAttr);
    yT->setDataType(convIOType);
  }
  yT->setOutput(true).setDataType(convIOType);

  // Validate, infer missing properties
  FUSILLI_CHECK_ERROR(graph.validate());

  // Compile
  FUSILLI_CHECK_ERROR(graph.compile(handle, /*remove=*/true));

  // Allocate input, weight and output buffers.
  auto xBuf = FUSILLI_TRY(allocateBufferOfType(handle, xT, convIOType, 1.0f));
  auto wBuf = FUSILLI_TRY(allocateBufferOfType(handle, wT, convIOType, 1.0f));
  auto yBuf = FUSILLI_TRY(allocateBufferOfType(handle, yT, convIOType, 0.0f));

  // Create variant pack.
  std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {xT, xBuf},
          {wT, wBuf},
          {yT, yBuf},
      };

  if (bias) {
    auto bBuf = FUSILLI_TRY(allocateBufferOfType(handle, bT, convIOType, 1.0f));
    variantPack.insert({bT, bBuf});
  }

  // Execute graph a few times.
  for (size_t i = 0; i < iter; i++)
    FUSILLI_CHECK_ERROR(graph.execute(handle, variantPack));

  return ok();
}

static ErrorObject
benchmarkConvWGrad(int64_t n, int64_t c, int64_t d, int64_t h, int64_t w,
                   int64_t g, int64_t k, int64_t z, int64_t y, int64_t x,
                   int64_t t, int64_t u, int64_t v, int64_t o, int64_t p,
                   int64_t q, int64_t m, int64_t l, int64_t j,
                   std::string_view imageLayout, std::string_view outputLayout,
                   std::string_view filterLayout, int64_t s, int64_t iter,
                   DataType convIOType) {
#ifdef FUSILLI_ENABLE_AMDGPU
  Handle handle = FUSILLI_TRY(Handle::create(Backend::AMDGPU));
#else
  Handle handle = FUSILLI_TRY(Handle::create(Backend::CPU));
#endif

  // Calculate filter channels
  auto fc = c / g;

  // Build attributes based on 2D/3D conv and layouts.
  auto xDims = (s == 2) ? std::vector<int64_t>{n, c, h, w}
                        : std::vector<int64_t>{n, c, d, h, w};
  auto wDims = (s == 2) ? std::vector<int64_t>{k, fc, y, x}
                        : std::vector<int64_t>{k, fc, z, y, x};
  auto convStride =
      (s == 2) ? std::vector<int64_t>{u, v} : std::vector<int64_t>{t, u, v};
  auto convPadding =
      (s == 2) ? std::vector<int64_t>{p, q} : std::vector<int64_t>{o, p, q};
  auto convDilation =
      (s == 2) ? std::vector<int64_t>{l, j} : std::vector<int64_t>{m, l, j};

  // Calculate output dimensions (DY shape) using the same inference as forward
  auto dyDims = getConvInferredOutputShape(xDims, wDims, convDilation,
                                           convPadding, convStride);

  auto xStride =
      (imageLayout == "NCHW" || imageLayout == "NCDHW")
          ? generateStrideFromDim(xDims, getContiguousStrideOrder(xDims.size()))
          : generateStrideFromDim(xDims,
                                  getChannelsLastStrideOrder(xDims.size()));
  auto dyStride = (outputLayout == "NCHW" || outputLayout == "NCDHW")
                      ? generateStrideFromDim(
                            dyDims, getContiguousStrideOrder(dyDims.size()))
                      : generateStrideFromDim(
                            dyDims, getChannelsLastStrideOrder(dyDims.size()));
  auto wStride =
      (filterLayout == "NCHW" || filterLayout == "NCDHW")
          ? generateStrideFromDim(wDims, getContiguousStrideOrder(wDims.size()))
          : generateStrideFromDim(wDims,
                                  getChannelsLastStrideOrder(wDims.size()));

  // Build graph for the given handle (device), validate and compile it.
  Graph graph;

  // Set unique name to prevent concurrent invocations from polluting cache.
  auto graphName =
      std::format("benchmark_conv_wgrad_n{}_c{}_d{}_h{}_w{}_g{}_k{"
                  "}_z{}_y{}_x{}_t{}_u{}_v{}_o{}"
                  "_p{}_q{}_m{}_l{}_j{}_S{}_I{}_O{}_F{}",
                  n, c, d, h, w, g, k, z, y, x, t, u, v, o, p, q, m, l, j, s,
                  imageLayout, outputLayout, filterLayout);
  graph.setName(graphName);

  graph.setIODataType(DataType::Float)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  auto dyT = graph.tensor(
      TensorAttr().setName("dy").setDim(dyDims).setStride(dyStride).setDataType(
          convIOType));

  auto xT = graph.tensor(TensorAttr()
                             .setName("input")
                             .setDim(xDims)
                             .setStride(xStride)
                             .setDataType(convIOType));

  auto convAttr = ConvWGradAttr()
                      .setStride(convStride)
                      .setPadding(convPadding)
                      .setDilation(convDilation)
                      .setName("conv_wgrad");

  auto dwT = graph.convWGrad(dyT, xT, convAttr);
  dwT->setDim(wDims).setOutput(true).setDataType(convIOType);

  // Validate, infer missing properties
  FUSILLI_CHECK_ERROR(graph.validate());

  // Compile
  FUSILLI_CHECK_ERROR(graph.compile(handle, /*remove=*/true));

  // Allocate buffers.
  auto dyBuf = FUSILLI_TRY(allocateBufferOfType(handle, dyT, convIOType, 1.0f));
  auto xBuf = FUSILLI_TRY(allocateBufferOfType(handle, xT, convIOType, 1.0f));
  auto dwBuf = FUSILLI_TRY(allocateBufferOfType(handle, dwT, convIOType, 0.0f));

  // Create variant pack.
  std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {dyT, dyBuf},
          {xT, xBuf},
          {dwT, dwBuf},
      };

  // Execute graph a few times.
  for (size_t i = 0; i < iter; i++)
    FUSILLI_CHECK_ERROR(graph.execute(handle, variantPack));

  return ok();
}

static ErrorObject
benchmarkConvDGrad(int64_t n, int64_t c, int64_t d, int64_t h, int64_t w,
                   int64_t g, int64_t k, int64_t z, int64_t y, int64_t x,
                   int64_t t, int64_t u, int64_t v, int64_t o, int64_t p,
                   int64_t q, int64_t m, int64_t l, int64_t j,
                   std::string_view imageLayout, std::string_view outputLayout,
                   std::string_view filterLayout, int64_t s, int64_t iter,
                   DataType convIOType) {
#ifdef FUSILLI_ENABLE_AMDGPU
  Handle handle = FUSILLI_TRY(Handle::create(Backend::AMDGPU));
#else
  Handle handle = FUSILLI_TRY(Handle::create(Backend::CPU));
#endif

  // Calculate filter channels
  auto fc = c / g;

  // Build attributes based on 2D/3D conv and layouts.
  auto xDims = (s == 2) ? std::vector<int64_t>{n, c, h, w}
                        : std::vector<int64_t>{n, c, d, h, w};
  auto wDims = (s == 2) ? std::vector<int64_t>{k, fc, y, x}
                        : std::vector<int64_t>{k, fc, z, y, x};
  auto convStride =
      (s == 2) ? std::vector<int64_t>{u, v} : std::vector<int64_t>{t, u, v};
  auto convPadding =
      (s == 2) ? std::vector<int64_t>{p, q} : std::vector<int64_t>{o, p, q};
  auto convDilation =
      (s == 2) ? std::vector<int64_t>{l, j} : std::vector<int64_t>{m, l, j};

  // Calculate output dimensions (DY shape) using the same inference as forward
  auto dyDims = getConvInferredOutputShape(xDims, wDims, convDilation,
                                           convPadding, convStride);

  auto xStride =
      (imageLayout == "NCHW" || imageLayout == "NCDHW")
          ? generateStrideFromDim(xDims, getContiguousStrideOrder(xDims.size()))
          : generateStrideFromDim(xDims,
                                  getChannelsLastStrideOrder(xDims.size()));
  auto dyStride = (outputLayout == "NCHW" || outputLayout == "NCDHW")
                      ? generateStrideFromDim(
                            dyDims, getContiguousStrideOrder(dyDims.size()))
                      : generateStrideFromDim(
                            dyDims, getChannelsLastStrideOrder(dyDims.size()));
  auto wStride =
      (filterLayout == "NCHW" || filterLayout == "NCDHW")
          ? generateStrideFromDim(wDims, getContiguousStrideOrder(wDims.size()))
          : generateStrideFromDim(wDims,
                                  getChannelsLastStrideOrder(wDims.size()));

  // Build graph for the given handle (device), validate and compile it.
  Graph graph;

  // Set unique name to prevent concurrent invocations from polluting cache.
  auto graphName =
      std::format("benchmark_conv_dgrad_n{}_c{}_d{}_h{}_w{}_g{}_k{"
                  "}_z{}_y{}_x{}_t{}_u{}_v{}_o{}"
                  "_p{}_q{}_m{}_l{}_j{}_S{}_I{}_O{}_F{}",
                  n, c, d, h, w, g, k, z, y, x, t, u, v, o, p, q, m, l, j, s,
                  imageLayout, outputLayout, filterLayout);
  graph.setName(graphName);

  graph.setIODataType(DataType::Float)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);

  auto dyT = graph.tensor(
      TensorAttr().setName("dy").setDim(dyDims).setStride(dyStride).setDataType(
          convIOType));

  auto wT = graph.tensor(TensorAttr()
                             .setName("filter")
                             .setDim(wDims)
                             .setStride(wStride)
                             .setDataType(convIOType));

  auto convAttr = ConvDGradAttr()
                      .setStride(convStride)
                      .setPadding(convPadding)
                      .setDilation(convDilation)
                      .setName("conv_dgrad");

  auto dxT = graph.convDGrad(dyT, wT, convAttr);
  dxT->setDim(xDims).setOutput(true).setDataType(convIOType);

  // Validate, infer missing properties
  FUSILLI_CHECK_ERROR(graph.validate());

  // Compile
  FUSILLI_CHECK_ERROR(graph.compile(handle, /*remove=*/true));

  // Allocate buffers.
  auto dyBuf = FUSILLI_TRY(allocateBufferOfType(handle, dyT, convIOType, 1.0f));
  auto wBuf = FUSILLI_TRY(allocateBufferOfType(handle, wT, convIOType, 1.0f));
  auto dxBuf = FUSILLI_TRY(allocateBufferOfType(handle, dxT, convIOType, 0.0f));

  // Create variant pack.
  std::unordered_map<std::shared_ptr<TensorAttr>, std::shared_ptr<Buffer>>
      variantPack = {
          {dyT, dyBuf},
          {wT, wBuf},
          {dxT, dxBuf},
      };

  // Execute graph a few times.
  for (size_t i = 0; i < iter; i++)
    FUSILLI_CHECK_ERROR(graph.execute(handle, variantPack));

  return ok();
}

static int benchmark(int argc, char **argv) {
  CLI::App mainApp{"Fusilli Benchmark Driver"};
  mainApp.require_subcommand(1);

  int64_t iter;
  mainApp.add_option("--iter,-i", iter, "Benchmark iterations")
      ->required()
      ->check(kIsPositiveInteger);

  // Conv flags are kept in sync with MIOpen's ConvDriver:
  // https://github.com/ROCm/rocm-libraries/blob/db0544fb61f2c7bd5a86dce98d4963420c1c741a/projects/miopen/driver/conv_driver.hpp#L878
  CLI::App *convApp =
      mainApp.add_subcommand("conv", "Fusilli Benchmark Convolution");

  // CLI Options:
  int64_t n, c, d, h, w, g, k, z, y, x, t, u, v, o, p, q, m, l, j, s;
  int64_t mode;
  std::string imageLayout, filterLayout, outputLayout;
  convApp
      ->add_option("--mode,-F", mode,
                   "Conv mode: 1=forward, 2=data_grad, 4=weight_grad")
      ->required()
      ->check(CLI::IsMember({1, 2, 4}));
  convApp->add_option("--batchsize,-n", n, "Input batch size")
      ->required()
      ->check(kIsPositiveInteger);
  convApp->add_option("--in_channels,-c", c, "Input channels")
      ->required()
      ->check(kIsPositiveInteger);
  convApp->add_option("--in_d", d, "Input depth")
      ->default_val("-1")
      ->check(kIsPositiveInteger);
  convApp->add_option("--in_h,-H", h, "Input height")
      ->required()
      ->check(kIsPositiveInteger);
  convApp->add_option("--in_w,-W", w, "Input width")
      ->required()
      ->check(kIsPositiveInteger);
  convApp->add_option("--group_count,-g", g, "Number of groups")
      ->default_val("1")
      ->check(kIsPositiveInteger);
  convApp->add_option("--out_channels,-k", k, "Output channels")
      ->required()
      ->check(kIsPositiveInteger);
  convApp->add_option("--fil_d", z, "Filter depth")
      ->default_val("-1")
      ->check(kIsPositiveInteger);
  convApp->add_option("--fil_h,-y", y, "Filter height")
      ->required()
      ->check(kIsPositiveInteger);
  convApp->add_option("--fil_w,-x", x, "Filter width")
      ->required()
      ->check(kIsPositiveInteger);
  convApp->add_option("--conv_stride_d", t, "Conv stride depth")
      ->default_val("-1")
      ->check(kIsPositiveInteger);
  convApp->add_option("--conv_stride_h,-u", u, "Conv stride height")
      ->required()
      ->check(kIsPositiveInteger);
  convApp->add_option("--conv_stride_w,-v", v, "Conv stride width")
      ->required()
      ->check(kIsPositiveInteger);
  convApp->add_option("--pad_d", o, "Conv padding depth")
      ->default_val("-1")
      ->check(kIsNonNegativeInteger);
  convApp->add_option("--pad_h,-p", p, "Conv padding height")
      ->required()
      ->check(kIsNonNegativeInteger);
  convApp->add_option("--pad_w,-q", q, "Conv padding width")
      ->required()
      ->check(kIsNonNegativeInteger);
  convApp->add_option("--dilation_d", m, "Conv dilation depth")
      ->default_val("-1")
      ->check(kIsPositiveInteger);
  convApp->add_option("--dilation_h,-l", l, "Conv dilation height")
      ->required()
      ->check(kIsPositiveInteger);
  convApp->add_option("--dilation_w,-j", j, "Conv dilation width")
      ->required()
      ->check(kIsPositiveInteger);
  convApp->add_option("--in_layout", imageLayout, "Input layout")
      ->required()
      ->check(kIsValidConvLayout);
  convApp->add_option("--fil_layout", filterLayout, "Filter layout")
      ->required()
      ->check(kIsValidConvLayout);
  convApp->add_option("--out_layout", outputLayout, "Output layout")
      ->required()
      ->check(kIsValidConvLayout);
  convApp
      ->add_option("--spatial_dim", s,
                   "Number of spatial dimensions (2 for conv2d, 3 for conv3d)")
      ->required()
      ->check(CLI::IsMember({2, 3}));

  // CLI Flags:
  bool fp16{false}, bf16{false}, bias{false};
  auto *f1 = convApp->add_flag("--fp16", fp16, "Run fp16 convolution");
  auto *f2 = convApp->add_flag("--bf16", bf16, "Run bf16 convolution");
  // Can't specify both flags.
  f1->excludes(f2);
  convApp->add_flag("--bias,-b", bias, "Run with bias (only for mode=1)");

  CLI11_PARSE(mainApp, argc, argv);

  std::cout << "Fusilli Benchmark started..." << std::endl;

  if (convApp->parsed()) {
    // Additional validation of convApp options (apart from default CLI checks)
    if (s == 2) {
      // Reject 3D layouts for 2D conv
      if (imageLayout.size() != 4 || filterLayout.size() != 4 ||
          outputLayout.size() != 4) {
        std::cerr << "Detected at least one invalid {input, filter, output} "
                     "layout for 2D convolution."
                  << std::endl;
        return 1;
      }
    }
    if (s == 3) {
      // Reject 2D layouts for 3D conv
      if (imageLayout.size() != 5 || filterLayout.size() != 5 ||
          outputLayout.size() != 5) {
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

    // Validation of group count
    if (c % g != 0 || k % g != 0) {
      std::cerr << "Detected invalid group count." << std::endl;
      return 1;
    }

    // Validate bias flag only works with forward mode
    if (bias && mode != 1) {
      std::cerr << "Bias flag (--bias) is only supported for forward "
                   "convolution (mode=1)."
                << std::endl;
      return 1;
    }

    DataType convIOType;
    if (fp16)
      convIOType = DataType::Half;
    else if (bf16)
      convIOType = DataType::BFloat16;
    else
      // When unspecified, default to fp32 conv.
      convIOType = DataType::Float;

    ErrorObject status = ok();
    if (mode == 1) {
      // Forward convolution
      status = benchmarkConvFprop(n, c, d, h, w, g, k, z, y, x, t, u, v, o, p,
                                  q, m, l, j, imageLayout, outputLayout,
                                  filterLayout, s, bias, iter, convIOType);
    } else if (mode == 2) {
      // Data gradient
      status = benchmarkConvDGrad(n, c, d, h, w, g, k, z, y, x, t, u, v, o, p,
                                  q, m, l, j, imageLayout, outputLayout,
                                  filterLayout, s, iter, convIOType);
    } else if (mode == 4) {
      // Weight gradient
      status = benchmarkConvWGrad(n, c, d, h, w, g, k, z, y, x, t, u, v, o, p,
                                  q, m, l, j, imageLayout, outputLayout,
                                  filterLayout, s, iter, convIOType);
    }

    if (isError(status)) {
      std::cerr << "Fusilli Benchmark failed: " << status << std::endl;
      return 1;
    }
  }

  std::cout << "Fusilli Benchmark complete!" << std::endl;
  return 0;
}

int main(int argc, char **argv) {
  try {
    return benchmark(argc, argv);
  } catch (const std::exception &e) {
    std::cerr << "Exception caught: " << e.what() << std::endl;
    return 1;
  } catch (...) {
    std::cerr << "Unknown exception caught" << std::endl;
    return 1;
  }
}
