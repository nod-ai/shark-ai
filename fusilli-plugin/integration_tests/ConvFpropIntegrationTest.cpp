// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cmath>
#include <gtest/gtest.h>
#include <hip/hip_runtime.h>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

#include <hipdnn_frontend/attributes/ConvolutionFpropAttributes.hpp>
#include <hipdnn_frontend/attributes/PointwiseAttributes.hpp>

#include <hipdnn_frontend/Graph.hpp>
#include <hipdnn_frontend/Utilities.hpp>
#include <hipdnn_frontend/attributes/TensorAttributes.hpp>
#include <hipdnn_sdk/test_utilities/CpuFpReferenceValidation.hpp>
#include <hipdnn_sdk/test_utilities/TestUtilities.hpp>
#include <hipdnn_sdk/utilities/MigratableMemory.hpp>
#include <hipdnn_sdk/utilities/Tensor.hpp>

using namespace hipdnn_frontend;
using namespace hipdnn_sdk::utilities;
using namespace hipdnn_sdk::test_utilities;

namespace {

struct ConvFpropTestCase {
  // batch size
  int64_t n;
  // input channels
  int64_t c;
  // height of input image
  int64_t h;
  // width of input image
  int64_t w;
  // number of output channels/filters
  int64_t k;
  // filter height (rows)
  int64_t r;
  // filter width (cols)
  int64_t s;

  friend std::ostream &operator<<(std::ostream &ss,
                                  const ConvFpropTestCase &tc) {
    return ss << "(n:" << tc.n << " c:" << tc.c << " h:" << tc.h
              << " w:" << tc.w << " k:" << tc.k << " r:" << tc.r
              << " s:" << tc.s << ")";
  }
};

struct ConvFpropTensorBundle {
  ConvFpropTensorBundle(const ConvFpropTestCase &dims, unsigned int seed = 1)
      : xTensor({dims.n, dims.c, dims.h, dims.w}),
        wTensor({dims.k, dims.c, dims.r, dims.s}),
        yTensor({dims.n, dims.k, dims.h, dims.w}) {
    std::ignore = seed;

    xTensor.fillWithValue(1.0f);
    wTensor.fillWithValue(1.0f);
    yTensor.fillWithValue(-100.0f);
  }

  PinnedTensor<float> xTensor; // input image
  PinnedTensor<float> wTensor; // filter/weights
  PinnedTensor<float> yTensor; // output
};

} // namespace

class ConvFpropInferenceIntegrationTest
    : public ::testing::TestWithParam<ConvFpropTestCase> {
protected:
  void SetUp() override {
    SKIP_IF_NO_DEVICES();

    // Uncomment if you want debug logging info.
    setenv("HIPDNN_LOG_LEVEL", "info", 1);

    // Initialize HIP
    ASSERT_EQ(hipInit(0), hipSuccess);
    ASSERT_EQ(hipGetDevice(&_deviceId), hipSuccess);

    // Note: The plugin paths has to be set before we create the hipdnn handle.
    const std::array<const char *, 1> paths = {PLUGIN_DIR};
    ASSERT_EQ(hipdnnSetEnginePluginPaths_ext(paths.size(), paths.data(),
                                             HIPDNN_PLUGIN_LOADING_ABSOLUTE),
              HIPDNN_STATUS_SUCCESS);

    // Create handle
    ASSERT_EQ(hipdnnCreate(&_handle), HIPDNN_STATUS_SUCCESS);

    // todo: bring back stream support once MigratableMemory supports it
    // ASSERT_EQ(hipStreamCreate(&stream), hipSuccess);
    // ASSERT_EQ(hipdnnSetStream(handle, stream), HIPDNN_STATUS_SUCCESS);
  }

  void TearDown() override {
    if (_handle != nullptr) {
      ASSERT_EQ(hipdnnDestroy(_handle), HIPDNN_STATUS_SUCCESS);
    }
    if (_stream != nullptr) {
      ASSERT_EQ(hipStreamDestroy(_stream), hipSuccess);
    }
  }

  static std::unordered_map<int64_t, void *>
  createVariantPack(const graph::TensorAttributes &xTensorAttr,
                    const graph::TensorAttributes &wTensorAttr,
                    const graph::TensorAttributes &yTensorAttr,
                    ConvFpropTensorBundle &tensorBundle) {
    std::unordered_map<int64_t, void *> variantPack;
    variantPack[xTensorAttr.get_uid()] =
        tensorBundle.xTensor.memory().deviceData();
    variantPack[wTensorAttr.get_uid()] =
        tensorBundle.wTensor.memory().deviceData();
    variantPack[yTensorAttr.get_uid()] =
        tensorBundle.yTensor.memory().deviceData();

    return variantPack;
  }

  void runConvFwd(ConvFpropTensorBundle &graphTensorBundle,
                  DataType_t inputDataType) {
    auto graph = std::make_shared<hipdnn_frontend::graph::Graph>();

    graph->set_name("fprop_sample");
    graph->set_io_data_type(DataType_t::FLOAT)
        .set_compute_data_type(DataType_t::FLOAT);

    int64_t uid = 1;

    // Create input tensor (image) with UID
    auto xAttr = graph::makeTensorAttributes("image", inputDataType,
                                             graphTensorBundle.xTensor);
    xAttr.set_uid(uid++);
    auto xTensorAttr =
        std::make_shared<graph::TensorAttributes>(std::move(xAttr));

    // Create weight/filter tensor with UID
    auto wAttr = graph::makeTensorAttributes("filter", inputDataType,
                                             graphTensorBundle.wTensor);
    wAttr.set_uid(uid++);
    auto wTensorAttr =
        std::make_shared<graph::TensorAttributes>(std::move(wAttr));

    // Create convolution attributes
    graph::ConvFpropAttributes convAttr;
    convAttr.set_name("conv_fprop")
        .set_padding({0, 0})
        .set_stride({1, 1})
        .set_dilation({1, 1});

    // Perform convolution
    auto yTensorAttr = graph->conv_fprop(xTensorAttr, wTensorAttr, convAttr);

    // Set UID for output tensor
    if (!yTensorAttr->has_uid()) {
      yTensorAttr->set_uid(uid++);
    }

    // Set output tensor dimensions and strides
    yTensorAttr->set_dim(graphTensorBundle.yTensor.dims())
        .set_stride(graphTensorBundle.yTensor.strides());
    yTensorAttr->set_output(true);

    auto result = graph->validate();
    ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

    result = graph->build_operation_graph(_handle);
    ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

    result = graph->create_execution_plans();
    ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

    result = graph->check_support();
    ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

    result = graph->build_plans();
    ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;

    // Create variant pack with all three tensors
    auto variantPack = createVariantPack(*xTensorAttr, *wTensorAttr,
                                         *yTensorAttr, graphTensorBundle);

    result = graph->execute(_handle, variantPack, nullptr);
    ASSERT_EQ(result.code, error_code_t::OK) << result.err_msg;
  }

  static void runCpuConvFpropFwd(ConvFpropTensorBundle &cpuTensorBundle) {
    auto *output = cpuTensorBundle.yTensor.memory().hostData();
    size_t size = cpuTensorBundle.yTensor.memory().count();

    for (size_t i = 0; i < size; i++) {
      output[i] = 128.0f;
    }
  }

  void runConvFpropTest(const ConvFpropTestCase &testCase,
                        float tolerance = 1e-4f) {
    auto inputDataType = getDataTypeEnumFromType<float>();

    unsigned int seed = std::random_device{}();
    // log the random seed in case we need to reproduce the test
    HIPDNN_LOG_INFO("Test is using {} for its random seed", seed);

    ConvFpropTensorBundle graphTensorBundle(testCase, seed);

    ConvFpropTensorBundle cpuTensorBundle(testCase, seed);

    runConvFwd(graphTensorBundle, inputDataType);
    graphTensorBundle.yTensor.memory().markDeviceModified();

    runCpuConvFpropFwd(cpuTensorBundle);

    CpuFpReferenceValidation<float> cpuRefValidation(tolerance, tolerance);
    EXPECT_TRUE(cpuRefValidation.allClose(cpuTensorBundle.yTensor.memory(),
                                          graphTensorBundle.yTensor.memory()));
  }

private:
  hipdnnHandle_t _handle = nullptr;
  hipStream_t _stream = nullptr;
  int _deviceId = 0;
};

namespace {

std::vector<ConvFpropTestCase> getConvFpropInferenceTestCases() {
  return {{.n = 16, .c = 128, .h = 64, .w = 64, .k = 256, .r = 1, .s = 1}};
}

} // namespace

TEST_P(ConvFpropInferenceIntegrationTest, RunFloatFwdBatchnormGraphNCHW) {
  ConvFpropTestCase testCase = GetParam();
  runConvFpropTest(testCase, 1e-6f);
}

INSTANTIATE_TEST_SUITE_P(RunFloatFwdBatchnormGraph,
                         ConvFpropInferenceIntegrationTest,
                         testing::ValuesIn(getConvFpropInferenceTestCases()));
