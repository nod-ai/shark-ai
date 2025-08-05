// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>
#include <optional>
#include <vector>

using namespace fusilli;

TEST_CASE("Graph getName correctly propagates the context name", "[graph]") {
  Graph g;
  g.setName("foo_graph");
  REQUIRE(g.getName() == "foo_graph");
}

TEST_CASE("Graph tensor() adds input tensor", "[graph]") {
  Graph g;
  g.setName("adds_input_tensor");
  auto t =
      g.tensor(TensorAttr().setName("input").setDim({2, 2}).setStride({2, 1}));
  REQUIRE(t->getName() == "input");
  REQUIRE(t->getDim() == std::vector<int64_t>({2, 2}));
  REQUIRE(t->getStride() == std::vector<int64_t>({2, 1}));
}

TEST_CASE("Graph conv_fprop() adds ConvFPropNode and output tensor",
          "[graph]") {
  Graph g;
  g.setName("adds_convfpropnoe_and_output_tensor");
  auto x =
      g.tensor(TensorAttr().setDim({1, 8, 8, 3}).setStride({192, 24, 3, 1}));
  auto w = g.tensor(TensorAttr().setDim({4, 3, 3, 3}).setStride({27, 9, 3, 1}));
  ConvFPropAttr attr;
  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1});
  auto y = g.convFProp(x, w, attr);

  // Names for inputs are auto-populated when not set
  REQUIRE(x->getName() == "conv_fprop_0_X");
  REQUIRE(w->getName() == "conv_fprop_0_W");
  REQUIRE(y->getName() == "conv_fprop_0_Y");

  // Y is virtual (intermediate tensor) unless specified as output
  REQUIRE(y->isVirtual() == true);
  y->setOutput(true);
  REQUIRE(y->isVirtual() == false);
}

TEST_CASE("Graph validate() fails if name is not set", "[graph]") {
  Graph g;
  ErrorObject err = g.validate();
  REQUIRE(isError(err));
  REQUIRE(err.getCode() == ErrorCode::AttributeNotSet);
  REQUIRE(err.getMessage() == "Graph name not set");
}

TEST_CASE("Graph validate() returns OK for valid graph", "[graph]") {
  Graph g;
  g.setName("validate_returns_ok_for_valid_graph");
  g.setIODataType(DataType::Half)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);
  auto x = g.tensor(TensorAttr()
                        .setName("X")
                        .setDim({1, 8, 8, 3})
                        .setStride({192, 24, 3, 1}));
  auto w = g.tensor(
      TensorAttr().setName("W").setDim({4, 3, 3, 3}).setStride({27, 9, 3, 1}));
  ConvFPropAttr attr;
  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1}).setName("conv");
  auto y = g.convFProp(x, w, attr);

  // Fails because y is underspecified (shape/stride inference unimplemented)
  auto status = g.validate();
  REQUIRE(isError(status));
  REQUIRE(status.getCode() == ErrorCode::NotImplemented);
  REQUIRE(status.getMessage() ==
          "ConvFProp node shape inference not implemented yet; please "
          "specify output tensor dimensions");

  // Specify y's shape and strides
  y->setDim({1, 8, 8, 4}).setStride({256, 32, 4, 1});
  REQUIRE(isOk(g.validate()));
}

TEST_CASE("Graph asm_emitter requires validation to be run first", "[graph]") {
  Graph g;
  g.setName("asm_emitter_requires_validation_first");
  g.setIODataType(DataType::Half)
      .setComputeDataType(DataType::Float)
      .setIntermediateDataType(DataType::Float);
  auto x = g.tensor(TensorAttr()
                        .setName("X")
                        .setDim({1, 8, 8, 3})
                        .setStride({192, 24, 3, 1}));
  auto w = g.tensor(
      TensorAttr().setName("W").setDim({4, 3, 3, 3}).setStride({27, 9, 3, 1}));
  ConvFPropAttr attr;
  attr.setPadding({0, 0}).setStride({1, 1}).setDilation({1, 1}).setName("conv");
  auto y = g.convFProp(x, w, attr);
  y->setDim({1, 8, 8, 4}).setStride({256, 32, 4, 1});

  // ASM emitter without validation should throw an error
  REQUIRE(isError(g.emitAsm()));
  // Validate the graph first
  REQUIRE(isOk(g.validate()));
  // ASM emitter should now work
  REQUIRE(isOk(g.emitAsm()));
}

// Helper function to create a valid graph for testing
Graph validGraph() {
  Graph g;
  g.setName("test_graph");
  int64_t n = 16, c = 128, h = 64, w = 64, k = 256, r = 1, s = 1;
  g.setName("test_graph");
  g.setBackend(Backend::CPU);
  g.setIODataType(DataType::Half).setComputeDataType(DataType::Float);
  auto X = g.tensor(TensorAttr()
                        .setName("image")
                        .setDim({n, c, h, w})
                        .setStride({c * h * w, h * w, w, 1}));
  auto W = g.tensor(TensorAttr()
                        .setName("filter")
                        .setDim({k, c, r, s})
                        .setStride({c * r * s, r * s, s, 1}));
  auto conv = ConvFPropAttr()
                  .setPadding({0, 0})
                  .setStride({1, 1})
                  .setDilation({1, 1})
                  .setName("conv_fprop");
  auto Y = g.convFProp(X, W, conv);
  Y->setDim({n, k, h, w}).setStride({k * h * w, h * w, w, 1});
  Y->setOutput(true);
  REQUIRE(isOk(g.validate()));
  return g;
};

TEST_CASE("Graph `readOrGenerateCompiledArtifacts`", "[graph]") {
  SECTION("cache generation and invalidation") {
    Graph g = validGraph();

    std::string generatedAsm = FUSILLI_REQUIRE_UNWRAP(g.emitAsm());

    // Cache should be empty, compilation artifacts should be generated.
    std::optional<bool> reCompiled = std::nullopt;
    REQUIRE(isOk(g.readOrGenerateCompiledArtifact(generatedAsm, /*remove=*/true,
                                                  /*reCompiled=*/&reCompiled)));
    REQUIRE(reCompiled.has_value());
    REQUIRE(reCompiled.value());

    // Cache should hit, no compilation should be required.
    reCompiled = std::nullopt;
    REQUIRE(isOk(g.readOrGenerateCompiledArtifact(generatedAsm, /*remove=*/true,
                                                  /*reCompiled=*/&reCompiled)));
    REQUIRE(reCompiled.has_value());
    REQUIRE(!reCompiled.value());

    // Cache should miss based on different compile command.
    // TODO(#1964): GFX942 compilation seems to be broken
    // g.setBackend(Backend::GFX942);
    // reCompiled = std::nullopt;
    // REQUIRE(isOk(g.readOrGenerateCompiledArtifact(generatedAsm,
    // &reCompiled))); REQUIRE(reCompiled.has_value());
    // REQUIRE(reCompiled.value());

    // Cache should miss because of different generated asm.
    reCompiled = std::nullopt;
    FUSILLI_REQUIRE_UNWRAP(g.readOrGenerateCompiledArtifact(
        generatedAsm + " ", /*remove=*/true, /*reCompiled=*/&reCompiled));
    REQUIRE(reCompiled.has_value());
    REQUIRE(reCompiled.value());

    // Cache should hit with the same generated asm.
    reCompiled = std::nullopt;
    FUSILLI_REQUIRE_UNWRAP(g.readOrGenerateCompiledArtifact(
        generatedAsm + " ", /*remove=*/true, /*reCompiled=*/&reCompiled));
    REQUIRE(reCompiled.has_value());
    REQUIRE(!reCompiled.value());

    // Cache should miss because graph name change.
    g.setName("new_graph_name");
    reCompiled = std::nullopt;
    FUSILLI_REQUIRE_UNWRAP(g.readOrGenerateCompiledArtifact(
        generatedAsm + " ", /*remove=*/true, /*reCompiled=*/&reCompiled));
    REQUIRE(reCompiled.has_value());
    REQUIRE(reCompiled.value());
  }

  SECTION("should not read cached items from other/previous Graph instances") {
    std::string generatedAsm;
    {
      Graph g = validGraph();

      generatedAsm = FUSILLI_REQUIRE_UNWRAP(g.emitAsm());

      // Cache should be empty.
      std::optional<bool> reCompiled = std::nullopt;
      REQUIRE(isOk(g.readOrGenerateCompiledArtifact(
          generatedAsm, /*remove=*/false, /*reCompiled=*/&reCompiled)));
      REQUIRE(reCompiled.has_value());
      REQUIRE(reCompiled.value());

      // Cache should hit with the same generated asm.
      reCompiled = std::nullopt;
      FUSILLI_REQUIRE_UNWRAP(g.readOrGenerateCompiledArtifact(
          generatedAsm, /*remove=*/false, /*reCompiled=*/&reCompiled));
      REQUIRE(reCompiled.has_value());
      REQUIRE(!reCompiled.value());
    }

    Graph g = validGraph();

    // Check that based on the generated asm the cache should hit.
    CacheFile asmCache = FUSILLI_REQUIRE_UNWRAP(
        CacheFile::open(g.getName(), IREE_COMPILE_INPUT_FILENAME));
    REQUIRE(FUSILLI_REQUIRE_UNWRAP(asmCache.read()) == generatedAsm);

    // New instance should regenerate cache.
    std::optional<bool> reCompiled = std::nullopt;
    REQUIRE(isOk(g.readOrGenerateCompiledArtifact(generatedAsm, /*remove=*/true,
                                                  /*reCompiled=*/&reCompiled)));
    REQUIRE(reCompiled.has_value());
    REQUIRE(reCompiled.value());
  }

  SECTION("Invalid input IR") {
    Graph g;
    g.setName("invalid_input_ir");
    ErrorObject err = g.readOrGenerateCompiledArtifact("invalid mlir");
    REQUIRE(isError(err));
    REQUIRE(err.getCode() == ErrorCode::CompileFailure);
    REQUIRE(err.getMessage() == "iree-compile command failed");
  }
}
