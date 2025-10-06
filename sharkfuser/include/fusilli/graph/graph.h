// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains definitions for the `Graph` class which derives from the
// `INode` class (like other nodes).
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_GRAPH_GRAPH_H
#define FUSILLI_GRAPH_GRAPH_H

#include "fusilli/attributes/tensor_attributes.h"
#include "fusilli/backend/backend.h"
#include "fusilli/backend/buffer.h"
#include "fusilli/backend/handle.h"
#include "fusilli/graph/context.h"
#include "fusilli/node/conv_node.h"
#include "fusilli/node/node.h"
#include "fusilli/support/cache.h"
#include "fusilli/support/external_tools.h"
#include "fusilli/support/extras.h"
#include "fusilli/support/logging.h"

#include <cstdlib>
#include <filesystem>
#include <memory>
#include <optional>
#include <set>
#include <string>
#include <unordered_set>
#include <utility>

#define IREE_COMPILE_INPUT_FILENAME "iree-compile-input.mlir"
#define IREE_COMPILE_OUTPUT_FILENAME "iree-compile-output.vmfb"
#define IREE_COMPILE_COMMAND_FILENAME "iree-compile-command.txt"
#define IREE_COMPILE_STATISTICS_FILENAME "iree-compile-statistics.json"

namespace fusilli {

class Graph : public INode {
public:
  Graph() : INode(Context{}) {}

  // Validates the graph for correctness and infers missing properties.
  ErrorObject validate() {
    FUSILLI_LOG_LABEL_ENDL("INFO: Validating Graph");
    FUSILLI_RETURN_ERROR_IF(getName().empty(), ErrorCode::AttributeNotSet,
                            "Graph name not set");
    // Validate nodes:
    // This infers missing tensor properties such as dims,
    // stride, dtype based on context.
    FUSILLI_CHECK_ERROR(validateSubtree());
    // Validate inputs:
    // This has to happen after `validateSubtree` to infer any
    // missing properties on inputs first.
    for (const auto &input : fullGraphInputs_) {
      FUSILLI_CHECK_ERROR(input->validate());
    }
    // Validate outputs:
    // This has to happen after `validateSubtree` to infer any
    // missing properties on outputs first.
    for (const auto &output : fullGraphOutputs_) {
      FUSILLI_CHECK_ERROR(output->validate());
    }
    FUSILLI_LOG_LABEL_ENDL("INFO: Graph validation completed successfully");
    isValidated_ = true;
    return ok();
  }

  // Compiles the graph using IREE compiler and sets up the IREE runtime
  // session context for future g->execute calls.
  //
  // Set `remove = true` to remove compilation artifacts (cache files) when
  // this `Graph` instance goes out of scope.
  ErrorObject compile(const Handle &handle, bool remove = false) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Compiling Graph");
    FUSILLI_RETURN_ERROR_IF(!isValidated_, ErrorCode::NotValidated,
                            "Graph must be validated before being compiled");

    // Generate MLIR assembly for this graph.
    std::string generatedAsm = FUSILLI_TRY(emitAsm());

    // Compile using IREE compiler or reuse cached artifact.
    std::string vmfbPath =
        FUSILLI_TRY(getCompiledArtifact(handle, generatedAsm, remove));

    // Create per-graph IREE runtime session and load the compiled artifact.
    FUSILLI_CHECK_ERROR(createPerGraphSession(handle, vmfbPath));

    return ok();
  }

  // Executes the graph using IREE runtime. Requires a `variantPack` which is a
  // map from `TensorAttr` to `Buffer` wrapping the `iree_hal_buffer_view_t *`.
  // Definition in `fusilli/backend/runtime.h`.
  ErrorObject
  execute(const std::unordered_map<std::shared_ptr<TensorAttr>,
                                   std::shared_ptr<Buffer>> &variantPack) const;

  // Delete copy constructors, keep default move constructor and destructor.
  Graph(const Graph &) = delete;
  Graph &operator=(const Graph &) = delete;
  Graph(Graph &&) noexcept = default;
  Graph &operator=(Graph &&) noexcept = default;
  ~Graph() = default;

  // Getters and setters for graph context.
  const std::string &getName() const override final {
    return context.getName();
  }
  Type getType() const override final { return Type::Composite; }

  Graph &setName(const std::string &name) {
    context.setName(name);
    return *this;
  }

  Graph &setIODataType(DataType type) {
    context.setIODataType(type);
    return *this;
  }

  Graph &setComputeDataType(DataType type) {
    context.setComputeDataType(type);
    return *this;
  }

  Graph &setIntermediateDataType(DataType type) {
    context.setIntermediateDataType(type);
    return *this;
  }

  // Declarations for tensor and op builder methods go here.
  // Definitions are towards the end of this file below.
  std::shared_ptr<TensorAttr> tensor(const TensorAttr &tensor);

  std::shared_ptr<TensorAttr> convFProp(const std::shared_ptr<TensorAttr> &x,
                                        const std::shared_ptr<TensorAttr> &w,
                                        ConvFPropAttr &attributes);

  // ASM emitter driver method.
  //
  // TODO(#2152): Make this private. It is public for now to aid testing and
  // debuggability, however the intended user facing API is `Graph::compile()`.
  ErrorOr<std::string> emitAsm() {
    FUSILLI_LOG_LABEL_ENDL("INFO: Emitting MLIR assembly for Graph");
    FUSILLI_RETURN_ERROR_IF(
        !isValidated_, ErrorCode::NotValidated,
        "Graph must be validated before emitting MLIR assembly");
    std::ostringstream oss;
    emitAsmSubtree(oss);
    FUSILLI_LOG_ENDL(oss.str());
    return ok(oss.str());
  }

  // Return compiled artifact. The first invocation will always generate
  // compiled artifact, subsequent invocations may return cached versions
  // assuming cache invalidation checks pass. Set `remove = true` to remove
  // cache files when this `Graph` instance goes out of scope.
  //
  // `reCompiled` will be set to true if a value is passed and the cache was
  // (re)generated; this parameter is useful for testing.
  //
  // TODO(#2152): Make this private. It is public for now to aid testing and
  // debuggability, however the intended user facing API is `Graph::compile()`.
  ErrorOr<std::filesystem::path>
  getCompiledArtifact(const Handle &handle, const std::string &generatedAsm,
                      bool remove, std::optional<bool> *reCompiled = nullptr) {
    // Check for cache hit.
    if (FUSILLI_TRY(validateCache(handle, generatedAsm))) {
      if (reCompiled)
        *reCompiled = false;
      return ok(cache_->output.path);
    }
    // (Re)generate cache.
    cache_ =
        FUSILLI_TRY(generateCompiledArtifact(handle, generatedAsm, remove));
    if (reCompiled)
      *reCompiled = true;
    return ok(cache_->output.path);
  }

  ErrorOr<std::string> readCompilationCacheFile(CachedAssetsType type) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Getting cached assets path");
    FUSILLI_RETURN_ERROR_IF(!cache_.has_value(), ErrorCode::FileSystemFailure,
                            "Cache not populated yet");

    // `CacheFile::read` already returns an `ErrorOr<std::string>`
    // so don't wrap it in another `ok()` here.
    switch (type) {
    case CachedAssetsType::Input:
      return cache_->input.read();
    case CachedAssetsType::Command:
      return cache_->command.read();
    case CachedAssetsType::Output:
      return cache_->output.read();
    case CachedAssetsType::Statistics:
      return cache_->statistics.read();
    default:
      return error(ErrorCode::InvalidAttribute, "Unknown CachedAssetsType");
    }
  }

private:
  // Definition in `fusilli/backend/runtime.h`.
  ErrorObject createPerGraphSession(const Handle &handle,
                                    const std::string &vmfbPath);

  std::string buildCompileCommand(const Handle &handle, const CacheFile &input,
                                  const CacheFile &output,
                                  const CacheFile &statistics) {
    std::vector<std::string> args = {IREE_COMPILE_PATH, input.path};
    auto &flags = backendFlags.at(handle.getBackend());
    args.insert(args.end(), flags.begin(), flags.end());
    // TODO(#2374): Make this conditional (enabled only for testing/debug).
    args.push_back("--iree-scheduling-dump-statistics-format=json");
    args.push_back("--iree-scheduling-dump-statistics-file=" +
                   statistics.path.string());
    args.push_back("-o");
    args.push_back(output.path);
    std::ostringstream cmdss;
    interleave(
        args.begin(), args.end(),
        // each_fn:
        [&](const std::string &name) { cmdss << name; },
        // between_fn:
        [&] { cmdss << " "; });
    return cmdss.str() + "\n";
  }

  // Create compiled artifacts from graph writing results to the cache. Set
  // `remove = true` to remove cache files when returned `CachedAssets` lifetime
  // ends.
  ErrorOr<CachedAssets>
  generateCompiledArtifact(const Handle &handle,
                           const std::string &generatedAsm, bool remove) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Generating compiled artifacts");

    // Create cache.
    CachedAssets cache = CachedAssets(
        /*in=*/
        FUSILLI_TRY(CacheFile::create(
            /*graphName=*/getName(),
            /*fileName=*/IREE_COMPILE_INPUT_FILENAME,
            /*remove=*/remove)),
        /*out=*/
        FUSILLI_TRY(CacheFile::create(
            /*graphName=*/getName(),
            /*fileName=*/IREE_COMPILE_OUTPUT_FILENAME,
            /*remove=*/remove)),
        /*cmd=*/
        FUSILLI_TRY(CacheFile::create(
            /*graphName=*/getName(),
            /*fileName=*/IREE_COMPILE_COMMAND_FILENAME,
            /*remove=*/remove)),
        /*stat=*/
        FUSILLI_TRY(CacheFile::create(
            /*graphName=*/getName(),
            /*fileName=*/IREE_COMPILE_STATISTICS_FILENAME,
            /*remove=*/remove)));

    // Write input asm to cache.
    FUSILLI_CHECK_ERROR(cache.input.write(generatedAsm));

    // Build + cache + log compile command.
    std::string cmd = buildCompileCommand(handle, cache.input, cache.output,
                                          cache.statistics);
    FUSILLI_CHECK_ERROR(cache.command.write(cmd));
    FUSILLI_LOG_LABEL_ENDL("INFO: iree-compile command");
    FUSILLI_LOG_ENDL(cmd);

    // Run iree-compile.
    // TODO(#1934): in the error case, std::system will dump to stderr, it would
    // be great to capture this for better logging + reproducer production.
    int returnCode = std::system(cmd.c_str());
    FUSILLI_RETURN_ERROR_IF(returnCode, ErrorCode::CompileFailure,
                            "iree-compile command failed");

    return ok(std::move(cache));
  }

  // Check for cache validity. Cache should be invalidated if:
  //  - Cache has not been generated for this instance yet
  //  - Graph name (and therefore cache path) has changed
  //  - Generated assembly differs
  //  - Compile commands have changed
  //  - Handle/backend (and therefore compile command) has changed
  ErrorOr<bool> validateCache(const Handle &handle,
                              const std::string &generatedAsm) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Validating cache");

    // Check for cache miss if cache hasn't been generated.
    if (!cache_.has_value()) {
      FUSILLI_LOG_ENDL("Cache not previously populated.");
      return ok(false);
    }

    // Check for cache miss if paths don't match (e.g., if graph name changed).
    if (cache_->input.path != CacheFile::getPath(
                                  /*graphName=*/getName(),
                                  /*fileName=*/IREE_COMPILE_INPUT_FILENAME)) {
      FUSILLI_LOG_ENDL("Cache input paths differ.");
      return ok(false);
    }
    if (cache_->output.path != CacheFile::getPath(
                                   /*graphName=*/getName(),
                                   /*fileName=*/IREE_COMPILE_OUTPUT_FILENAME)) {
      FUSILLI_LOG_ENDL("Cache output paths differ.");
      return ok(false);
    }
    if (cache_->command.path !=
        CacheFile::getPath(
            /*graphName=*/getName(),
            /*fileName=*/IREE_COMPILE_COMMAND_FILENAME)) {
      FUSILLI_LOG_ENDL("Cache compile command paths differ.");
      return ok(false);
    }
    if (cache_->statistics.path !=
        CacheFile::getPath(
            /*graphName=*/getName(),
            /*fileName=*/IREE_COMPILE_STATISTICS_FILENAME)) {
      FUSILLI_LOG_ENDL("Cache compile statistics paths differ.");
      return ok(false);
    }

    // Open expected files.
    CacheFile input = FUSILLI_TRY(CacheFile::open(
        /*graphName=*/getName(),
        /*fileName=*/IREE_COMPILE_INPUT_FILENAME));
    CacheFile output = FUSILLI_TRY(CacheFile::open(
        /*graphName=*/getName(),
        /*fileName=*/IREE_COMPILE_OUTPUT_FILENAME));
    CacheFile command = FUSILLI_TRY(CacheFile::open(
        /*graphName=*/getName(),
        /*fileName=*/IREE_COMPILE_COMMAND_FILENAME));
    CacheFile statistics = FUSILLI_TRY(CacheFile::open(
        /*graphName=*/getName(),
        /*fileName=*/IREE_COMPILE_STATISTICS_FILENAME));

    // Check for a cache miss on generated assembly.
    if (FUSILLI_TRY(input.read()) != generatedAsm) {
      FUSILLI_LOG_ENDL("Generated assembly does not match");
      return ok(false);
    }

    // Check for a cache miss on compile command.
    std::string cmd = buildCompileCommand(handle, input, output, statistics);
    if (FUSILLI_TRY(command.read()) != cmd) {
      FUSILLI_LOG_ENDL("Compile command does not match");
      return ok(false);
    }

    return ok(true);
  }

  std::shared_ptr<TensorAttr> outputTensor(const std::string &name) {
    FUSILLI_LOG_LABEL_ENDL("INFO: Adding output tensor '"
                           << name << "' to Graph outputs");
    auto tensor = std::make_shared<TensorAttr>();
    tensor->setName(name).setIsVirtual(true);
    fullGraphOutputs_.insert(tensor);
    return tensor;
  }

  ErrorObject preValidateNode() const override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Pre-Validating Graph");
    // Validate input/output names are unique (requirement for SSA).
    std::unordered_set<std::string> usedSymbols;
    for (const auto &t : fullGraphInputs_) {
      FUSILLI_RETURN_ERROR_IF(usedSymbols.contains(t->getName()), // C++20
                              ErrorCode::InvalidAttribute,
                              "Symbol name '" + t->getName() +
                                  "' already in use");
      usedSymbols.insert(t->getName());
    }
    for (const auto &t : fullGraphOutputs_) {
      FUSILLI_RETURN_ERROR_IF(usedSymbols.contains(t->getName()), // C++20
                              ErrorCode::InvalidAttribute,
                              "Symbol name '" + t->getName() +
                                  "' already in use");
      usedSymbols.insert(t->getName());
    }
    // Recursively validate node names are unique (requirement for SSA).
    FUSILLI_CHECK_ERROR(checkNodeNamesAreUnique(usedSymbols));

    return ok();
  }

  ErrorObject inferPropertiesNode() override final {
    FUSILLI_LOG_LABEL_ENDL("INFO: Inferring properties for Graph");
    // Populate sorted inputs / outputs after graph is fully constructed
    // and pre-validated (to ensure no symbol conflict).
    fullGraphInputsSorted_.insert(fullGraphInputs_.begin(),
                                  fullGraphInputs_.end());
    fullGraphOutputsSorted_.insert(fullGraphOutputs_.begin(),
                                   fullGraphOutputs_.end());
    return ok();
  }

  ErrorObject postValidateNode() const override final { return ok(); }

  // MLIR assembly emitter helper methods.
  std::string emitNodePreAsm() const override final;
  std::string emitNodePostAsm() const override final;
  std::string getOperandNamesAndTypesAsm() const;
  std::string getResultNamesAndTypesAsm() const;

  // This is set after `validate()` is run at least once successfully.
  bool isValidated_ = false;

  // IREE runtime session lifetime managed by the `Graph` object
  // (deleted when the `Graph` object goes out of scope).
  IreeRuntimeSessionUniquePtrType session_;

  // Cache set by `getCompiledArtifact()`.
  //
  // Note: new instances should always re-generate cache even if the results
  // could be read from the file system. Old results may have been generated
  // with a different version of IREE, it would not be safe to use them.
  std::optional<CachedAssets> cache_ = std::nullopt;

  // This is safe for post-insertion updates of TensorAttr (e.g. setting name
  // or other properties) since it uses the pointer value itself for hashing.
  std::unordered_set<std::shared_ptr<TensorAttr>> fullGraphInputs_;
  std::unordered_set<std::shared_ptr<TensorAttr>> fullGraphOutputs_;

  // These are sorted by the TensorAttr name, so post-insertion modification is
  // UB (undefined behavior). These are to be populated after the graph is fully
  // constructed and validated, and no further updates are expected.
  std::set<std::shared_ptr<TensorAttr>, TensorAttrSortByName>
      fullGraphInputsSorted_;
  std::set<std::shared_ptr<TensorAttr>, TensorAttrSortByName>
      fullGraphOutputsSorted_;
};

// Given a TensorAttr, create a shared pointer and add it to the graph's
// inputs. This allows the graph to manage the lifetime of the input tensor.
inline std::shared_ptr<TensorAttr> Graph::tensor(const TensorAttr &tensor) {
  FUSILLI_LOG_LABEL_ENDL("INFO: Adding input tensor '" << tensor.getName()
                                                       << "' to Graph inputs");
  auto tensorPtr = std::make_shared<TensorAttr>(tensor);
  fullGraphInputs_.insert(tensorPtr);
  return tensorPtr;
}

// Create a ConvFPropNode, populate it with the specified attributes, create
// output tensors and add the node to the graph's sub nodes.
inline std::shared_ptr<TensorAttr>
Graph::convFProp(const std::shared_ptr<TensorAttr> &x,
                 const std::shared_ptr<TensorAttr> &w,
                 ConvFPropAttr &convAttr) {
  // Populate names when not set.
  if (convAttr.getName().empty())
    convAttr.setName("conv_fprop_" + std::to_string(subNodes_.size()));
  if (x->getName().empty())
    x->setName(convAttr.getName() + "_X");
  if (w->getName().empty())
    w->setName(convAttr.getName() + "_W");

  FUSILLI_LOG_LABEL_ENDL("INFO: Adding ConvFPropNode '" << convAttr.getName()
                                                        << "' to Graph");

  // Set inputs.
  convAttr.setX(x).setW(w);

  // Set outputs.
  auto y = outputTensor(convAttr.getName() + "_Y");
  convAttr.setY(y);

  // Create node and add to Graph's subNodes_.
  subNodes_.emplace_back(
      std::make_unique<ConvFPropNode>(std::move(convAttr), context));

  return y;
}

} // namespace fusilli

#endif // FUSILLI_GRAPH_GRAPH_H
