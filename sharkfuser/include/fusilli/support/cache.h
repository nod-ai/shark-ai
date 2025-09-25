// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains classes for cache file handling of generated artifacts.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILLI_SUPPORT_CACHE_H
#define FUSILLI_SUPPORT_CACHE_H

#include "fusilli/support/logging.h"

#include <algorithm>
#include <cassert>
#include <filesystem>
#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <unistd.h>

namespace fusilli {

// An RAII type for creating + destroying cache files in
// `${HOME}/.cache/fusilli`.
//
//  void example() {
//    // `remove = true`
//    {
//      // Create ${HOME}/.cache/fusilli/example_graph/input
//      ErrorOr<CacheFile> cacheFile = CacheFile::create(
//          /*graphName=*/"example_graph", /*filename=*/"input",
//          /*remove=*/true);
//      assert(isOk(cacheFile));
//
//      assert(isOk(CacheFile::open(/*graphName=*/"example_graph",
//                                  /*filename=*/"input")));
//    }
//    // Try to open the same (now removed) cache file.
//    assert(isError(CacheFile::open(/*graphName=*/"example_graph",
//                                   /*filename=*/"input")));
//
//    // `remove = false`
//    {
//      ErrorOr<CacheFile> cacheFile = CacheFile::create(
//          /*graphName=*/"example_graph", /*filename=*/"input",
//          /*remove=*/false);
//      assert(isOk(cacheFile));
//    }
//    // Try to open the same cache file. This time it's found.
//    assert(isOk(CacheFile::open(/*graphName=*/"example_graph",
//                                /*filename=*/"input")));
//  }
class CacheFile {
public:
  // Factory constructor that creates file, overwriting an existing file, and
  // returns an ErrorObject if file could not be created.
  static ErrorOr<CacheFile> create(const std::string &graphName,
                                   const std::string &fileName, bool remove) {
    std::filesystem::path path = CacheFile::getPath(graphName, fileName);
    FUSILLI_LOG_LABEL_ENDL("INFO: Creating Cache file");
    FUSILLI_LOG_ENDL(path);

    // Create directory: ${HOME}/.cache/fusilli/<graphName>
    std::filesystem::path cacheDir = path.parent_path();
    std::error_code ec;
    std::filesystem::create_directories(cacheDir, ec);
    FUSILLI_RETURN_ERROR_IF(ec, ErrorCode::FileSystemFailure,
                            "Failed to create cache directory: " +
                                cacheDir.string() + " - " + ec.message());

    // Create file: ${HOME}/.cache/fusilli/<graphName>/<fileName>
    std::ofstream file(path);
    FUSILLI_RETURN_ERROR_IF(!file.is_open(), ErrorCode::FileSystemFailure,
                            "Failed to create file: " + path.string());
    file.close();

    return ok(CacheFile(path, remove));
  }

  // Factory constructor that opens an existing file and returns ErrorObject if
  // the file does not exist.
  static ErrorOr<CacheFile> open(const std::string &graphName,
                                 const std::string &fileName) {
    std::filesystem::path path = CacheFile::getPath(graphName, fileName);

    // Check if the file exists.
    FUSILLI_RETURN_ERROR_IF(!std::filesystem::exists(path),
                            ErrorCode::FileSystemFailure,
                            "File does not exist: " + path.string());

    return ok(CacheFile(path, /*remove=*/false));
  }

  // Utility method to build the path to cache file given `graphName` and
  // `fileName`.
  //
  // Format: ${HOME}/.cache/fusilli/<sanitized version of graphName>/<fileName>
  static std::filesystem::path getPath(const std::string &graphName,
                                       const std::string &fileName) {
    // Ensure graphName is safe to use as a directory name, we assume fileName
    // is safe.
    std::string sanitizedGraphName = graphName;
    std::transform(sanitizedGraphName.begin(), sanitizedGraphName.end(),
                   sanitizedGraphName.begin(),
                   [](char c) { return c == ' ' ? '_' : c; });
    std::erase_if(sanitizedGraphName, [](unsigned char c) { // C++20
      return !(std::isalnum(c) || c == '_');
    });

    // Ensure graphName has a value.
    if (sanitizedGraphName.empty())
      sanitizedGraphName = "unnamed_graph";

    // Defaults to "${HOME}/.cache/fusilli" but having it set via
    // ${FUSILLI_CACHE_DIR} to "/tmp" helps bypass permission issues
    // on the GHA CI runners.
    const char *cacheDir = std::getenv("FUSILLI_CACHE_DIR");
    if (!cacheDir)
      cacheDir = std::getenv("HOME");
    return std::filesystem::path(cacheDir) / ".cache" / "fusilli" /
           sanitizedGraphName / fileName;
  }

  // Move constructors.
  CacheFile(CacheFile &&other) noexcept
      : path(std::move(other.path)), remove_(other.remove_) {
    other.path.clear();
    other.remove_ = false;
  }
  CacheFile &operator=(CacheFile &&other) noexcept {
    if (this == &other)
      return *this;
    // If ownership of the cached file is simply changing, we aren't creating a
    // dangling resource that might to be removed.
    bool samePath = path == other.path;
    // Remove current resource if needed.
    if (remove_ && !path.empty() && !samePath)
      std::filesystem::remove(path);
    // Move from other.
    path = std::move(other.path);
    remove_ = other.remove_;
    other.path.clear();
    other.remove_ = false;
    return *this;
  }

  // Delete copy constructors. A copy constructor would likely not be safe, as
  // the destructor for a copy could remove the underlying file while the
  // original is still expecting it to exist.
  CacheFile(const CacheFile &) = delete;
  CacheFile &operator=(const CacheFile &) = delete;

  ~CacheFile() {
    if (remove_ && !path.empty()) {
      std::filesystem::remove(path);
    }
  }

  // Path of file this class wraps.
  std::filesystem::path path;

  // Write to cache file.
  ErrorObject write(const std::string &content) {
    std::ofstream file(path);
    FUSILLI_RETURN_ERROR_IF(!file.is_open(), ErrorCode::FileSystemFailure,
                            "Failed to open file: " + path.string());

    file << content;
    FUSILLI_RETURN_ERROR_IF(!file.good(), ErrorCode::FileSystemFailure,
                            "Failed to write to file: " + path.string())

    return ok();
  }

  // Read contents of cache file.
  ErrorOr<std::string> read() {
    // std::ios::ate opens file and moves the cursor to the end, allowing us
    // to get the file size with tellg().
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    FUSILLI_RETURN_ERROR_IF(!file.is_open(), ErrorCode::FileSystemFailure,
                            "Failed to open file: " + path.string());

    // Copy the contents of the file into a string.
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::string buffer(size, '\0');
    file.read(buffer.data(), size);
    FUSILLI_RETURN_ERROR_IF(!file.good(), ErrorCode::FileSystemFailure,
                            "Failed to read file: " + path.string());

    return ok(buffer);
  }

private:
  // Class should be constructed using one of the factory functions.
  CacheFile(std::filesystem::path path, bool remove)
      : path(path), remove_(remove) {}

  // Whether to remove the file on destruction or not.
  bool remove_;
};

// CleanupCacheDirectory removes a sub-directory from the main cache directory
// if it's empty. When used as a base class, C++ destructor ordering (explained
// below) ensures that the directory cleanup in CleanupCacheDirectory destructor
// will happen after any CacheFiles member variables have been destroyed.
//
// Destructor ordering example:
//   struct A   { ~A()  {std::cout << "A";} };
//   struct M1  { ~M1() {std::cout << "M1, ";} };
//   struct M2  { ~M2() {std::cout << "M2, ";} };
//
//   struct B : A {
//       M1 m1;
//       M2 m2;
//       ~B() { std::cout << "B, "; }
//   };
//   // output -> "B, M2, M1, A"
//
// If member destructors (~M1, ~M2) are called inside ~B; the compiler will
// still destroy members afterward, leading to double-destruction (UB).
struct CleanupCacheDirectory {
  std::filesystem::path cacheDir;
  explicit CleanupCacheDirectory(std::filesystem::path dir)
      : cacheDir(std::move(dir)) {}

  ~CleanupCacheDirectory() {
    // This likely indicates the instance in question has been moved from.
    if (cacheDir.empty())
      return;

    if (std::filesystem::exists(cacheDir) &&
        std::filesystem::is_empty(cacheDir))
      std::filesystem::remove(cacheDir);
  }
};

// Holds cached assets. If `CacheFiles` are set to be removed RAII based removal
// will be tied to the lifetime of this object.
struct CachedAssets : CleanupCacheDirectory {
  CacheFile input;
  CacheFile output;
  CacheFile compileCommand;

  CachedAssets(CacheFile &&in, CacheFile &&out, CacheFile &&cmd)
      : CleanupCacheDirectory(in.path.parent_path()), input(std::move(in)),
        output(std::move(out)), compileCommand(std::move(cmd)) {
    // sanity checks:
    assert(input.path.parent_path() == output.path.parent_path() &&
           input.path.parent_path() == compileCommand.path.parent_path() &&
           "Cached assets should be in the same directory.");
    assert(std::filesystem::is_directory(input.path.parent_path()));
  }

  // Default move constructors + destructor.
  CachedAssets(CachedAssets &&) noexcept = default;
  CachedAssets &operator=(CachedAssets &&) noexcept = default;
  ~CachedAssets() = default;

  // Delete copy constructors.
  CachedAssets(const CachedAssets &) = delete;
  CachedAssets &operator=(const CachedAssets &) = delete;
};

} // namespace fusilli

#endif // FUSILLI_SUPPORT_CACHE_H
