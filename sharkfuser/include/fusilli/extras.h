// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FUSILLI_UTILS_H
#define FUSILLI_UTILS_H

#include "fusilli/logging.h"

#include <filesystem>
#include <format>
#include <fstream>
#include <ios>
#include <iostream>
#include <string>
#include <unistd.h>

namespace fusilli {

// A RAII type for creating + destroying cache files in $HOME/.cache/fusilli.
//
//  void example() {
//    {
//      ErrorOr<CacheFile> cacheFile = CacheFile::create(
//          /*graphName=*/"example_graph", /*filename=*/"input",
//          /*remove=*/true);
//      assert(isOk(cacheFile));
//
//      assert(isOk(CacheFile::open(/*graphName=*/"example_graph",
//                                  /*filename=*/"input")));
//    }
//    assert(isError(CacheFile::open(/*graphName=*/"example_graph",
//                                   /*filename=*/"input")));
//
//    {
//      ErrorOr<CacheFile> cacheFile = CacheFile::create(
//          /*graphName=*/"example_graph", /*filename=*/"input",
//          /*remove=*/false);
//      assert(isOk(cacheFile));
//    }
//    assert(isOk(CacheFile::open(/*graphName=*/"example_graph",
//                                /*filename=*/"input")));
//  }
class CacheFile {
public:
  // Factory constructor that creates file, overwriting an existing file, and
  // returns an ErrorObject if file could not be created.
  static ErrorOr<CacheFile> create(const std::string &graphName,
                                   const std::string &filename, bool remove);

  // Factory constructor that opens an existing file and returns ErrorObject if
  // the file does not exist.
  static ErrorOr<CacheFile> open(const std::string &graphName,
                                 const std::string &filename);

  CacheFile(CacheFile &&other) noexcept
      : path(std::move(other.path)), remove_(other.remove_) {
    other.path.clear();
    other.remove_ = false;
  }

  ~CacheFile() {
    if (remove_ && !path.empty()) {
      std::filesystem::remove(path);
    }
  }

  // Path of file this class wraps.
  std::filesystem::path path;

  // Delete copy constructor + assignment operators. A copy constructor would
  // likely not be safe, as the destructor for a copy could remove the
  // underlying file while the original is still expecting it to exist.
  CacheFile(const CacheFile &) = delete;
  CacheFile &operator=(const CacheFile &) = delete;
  CacheFile &operator=(CacheFile &&other) noexcept = delete;

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
  // Class should be constructed using the `create` factory function.
  CacheFile(std::filesystem::path path, bool remove)
      : path(path), remove_(remove) {}

  // Whether to remove the file on destruction or not.
  bool remove_;

  // Utility method to get the cache file path.
  static std::filesystem::path getCacheFilePath(const std::string &graphName,
                                                const std::string &filename) {
    // Ensure graph graphName is safe to use as a directory name, we assume
    // filename is safe.
    std::string sanitizedGraphName = graphName;
    std::transform(sanitizedGraphName.begin(), sanitizedGraphName.end(),
                   sanitizedGraphName.begin(),
                   [](char c) { return c == ' ' ? '_' : c; });
    std::erase_if(sanitizedGraphName, [](unsigned char c) {
      return !(std::isalnum(c) || c == '_');
    });

    const char *homeDir = std::getenv("HOME");
    return std::filesystem::path(homeDir) / ".cache" / "fusilli" /
           sanitizedGraphName / filename;
  }
};

inline ErrorOr<CacheFile> CacheFile::create(const std::string &graphName,
                                            const std::string &filename,
                                            bool remove) {
  std::filesystem::path cacheFilePath = getCacheFilePath(graphName, filename);
  FUSILLI_LOG_LABEL_ENDL("Createing Cache file");
  FUSILLI_LOG_ENDL(cacheFilePath);

  // Create directory $HOME/.cache/fusilli/<graphName>
  std::filesystem::path cacheDir = cacheFilePath.parent_path();
  std::error_code ec;
  std::filesystem::create_directories(cacheDir, ec);
  FUSILLI_RETURN_ERROR_IF(ec, ErrorCode::FileSystemFailure,
                          "Failed to create cache directory: " +
                              cacheDir.string() + " - " + ec.message());

  // Create file $HOME/.cache/fusilli/<graphName>/<filename>
  std::ofstream file(cacheFilePath);
  FUSILLI_RETURN_ERROR_IF(!file.is_open(), ErrorCode::FileSystemFailure,
                          "Failed to create file: " + cacheFilePath.string());
  file.close();

  return ok(CacheFile(cacheFilePath, remove));
}

inline ErrorOr<CacheFile> CacheFile::open(const std::string &graphName,
                                          const std::string &filename) {
  std::filesystem::path cacheFilePath = getCacheFilePath(graphName, filename);

  // Check if the file exists.
  FUSILLI_RETURN_ERROR_IF(!std::filesystem::exists(cacheFilePath),
                          ErrorCode::FileSystemFailure,
                          "File does not exist: " + cacheFilePath.string());

  return ok(CacheFile(cacheFilePath, false));
}

// An STL-style algorithm similar to std::for_each that applies a second
// functor between every pair of elements.
//
// This provides the control flow logic to, for example, print a
// comma-separated list:
//
//   interleave(names.begin(), names.end(),
//              [&](std::string name) { os << name; },
//              [&] { os << ", "; });
//
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor>
inline void interleave(ForwardIterator begin, ForwardIterator end,
                       UnaryFunctor each_fn, NullaryFunctor between_fn) {
  if (begin == end)
    return;
  each_fn(*begin);
  ++begin;
  for (; begin != end; ++begin) {
    between_fn();
    each_fn(*begin);
  }
}

// An overload of `interleave` which additionally accepts a SkipFunctor
// to skip certain elements based on a predicate.
//
// This provides the control flow logic to, for example, print a
// comma-separated list excluding "foo":
//
//   interleave(names.begin(), names.end(),
//              [&](std::string name) { os << name; },
//              [&] { os << ", "; },
//              [&](std::string name) { return name == "foo"; });
//
template <typename ForwardIterator, typename UnaryFunctor,
          typename NullaryFunctor, typename SkipFunctor>
inline void interleave(ForwardIterator begin, ForwardIterator end,
                       UnaryFunctor each_fn, NullaryFunctor between_fn,
                       SkipFunctor skip_fn) {
  if (begin == end)
    return;
  bool first = true;
  for (; begin != end; ++begin) {
    if (!skip_fn(*begin)) {
      if (!first)
        between_fn();
      first = false;
      each_fn(*begin);
    }
  }
}

} // namespace fusilli
#endif // FUSILLI_UTILS_H
