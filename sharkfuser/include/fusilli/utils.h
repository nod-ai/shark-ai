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
#include <iostream>
#include <string>
#include <unistd.h>

namespace fusilli {

// A RAII type for creating + destroying temporary files.
//
//   ErrorOr<std::string> runCmd(const std::string &input) {
//     auto inputFile = FUSILLI_TRY(TempFile::create("input"));
//     FUSILLI_CHECK_ERROR(inputFile.write(input));
//     auto outputFile = FUSILLI_TRY(TempFile::create("output"));
//     auto cmd = std::format("compiler {} -o {}",
//                            inputFile.path.string(),
//                            outputFile.path.string());
//     std::system(cmd.c_str());
//     return outputFile.read();
//   }
struct TempFile {
private:
  // Whether to remove the temporary file on destruction.
  bool remove;

  // Class should be constructed using the `create` factory function.
  TempFile(std::filesystem::path path, bool remove)
      : path(path), remove(remove) {}

public:
  // Factory constructor, returns ErrorObject if temporary file could not be
  // created. `prefix` will be prefixed to temp file name for disambiguation,
  // `remove` controls if the temp file is removed on object destruction.
  static ErrorOr<TempFile> create(const std::string &prefix = "",
                                  bool remove = true);

  TempFile(TempFile &&other) noexcept
      : path(std::move(other.path)), remove(other.remove) {
    other.path.clear();
    other.remove = false;
  }

  ~TempFile() {
    if (remove && !path.empty()) {
      std::filesystem::remove(path);
    }
  }

  // Path of temporary file this class wraps.
  std::filesystem::path path;

  // Delete copy constructor + assignment operators. A copy constructor would
  // likely not be safe, as the destructor for a copy could remove the
  // underlying temp file while the original is still expecting it to exist.
  TempFile(const TempFile &) = delete;
  TempFile &operator=(const TempFile &) = delete;
  TempFile &operator=(TempFile &&other) noexcept = delete;

  // Write to temporary file.
  ErrorObject write(const std::string &content) {
    std::ofstream file(path);
    FUSILI_RETURN_ERROR_IF(!file.is_open(), ErrorCode::FileSystem,
                           "Failed to open file: " + path.string());

    file << content;
    FUSILI_RETURN_ERROR_IF(!file.good(), ErrorCode::FileSystem,
                           "Failed to write to file: " + path.string())

    return ok();
  }

  // Read from temporary file.
  ErrorOr<std::string> read() {
    // std::ios::ate opens file and moves the cursor to the end, allowing us
    // to get the file size with tellg().
    std::ifstream file(path, std::ios::binary | std::ios::ate);
    FUSILI_RETURN_ERROR_IF(!file.is_open(), ErrorCode::FileSystem,
                           "Failed to open file: " + path.string());

    // Copy the contents of the file into a string.
    const std::streamsize size = file.tellg();
    file.seekg(0, std::ios::beg);
    std::string buffer(size, '\0');
    file.read(buffer.data(), size);
    FUSILI_RETURN_ERROR_IF(!file.good(), ErrorCode::FileSystem,
                           "Failed to read file: " + path.string());

    return ok(buffer);
  }
};

inline ErrorOr<TempFile> TempFile::create(const std::string &prefix,
                                          bool remove) {
  // create temp file using mkstemp
  auto tempDir = std::filesystem::temp_directory_path();
  auto tempFilename = std::format("sharkfuser_{}_XXXXXX", prefix);
  auto tempPath = tempDir / tempFilename;
  std::string tmpFile = tempPath.string();
  int fileDescriptor = mkstemp(tmpFile.data());
  FUSILI_RETURN_ERROR_IF(fileDescriptor == -1, ErrorCode::FileSystem,
                         "failed to create temp file");
  close(fileDescriptor);

  return ok(TempFile(tmpFile, remove));
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
