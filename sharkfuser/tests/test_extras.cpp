// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <filesystem>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

using namespace fusilli;

TEST_CASE("CacheFile::create", "[CacheFile]") {
  SECTION("remove = true") {
    std::filesystem::path cacheFilePath;
    {
      CacheFile cf = FUSILLI_REQUIRE_UNWRAP(CacheFile::create(
          /*graphName=*/"graph", /*filename=*/"test_temp_file",
          /*remove=*/true));

      // Double check the path exists
      cacheFilePath = cf.path;
      REQUIRE(std::filesystem::exists(cacheFilePath));

      // Roundtrip writing and reading.
      REQUIRE(isOk(cf.write("test content")));
      std::string content = FUSILLI_REQUIRE_UNWRAP(cf.read());
      REQUIRE(content == "test content");
    }

    // Cache file should be removed
    REQUIRE(!std::filesystem::exists(cacheFilePath));
  }

  SECTION("remove = false") {
    std::filesystem::path cacheFilePath;
    {
      CacheFile cf = FUSILLI_REQUIRE_UNWRAP(CacheFile::create(
          /*graphName=*/"graph", /*filename=*/"test_temp_file",
          /*remove=*/false));

      // Double check the path exists
      cacheFilePath = cf.path;
      REQUIRE(std::filesystem::exists(cacheFilePath));
    }

    // Cache file should not have been removed
    REQUIRE(std::filesystem::exists(cacheFilePath));

    // Remote test artifacts.
    std::filesystem::remove_all(cacheFilePath.parent_path());
  }
}

TEST_CASE("CacheFile::open", "[CacheFile]") {
  // Try to open a file that doesn't exist
  ErrorOr<CacheFile> failOpen = CacheFile::open("test_graph", "test_file.txt");
  REQUIRE(isError(failOpen));
  ErrorObject err(failOpen);
  REQUIRE(err.getCode() == ErrorCode::FileSystemFailure);
  REQUIRE_THAT(err.getMessage(),
               Catch::Matchers::ContainsSubstring("File does not exist"));

  // Create the file
  CacheFile cacheFile = FUSILLI_REQUIRE_UNWRAP(CacheFile::create(
      /*graphName=*/"test_graph",
      /*filename=*/"test_file.txt",
      /*remove=*/true));
  REQUIRE(isOk(cacheFile.write("test data")));

  // Now open the existing file
  CacheFile opened =
      FUSILLI_REQUIRE_UNWRAP(CacheFile::open("test_graph", "test_file.txt"));

  // Verify we can read the content
  std::string content = FUSILLI_REQUIRE_UNWRAP(opened.read());
  REQUIRE(content == "test data");

  // Remote test artifacts.
  std::filesystem::remove_all(cacheFile.path.parent_path());
}

TEST_CASE("CacheFile directory sanitization", "[CacheFile]") {
  // Test that special characters in graph name are sanitized
  CacheFile cacheFile = FUSILLI_REQUIRE_UNWRAP(
      CacheFile::create(/*graphName=*/"test / gr@ph!",
                        /*filename=*/"test_file.txt", /*remove=*/true));

  // Extract the sanitized directory name from the path
  std::filesystem::path dirPath = cacheFile.path.parent_path();
  std::string actualDirName = dirPath.filename().string();

  // Spaces should be replaced with underscores, special chars removed
  REQUIRE(actualDirName == "test__grph");

  // Verify the file was actually created
  REQUIRE(std::filesystem::exists(cacheFile.path));

  // Remote test artifacts.
  std::filesystem::remove_all(cacheFile.path.parent_path());
}
