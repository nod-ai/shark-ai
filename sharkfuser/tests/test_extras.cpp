// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include <catch2/catch_test_macros.hpp>

using namespace fusilli;

TEST_CASE("TempFile", "[TempFile]") {
  ErrorOr<TempFile> t = TempFile::create("test_temp_file");
  REQUIRE(isOk(t));

  // Roundtrip writing and reading from the temporary file
  REQUIRE(isOk(t->write("test content")));
  ErrorOr<std::string> content = t->read();
  REQUIRE(isOk(content));
  REQUIRE(*content == "test content");

  // Check that the file is removed on destruction
  {
    TempFile tempFile = std::move(*t);
    REQUIRE(std::filesystem::exists(tempFile.path) == true);
  }
  REQUIRE(std::filesystem::exists(t->path) == false);
}
