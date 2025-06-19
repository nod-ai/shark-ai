// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusili.h>

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <sstream>

using namespace fusili::logging;

TEST_CASE("fusili::logging::isLoggingEnabled", "[logging]") {
  // Create a string stream to capture the output
  std::ostringstream oss;
  ConditionalStreamer logger(oss);

  // When env variable is set to 0, disable logging
  setenv("FUSILI_LOG_INFO", "0", 1);
  oss.str("");
  logger << "Hello World";
  REQUIRE(oss.str().empty());
  REQUIRE(!isLoggingEnabled());

  // When env variable is set to 1, enable logging
  isLoggingEnabled() = true;
  // ^ force mimics the effect of setenv("FUSILI_LOG_INFO", "1", 1);
  oss.str("");
  logger << "Hello World";
  REQUIRE(oss.str() == "Hello World");
  REQUIRE(isLoggingEnabled());

  // When env variable is not set, disable logging
  isLoggingEnabled() = false;
  // ^ force mimics the effect of unsetenv("FUSILI_LOG_INFO");
  oss.str("");
  logger << "Hello World";
  REQUIRE(oss.str().empty());
  REQUIRE(!isLoggingEnabled());
}

TEST_CASE("fusili::logging macros", "[logging]") {
  std::ostringstream oss;
  ConditionalStreamer logger(oss);

  // Enable logging
  isLoggingEnabled() = true;

  // Test logging macros
  oss.str("");
  _FUSILI_LOG("Test", logger);
  REQUIRE(oss.str() == "Test");

  oss.str("");
  _FUSILI_LOG_LABEL("Test2", logger);
  REQUIRE(oss.str() == "[FUSILI] Test2");

  oss.str("");
  _FUSILI_LOG_LABEL_ENDL("Test3", logger);
  REQUIRE(oss.str() == "[FUSILI] Test3\n");
}

TEST_CASE("fusili::logging::getStream file mode", "[logging]") {
  const char *test_file = "/tmp/test_fusili_log.txt";
  setenv("FUSILI_LOG_FILE", test_file, 1);
  std::ostream &stream = getStream();
  REQUIRE(&stream != &std::cout);
  REQUIRE(&stream != &std::cerr);
  // Check that the stream reference is indeed pointing to
  // a file stream and not cout / cerr.
  REQUIRE(dynamic_cast<std::ofstream *>(&stream));

  // Cleanup
  unsetenv("FUSILI_LOG_FILE");
  std::remove(test_file);
}

// This test is disabled because getStream() statically initializes
// the stream ref picking the first snapshot of FUSILI_LOG_FILE
// env variable. So subsequent tests that change the env variable (in
// the same process) will not affect the stream returned by getStream().
TEST_CASE("fusili::logging::getStream stdout mode", "[logging][.]") {
  setenv("FUSILI_LOG_FILE", "stdcout", 1);
  std::ostream &stream = getStream();
  REQUIRE(&stream == &std::cout);

  unsetenv("FUSILI_LOG_FILE");
}

// This test is disabled because getStream() statically initializes
// the stream ref picking the first snapshot of FUSILI_LOG_FILE
// env variable. So subsequent tests that change the env variable (in
// the same process) will not affect the stream returned by getStream().
TEST_CASE("fusili::logging::getStream stderr mode", "[logging][.]") {
  setenv("FUSILI_LOG_FILE", "stderr", 1);
  std::ostream &stream = getStream();
  REQUIRE(&stream == &std::cerr);

  unsetenv("FUSILI_LOG_FILE");
}
