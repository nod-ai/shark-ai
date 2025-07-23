// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusili.h>

#include <catch2/catch_test_macros.hpp>
#include <cstdlib>
#include <sstream>

using namespace fusili;

TEST_CASE("ConditionalStreamer conditioned on isLoggingEnabled", "[logging]") {
  // Create a string stream to capture the output
  std::ostringstream oss;
  ConditionalStreamer logger(oss);

  // When env variable is set to 0, disable logging
  isLoggingEnabled() = false;
  // ^ force mimics the effect of setenv("FUSILI_LOG_INFO", "0", 1);
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

// This test is disabled because getStream() statically initializes
// the stream ref picking the first snapshot of FUSILI_LOG_FILE
// env variable. So subsequent tests that change the env variable (in
// the same process) will not affect the stream returned by getStream().
TEST_CASE("getStream stdout mode", "[logging][.]") {
  setenv("FUSILI_LOG_FILE", "stdout", 1);
  std::ostream &stream = getStream();
  REQUIRE(&stream == &std::cout);

  unsetenv("FUSILI_LOG_FILE");
}

// This test is disabled because getStream() statically initializes
// the stream ref picking the first snapshot of FUSILI_LOG_FILE
// env variable. So subsequent tests that change the env variable (in
// the same process) will not affect the stream returned by getStream().
TEST_CASE("getStream stderr mode", "[logging][.]") {
  setenv("FUSILI_LOG_FILE", "stderr", 1);
  std::ostream &stream = getStream();
  REQUIRE(&stream == &std::cerr);

  unsetenv("FUSILI_LOG_FILE");
}

// This test is disabled because getStream() statically initializes
// the stream ref picking the first snapshot of FUSILI_LOG_FILE
// env variable. So subsequent tests that change the env variable (in
// the same process) will not affect the stream returned by getStream().
TEST_CASE("getStream file mode", "[logging][.]") {
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

TEST_CASE("error_t and error_code_t operators and methods", "[logging]") {
  SECTION("Default constructed error_t is OK") {
    fusili::error_t err;
    REQUIRE(err.code == error_code_t::OK);
    REQUIRE(err.getCode() == error_code_t::OK);
    REQUIRE(err.getMessage() == "");
    REQUIRE(err.isOk());
    REQUIRE(!err.isFailure());
    REQUIRE(err == error_code_t::OK);
  }

  SECTION("Custom error_t construction and comparison") {
    fusili::error_t err(error_code_t::AttributeNotSet, "missing attribute");
    REQUIRE(err.code == error_code_t::AttributeNotSet);
    REQUIRE(err.getCode() == error_code_t::AttributeNotSet);
    REQUIRE(err.getMessage() == "missing attribute");
    REQUIRE(!err.isOk());
    REQUIRE(err.isFailure());
    REQUIRE(err == error_code_t::AttributeNotSet);
  }

  SECTION("operator<< for error_code_t") {
    std::ostringstream oss;
    oss << error_code_t::OK;
    REQUIRE(oss.str() == "OK");
    oss.str("");
    oss << error_code_t::AttributeNotSet;
    REQUIRE(oss.str() == "ATTRIBUTE_NOT_SET");
    oss.str("");
    oss << static_cast<error_code_t>(9999); // Unknown code
    REQUIRE(oss.str() == "UNKNOWN_ERROR_CODE");
  }

  SECTION("operator<< for error_t") {
    fusili::error_t err(error_code_t::InvalidAttribute, "bad attr");
    std::ostringstream oss;
    oss << err;
    // Should contain both code and message
    REQUIRE(oss.str().find("INVALID_ATTRIBUTE") != std::string::npos);
    REQUIRE(oss.str().find("bad attr") != std::string::npos);
  }
}
