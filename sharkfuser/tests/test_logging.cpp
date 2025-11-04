// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <fusilli.h>

#include "utils.h"

#include <catch2/catch_test_macros.hpp>

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <format>
#include <fstream>
#include <iostream>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdlib.h>
#include <utility>

using namespace fusilli;

TEST_CASE("ConditionalStreamer conditioned on isLoggingEnabled", "[logging]") {
  // Create a string stream to capture the output.
  std::ostringstream oss;
  ConditionalStreamer logger(oss);

  // When env variable is set to 0, disable logging.
  isLoggingEnabled() = false;
  // ^ force mimics the effect of setenv("FUSILLI_LOG_INFO", "0", 1);
  oss.str("");
  logger << "Hello World";
  REQUIRE(oss.str().empty());
  REQUIRE(!isLoggingEnabled());

  // When env variable is set to 1, enable logging.
  isLoggingEnabled() = true;
  // ^ force mimics the effect of setenv("FUSILLI_LOG_INFO", "1", 1);
  oss.str("");
  logger << "Hello World";
  REQUIRE(oss.str() == "Hello World");
  REQUIRE(isLoggingEnabled());

  // When env variable is not set, disable logging.
  isLoggingEnabled() = false;
  // ^ force mimics the effect of unsetenv("FUSILLI_LOG_INFO");
  oss.str("");
  logger << "Hello World";
  REQUIRE(oss.str().empty());
  REQUIRE(!isLoggingEnabled());
}

// This test is disabled because getStream() statically initializes
// the stream ref picking the first snapshot of FUSILLI_LOG_FILE
// env variable. So subsequent tests that change the env variable (in
// the same process) will not affect the stream returned by getStream().
TEST_CASE("getStream stdout mode", "[logging][.]") {
  setenv("FUSILLI_LOG_FILE", "stdout", 1);
  std::ostream &stream = getStream();
  REQUIRE(&stream == &std::cout);

  unsetenv("FUSILLI_LOG_FILE");
}

// This test is disabled because getStream() statically initializes
// the stream ref picking the first snapshot of FUSILLI_LOG_FILE
// env variable. So subsequent tests that change the env variable (in
// the same process) will not affect the stream returned by getStream().
TEST_CASE("getStream stderr mode", "[logging][.]") {
  setenv("FUSILLI_LOG_FILE", "stderr", 1);
  std::ostream &stream = getStream();
  REQUIRE(&stream == &std::cerr);

  unsetenv("FUSILLI_LOG_FILE");
}

// This test is disabled because getStream() statically initializes
// the stream ref picking the first snapshot of FUSILLI_LOG_FILE
// env variable. So subsequent tests that change the env variable (in
// the same process) will not affect the stream returned by getStream().
TEST_CASE("getStream file mode", "[logging][.]") {
  const char *testFile = "/tmp/test_fusilli_log.txt";
  setenv("FUSILLI_LOG_FILE", testFile, 1);
  std::ostream &stream = getStream();
  REQUIRE(&stream != &std::cout);
  REQUIRE(&stream != &std::cerr);
  // Check that the stream reference is indeed pointing to
  // a file stream and not cout / cerr.
  REQUIRE(dynamic_cast<std::ofstream *>(&stream));

  // Cleanup.
  unsetenv("FUSILLI_LOG_FILE");
  std::remove(testFile);
}

TEST_CASE("error_t and ErrorCode operators and methods", "[logging]") {
  SECTION("Default constructed error_t is OK") {
    ErrorObject err;
    REQUIRE(err.code == ErrorCode::OK);
    REQUIRE(err.getCode() == ErrorCode::OK);
    REQUIRE(err.getMessage() == "");
    FUSILLI_REQUIRE_OK(err);
    REQUIRE(!isError(err));
    REQUIRE(err == ErrorCode::OK);
  }

  SECTION("Custom error_t construction and comparison") {
    ErrorObject err(ErrorCode::AttributeNotSet, "missing attribute");
    REQUIRE(err.code == ErrorCode::AttributeNotSet);
    REQUIRE(err.getCode() == ErrorCode::AttributeNotSet);
    REQUIRE(err.getMessage() == "missing attribute");
    REQUIRE(!isOk(err));
    REQUIRE(isError(err));
    REQUIRE(err == ErrorCode::AttributeNotSet);
  }

  SECTION("operator<< for ErrorCode") {
    std::ostringstream oss;
    oss << ErrorCode::OK;
    REQUIRE(oss.str() == "OK");
    oss.str("");
    oss << ErrorCode::AttributeNotSet;
    REQUIRE(oss.str() == "ATTRIBUTE_NOT_SET");
    oss.str("");
    // Unknown code.
    oss << static_cast<ErrorCode>(9999);
    REQUIRE(oss.str() == "UNKNOWN_ERROR_CODE");
  }

  SECTION("operator<< for error_t") {
    ErrorObject err(ErrorCode::InvalidAttribute, "bad attr");
    std::ostringstream oss;
    oss << err;
    // Should contain both code and message.
    REQUIRE(oss.str().find("INVALID_ATTRIBUTE") != std::string::npos);
    REQUIRE(oss.str().find("bad attr") != std::string::npos);
  }
}

TEST_CASE("ErrorOr construction", "[logging][erroror]") {
  SECTION("Construct from value") {
    ErrorOr<int> result = ok(42);
    FUSILLI_REQUIRE_OK(result);
    REQUIRE(!isError(result));
    REQUIRE(*result == 42);
  }

  SECTION("Construct from rvalue reference") {
    {
      ErrorOr<std::string> strResult = ok("hello");
      FUSILLI_REQUIRE_OK(strResult);
      REQUIRE(*strResult == "hello");
    }
    {
      auto ptr = std::make_unique<int>(42);
      ErrorOr<std::unique_ptr<int>> result(std::move(ptr));
      FUSILLI_REQUIRE_OK(result);
      REQUIRE(**result == 42);
    }
  }

  SECTION("Construct from lvalue reference") {
    std::string str = "This is a very long string that should be moved";
    ErrorOr<std::string> result = ok(str);
    FUSILLI_REQUIRE_OK(result);
    REQUIRE(*result == str);
  }

  SECTION("Construct from error") {
    ErrorOr<int> result = error(ErrorCode::NotImplemented, "not impl");
    REQUIRE(!isOk(result));
    REQUIRE(isError(result));
  }

  SECTION("Move construction") {
    { // Different types value case.
      ErrorOr<const char *> source = ok("hello");
      ErrorOr<std::string> destination = std::move(source);
      FUSILLI_REQUIRE_OK(destination);
      REQUIRE(*destination == "hello");
    }
    { // Different types error case.
      ErrorOr<const char *> source =
          error(ErrorCode::NotImplemented, "test case");
      ErrorOr<std::string> destination = std::move(source);
      ErrorObject err = destination; // Convert to ErrorObject
      REQUIRE(isError(err));
      REQUIRE(err.getCode() == ErrorCode::NotImplemented);
      REQUIRE(err.getMessage() == "test case");
    }
    { // Same types value case.
      ErrorOr<std::string> source = ok("hello");
      ErrorOr<std::string> destination = std::move(source);
      FUSILLI_REQUIRE_OK(destination);
      REQUIRE(*destination == "hello");
    }
    { // Same types error case.
      ErrorOr<std::string> source =
          error(ErrorCode::NotImplemented, "test case");
      ErrorOr<std::string> destination = std::move(source);
      ErrorObject err = destination; // Convert to ErrorObject
      REQUIRE(isError(err));
      REQUIRE(err.getCode() == ErrorCode::NotImplemented);
      REQUIRE(err.getMessage() == "test case");
    }
  }
}

TEST_CASE("ErrorOr accessors", "[logging][erroror]") {
  SECTION("Dereference operator") {
    ErrorOr<int> result = ok(100);
    REQUIRE(*result == 100);
    *result = 200;
    REQUIRE(*result == 200);
  }

  SECTION("Arrow operator") {
    struct TestStruct {
      int value;
      std::string name;
    };

    ErrorOr<TestStruct> result = ok(TestStruct{42, "test"});
    REQUIRE(result->value == 42);
    REQUIRE(result->name == "test");
  }
}

TEST_CASE("ErrorOr conversion to ErrorObject", "[logging][erroror]") {
  SECTION("Success case") {
    ErrorOr<int> result = ok(42);
    ErrorObject err = result;
    FUSILLI_REQUIRE_OK(err);
    REQUIRE(err.getCode() == ErrorCode::OK);
    REQUIRE(err.getMessage().empty());
  }

  SECTION("Error case") {
    ErrorOr<int> result =
        error(ErrorCode::VariantPackError, "variant pack error");
    ErrorObject err = result;
    REQUIRE(isError(err));
    REQUIRE(err.getCode() == ErrorCode::VariantPackError);
    REQUIRE(err.getMessage() == "variant pack error");
  }
}

// Test error propagation when a function returning an `ErrorOr` type is called
// by another function returning `ErrorOr` type.
TEST_CASE("ErrorOr -> ErrorOr error propagation", "[logging][erroror]") {
  int failingFunctionCallCount = 0;
  auto failingFunction = [&failingFunctionCallCount]() -> ErrorOr<int> {
    failingFunctionCallCount++;
    return error(ErrorCode::NotImplemented, "not implemented");
  };

  int successFunctionCallCount = 0;
  auto successFunction = [&successFunctionCallCount]() -> ErrorOr<int> {
    successFunctionCallCount++;
    return ok(42);
  };

  auto consumerFunction = [&]() -> ErrorOr<std::string> {
    return ok(std::format("got {}", FUSILLI_TRY(successFunction())));
  };

  auto failingConsumer = [&]() -> ErrorOr<std::string> {
    return ok(std::format("got {}", FUSILLI_TRY(failingFunction())));
  };

  SECTION("Success propagation") {
    successFunctionCallCount = 0;
    ErrorOr<std::string> result = consumerFunction();
    // Check that call count is what's expected. The count might be incorrect
    // if, for example, the macros double evaluate an expression.
    REQUIRE(successFunctionCallCount == 1);
    FUSILLI_REQUIRE_OK(result);
    REQUIRE(*result == "got 42");
  }

  SECTION("Error propagation") {
    failingFunctionCallCount = 0;
    ErrorOr<std::string> result = failingConsumer();
    REQUIRE(failingFunctionCallCount == 1);
    REQUIRE(isError(result));
    ErrorObject err = result;
    REQUIRE(err.getCode() == ErrorCode::NotImplemented);
    REQUIRE(err.getMessage() == "not implemented");
  }
}

// Test error propagation when a function returning an `ErrorOr` type is called
// by another function returning `ErrorObject` type. `ErrorOr`s should freely
// convert to `ErrorObjects`.
TEST_CASE("ErrorOr -> ErrorObject error propagation", "[logging][erroror]") {
  int failingFunctionCallCount = 0;
  auto failingFunction = [&failingFunctionCallCount]() -> ErrorOr<int> {
    failingFunctionCallCount++;
    return error(ErrorCode::NotImplemented, "not implemented");
  };

  int successFunctionCallCount = 0;
  auto successFunction = [&successFunctionCallCount]() -> ErrorOr<int> {
    successFunctionCallCount++;
    return ok(42);
  };

  auto consumerFunction = [&]() -> ErrorObject {
    FUSILLI_CHECK_ERROR(successFunction());

    return ok();
  };

  auto failingConsumer = [&]() -> ErrorObject {
    FUSILLI_CHECK_ERROR(failingFunction());

    assert(false && "unreachable");
    return ok();
  };

  SECTION("Success propagation") {
    successFunctionCallCount = 0;
    ErrorObject result = consumerFunction();
    REQUIRE(successFunctionCallCount == 1);
    FUSILLI_REQUIRE_OK(result);
  }

  SECTION("Error propagation") {
    failingFunctionCallCount = 0;
    ErrorObject result = failingConsumer();
    REQUIRE(failingFunctionCallCount == 1);
    REQUIRE(isError(result));
    REQUIRE(result.getCode() == ErrorCode::NotImplemented);
    REQUIRE(result.getMessage() == "not implemented");
  }
}

// Test error propagation when a function returning an `ErrorObject` type is
// called by another function returning `ErrorOr` type. Failed `ErrorObjects`
// should be free to convert to `ErrorOr`s.
TEST_CASE("ErrorObject -> ErrorOr error propagation", "[logging][erroror]") {
  int failingFunctionCallCount = 0;
  auto failingFunction = [&failingFunctionCallCount]() -> ErrorObject {
    failingFunctionCallCount++;
    return error(ErrorCode::NotImplemented, "not implemented");
  };

  int successFunctionCallCount = 0;
  auto successFunction = [&successFunctionCallCount]() -> ErrorObject {
    successFunctionCallCount++;
    return ok();
  };

  auto consumerFunction = [&]() -> ErrorOr<std::string> {
    FUSILLI_CHECK_ERROR(successFunction());
    return ok("success!");
  };

  auto failingConsumer = [&]() -> ErrorOr<std::string> {
    FUSILLI_CHECK_ERROR(failingFunction());

    assert(false && "unreachable");
    return ok("unreachable");
  };

  SECTION("Success propagation") {
    successFunctionCallCount = 0;
    ErrorOr<std::string> result = consumerFunction();
    REQUIRE(successFunctionCallCount == 1);
    FUSILLI_REQUIRE_OK(result);
    REQUIRE(*result == "success!");
  }

  SECTION("Error propagation") {
    failingFunctionCallCount = 0;
    ErrorOr<std::string> result = failingConsumer();
    REQUIRE(failingFunctionCallCount == 1);
    REQUIRE(isError(result));
    ErrorObject err = result;
    REQUIRE(err.getCode() == ErrorCode::NotImplemented);
    REQUIRE(err.getMessage() == "not implemented");
  }
}
