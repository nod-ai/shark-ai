// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//===----------------------------------------------------------------------===//
//
// This file contains utilities for logging and error codes.
//
//===----------------------------------------------------------------------===//

#ifndef FUSILI_LOGGING_H
#define FUSILI_LOGGING_H

#include <cassert>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <variant>

namespace fusili {

enum class [[nodiscard]] ErrorCode {
  OK,
  NotImplemented,
  AttributeNotSet,
  InvalidAttribute,
  TensorNotFound,
};

static const std::unordered_map<ErrorCode, std::string> ErrorCodeToStr = {
    {ErrorCode::OK, "OK"},
    {ErrorCode::NotImplemented, "NOT_IMPLEMENTED"},
    {ErrorCode::AttributeNotSet, "ATTRIBUTE_NOT_SET"},
    {ErrorCode::InvalidAttribute, "INVALID_ATTRIBUTE"},
    {ErrorCode::TensorNotFound, "TENSOR_NOT_FOUND"}};

struct [[nodiscard]] ErrorObject {
  ErrorCode code;
  std::string errMsg;

  ErrorObject() : code(ErrorCode::OK), errMsg("") {}
  ErrorObject(ErrorCode err, std::string msg)
      : code(err), errMsg(std::move(msg)) {}

  ErrorCode getCode() const { return code; }
  const std::string &getMessage() const { return errMsg; }
  bool isOk() const { return code == ErrorCode::OK; }
  bool isError() const { return !isOk(); }

  bool operator==(ErrorCode compareCode) const { return code == compareCode; }
  bool operator!=(ErrorCode compareCode) const { return code != compareCode; }
};

// Utility function that returns true if an ErrorObject represents a successful
// result.
inline bool isOk(ErrorObject result) { return result.isOk(); }

// Utility function that returns true if an ErrorObject represents an
// unsuccessful result.
inline bool isError(ErrorObject result) { return result.isError(); }

// Utility function to generate an ErrorObject representing a successful result.
inline ErrorObject ok() { return {ErrorCode::OK, ""}; }

// Utility function to generate an ErrorObject representing an unsuccessful
// result.
template <typename S>
  requires std::convertible_to<S, std::string>
inline ErrorObject error(ErrorCode err, S &&errMsg) {
  return ErrorObject(err, std::forward<S>(errMsg));
}

// Represents either an error or a value T.
//
// ErrorOr<T> represents the results of an operation. In the successful case it
// provides pointer/optional<T> like semantics to the underlying T, on the
// failure path it is convertible to an ErrorObject describing the failure
// preventing T from being created.
//
// usage:
//   ErrorOr<AST> buildAST() {
//      ErrorOr<std::string> maybeBuffer = getBuffer();
//      FUSILI_CHECK_ERROR(maybeBuffer);
//      AST ast = buildAST(*maybeBuffer);
//      return ok(ast);
//   }
template <typename T> class [[nodiscard]] ErrorOr {
private:
  using Storage = std::variant<T, ErrorObject>;

  Storage storage;

  // The intended consumption pattern is to use `isOk` and `isError` utility
  // methods to check this class when converted to an ErrorObject.
  bool has_value() const noexcept { return std::holds_alternative<T>(storage); }

  // Friend declaration to allow ErrorOr<U> to access ErrorOr<T>'s private
  // members
  template <typename U> friend class ErrorOr;

public:
  // Construct successful case from anything T is constructable from.
  template <typename U>
    requires std::constructible_from<T, U &&>
  ErrorOr(U &&val) : storage(std::in_place_type<T>, std::forward<U>(val)) {}

  // Move constructor.
  ErrorOr(ErrorOr &&other) noexcept : storage(std::move(other.storage)) {}

  // Move constructor for differing types, to allow for ErrorOr<const char *> to
  // ErrorOr<std::string> for example.
  template <typename U>
    requires std::is_constructible_v<T, U>
  ErrorOr(ErrorOr<U> &&other) {
    if (isOk(other)) {
      storage = Storage(std::in_place_type<T>, std::move(*other));
    } else {
      storage = Storage(std::in_place_type<ErrorObject>,
                        std::move(std::get<ErrorObject>(other.storage)));
    }
  }

  // Construct implicitly from ErrorObject to support returning `error(...)`
  // i.e.
  //   ErrorOr<string> method() {
  //     std::string output;
  //     if(int returnCode = runShellScript(&output)) {
  //        return error(ErrorCode::ScriptFail, "shell script error");
  //     }
  //     return ok(output);
  //   }
  ErrorOr(ErrorObject errorObject)
      : storage(std::in_place_type<ErrorObject>, errorObject) {
    assert(isError(std::get<ErrorObject>(storage)) &&
           "successful results should be constructed with T type");
  }

  // Delete copy constructor + all assignment operators.
  ErrorOr(const ErrorOr &other) = delete;
  ErrorOr &operator=(const ErrorOr &) = delete;
  ErrorOr &operator=(ErrorOr &&) = delete;
  template <typename U> ErrorOr &operator=(const ErrorOr<U> &) = delete;
  template <typename U> ErrorOr &operator=(ErrorOr<U> &&) = delete;

  // Convert to error object.
  operator ErrorObject() const {
    if (std::holds_alternative<T>(storage)) {
      return ok();
    }
    return std::get<ErrorObject>(storage);
  }

#define ACCESSOR_ERROR                                                         \
  "ErrorOr<T> does not hold a value, ErrorOr<T> should be checked with "       \
  "isOk() before dereferencing." // Error string for asserts below.

  // Dereference operator - returns a reference to the contained value The
  // ErrorOr must be in success state (checked via isOk()) before calling
  // accessor methods.
  T &operator*() {
    assert(has_value() && ACCESSOR_ERROR);
    return std::get<T>(storage);
  }

  // Const dereference operator. The ErrorOr must be in success state (checked
  // via isOk()) before calling accessor methods.
  const T &operator*() const {
    assert(has_value() && ACCESSOR_ERROR);
    return std::get<T>(storage);
  }

  // Member access operator - returns a pointer to the contained value. The
  // ErrorOr must be in success state (checked via isOk()) before calling
  // accessor methods.
  T *operator->() {
    assert(has_value() && ACCESSOR_ERROR);
    return &std::get<T>(storage);
  }

  // Const member access operator. The ErrorOr must be in success state (checked
  // via isOk()) before calling accessor methods.
  const T *operator->() const {
    assert(has_value() && ACCESSOR_ERROR);
    return &std::get<T>(storage);
  }
#undef ACCESSOR_ERROR
};

// Override of ok utility method allowing for a similar consumption pattern
// between ErrorOr and ErrorObject.
//
//   ErrorOr<int> get42() {
//     int i = getInt();
//     if (i != 42) {
//        return error(ErrorCode::InvalidAttribute, "expected 42");
//     }
//     return ok(i);
//   }
template <typename T> inline auto ok(T &&y) {
  return ErrorOr<std::decay_t<T>>(std::forward<T>(y));
}

// Stream operator for ErrorCode
inline std::ostream &operator<<(std::ostream &os, const ErrorCode &code) {
  auto it = ErrorCodeToStr.find(code);
  if (it != ErrorCodeToStr.end())
    os << it->second;
  else
    os << "UNKNOWN_ERROR_CODE";
  return os;
}

// Stream operator for ErrorObject
inline std::ostream &operator<<(std::ostream &os, const ErrorObject &err) {
  os << err.getCode() << ": " << err.getMessage();
  return os;
}

inline bool &isLoggingEnabled() {
  static bool logEnabled = []() -> bool {
    const char *envVal = std::getenv("FUSILI_LOG_INFO");
    // Disabled when FUSILI_LOG_INFO is not set
    if (!envVal) {
      return false;
    }
    std::string envValStr(envVal);
    // Disabled when FUSILI_LOG_INFO == "" (empty string)
    // Disabled when FUSILI_LOG_INFO == "0", any other value enables it
    return !envValStr.empty() && envValStr[0] != '0';
  }();
  return logEnabled;
}

// Get the logging stream based on `FUSILI_LOG_FILE`
//   When not set, logging is disabled.
//   When set to `stdout`, uses `std::cout`.
//   When set to `stderr`, uses `std::cerr`.
//   When set to /some/file/path.txt, uses that.
inline std::ostream &getStream() {
  static std::ofstream outFile;
  static std::ostream &stream = []() -> std::ostream & {
    const char *logFile = std::getenv("FUSILI_LOG_FILE");
    if (!logFile) {
      isLoggingEnabled() = false;
      return std::cout;
    }
    std::string filePath(logFile);
    if (filePath == "stdout") {
      return std::cout;
    } else if (filePath == "stderr") {
      return std::cerr;
    } else {
      outFile.open(logFile, std::ios::out);
      return outFile;
    }
  }();
  return stream;
}

class ConditionalStreamer {
public:
  explicit ConditionalStreamer(std::ostream &stream) : stream_(stream) {}

  template <typename T>
  const ConditionalStreamer &operator<<(const T &t) const {
    if (isLoggingEnabled()) {
      stream_ << t;
    }
    return *this;
  }

  const ConditionalStreamer &
  operator<<(std::ostream &(*spl)(std::ostream &)) const {
    if (isLoggingEnabled()) {
      stream_ << spl;
    }
    return *this;
  }

private:
  std::ostream &stream_;
};

inline ConditionalStreamer &getLogger() {
  static ConditionalStreamer logger(getStream());
  return logger;
}

} // namespace fusili

// Macros for logging and error handling
#define FUSILI_COLOR_RED "\033[31m"
#define FUSILI_COLOR_GREEN "\033[32m"
#define FUSILI_COLOR_YELLOW "\033[33m"
#define FUSILI_COLOR_RESET "\033[0m"

#define FUSILI_LOG(X) fusili::getLogger() << X
#define FUSILI_LOG_ENDL(X) fusili::getLogger() << X << std::endl
#define FUSILI_LOG_LABEL_RED(X)                                                \
  fusili::getLogger() << FUSILI_COLOR_RED << "[FUSILI] " << X                  \
                      << FUSILI_COLOR_RESET
#define FUSILI_LOG_LABEL_GREEN(X)                                              \
  fusili::getLogger() << FUSILI_COLOR_GREEN << "[FUSILI] " << X                \
                      << FUSILI_COLOR_RESET
#define FUSILI_LOG_LABEL_YELLOW(X)                                             \
  fusili::getLogger() << FUSILI_COLOR_YELLOW << "[FUSILI] " << X               \
                      << FUSILI_COLOR_RESET
#define FUSILI_LOG_LABEL_ENDL(X)                                               \
  fusili::getLogger() << "[FUSILI] " << X << std::endl

#define FUSILI_RETURN_ERROR_IF(cond, retval, message)                          \
  do {                                                                         \
    if (cond) {                                                                \
      if (retval == fusili::ErrorCode::OK)                                     \
        FUSILI_LOG_LABEL_YELLOW("INFO: ");                                     \
      else                                                                     \
        FUSILI_LOG_LABEL_RED("ERROR: ");                                       \
      FUSILI_LOG_ENDL(retval << ": " << message << ": (" << #cond ") at "      \
                             << __FILE__ << ":" << __LINE__);                  \
      return error(retval, message);                                           \
    }                                                                          \
  } while (false);

#define FUSILI_CHECK_ERROR(x)                                                  \
  do {                                                                         \
    if (isError(x)) {                                                          \
      FUSILI_LOG_LABEL_RED("ERROR: ");                                         \
      FUSILI_LOG_ENDL(#x << " at " << __FILE__ << ":" << __LINE__);            \
      return ErrorObject(x);                                                   \
    }                                                                          \
  } while (false);

#endif // FUSILI_LOGGING_H
