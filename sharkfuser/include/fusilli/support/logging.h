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

#ifndef FUSILLI_SUPPORT_LOGGING_H
#define FUSILLI_SUPPORT_LOGGING_H

#include <iree/base/status.h>

#include <cassert>
#include <cstddef>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>
#include <variant>

namespace fusilli {

// An RAII based adapter for writing to C++ `std::strings` using C-style
// `fprint`s.
//
// example usage:
//   // C functions from IREE runtime api.
//   status_t try_thing();
//   bool is_ok(status_t status);
//   void error_message_fprint(FILE *stream, status_t error_status);
//
//   void CPPCodeUsingCAPI() {
//     status_t status = try_thing();
//     if (!is_ok(status)) {
//       std::string err;
//       error_message_fprint(FprintToString(err), status);
//       FUSILLI_LOG_LABEL_RED("try_thing failure: " << err);
//     }
//   }
struct FprintToString {
  char *buffer;
  size_t size;
  FILE *stream;
  std::string &output;

  explicit FprintToString(std::string &output)
      : buffer(nullptr), size(0), stream(open_memstream(&buffer, &size)),
        output(output) {}

  ~FprintToString() {
    fclose(stream);
    output = std::string(buffer);
    free(buffer);
  }

  operator FILE *() { return stream; }

  // Delete all other constructors.
  FprintToString(FprintToString &&other) noexcept = delete;
  FprintToString operator=(const FprintToString &&) noexcept = delete;
  FprintToString(const FprintToString &) = delete;
  FprintToString &operator=(const FprintToString &) = delete;
};

enum class [[nodiscard]] ErrorCode {
  OK,
  NotImplemented,
  NotValidated,
  NotCompiled,
  AttributeNotSet,
  InvalidAttribute,
  TensorNotFound,
  CompileFailure,
  RuntimeFailure,
  FileSystemFailure,
};

static const std::unordered_map<ErrorCode, std::string> ErrorCodeToStr = {
    {ErrorCode::OK, "OK"},
    {ErrorCode::NotImplemented, "NOT_IMPLEMENTED"},
    {ErrorCode::NotValidated, "NOT_VALIDATED"},
    {ErrorCode::NotCompiled, "NOT_COMPILED"},
    {ErrorCode::AttributeNotSet, "ATTRIBUTE_NOT_SET"},
    {ErrorCode::InvalidAttribute, "INVALID_ATTRIBUTE"},
    {ErrorCode::TensorNotFound, "TENSOR_NOT_FOUND"},
    {ErrorCode::CompileFailure, "COMPILE_FAILURE"},
    {ErrorCode::RuntimeFailure, "RUNTIME_FAILURE"},
    {ErrorCode::FileSystemFailure, "FILE_SYSTEM_FAILURE"},
};

struct [[nodiscard]] ErrorObject {
  ErrorCode code;
  std::string errMsg;

  ErrorObject() : code(ErrorCode::OK), errMsg("") {}
  ErrorObject(ErrorCode err, std::string msg)
      : code(err), errMsg(std::move(msg)) {}
  ErrorObject(iree_status_t status) {
    if (iree_status_is_ok(status)) {
      code = ErrorCode::OK;
      errMsg = "";
      return;
    }
    code = ErrorCode::RuntimeFailure;
    // Write error message into `errMsg` variable using the runtime's fprint
    // based reporting.
    iree_status_fprint(FprintToString(errMsg), status);
  }

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
//      FUSILLI_CHECK_ERROR(maybeBuffer);
//      AST ast = buildAST(*maybeBuffer);
//      return ok(ast);
//   }
template <typename T> class [[nodiscard]] ErrorOr {
public:
  // Construct successful case from anything T is constructable from.
  template <typename U>
    requires std::constructible_from<T, U &&>
  ErrorOr(U &&val) : storage_(std::in_place_type<T>, std::forward<U>(val)) {}

  // Move constructor.
  ErrorOr(ErrorOr &&other) noexcept : storage_(std::move(other.storage_)) {}

  // Move constructor for differing types, to allow for ErrorOr<const char *> to
  // ErrorOr<std::string> for example.
  template <typename U>
    requires std::is_constructible_v<T, U>
  ErrorOr(ErrorOr<U> &&other) {
    if (isOk(other))
      storage_ = Storage(std::in_place_type<T>, std::move(*other));
    else
      storage_ = Storage(std::in_place_type<ErrorObject>,
                         std::move(std::get<ErrorObject>(other.storage_)));
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
      : storage_(std::in_place_type<ErrorObject>, errorObject) {
    assert(isError(std::get<ErrorObject>(storage_)) &&
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
    if (std::holds_alternative<T>(storage_))
      return ok();
    return std::get<ErrorObject>(storage_);
  }

#define ACCESSOR_ERROR_MSG                                                     \
  "ErrorOr<T> is in error state (it holds an ErrorObject rather than T) and "  \
  "cannot be dereferenced. ErrorOr<T> state should be checked with "           \
  "isOk() or isError() utility methods before dereferencing."

  // Dereference operator - returns a reference to the contained value. The
  // ErrorOr must be in success state (checked via isOk()) before calling
  // accessor methods.
  T &operator*() {
    assert(hasValue() && ACCESSOR_ERROR_MSG);
    return std::get<T>(storage_);
  }

  // Const dereference operator. The ErrorOr must be in success state (checked
  // via isOk()) before calling accessor methods.
  const T &operator*() const {
    assert(hasValue() && ACCESSOR_ERROR_MSG);
    return std::get<T>(storage_);
  }

  // Member access operator - returns a pointer to the contained value. The
  // ErrorOr must be in success state (checked via isOk()) before calling
  // accessor methods.
  T *operator->() {
    assert(hasValue() && ACCESSOR_ERROR_MSG);
    return &std::get<T>(storage_);
  }

  // Const member access operator. The ErrorOr must be in success state (checked
  // via isOk()) before calling accessor methods.
  const T *operator->() const {
    assert(hasValue() && ACCESSOR_ERROR_MSG);
    return &std::get<T>(storage_);
  }
#undef ACCESSOR_ERROR_MSG

private:
  using Storage = std::variant<T, ErrorObject>;

  Storage storage_;

  // The intended consumption pattern is to use `isOk` and `isError` utility
  // methods to check this class when converted to an ErrorObject.
  bool hasValue() const noexcept { return std::holds_alternative<T>(storage_); }

  // Friend declaration to allow ErrorOr<U> to access ErrorOr<T>'s private
  // members.
  template <typename U> friend class ErrorOr;
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

// Stream operator for ErrorCode.
inline std::ostream &operator<<(std::ostream &os, const ErrorCode &code) {
  if (ErrorCodeToStr.contains(code)) // C++20
    os << ErrorCodeToStr.at(code);
  else
    os << "UNKNOWN_ERROR_CODE";
  return os;
}

// Stream operator for ErrorObject.
inline std::ostream &operator<<(std::ostream &os, const ErrorObject &err) {
  os << err.getCode() << ": " << err.getMessage();
  return os;
}

inline bool &isLoggingEnabled() {
  static bool logEnabled = []() -> bool {
    const char *envVal = std::getenv("FUSILLI_LOG_INFO");
    // Disabled when FUSILLI_LOG_INFO is not set
    if (!envVal)
      return false;
    std::string envValStr(envVal);
    // Disabled when FUSILLI_LOG_INFO == "" (empty string)
    // Disabled when FUSILLI_LOG_INFO == "0", any other value enables it
    return !envValStr.empty() && envValStr[0] != '0';
  }();
  return logEnabled;
}

// Get the logging stream based on `FUSILLI_LOG_FILE`:
//   - When not set, logging is disabled.
//   - When set to `stdout`, uses `std::cout`.
//   - When set to `stderr`, uses `std::cerr`.
//   - When set to /some/file/path.txt, uses that.
inline std::ostream &getStream() {
  static std::ofstream outFile;
  static std::ostream &stream = []() -> std::ostream & {
    const char *logFile = std::getenv("FUSILLI_LOG_FILE");
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
    if (isLoggingEnabled())
      stream_ << t;
    return *this;
  }

  const ConditionalStreamer &
  operator<<(std::ostream &(*spl)(std::ostream &)) const {
    if (isLoggingEnabled())
      stream_ << spl;
    return *this;
  }

private:
  std::ostream &stream_;
};

inline ConditionalStreamer &getLogger() {
  static ConditionalStreamer logger(getStream());
  return logger;
}

} // namespace fusilli

// Macros for logging and error handling
#define FUSILLI_COLOR_RED "\033[31m"
#define FUSILLI_COLOR_GREEN "\033[32m"
#define FUSILLI_COLOR_YELLOW "\033[33m"
#define FUSILLI_COLOR_RESET "\033[0m"

#define FUSILLI_LOG(X) fusilli::getLogger() << X
#define FUSILLI_LOG_ENDL(X) fusilli::getLogger() << X << std::endl
#define FUSILLI_LOG_LABEL_RED(X)                                               \
  fusilli::getLogger() << FUSILLI_COLOR_RED << "[FUSILLI] " << X               \
                       << FUSILLI_COLOR_RESET
#define FUSILLI_LOG_LABEL_GREEN(X)                                             \
  fusilli::getLogger() << FUSILLI_COLOR_GREEN << "[FUSILLI] " << X             \
                       << FUSILLI_COLOR_RESET
#define FUSILLI_LOG_LABEL_YELLOW(X)                                            \
  fusilli::getLogger() << FUSILLI_COLOR_YELLOW << "[FUSILLI] " << X            \
                       << FUSILLI_COLOR_RESET
#define FUSILLI_LOG_LABEL_ENDL(X)                                              \
  fusilli::getLogger() << "[FUSILLI] " << X << std::endl

#define FUSILLI_RETURN_ERROR_IF(cond, retval, message)                         \
  do {                                                                         \
    if (cond) {                                                                \
      if (retval == fusilli::ErrorCode::OK)                                    \
        FUSILLI_LOG_LABEL_YELLOW("INFO: ");                                    \
      else                                                                     \
        FUSILLI_LOG_LABEL_RED("ERROR: ");                                      \
      FUSILLI_LOG_ENDL(retval << ": " << message << ": (" << #cond ") at "     \
                              << __FILE__ << ":" << __LINE__);                 \
      return error(retval, message);                                           \
    }                                                                          \
  } while (false);

// Checks if the expression that evaluates to an ErrorObject (or an ErrorOr<T>
// that contains an error) resulted in the error state and propagates the error.
//
// Usage:
//   ErrorObject doBar();
//
//   ErrorObject processFoo() {
//     // Returns error if doBar() fails
//     FUSILLI_CHECK_ERROR(doBar());
//     return ok();
//   }
#define FUSILLI_CHECK_ERROR(expr)                                              \
  do {                                                                         \
    ErrorObject _error = (expr);                                               \
    if (isError(_error)) {                                                     \
      FUSILLI_LOG_LABEL_RED("ERROR: ");                                        \
      FUSILLI_LOG_ENDL(#expr << " at " << __FILE__ << ":" << __LINE__);        \
      return _error;                                                           \
    }                                                                          \
  } while (false);

// Unwrap the type returned from an expression that evaluates to an ErrorOr,
// returning an error from the enclosing function in the error case, and the
// value otherwise.
//
// Usage:
//   ErrorOr<std::string> getString();
//
//   ErrorOr<int> processString() {
//     // Either gets the string or returns error.
//     std::string str = FUSILLI_TRY(getString());
//     return ok(str.length());
//   }
#define FUSILLI_TRY(expr)                                                      \
  ({                                                                           \
    auto _errorOr = (expr);                                                    \
    if (isError(_errorOr)) {                                                   \
      FUSILLI_LOG_LABEL_RED("ERROR: ");                                        \
      FUSILLI_LOG_ENDL(#expr << " at " << __FILE__ << ":" << __LINE__);        \
      return ErrorObject(_errorOr);                                            \
    }                                                                          \
    std::move(*_errorOr);                                                      \
  })

#endif // FUSILLI_SUPPORT_LOGGING_H
