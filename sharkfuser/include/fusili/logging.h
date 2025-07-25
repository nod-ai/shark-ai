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

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

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
  bool isFailure() const { return !isOk(); }

  bool operator==(ErrorCode compareCode) const { return code == compareCode; }
  bool operator!=(ErrorCode compareCode) const { return code != compareCode; }
};

using error_code_t = ErrorCode;
using error_t = ErrorObject;

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
  os << err.getCode() << err.getMessage();
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
      return {retval, message};                                                \
    }                                                                          \
  } while (false);

#define FUSILI_CHECK_ERROR(x)                                                  \
  do {                                                                         \
    if (auto retval = x; retval.isFailure()) {                                 \
      FUSILI_LOG_LABEL_RED("ERROR: ");                                         \
      FUSILI_LOG_ENDL(#x << " at " << __FILE__ << ":" << __LINE__);            \
      return retval;                                                           \
    }                                                                          \
  } while (false);

#endif // FUSILI_LOGGING_H
