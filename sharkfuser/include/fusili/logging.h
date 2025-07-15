// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef FUSILI_LOGGING_H
#define FUSILI_LOGGING_H

#include <fstream>
#include <iostream>
#include <string>
#include <unordered_map>

namespace fusili {

enum class [[nodiscard]] error_code_t {
  // Add error codes as needed
  OK,
  NOT_IMPLEMENTED,
  ATTRIBUTE_NOT_SET,
  INVALID_ATTRIBUTE,
  TENSOR_NOT_FOUND,
};

static const std::unordered_map<error_code_t, std::string> error_code_str = {
    {error_code_t::OK, "OK"},
    {error_code_t::NOT_IMPLEMENTED, "NOT_IMPLEMENTED"},
    {error_code_t::ATTRIBUTE_NOT_SET, "ATTRIBUTE_NOT_SET"},
    {error_code_t::INVALID_ATTRIBUTE, "INVALID_ATTRIBUTE"},
    {error_code_t::TENSOR_NOT_FOUND, "TENSOR_NOT_FOUND"}};

typedef struct [[nodiscard]] error_object {
  error_code_t code;
  std::string err_msg;

  error_object() : code(error_code_t::OK), err_msg("") {};
  error_object(error_code_t err, std::string msg)
      : code(err), err_msg(std::move(msg)) {};

  error_code_t get_code() const { return code; }

  const std::string &get_message() const { return err_msg; }

  bool is_ok() const { return code == error_code_t::OK; }

  bool is_failure() const { return !is_ok(); }

  bool operator==(error_code_t compare_code) const {
    return code == compare_code;
  }

  bool operator!=(error_code_t compare_code) const {
    return code != compare_code;
  }

} error_t;

static inline std::ostream &operator<<(std::ostream &os,
                                       const error_code_t &code) {
  auto it = error_code_str.find(code);
  if (it != error_code_str.end())
    os << it->second;
  else
    os << "UNKNOWN_ERROR_CODE";
  return os;
}

static inline std::ostream &operator<<(std::ostream &os, error_object &err) {
  os << err.get_code() << err.get_message();
  return os;
}

inline bool &isLoggingEnabled() {
  static bool log_enabled = []() -> bool {
    const char *env_val = std::getenv("FUSILI_LOG_INFO");
    // Disabled when FUSILI_LOG_INFO is not set
    if (!env_val) {
      return false;
    }
    std::string env_val_str(env_val);
    // Disabled when FUSILI_LOG_INFO == "" (empty string)
    // Disabled when FUSILI_LOG_INFO == "0", any other value enables it
    return !env_val_str.empty() && env_val_str[0] != '0';
  }();
  return log_enabled;
}

inline std::ostream &getStream() {
  static std::ofstream outFile;
  static std::ostream &stream = []() -> std::ostream & {
    const char *log_file = std::getenv("FUSILI_LOG_FILE");
    if (!log_file) {
      isLoggingEnabled() = false;
      return std::cout;
    }

    std::string file_path(log_file);
    if (file_path == "stdout") {
      return std::cout;
    } else if (file_path == "stderr") {
      return std::cerr;
    } else {
      outFile.open(log_file, std::ios::out);
      return outFile;
    }
  }();
  return stream;
}

class ConditionalStreamer {
private:
  std::ostream &stream;

public:
  ConditionalStreamer(std::ostream &stream_) : stream(stream_) {}

  template <typename T>
  const ConditionalStreamer &operator<<(const T &t) const {
    if (isLoggingEnabled()) {
      stream << t;
    }
    return *this;
  }

  const ConditionalStreamer &
  operator<<(std::ostream &(*spl)(std::ostream &)) const {
    if (isLoggingEnabled()) {
      stream << spl;
    }
    return *this;
  }
};

inline ConditionalStreamer &getLogger() {
  static ConditionalStreamer logger(getStream());
  return logger;
}

} // namespace fusili

#define FUSILI_COLOR_RED "\033[31m"
#define FUSILI_COLOR_GREEN "\033[32m"
#define FUSILI_COLOR_YELLOW "\033[33m"
#define FUSILI_COLOR_RESET "\033[0m"

#define FUSILI_LOG(X) getLogger() << X
#define FUSILI_LOG_ENDL(X) getLogger() << X << std::endl
#define FUSILI_LOG_LABEL_RED(X)                                                \
  getLogger() << FUSILI_COLOR_RED << "[FUSILI] " << X << FUSILI_COLOR_RESET
#define FUSILI_LOG_LABEL_GREEN(X)                                              \
  getLogger() << FUSILI_COLOR_GREEN << "[FUSILI] " << X << FUSILI_COLOR_RESET
#define FUSILI_LOG_LABEL_YELLOW(X)                                             \
  getLogger() << FUSILI_COLOR_YELLOW << "[FUSILI] " << X << FUSILI_COLOR_RESET
#define FUSILI_LOG_LABEL_ENDL(X) getLogger() << "[FUSILI] " << X << std::endl

#define FUSILI_RETURN_ERROR_IF(cond, retval, message)                          \
  do {                                                                         \
    if (cond) {                                                                \
      if (retval == error_code_t::OK)                                          \
        FUSILI_LOG_LABEL_YELLOW("INFO: ");                                     \
      else                                                                     \
        FUSILI_LOG_LABEL_RED("ERROR: ");                                       \
                                                                               \
      FUSILI_LOG_ENDL(retval << ": " << message << ": (" << #cond ") at "      \
                             << __FILE__ << ":" << __LINE__);                  \
      return {retval, message};                                                \
    }                                                                          \
  } while (false);

#define FUSILI_CHECK_ERROR(x)                                                  \
  do {                                                                         \
    if (auto retval = x; retval.is_failure()) {                                \
      FUSILI_LOG_LABEL_RED("ERROR: ");                                         \
      FUSILI_LOG_ENDL(#x << " at " << __FILE__ << ":" << __LINE__);            \
      return retval;                                                           \
    }                                                                          \
  } while (false);

#endif // FUSILI_LOGGING_H
