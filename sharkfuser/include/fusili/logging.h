// Copyright 2025 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <fstream>
#include <iostream>
#include <string>

namespace fusili {
namespace logging {

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

} // namespace logging
} // namespace fusili

// Macros starting with _ are for testing purposes as they allow
// passing a custom logger which can be intercepted for testing
#define FUSILI_LOG(X) getLogger() << X
#define _FUSILI_LOG(X, logger) logger << X

#define FUSILI_LOG_LABEL(X) getLogger() << "[FUSILI] " << X
#define _FUSILI_LOG_LABEL(X, logger) logger << "[FUSILI] " << X

#define FUSILI_LOG_LABEL_ENDL(X) getLogger() << "[FUSILI] " << X << std::endl
#define _FUSILI_LOG_LABEL_ENDL(X, logger)                                      \
  logger << "[FUSILI] " << X << std::endl
