// Copyright 2024 Advanced Micro Devices, Inc.
//
// Licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "shortfin/support/config.h"

#include <algorithm>
#include <cctype>
#include <charconv>
#include <cstdlib>

#include "fmt/format.h"
#include "fmt/xchar.h"
#include "shortfin/support/logging.h"

namespace shortfin {

void ConfigOptions::SetOption(std::string_view key, std::string value) {
  options_[intern_.intern(key)] = std::move(value);
}

const std::optional<std::string_view> ConfigOptions::GetOption(
    std::string_view key) const {
  // Get explicit option.
  auto found_it = options_.find(key);
  consumed_keys_.insert(key);
  if (found_it != options_.end()) {
    return found_it->second;
  }

  // Consult environment.
  if (env_lookup_prefix_) {
    std::string env_key;
    env_key.reserve(env_lookup_prefix_->size() + key.size());
    env_key.append(*env_lookup_prefix_);
    for (char c : key) {
      env_key.push_back(std::toupper(c));
    }
    auto env_value = GetRawEnv(env_key.c_str());
    if (env_value) {
      return env_value;
    }
  }

  return {};
}

std::optional<int64_t> ConfigOptions::GetInt(std::string_view key,
                                             bool non_negative) const {
  auto value = GetOption(key);
  if (!value) return {};
  int64_t result;
  auto last = value->data() + value->size();
  auto err = std::from_chars(value->data(), last, result);
  if (err.ec != std::errc{} || err.ptr != last) {
    throw std::invalid_argument(
        fmt::format("Could not parse '{}' as an integer from config option "
                    "{}",
                    *value, key));
  }
  if (non_negative && result < 0) {
    throw std::invalid_argument(fmt::format(
        "Could not parse '{}' as a non-negative integer from config option "
        "{}",
        *value, key));
  }
  return result;
}

bool ConfigOptions::GetBool(std::string_view key, bool default_value) const {
  auto svalue = GetOption(key);
  if (!svalue) return default_value;
  auto iequal = [](std::string_view a, std::string_view b) -> bool {
    return std::ranges::equal(a, b, [](char c1, char c2) {
      return std::toupper(c1) == std::toupper(c2);
    });
  };
  if (iequal(*svalue, "1") || iequal(*svalue, "TRUE") ||
      iequal(*svalue, "ON")) {
    return true;
  } else if (iequal(*svalue, "0") || iequal(*svalue, "FALSE") ||
             iequal(*svalue, "OFF")) {
    return false;
  } else {
    throw std::invalid_argument(
        fmt::format("Cannot interpret {} = '{}' as bool: must be one of '1', "
                    "'TRUE', 'ON', '0', 'FALSE', 'OFF'",
                    key, *svalue));
  }
}

std::optional<std::vector<int64_t>> ConfigOptions::GetIntList(
    std::string_view key, bool non_negative) const {
  auto value = GetOption(key);
  if (!value) return {};

  std::vector<int64_t> results;
  auto Consume = [&](std::string_view atom) {
    int64_t result;
    auto last = atom.data() + atom.size();
    auto err = std::from_chars(atom.data(), last, result);
    if (err.ec != std::errc{} || err.ptr != last) {
      throw std::invalid_argument(
          fmt::format("Could not parse '{}' as an integer from config option "
                      "{} (full value: {})",
                      atom, key, *value));
    }
    if (non_negative && result < 0) {
      throw std::invalid_argument(fmt::format(
          "Could not parse '{}' as a non-negative integer from config option "
          "{} (full value: {})",
          atom, key, *value));
    }
    results.push_back(result);
  };
  std::string_view sv_value = *value;
  for (;;) {
    auto found_it = sv_value.find(',');
    if (found_it == std::string_view::npos) {
      Consume(sv_value);
      break;
    }

    Consume(sv_value.substr(0, found_it));
    sv_value.remove_prefix(found_it + 1);
  }

  return results;
}

void ConfigOptions::ValidateUndef() const {
  std::vector<std::string_view> unused_options;
  for (auto it : options_) {
    const auto &key = it.first;
    if (!consumed_keys_.contains(key)) {
      unused_options.push_back(key);
    }
  }
  if (!unused_options.empty()) {
    std::string message = fmt::format(
        "Specified options were not used: {} (available: {})",
        fmt::join(unused_options, ", "), fmt::join(consumed_keys_, ", "));
    switch (validation_) {
      case ValidationLevel::UNDEF_DEBUG:
        logging::debug("{}", message);
        break;
      case ValidationLevel::UNDEF_WARN:
        logging::warn("{}", message);
        break;
      case ValidationLevel::UNDEF_ERROR:
        throw std::invalid_argument(std::move(message));
    }
  }
}

std::optional<std::string_view> ConfigOptions::GetRawEnv(
    const char *key) const {
  char *env_value = std::getenv(key);
  if (env_value && std::strlen(env_value) > 0) {
    return intern_.intern(env_value);
  }
  return {};
}

// Helper to split on a delimitter.
std::vector<std::string_view> ConfigOptions::Split(std::string_view value,
                                                   char delim) {
  std::vector<std::string_view> results;
  std::string_view rest(value);
  for (;;) {
    auto pos = rest.find(delim);
    if (pos == std::string_view::npos) {
      results.push_back(rest);
      break;
    }
    std::string_view first = rest.substr(0, pos);
    rest = rest.substr(pos + 1);
    results.push_back(first);
  }
  return results;
}

}  // namespace shortfin
