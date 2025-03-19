#include "xtensor/xmath.hpp"

#ifndef QUANT_DTYPES_HPP
#define QUANT_DTYPES_HPP

#include <bit>
#include <cstdint>
#include <limits>
#include <type_traits>

struct float8_e4m3fnuz_t {
  uint8_t value;

  constexpr float8_e4m3fnuz_t() noexcept : value(0) {}

  explicit constexpr float8_e4m3fnuz_t(char c) noexcept {
    uint8_t temp = std::bit_cast<uint8_t>(c);
    value = temp;
  }

  template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T> &&
                                                    !std::is_same_v<T, float>>>
  constexpr float8_e4m3fnuz_t(T value) noexcept
      : float8_e4m3fnuz_t(static_cast<float>(value)) {}

  constexpr operator float() const noexcept {
    uint32_t temp = value;
    return std::bit_cast<float>(temp);
  }
};

// Mark float8_e4m3fnuz_t as a trivial, standard-layout type so that xtensor can
// use it.
namespace std {
template <>
struct is_trivial<float8_e4m3fnuz_t> : std::true_type {};
template <>
struct is_standard_layout<float8_e4m3fnuz_t> : std::true_type {};
template <>
struct is_trivially_copyable<float8_e4m3fnuz_t> : std::true_type {};
}  // namespace std

struct bfloat16_t {
  uint16_t value;

  constexpr bfloat16_t() noexcept : value(0) {}

  explicit constexpr bfloat16_t(float f) noexcept {
    uint32_t temp = std::bit_cast<uint32_t>(f);
    value = static_cast<uint16_t>(temp >> 16);
  }

  template <typename T, typename = std::enable_if_t<std::is_arithmetic_v<T> &&
                                                    !std::is_same_v<T, float>>>
  constexpr bfloat16_t(T value) noexcept
      : bfloat16_t(static_cast<float>(value)) {}

  constexpr operator float() const noexcept {
    uint32_t temp = static_cast<uint32_t>(value) << 16;
    return std::bit_cast<float>(temp);
  }

  // Arithmetic operators (implemented via conversion to float)
  constexpr bfloat16_t operator+(const bfloat16_t &other) const noexcept {
    return bfloat16_t(float(*this) + float(other));
  }
  constexpr bfloat16_t operator-(const bfloat16_t &other) const noexcept {
    return bfloat16_t(float(*this) - float(other));
  }
  constexpr bfloat16_t operator*(const bfloat16_t &other) const noexcept {
    return bfloat16_t(float(*this) * float(other));
  }
  constexpr bfloat16_t operator/(const bfloat16_t &other) const noexcept {
    return bfloat16_t(float(*this) / float(other));
  }

  constexpr bfloat16_t &operator+=(const bfloat16_t &other) noexcept {
    *this = *this + other;
    return *this;
  }
  constexpr bfloat16_t &operator-=(const bfloat16_t &other) noexcept {
    *this = *this - other;
    return *this;
  }
  constexpr bfloat16_t &operator*=(const bfloat16_t &other) noexcept {
    *this = *this * other;
    return *this;
  }
  constexpr bfloat16_t &operator/=(const bfloat16_t &other) noexcept {
    *this = *this / other;
    return *this;
  }

  // Comparison operators (using conversion to float)
  constexpr bool operator==(const bfloat16_t &other) const noexcept {
    return float(*this) == float(other);
  }
  constexpr bool operator!=(const bfloat16_t &other) const noexcept {
    return !(*this == other);
  }
  constexpr bool operator<(const bfloat16_t &other) const noexcept {
    return float(*this) < float(other);
  }
  constexpr bool operator<=(const bfloat16_t &other) const noexcept {
    return float(*this) <= float(other);
  }
  constexpr bool operator>(const bfloat16_t &other) const noexcept {
    return float(*this) > float(other);
  }
  constexpr bool operator>=(const bfloat16_t &other) const noexcept {
    return float(*this) >= float(other);
  }
};

// Mark bfloat16_t as a trivial, standard-layout type so that xtensor can use
// it.
namespace std {
template <>
struct is_trivial<bfloat16_t> : std::true_type {};
template <>
struct is_standard_layout<bfloat16_t> : std::true_type {};
template <>
struct is_trivially_copyable<bfloat16_t> : std::true_type {};
}  // namespace std

// Math functions needed by xtensor for bfloat16_t
inline constexpr bfloat16_t round(bfloat16_t x) noexcept {
  return bfloat16_t(std::round(float(x)));
}

inline constexpr bfloat16_t ceil(bfloat16_t x) noexcept {
  return bfloat16_t(std::ceil(float(x)));
}

inline constexpr bfloat16_t floor(bfloat16_t x) noexcept {
  return bfloat16_t(std::floor(float(x)));
}

inline constexpr bfloat16_t trunc(bfloat16_t x) noexcept {
  return bfloat16_t(std::trunc(float(x)));
}

#endif  // QUANT_DTYPES_HPP
