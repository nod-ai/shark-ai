#pragma once

#include <numeric>

#define FP16_EXP_BITS (5)

#ifndef CHECK_HIP_ERROR
#define CHECK_HIP_ERROR(status)                                                \
  if (status != hipSuccess) {                                                  \
    fprintf(stderr, "hip error: '%s'(%d) at %s:%d\n",                          \
            hipGetErrorString(status), status, __FILE__, __LINE__);            \
    exit(EXIT_FAILURE);                                                        \
  }
#endif

// Queries for [[attribute]] identifiers in modern compilers.
#if defined(__has_attribute)
#define IREE_HAVE_ATTRIBUTE(x) __has_attribute(x)
#else
#define IREE_HAVE_ATTRIBUTE(x) 0
#endif // __has_attribute

#if IREE_HAVE_ATTRIBUTE(maybe_unused) && defined(__clang__)
#define IREE_ATTRIBUTE_UNUSED __attribute__((maybe_unused))
#elif IREE_HAVE_ATTRIBUTE(unused) || (defined(__GNUC__) && !defined(__clang__))
#define IREE_ATTRIBUTE_UNUSED __attribute__((unused))
#else
#define IREE_ATTRIBUTE_UNUSED
#endif // IREE_HAVE_ATTRIBUTE(maybe_unused / unused)

#define IREE_MATH_FP_FORMAT_CONSTANTS(prefix, bits, ebits)                     \
  const int prefix##exp_bits IREE_ATTRIBUTE_UNUSED = ebits;                    \
  const int prefix##mantissa_bits IREE_ATTRIBUTE_UNUSED =                      \
      bits - 1 - prefix##exp_bits;                                             \
  const int prefix##sign_shift IREE_ATTRIBUTE_UNUSED = bits - 1;               \
  const int prefix##exp_shift IREE_ATTRIBUTE_UNUSED = prefix##mantissa_bits;   \
  const int prefix##sign_mask IREE_ATTRIBUTE_UNUSED = 1u                       \
                                                      << prefix##sign_shift;   \
  const int prefix##mantissa_mask IREE_ATTRIBUTE_UNUSED =                      \
      (1u << prefix##exp_shift) - 1;                                           \
  const int prefix##exp_mask IREE_ATTRIBUTE_UNUSED =                           \
      (1u << prefix##sign_shift) - (1u << prefix##exp_shift);

static inline float half2float(uint16_t f16_value, int exp_bits) {
  IREE_MATH_FP_FORMAT_CONSTANTS(f16_, 16, exp_bits)
  IREE_MATH_FP_FORMAT_CONSTANTS(f32_, 32, 8)
  const uint32_t f16_sign = f16_value & f16_sign_mask;
  const uint32_t f32_sign = f16_sign << (f32_sign_shift - f16_sign_shift);
  const uint32_t f16_exp = f16_value & f16_exp_mask;
  const uint32_t f16_mantissa = f16_value & f16_mantissa_mask;
  uint32_t f32_exp = 0;
  uint32_t f32_mantissa = 0;
  if (f16_exp == f16_exp_mask) {
    // NaN or Inf case.
    f32_exp = f32_exp_mask;
    if (f16_mantissa) {
      // NaN. Generate a quiet NaN.
      f32_mantissa = f32_mantissa_mask;
    } else {
      // Inf. Leave zero mantissa.
    }
  } else if (f16_exp == 0) {
    // Zero or subnormal. Generate zero. Leave zero mantissa.
  } else {
    // Normal finite value.
    int arithmetic_f16_exp = f16_exp >> f16_exp_shift;
    int arithmetic_f32_exp = arithmetic_f16_exp + (1 << (f32_exp_bits - 1)) -
                             (1 << (f16_exp_bits - 1));
    f32_exp = arithmetic_f32_exp << f32_exp_shift;
    f32_mantissa = f16_mantissa << (f32_mantissa_bits - f16_mantissa_bits);
  }
  const uint32_t u32_value = f32_sign | f32_exp | f32_mantissa;
  float f32_value = std::bit_cast<float>(u32_value);
  return f32_value;
}

static inline uint16_t float2half(float value, int exp_bits) {
  IREE_MATH_FP_FORMAT_CONSTANTS(f16_, 16, exp_bits)
  IREE_MATH_FP_FORMAT_CONSTANTS(f32_, 32, 8)
  uint32_t u32_value = std::bit_cast<uint32_t>(value);
  const uint32_t f32_sign = u32_value & f32_sign_mask;
  const uint32_t f16_sign = f32_sign >> (f32_sign_shift - f16_sign_shift);
  const uint32_t f32_exp = u32_value & f32_exp_mask;
  const uint32_t f32_mantissa = u32_value & f32_mantissa_mask;
  uint32_t f16_exp = 0;
  uint32_t f16_mantissa = 0;
  if (f32_exp == f32_exp_mask) {
    // NaN or Inf case.
    f16_exp = f16_exp_mask;
    if (f32_mantissa) {
      // NaN. Generate a quiet NaN.
      f16_mantissa = f16_mantissa_mask;
    } else {
      // Inf. Leave zero mantissa.
    }
  } else if (f32_exp == 0) {
    // Zero or subnormal. Generate zero. Leave zero mantissa.
  } else {
    // Normal finite value.
    int arithmetic_exp = (f32_exp >> f32_exp_shift) - (1 << (f32_exp_bits - 1));
    if (arithmetic_exp >= (1 << (f16_exp_bits - 1))) {
      // Overflow. Generate Inf. Leave zero mantissa.
      f16_exp = f16_exp_mask;
    } else if (arithmetic_exp < -(1 << (f16_exp_bits - 1))) {
      // Underflow. Generate zero. Leave zero mantissa.
      f16_exp = 0;
    } else {
      // Normal case.
      // Implement round-to-nearest-even, by adding a bias before truncating.
      // truncating.
      int even_bit = 1u << (f32_mantissa_bits - f16_mantissa_bits);
      int odd_bit = even_bit >> 1;
      uint32_t biased_f32_mantissa =
          f32_mantissa +
          ((f32_mantissa & even_bit) ? (odd_bit) : (odd_bit - 1));
      // Adding the bias may cause an exponent increment.
      if (biased_f32_mantissa > f32_mantissa_mask) {
        // Note: software implementations that try to be fast tend to get this
        // conditional increment of exp and zeroing of mantissa for free by
        // simplying incrementing the whole uint32 encoding of the float value,
        // so that the mantissa overflows into the exponent bits.
        // This results in magical-looking code like in the following links.
        // We'd rather not care too much about performance of this function;
        // we should only care about fp16 performance on fp16 hardware, and
        // then, we should use hardware instructions.
        // https://github.com/pytorch/pytorch/blob/e1502c0cdbfd17548c612f25d5a65b1e4b86224d/c10/util/BFloat16.h#L76
        // https://gitlab.com/libeigen/eigen/-/blob/21cd3fe20990a5ac1d683806f605110962aac3f1/Eigen/src/Core/arch/Default/BFloat16.h#L565
        biased_f32_mantissa = 0;
        ++arithmetic_exp;
      }
      // The exponent increment in the above if() branch may cause overflow.
      // This is exercised by converting 65520.0f from f32 to f16. No special
      // handling is needed for this case: the above if() branch already set
      // biased_f32_mantissa=0, so we will be generating a 0 mantissa, as
      // needed for infinite values.
      f16_exp = (arithmetic_exp + (1 << (f16_exp_bits - 1))) << f16_exp_shift;
      f16_mantissa =
          biased_f32_mantissa >> (f32_mantissa_bits - f16_mantissa_bits);
    }
  }
  uint16_t f16_value = f16_sign | f16_exp | f16_mantissa;
  return f16_value;
}

using float16_t = uint16_t;
using float32_t = float;
template <typename DataT>
static inline void fillRand(DataT *mat, uint32_t m, uint32_t n) {
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; j++) {
      // Random values normalized such that output is between 0 and 1
      float original =
          static_cast<float>(rand()) / static_cast<float>(RAND_MAX);
      float16_t value = float2half(original, FP16_EXP_BITS);
      mat[i * n + j] = static_cast<DataT>(value);
    }
  }
}
