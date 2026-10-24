#pragma once

#include "native_math.hpp"
#include <array>

namespace uni20::detail::native_math
{
// A bounded expansion used only by compound. Guard words cover amplification
// by a signed-64-bit exponent: 4 words for binary32, 3 for binary64/80, 2 for
// binary128. Components are nonoverlapping, in increasing magnitude order.
// See docs/tensor/native_compound.md for the error model and acceptance tests.
template <NativeReal R> class compound_expansion {
  public:
    void add(R value)
    {
      int size = 0;
      for (int i = 0; i < size_; ++i)
      {
        R sum = value + terms_[i];
        R part = sum - value;
        R error = (value - (sum - part)) + (terms_[i] - part);
        if (error != 0) terms_[size++] = error;
        value = sum;
      }
      if (value != 0 || size == 0) terms_[size++] = value;
      size_ = size;
    }

    static compound_expansion multiply(compound_expansion const& a, compound_expansion const& b)
    {
      compound_expansion result;
      for (int i = 0; i < a.size_; ++i)
        for (int j = 0; j < b.size_; ++j)
        {
          R product = a.terms_[i] * b.terms_[j];
          result.add(std::fma(a.terms_[i], b.terms_[j], -product));
          result.add(product);
        }
      result.compress();
      return result;
    }

    // The leading component is positive. Scaling every word keeps the product
    // arithmetic near unity even when the final power is outside native range.
    int normalize()
    {
      int exponent;
      std::frexp(this->high(), &exponent);
      for (int i = 0; i < size_; ++i)
        terms_[i] = std::ldexp(terms_[i], -exponent);
      return exponent;
    }

    R rounded(bool reciprocal) const
    {
      if (!reciprocal)
      {
        R sum = 0;
        for (int i = 0; i < size_; ++i)
          sum += terms_[i];
        return sum;
      }
      // One quotient correction suffices for a native result. Apply it before
      // the binary scale, so an intermediate reciprocal cannot overflow.
      R quotient = R(1) / this->high();
      R residual = std::fma(-quotient, this->high(), R(1));
      for (int i = size_ - 2; i >= 0; --i)
        residual = std::fma(-quotient, terms_[i], residual);
      return quotient + residual / this->high();
    }

  private:
    static constexpr int digits_ = numeric_limits<R>::digits();
    static constexpr int words_ = (digits_ + 72 + digits_ - 1) / digits_;
    // A product adds at most two terms per pair of retained input words.
    std::array<R, 2 * words_ * words_ + 2> terms_{};
    int size_ = 0;

    R high() const { return terms_[size_ - 1]; }

    void compress()
    {
      // Two passes of FastTwoSum pack the expansion before discarding its
      // smallest words. Truncating a sparse, uncompressed expansion would not
      // retain the intended number of significand bits.
      R carry = this->high();
      int bottom = size_;
      for (int i = size_ - 2; i >= 0; --i)
      {
        R sum = carry + terms_[i];
        R error = terms_[i] - (sum - carry);
        if (error != 0)
        {
          terms_[--bottom] = sum;
          carry = error;
        }
        else
          carry = sum;
      }
      int size = 0;
      for (int i = bottom; i < size_; ++i)
      {
        R sum = terms_[i] + carry;
        R error = carry - (sum - terms_[i]);
        if (error != 0) terms_[size++] = error;
        carry = sum;
      }
      terms_[size++] = carry;
      size_ = size;
      if (size_ > words_)
      {
        std::copy(terms_.begin() + size_ - words_, terms_.begin() + size_, terms_.begin());
        size_ = words_;
      }
    }
};

template <NativeReal R> R portable_compound(R x, std::int64_t exponent)
{
  using L = numeric_limits<R>;
  if (x < R(-1)) return L::quiet_NaN();
  if (exponent == 0 || x == 0) return R(1);
  if (std::isnan(x)) return x;
  if (x == R(-1)) return exponent < 0 ? L::infinity() : R(0);
  if (std::isinf(x)) return exponent > 0 ? L::infinity() : R(0);
  if (exponent == 1) return R(1) + x;

  compound_expansion<R> base;
  base.add(R(1));
  base.add(x);
  int base_exponent = base.normalize();
  compound_expansion<R> result;
  result.add(R(0.5));
  int result_exponent = 1;
  // Unsigned subtraction handles INT64_MIN without signed negation.
  auto remaining = exponent < 0 ? std::uint64_t(0) - std::uint64_t(exponent) : std::uint64_t(exponent);
  constexpr int limit = 2 * (L::max_exponent - L::min_exponent + L::digits());
  while (remaining)
  {
    if (remaining & 1)
    {
      result = compound_expansion<R>::multiply(result, base);
      result_exponent += base_exponent + result.normalize();
    }
    // All positive powers of the base lie on the same side of one. Once a
    // factor exceeds this conservative scale bound, any remaining power also
    // exceeds the result range, including after a reciprocal. Checking here
    // bounds both exponent accumulators before the next doubling/addition.
    if (result_exponent > limit || base_exponent > limit) return exponent > 0 ? L::infinity() : R(0);
    if (result_exponent < -limit || base_exponent < -limit) return exponent > 0 ? R(0) : L::infinity();
    remaining >>= 1;
    if (remaining)
    {
      base = compound_expansion<R>::multiply(base, base);
      base_exponent = 2 * base_exponent + base.normalize();
    }
  }
  return std::ldexp(result.rounded(exponent < 0), exponent < 0 ? -result_exponent : result_exponent);
}

template <NativeReal R> struct compound_math_provider
{
    static R evaluate(R x, std::int64_t exponent) { return portable_compound(x, exponent); }
};
#if UNI20_HAS_COMPOUNDN_FLOAT
template <> struct compound_math_provider<float>
{
    static float evaluate(float x, std::int64_t exponent) { return ::compoundnf(x, exponent); }
};
#endif
#if UNI20_HAS_COMPOUNDN_DOUBLE
template <> struct compound_math_provider<double>
{
    static double evaluate(double x, std::int64_t exponent) { return ::compoundn(x, exponent); }
};
#endif
#if UNI20_HAS_COMPOUNDN_LONG_DOUBLE
template <> struct compound_math_provider<long double>
{
    static long double evaluate(long double x, std::int64_t exponent) { return ::compoundnl(x, exponent); }
};
#endif
#if UNI20_HAS_FLOAT128 && UNI20_HAS_COMPOUNDN_FLOAT128
template <> struct compound_math_provider<float128>
{
    static float128 evaluate(float128 x, std::int64_t exponent) { return ::compoundnf128(x, exponent); }
};
#endif

template <NativeReal R, MathInteger I> R compound(R x, I exponent)
{
  return compound_math_provider<R>::evaluate(x, math_integer(exponent));
}
} // namespace uni20::detail::native_math
