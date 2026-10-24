#pragma once

#include "native_math.hpp"
#include <uni20/core/math_constants.hpp>

namespace uni20::detail::native_math
{
// Keep the provider evaluation away from its extreme exponent boundary. The
// final multiply supplies native overflow/subnormal rounding. This also avoids
// premature overflow in binary128 pow and premature nonzero exp10 underflow.
template <NativeReal R, class Evaluate> R scaled_decimal_exponential(R x, Evaluate evaluate)
{
  if (x > R(numeric_limits<R>::max_exponent / 4) || x < R(numeric_limits<R>::min_exponent / 4))
  {
    R half = evaluate(x / 2);
    return half * half;
  }
  return evaluate(x);
}

template <NativeReal R> struct portable_exponential_math
{
    static R exp10(R x)
    {
      return scaled_decimal_exponential(x, [](R value) { return std::pow(R(10), value); });
    }

    static R exp2m1(R x)
    {
      if (std::abs(x) <= R(0.5)) return std::expm1(x * log_two<R>);
      return std::exp2(x) - R(1);
    }

    static R exp10m1(R x)
    {
      if (std::abs(x) <= R(0.5)) return std::expm1(x * std::numbers::ln10_v<R>);
      return portable_exponential_math::exp10(x) - R(1);
    }

    template <bool Binary> static R log_base_p1(R x)
    {
      constexpr R inverse_log = Binary ? std::numbers::log2e_v<R> : std::numbers::log10e_v<R>;
      if (std::abs(x) <= R(0.25) || !std::isfinite(x) || x <= R(-1)) return std::log1p(x) * inverse_log;
      // TwoSum(1,x) retains the part discarded by the rounded addition.
      // Here the result is bounded away from zero. The omitted quadratic
      // correction in log1p(low/sum) is O(epsilon^2).
      R sum = R(1) + x;
      R part = sum - R(1);
      R low = (R(1) - (sum - part)) + (x - part);
      R logarithm = Binary ? std::log2(sum) : std::log10(sum);
      return std::fma(low / sum, inverse_log, logarithm);
    }

    static R log2p1(R x) { return log_base_p1<true>(x); }
    static R log10p1(R x) { return log_base_p1<false>(x); }
};

template <NativeReal R> struct exponential_math_provider : portable_exponential_math<R>
{};

#if UNI20_HAS_EXP10_FLOAT || UNI20_HAS_EXP2M1_FLOAT || UNI20_HAS_EXP10M1_FLOAT || UNI20_HAS_LOG2P1_FLOAT ||            \
    UNI20_HAS_LOG10P1_FLOAT
template <> struct exponential_math_provider<float> : portable_exponential_math<float>
{
#if UNI20_HAS_EXP10_FLOAT
    static float exp10(float x)
    {
      return scaled_decimal_exponential(x, [](float value) { return ::exp10f(value); });
    }
#endif
#if UNI20_HAS_EXP2M1_FLOAT
    static float exp2m1(float x) { return ::exp2m1f(x); }
#endif
#if UNI20_HAS_EXP10M1_FLOAT
    static float exp10m1(float x) { return ::exp10m1f(x); }
#endif
#if UNI20_HAS_LOG2P1_FLOAT
    static float log2p1(float x) { return ::log2p1f(x); }
#endif
#if UNI20_HAS_LOG10P1_FLOAT
    static float log10p1(float x) { return ::log10p1f(x); }
#endif
};
#endif

#if UNI20_HAS_EXP10_DOUBLE || UNI20_HAS_EXP2M1_DOUBLE || UNI20_HAS_EXP10M1_DOUBLE || UNI20_HAS_LOG2P1_DOUBLE ||        \
    UNI20_HAS_LOG10P1_DOUBLE
template <> struct exponential_math_provider<double> : portable_exponential_math<double>
{
#if UNI20_HAS_EXP10_DOUBLE
    static double exp10(double x)
    {
      return scaled_decimal_exponential(x, [](double value) { return ::exp10(value); });
    }
#endif
#if UNI20_HAS_EXP2M1_DOUBLE
    static double exp2m1(double x) { return ::exp2m1(x); }
#endif
#if UNI20_HAS_EXP10M1_DOUBLE
    static double exp10m1(double x) { return ::exp10m1(x); }
#endif
#if UNI20_HAS_LOG2P1_DOUBLE
    static double log2p1(double x) { return ::log2p1(x); }
#endif
#if UNI20_HAS_LOG10P1_DOUBLE
    static double log10p1(double x) { return ::log10p1(x); }
#endif
};
#endif

#if UNI20_HAS_EXP10_LONG_DOUBLE || UNI20_HAS_EXP2M1_LONG_DOUBLE || UNI20_HAS_EXP10M1_LONG_DOUBLE ||                    \
    UNI20_HAS_LOG2P1_LONG_DOUBLE || UNI20_HAS_LOG10P1_LONG_DOUBLE
template <> struct exponential_math_provider<long double> : portable_exponential_math<long double>
{
#if UNI20_HAS_EXP10_LONG_DOUBLE
    static long double exp10(long double x)
    {
      return scaled_decimal_exponential(x, [](long double value) { return ::exp10l(value); });
    }
#endif
#if UNI20_HAS_EXP2M1_LONG_DOUBLE
    static long double exp2m1(long double x) { return ::exp2m1l(x); }
#endif
#if UNI20_HAS_EXP10M1_LONG_DOUBLE
    static long double exp10m1(long double x) { return ::exp10m1l(x); }
#endif
#if UNI20_HAS_LOG2P1_LONG_DOUBLE
    static long double log2p1(long double x) { return ::log2p1l(x); }
#endif
#if UNI20_HAS_LOG10P1_LONG_DOUBLE
    static long double log10p1(long double x) { return ::log10p1l(x); }
#endif
};
#endif

#if UNI20_HAS_FLOAT128 && (UNI20_HAS_EXP10_FLOAT128 || UNI20_HAS_EXP2M1_FLOAT128 || UNI20_HAS_EXP10M1_FLOAT128 ||      \
                           UNI20_HAS_LOG2P1_FLOAT128 || UNI20_HAS_LOG10P1_FLOAT128)
template <> struct exponential_math_provider<float128> : portable_exponential_math<float128>
{
#if UNI20_HAS_EXP10_FLOAT128
    static float128 exp10(float128 x)
    {
      return scaled_decimal_exponential(x, [](float128 value) { return ::exp10f128(value); });
    }
#endif
#if UNI20_HAS_EXP2M1_FLOAT128
    static float128 exp2m1(float128 x) { return ::exp2m1f128(x); }
#endif
#if UNI20_HAS_EXP10M1_FLOAT128
    static float128 exp10m1(float128 x) { return ::exp10m1f128(x); }
#endif
#if UNI20_HAS_LOG2P1_FLOAT128
    static float128 log2p1(float128 x) { return ::log2p1f128(x); }
#endif
#if UNI20_HAS_LOG10P1_FLOAT128
    static float128 log10p1(float128 x) { return ::log10p1f128(x); }
#endif
};
#endif

template <NativeReal R> R exp10(R x) { return exponential_math_provider<R>::exp10(x); }
template <NativeReal R> R exp2m1(R x) { return exponential_math_provider<R>::exp2m1(x); }
template <NativeReal R> R exp10m1(R x) { return exponential_math_provider<R>::exp10m1(x); }
template <NativeReal R> R log2p1(R x) { return exponential_math_provider<R>::log2p1(x); }
template <NativeReal R> R log10p1(R x) { return exponential_math_provider<R>::log10p1(x); }
} // namespace uni20::detail::native_math
