#pragma once

#include "native_math.hpp"
#include <uni20/core/math_constants.hpp>

namespace uni20::detail::native_math
{
template <NativeReal R> R sec(R x) { return R(1) / std::cos(x); }
template <NativeReal R> R csc(R x) { return R(1) / std::sin(x); }
template <NativeReal R> R cot(R x) { return R(1) / std::tan(x); }
template <NativeReal R> R coth(R x) { return R(1) / std::tanh(x); }

template <bool Sine, NativeReal R> R reciprocal_hyperbolic(R x)
{
  using L = numeric_limits<R>;
  R magnitude = std::abs(x);
  if (magnitude <= R(L::digits())) return R(1) / (Sine ? std::sinh(x) : std::cosh(x));

  // Beyond digits(), the relative denominator correction exp(-2*abs(x)) is
  // smaller than epsilon squared. Evaluate the shared tail 2*exp(-abs(x)).
  R result;
  if (magnitude < R(-L::min_exponent) * log_two<R>)
    result = R(2) * std::exp(-magnitude);
  else
  {
    // Squaring an exponential at half the argument preserves representable
    // final subnormals which exp(-abs(x)) might prematurely round to zero.
    R half = std::exp(-magnitude / 2);
    result = half * (R(2) * half);
  }
  if constexpr (Sine) return std::copysign(result, x);
  return result;
}
template <NativeReal R> R sech(R x) { return reciprocal_hyperbolic<false>(x); }
template <NativeReal R> R csch(R x) { return reciprocal_hyperbolic<true>(x); }
} // namespace uni20::detail::native_math
