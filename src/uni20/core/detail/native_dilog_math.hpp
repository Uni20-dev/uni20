#pragma once

#include "native_math.hpp"
#include <uni20/core/math_constants.hpp>

namespace uni20::detail::native_math
{
// Defining series on [0,1/2]. The tail after n terms is bounded by
// t^(n+1)/((n+1)^2*(1-t)); digits+4 terms suffice even at t=1/2.
template <NativeReal R> R dilog_series(R t)
{
  R power = t, sum = t, low = 0;
  for (int k = 2; k <= numeric_limits<R>::digits() + 4; ++k)
  {
    power *= t;
    R term = power / R(k * k);
    R next = sum + term;
    low += (sum - next) + term;
    sum = next;
    if (term <= numeric_limits<R>::epsilon() * sum / 8) break;
  }
  return sum + low;
}

template <NativeReal R> R dilog_real(R x)
{
  using L = numeric_limits<R>;
  if (std::isnan(x)) return x;
  if (std::isinf(x)) return -L::infinity();
  if (x == R(0)) return x;
  constexpr R sixth = pi<R> * pi<R> / 6;
  if (x == R(1)) return sixth;
  if (x < R(-1))
  {
    R inverse = R(1) / x, logarithm = std::log(-x), small = std::log1p(-inverse);
    R reduced = native_math::dilog_series(inverse / (inverse - R(1)));
    return std::fma(-logarithm / 2, logarithm, std::fma(small / 2, small, reduced) - sixth);
  }
  if (x < R(0))
  {
    R logarithm = std::log1p(-x);
    return -std::fma(logarithm / 2, logarithm, native_math::dilog_series(x / (x - R(1))));
  }
  if (x <= R(0.5)) return native_math::dilog_series(x);
  R logarithm = std::log(x);
  if (x < R(1)) return std::fma(-logarithm, std::log1p(-x), sixth - native_math::dilog_series(R(1) - x));
  if (x < R(2))
    return std::fma(logarithm, logarithm / 2 - std::log(x - R(1)), sixth + native_math::dilog_series((x - R(1)) / x));
  return std::fma(-logarithm / 2, logarithm, 2 * sixth - native_math::dilog_series(R(1) / x));
}
} // namespace uni20::detail::native_math
