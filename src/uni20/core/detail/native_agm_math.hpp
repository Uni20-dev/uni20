#pragma once

#include "native_math.hpp"
#include <bit>

namespace uni20::detail::native_math
{
template <NativeReal R> R agm(R a, R b)
{
  using L = numeric_limits<R>;
  if (std::isnan(a) || std::isnan(b)) return L::quiet_NaN();
  if (a == 0 || b == 0) return std::isfinite(a) && std::isfinite(b) ? R(0) : L::quiet_NaN();
  if (a < 0 || b < 0) return L::quiet_NaN();
  if (a == b) return a;
  if (std::isinf(a) || std::isinf(b)) return L::infinity();
  if (a < b) std::swap(a, b);

  int ae, be;
  a = std::frexp(a, &ae);
  b = std::frexp(b, &be);
  while (ae - be > L::digits() + 2)
  {
    // Keep separate scales until neither input can be erased by a common
    // scale. Here the smaller term changes the arithmetic mean by <u/4,
    // where u=2^-digits. The larger significand is unchanged when halved.
    int sum = ae + be;
    int odd = (sum % 2 + 2) % 2;
    int exponent = (sum - odd) / 2;
    R geometric = std::sqrt(a * b * R(odd ? 2 : 1));
    int shift;
    b = std::frexp(geometric, &shift);
    be = exponent + shift;
    --ae;
  }

  // Both scaled values and their product are normal for every supported
  // native format. Restore the original exponent only at the final rounding.
  b = std::ldexp(b, be - ae);
  constexpr int limit = 2 * std::bit_width(static_cast<unsigned>(L::digits())) + 8;
  for (int iteration = 0; iteration < limit; ++iteration)
  {
    R arithmetic = (a + b) / 2;
    if (std::abs(a - b) <= std::ldexp(a, 1 - L::digits())) return std::ldexp(arithmetic, ae);
    b = std::sqrt(a * b);
    a = arithmetic;
  }
  // Quadratic convergence should reach the spacing test well before this
  // conservative bound. Never return a silently unfinished approximation.
  return L::quiet_NaN();
}
} // namespace uni20::detail::native_math
