#pragma once

#include "native_math.hpp"
#include <uni20/core/math_constants.hpp>

namespace uni20::detail::native_math
{
template <NativeReal R> R expint_series(R x)
{
  using L = numeric_limits<R>;
  R sum = euler_gamma<R>, low = 0;
  auto add = [&](R value) {
    R next = sum + value;
    low += std::abs(sum) >= std::abs(value) ? (sum - next) + value : (value - next) + sum;
    sum = next;
  };
  add(std::log(std::abs(x)));
  R power = R(1);
  for (int k = 1; k <= 8 * L::digits() + 64; ++k)
  {
    power *= x / R(k);
    R term = power / R(k);
    add(term);
    if (R(k) >= 2 * std::abs(x) && std::abs(term) <= L::epsilon() * std::max(R(1), std::abs(sum)) / 16)
      return sum + low;
  }
  return L::quiet_NaN();
}

// Positive E1 continued fraction, divided by exp(-x). Neighboring positive
// truncations bracket the exact value; backwards evaluation avoids products
// of successive forward-update factors and cancellation in denominators.
template <NativeReal R> R scaled_e1(R x)
{
  using L = numeric_limits<R>;
  for (int depth = 16; depth <= 16 * L::digits(); depth *= 2)
  {
    R lo = R(depth) / (R(1) + R(depth) / x), hi = R(depth);
    for (int k = depth - 1; k >= 1; --k)
    {
      lo = R(k) / (R(1) + R(k) / (x + lo));
      hi = R(k) / (R(1) + R(k) / (x + hi));
    }
    lo = R(1) / (x + lo);
    hi = R(1) / (x + hi);
    if (std::abs(lo - hi) <= 4 * L::epsilon() * lo) return (lo + hi) / 2;
  }
  return L::quiet_NaN();
}

template <NativeReal R> R scaled_positive_ei(R x)
{
  using L = numeric_limits<R>;
  R sum = 1, low = 0, term = 1;
  for (int k = 1; k <= L::digits(); ++k)
  {
    term *= R(k) / x;
    R next = sum + term;
    low += (sum - next) + term;
    sum = next;
    // Here x>=2*digits and k<=digits, so the next term is smaller.
    // The DLMF remainder factor is <k+4, giving truncation <eps*sum/2.
    if (term <= L::epsilon() * sum / R(4 * (k + 2))) return (sum + low) / x;
  }
  return L::quiet_NaN();
}

template <NativeReal R> R expint(R x)
{
  using L = numeric_limits<R>;
  if (std::isnan(x)) return x;
  if (x == 0) return -L::infinity();
  // These conservative cutoffs are far beyond final native range boundaries.
  if (x > R(2 * L::max_exponent)) return L::infinity();
  if (x < R(2 * (L::min_exponent - L::digits()))) return -R(0);
  if (x >= R(-1) && x < R(2 * L::digits())) return native_math::expint_series(x);
  R factor = x < 0 ? -native_math::scaled_e1(-x) : native_math::scaled_positive_ei(x);
  R half = std::exp(x / 2);
  return half * (half * factor);
}
} // namespace uni20::detail::native_math
