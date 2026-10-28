#pragma once

#include "native_math.hpp"
#include "stirling_coefficients.hpp"

namespace uni20::detail::native_math
{
// Stirling log-Gamma correction at x=1/inverse. Callers use x>=8 for
// float32, x>=16 for binary64/80 and x>=32 for binary128; the coefficient
// generator records the first omitted term at those thresholds.
template <NativeReal R> R stirling_correction(R inverse)
{
  constexpr int bits = numeric_limits<R>::digits();
  constexpr int terms = bits <= 24 ? 4 : bits <= 53 ? 8 : bits <= 64 ? 10 : 16;
  constexpr auto coefficients = uni20::detail::native_stirling_coefficients<R>();
  R square = inverse * inverse, result = coefficients[terms - 1];
  for (int k = terms - 2; k >= 0; --k)
    result = std::fma(result, square, coefficients[k]);
  return inverse * result;
}
} // namespace uni20::detail::native_math
