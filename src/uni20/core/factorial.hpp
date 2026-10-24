#pragma once

#include "detail/native_math.hpp"
#include "runtime_precision.hpp"
#if UNI20_HAS_MPREAL
#include "mpreal.hpp"
#endif

namespace uni20
{
/// \brief Factorial of a checked nonnegative integer, with an explicit native result type.
/// \details Returns positive infinity on native overflow. Uses compensated multiplication
///          and work bounded by the native exponent range, even for a huge integer order.
/// \throws std::domain_error for a negative order; std::out_of_range beyond signed64.
template <detail::native_math::NativeReal R, detail::MathInteger I> R factorial(I n)
{
  auto order = detail::math_integer(n);
  if (order < 0) throw std::domain_error("factorial requires a nonnegative order");
  using L = numeric_limits<R>;
  if (order > L::max_exponent) return L::infinity(); // n! >= 2^(n-1).
  R high = 1, low = 0;
  for (std::int64_t k = 2; k <= order; ++k)
  {
    R factor = R(k);
    R product = high * factor;
    if (std::isinf(product)) return product;
    // Recover the high product's rounding residual and propagate the low word.
    // FastTwoSum renormalizes the much smaller correction into a new low word.
    R error = std::fma(high, factor, -product) + low * factor;
    R sum = product + error;
    low = error - (sum - product);
    high = sum;
  }
  return high;
}

/// \brief Typed factorial factory with a checked native precision argument.
/// \throws std::invalid_argument if p differs from the native result precision.
template <detail::native_math::NativeReal R, detail::MathInteger I> R factorial(I n, Precision p)
{
  require_precision<R>(p);
  return uni20::factorial<R>(n);
}

#if UNI20_HAS_MPREAL
/// \brief Typed MPFR factorial factory at explicit finite or exact precision.
/// \details Delegates to the MPFR scalar factory; exact factorials retain their full
///          integer value and can require large storage. No ambient precision is used.
template <class R, detail::MathInteger I>
  requires std::same_as<R, mpreal>
R factorial(I n, Precision p)
{
  return uni20::factorial(n, p);
}
#endif
} // namespace uni20
