#pragma once

#include "math_integer.hpp"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <uni20/core/math_results.hpp>
#include <uni20/core/scalar_traits.hpp>

namespace uni20::detail::native_math
{
template <class T>
concept NativeReal = std::same_as<T, float> || std::same_as<T, double> || std::same_as<T, long double>
#if UNI20_HAS_FLOAT128
                     || std::same_as<T, uni20::float128>
#endif
    ;

template <NativeReal R> R round_even(R x)
{
  R lower = std::floor(x);
  R fraction = x - lower;
  if (fraction > R(0.5) || (fraction == R(0.5) && std::fmod(lower, R(2)) != R(0))) lower += R(1);
  return lower == R(0) ? std::copysign(lower, x) : lower;
}

template <NativeReal R> frexp_result<R> frexp(R x)
{
  if (!std::isfinite(x) || x == R(0)) return {x, 0};
  int exponent;
  R fraction = std::frexp(x, &exponent);
  return {fraction, exponent};
}
template <NativeReal R> modf_result<R> modf(R x)
{
  R integer;
  R fraction = std::modf(x, &integer);
  return {fraction, integer};
}
template <NativeReal R> remquo_result<R> remquo(R x, R y)
{
  if (!std::isfinite(x) || std::isnan(y) || y == R(0)) return {numeric_limits<R>::quiet_NaN(), 0};
  int quotient = 0;
  R remainder = std::remquo(x, y, &quotient);
  return {remainder, quotient % 8};
}
template <NativeReal R> std::int64_t ilogb(R x)
{
  if (!std::isfinite(x) || x == R(0)) throw std::domain_error("ilogb requires a finite nonzero value");
  return native_math::frexp(x).exponent - 1;
}
template <NativeReal R, MathInteger I> R ldexp(R x, I exponent)
{
  // Native floating exponents fit int. Clamping an even larger requested
  // shift preserves overflow/underflow, including signed zero and infinity.
  auto n = std::clamp<std::int64_t>(math_integer(exponent), numeric_limits<int>::min(), numeric_limits<int>::max());
  return std::ldexp(x, static_cast<int>(n));
}
template <NativeReal R, MathInteger I> R scalbn(R x, I exponent) { return native_math::ldexp(x, exponent); }

template <NativeReal R, MathInteger I> R pown(R x, I value)
{
  auto exponent = math_integer(value);
  std::uint64_t magnitude = exponent < 0 ? std::uint64_t(0) - std::uint64_t(exponent) : std::uint64_t(exponent);
  R result(1);
  if (exponent < 0) x = result / x;
  while (magnitude)
  {
    if (magnitude & 1) result *= x;
    magnitude >>= 1;
    if (magnitude) x *= x;
  }
  return result;
}
template <NativeReal R, MathInteger I> R rootn(R x, I value)
{
  auto degree = math_integer(value);
  if (degree == 0) return numeric_limits<R>::quiet_NaN();
  if (degree == 1) return x;
  if (degree == -1) return R(1) / x;
  bool odd = degree % 2 != 0;
  if (std::signbit(x) && !odd && x != R(0)) return numeric_limits<R>::quiet_NaN();
  R magnitude = std::abs(x);
  R result;
  if (degree == 2 || degree == -2)
    result = std::sqrt(magnitude);
  else if (degree == 3 || degree == -3)
    result = std::cbrt(magnitude);
  else
    return odd ? std::copysign(std::pow(magnitude, R(1) / R(degree)), x) : std::pow(magnitude, R(1) / R(degree));
  if (degree < 0) result = R(1) / result;
  return odd ? std::copysign(result, x) : result;
}
template <NativeReal R> sincos_result<R> sincos(R x) { return {std::sin(x), std::cos(x)}; }
template <NativeReal R> sinhcosh_result<R> sinhcosh(R x) { return {std::sinh(x), std::cosh(x)}; }

template <NativeReal R, NativeReal S> R copysign(R magnitude, S sign)
{
  return std::copysign(magnitude, std::signbit(sign) ? R(-1) : R(1));
}
template <NativeReal R, NativeReal S> auto fmin(R a, S b)
{
  auto result = std::fmin(a, b);
  if (a == 0 && b == 0) return std::copysign(result, decltype(result)((std::signbit(a) || std::signbit(b)) ? -1 : 1));
  return result;
}
template <NativeReal R, NativeReal S> auto fmax(R a, S b)
{
  auto result = std::fmax(a, b);
  if (a == 0 && b == 0) return std::copysign(result, decltype(result)((std::signbit(a) && std::signbit(b)) ? -1 : 1));
  return result;
}
template <NativeReal R> R next_up(R x) { return std::nextafter(x, numeric_limits<R>::infinity()); }
template <NativeReal R> R next_down(R x) { return std::nextafter(x, -numeric_limits<R>::infinity()); }
template <NativeReal R, NativeReal S> R nextafter(R x, S direction)
{
  if (std::isnan(x) || std::isnan(direction)) return numeric_limits<R>::quiet_NaN();
  if (x == direction) return native_math::copysign(x, direction);
  return x < direction ? native_math::next_up(x) : native_math::next_down(x);
}
} // namespace uni20::detail::native_math
