#pragma once

#include "native_math.hpp"

namespace uni20::detail::native_math
{
template <class R> struct lgamma_provider
{};
template <class R> struct bessel_provider
{};

#define UNI20_NATIVE_LGAMMA_PROVIDER(TYPE, FUNCTION)                                                                   \
  template <> struct lgamma_provider<TYPE>                                                                             \
  {                                                                                                                    \
      static TYPE call(TYPE x, int* sign) { return FUNCTION(x, sign); }                                                \
  };
#if UNI20_HAS_LGAMMA_R_FLOAT
UNI20_NATIVE_LGAMMA_PROVIDER(float, ::lgammaf_r)
#endif
#if UNI20_HAS_LGAMMA_R_DOUBLE
UNI20_NATIVE_LGAMMA_PROVIDER(double, ::lgamma_r)
#endif
#if UNI20_HAS_LGAMMA_R_LONG_DOUBLE
UNI20_NATIVE_LGAMMA_PROVIDER(long double, ::lgammal_r)
#endif
#if UNI20_HAS_FLOAT128 && UNI20_HAS_LGAMMA_R_FLOAT128
UNI20_NATIVE_LGAMMA_PROVIDER(uni20::float128, ::lgammaf128_r)
#endif
#undef UNI20_NATIVE_LGAMMA_PROVIDER

#define UNI20_NATIVE_BESSEL_PROVIDER(TYPE, J, Y)                                                                       \
  template <> struct bessel_provider<TYPE>                                                                             \
  {                                                                                                                    \
      static TYPE first(int n, TYPE x) { return J(n, x); }                                                             \
      static TYPE second(int n, TYPE x) { return Y(n, x); }                                                            \
  };
#if UNI20_HAS_BESSEL_FLOAT
UNI20_NATIVE_BESSEL_PROVIDER(float, ::jnf, ::ynf)
#endif
#if UNI20_HAS_BESSEL_DOUBLE
UNI20_NATIVE_BESSEL_PROVIDER(double, ::jn, ::yn)
#endif
#if UNI20_HAS_BESSEL_LONG_DOUBLE
UNI20_NATIVE_BESSEL_PROVIDER(long double, ::jnl, ::ynl)
#endif
#if UNI20_HAS_FLOAT128 && UNI20_HAS_BESSEL_FLOAT128
UNI20_NATIVE_BESSEL_PROVIDER(uni20::float128, ::jnf128, ::ynf128)
#endif
#undef UNI20_NATIVE_BESSEL_PROVIDER

template <NativeReal R>
  requires requires(R x, int* sign) { lgamma_provider<R>::call(x, sign); }
lgamma_result<R> lgamma_sign(R x)
{
  int sign = 0;
  R value = lgamma_provider<R>::call(x, &sign);
  // The sign at a Gamma pole is undefined; normalize provider conventions to
  // the shared result contract. Signed zero retains its one-sided pole sign.
  if (std::isnan(x) || (x < R(0) && (std::isinf(x) || std::floor(x) == x)))
    sign = 0;
  else if (x == R(0))
    sign = std::signbit(x) ? -1 : 1;
  return {value, sign};
}

template <NativeReal R>
  requires requires(R x) { native_math::lgamma_sign(x); }
R lgamma(R x)
{
  return native_math::lgamma_sign(x).value;
}

template <MathInteger I> int native_bessel_order(I value)
{
  auto n = math_integer(value);
  // Normalize negative orders before calling providers, which may negate an
  // int internally. In particular INT_MIN cannot be lowered safely.
  constexpr auto maximum = numeric_limits<int>::max();
  if (n < -std::int64_t(maximum) || n > maximum)
    throw std::out_of_range("native Bessel absolute order does not fit int");
  return static_cast<int>(n);
}

template <MathInteger I, NativeReal R>
  requires requires(R x) { bessel_provider<R>::first(0, x); }
R bessel_j(I order, R x)
{
  int n = native_bessel_order(order);
  R result = bessel_provider<R>::first(n < 0 ? -n : n, x);
  return n < 0 && n % 2 != 0 ? -result : result;
}

template <MathInteger I, NativeReal R>
  requires requires(R x) { bessel_provider<R>::second(0, x); }
R bessel_y(I order, R x)
{
  int n = native_bessel_order(order);
  R result = bessel_provider<R>::second(n < 0 ? -n : n, x);
  return n < 0 && n % 2 != 0 ? -result : result;
}
} // namespace uni20::detail::native_math
