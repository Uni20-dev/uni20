#pragma once

#include "native_math.hpp"
#include <uni20/core/math_constants.hpp>

namespace uni20::detail::native_math
{
// The remainder on division by 1/2 is exact for binary inputs. remquo gives
// enough quotient bits to recover the quadrant, even when 2*x would overflow
// or when x is far too large for an integer carrier. Only the reduced argument
// (in [-1/4,1/4]) is multiplied by pi.
template <NativeReal R> struct portable_pi_math
{
    static R sinpi(R x)
    {
      if (!std::isfinite(x)) return numeric_limits<R>::quiet_NaN();
      int q;
      R r = std::remquo(x, R(0.5), &q);
      q &= 3;
      if (r == R(0)) return q % 2 == 0 ? std::copysign(R(0), x) : (q == 1 ? R(1) : R(-1));
      R angle = pi<R> * r;
      switch (q)
      {
        case 0:
          return std::sin(angle);
        case 1:
          return std::cos(angle);
        case 2:
          return -std::sin(angle);
        default:
          return -std::cos(angle);
      }
    }

    static R cospi(R x)
    {
      if (!std::isfinite(x)) return numeric_limits<R>::quiet_NaN();
      int q;
      R r = std::remquo(x, R(0.5), &q);
      q &= 3;
      if (r == R(0)) return q % 2 != 0 ? R(0) : (q == 0 ? R(1) : R(-1));
      R angle = pi<R> * r;
      switch (q)
      {
        case 0:
          return std::cos(angle);
        case 1:
          return -std::sin(angle);
        case 2:
          return -std::cos(angle);
        default:
          return std::sin(angle);
      }
    }

    static R tanpi(R x)
    {
      if (!std::isfinite(x)) return numeric_limits<R>::quiet_NaN();
      int q;
      R r = std::remquo(x, R(0.5), &q);
      q &= 3;
      if (r == R(0))
      {
        if (q % 2 == 0) return std::copysign(R(0), x);
        return q == 1 ? numeric_limits<R>::infinity() : -numeric_limits<R>::infinity();
      }
      // At a quarter-integer, remquo's ties-to-even quotient is even.
      // Preserve the exact result independently of libm's tan.
      if (std::abs(r) == R(0.25)) return std::copysign(R(1), r);
      R value = std::tan(pi<R> * r);
      return q % 2 == 0 ? value : -R(1) / value;
    }

    static R asinpi(R x)
    {
      if (x == R(0)) return x;
      if (std::abs(x) == R(1)) return std::copysign(R(0.5), x);
      return std::asin(x) / pi<R>;
    }

    static R acospi(R x)
    {
      if (x == R(0)) return R(0.5);
      if (x == R(1)) return R(0);
      if (x == R(-1)) return R(1);
      return std::acos(x) / pi<R>;
    }

    static R atanpi(R x)
    {
      if (x == R(0)) return x;
      if (std::abs(x) == R(1)) return std::copysign(R(0.25), x);
      if (std::isinf(x)) return std::copysign(R(0.5), x);
      return std::atan(x) / pi<R>;
    }

    static R atan2pi(R y, R x)
    {
      if (std::isnan(x) || std::isnan(y)) return numeric_limits<R>::quiet_NaN();
      if (y == R(0)) return std::copysign(std::signbit(x) ? R(1) : R(0), y);
      if (x == R(0)) return std::copysign(R(0.5), y);
      // This also handles all four infinite diagonals, before either infinity
      // is treated as an axis. No y/x is formed, so extreme ratios remain safe.
      if (std::abs(x) == std::abs(y)) return std::copysign(x > R(0) ? R(0.25) : R(0.75), y);
      if (std::isinf(y)) return std::copysign(R(0.5), y);
      if (std::isinf(x)) return std::copysign(x > R(0) ? R(0) : R(1), y);
      return std::atan2(y, x) / pi<R>;
    }
};

// Each native family is probed for its exact argument/result type and linked
// symbol. A missing family inherits the portable implementation at that type.
template <NativeReal R> struct pi_math_provider : portable_pi_math<R>
{};

#define UNI20_NATIVE_PI_FORWARD(TYPE, SUFFIX)                                                                          \
  static TYPE sinpi(TYPE x) { return ::sinpi##SUFFIX(x); }                                                             \
  static TYPE cospi(TYPE x) { return ::cospi##SUFFIX(x); }                                                             \
  static TYPE tanpi(TYPE x) { return ::tanpi##SUFFIX(x); }
#define UNI20_NATIVE_PI_INVERSE(TYPE, SUFFIX)                                                                          \
  static TYPE asinpi(TYPE x) { return ::asinpi##SUFFIX(x); }                                                           \
  static TYPE acospi(TYPE x) { return ::acospi##SUFFIX(x); }                                                           \
  static TYPE atanpi(TYPE x) { return ::atanpi##SUFFIX(x); }                                                           \
  static TYPE atan2pi(TYPE y, TYPE x) { return ::atan2pi##SUFFIX(y, x); }

#if (UNI20_HAS_PI_TRIG_FLOAT || UNI20_HAS_PI_INVERSE_FLOAT)
template <> struct pi_math_provider<float> : portable_pi_math<float>
{
#if UNI20_HAS_PI_TRIG_FLOAT
    UNI20_NATIVE_PI_FORWARD(float, f)
#endif
#if UNI20_HAS_PI_INVERSE_FLOAT
    UNI20_NATIVE_PI_INVERSE(float, f)
#endif
};
#endif

#if (UNI20_HAS_PI_TRIG_DOUBLE || UNI20_HAS_PI_INVERSE_DOUBLE)
template <> struct pi_math_provider<double> : portable_pi_math<double>
{
#if UNI20_HAS_PI_TRIG_DOUBLE
    UNI20_NATIVE_PI_FORWARD(double, )
#endif
#if UNI20_HAS_PI_INVERSE_DOUBLE
    UNI20_NATIVE_PI_INVERSE(double, )
#endif
};
#endif

#if (UNI20_HAS_PI_TRIG_LONG_DOUBLE || UNI20_HAS_PI_INVERSE_LONG_DOUBLE)
template <> struct pi_math_provider<long double> : portable_pi_math<long double>
{
#if UNI20_HAS_PI_TRIG_LONG_DOUBLE
    UNI20_NATIVE_PI_FORWARD(long double, l)
#endif
#if UNI20_HAS_PI_INVERSE_LONG_DOUBLE
    UNI20_NATIVE_PI_INVERSE(long double, l)
#endif
};
#endif

#if UNI20_HAS_FLOAT128 && (UNI20_HAS_PI_TRIG_FLOAT128 || UNI20_HAS_PI_INVERSE_FLOAT128)
template <> struct pi_math_provider<float128> : portable_pi_math<float128>
{
#if UNI20_HAS_PI_TRIG_FLOAT128
    UNI20_NATIVE_PI_FORWARD(float128, f128)
#endif
#if UNI20_HAS_PI_INVERSE_FLOAT128
    UNI20_NATIVE_PI_INVERSE(float128, f128)
#endif
};
#endif

#undef UNI20_NATIVE_PI_FORWARD
#undef UNI20_NATIVE_PI_INVERSE

template <NativeReal R> R sinpi(R x) { return pi_math_provider<R>::sinpi(x); }
template <NativeReal R> R cospi(R x) { return pi_math_provider<R>::cospi(x); }
template <NativeReal R> R tanpi(R x) { return pi_math_provider<R>::tanpi(x); }
template <NativeReal R> R asinpi(R x) { return pi_math_provider<R>::asinpi(x); }
template <NativeReal R> R acospi(R x) { return pi_math_provider<R>::acospi(x); }
template <NativeReal R> R atanpi(R x) { return pi_math_provider<R>::atanpi(x); }
template <NativeReal R, NativeReal S> auto atan2pi(R y, S x)
{
  using T = std::common_type_t<R, S>;
  return pi_math_provider<T>::atan2pi(T(y), T(x));
}
} // namespace uni20::detail::native_math
