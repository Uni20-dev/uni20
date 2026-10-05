// Separate executable: simulate missing standard special functions and missing
// float/extended Gamma/Bessel providers, without changing other translation units.
#include <uni20/config.hpp>
#include <cmath>
#include <version>
#undef __cpp_lib_math_special_functions
#undef UNI20_HAS_LGAMMA_R_FLOAT
#undef UNI20_HAS_LGAMMA_R_LONG_DOUBLE
#undef UNI20_HAS_LGAMMA_R_FLOAT128
#undef UNI20_HAS_BESSEL_FLOAT
#undef UNI20_HAS_BESSEL_LONG_DOUBLE
#undef UNI20_HAS_BESSEL_FLOAT128
#define UNI20_HAS_LGAMMA_R_FLOAT 0
#define UNI20_HAS_LGAMMA_R_LONG_DOUBLE 0
#define UNI20_HAS_LGAMMA_R_FLOAT128 0
#define UNI20_HAS_BESSEL_FLOAT 0
#define UNI20_HAS_BESSEL_LONG_DOUBLE 0
#define UNI20_HAS_BESSEL_FLOAT128 0
#include <uni20/core/math.hpp>

namespace m = uni20::math;
template <class R> constexpr bool unavailable =
    !std::invocable<decltype(m::lgamma), R> && !std::invocable<decltype(m::lgamma_sign), R> &&
    !std::invocable<decltype(m::bessel_j), int, R> && !std::invocable<decltype(m::bessel_y), int, R>;
static_assert(unavailable<float>);
static_assert(unavailable<long double>);
#if UNI20_HAS_FLOAT128
static_assert(unavailable<uni20::float128>);
#endif
static_assert(std::invocable<decltype(m::lgamma), double> == bool(UNI20_HAS_LGAMMA_R_DOUBLE));
static_assert(std::invocable<decltype(m::bessel_j), int, double> == bool(UNI20_HAS_BESSEL_DOUBLE));
#if UNI20_ENABLE_MPFR
static_assert(std::invocable<decltype(m::lgamma), uni20::mpreal>);
static_assert(std::invocable<decltype(m::bessel_j), int, uni20::mpreal>);
#endif
template <class R> constexpr bool standard_unavailable =
    !std::invocable<decltype(m::beta), R, R> && !std::invocable<decltype(m::zeta), R> &&
    !std::invocable<decltype(m::expint), R>;
static_assert(standard_unavailable<float>);
static_assert(standard_unavailable<double>);
static_assert(standard_unavailable<long double>);
#if UNI20_HAS_FLOAT128
static_assert(standard_unavailable<uni20::float128>);
#endif
#if UNI20_ENABLE_MPFR
static_assert(std::invocable<decltype(m::beta), uni20::mpreal, uni20::mpreal>);
static_assert(std::invocable<decltype(m::zeta), uni20::mpreal>);
static_assert(std::invocable<decltype(m::expint), uni20::mpreal>);
#endif
int main() {}
