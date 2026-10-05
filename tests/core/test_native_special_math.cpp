#include <gtest/gtest.h>
#include <uni20/core/math.hpp>

namespace
{
namespace m = uni20::math;
using uni20::numeric_limits;

struct ConversionOnly
{
    operator double() const { return 1; }
};
static_assert(!std::invocable<decltype(m::lgamma), ConversionOnly>);
static_assert(!std::invocable<decltype(m::lgamma_sign), ConversionOnly>);
static_assert(!std::invocable<decltype(m::bessel_j), int, ConversionOnly>);
static_assert(!std::invocable<decltype(m::bessel_y), int, ConversionOnly>);
static_assert(!std::invocable<decltype(m::bessel_j), bool, double>);
static_assert(!std::invocable<decltype(m::bessel_y), double, double>);

template <class R, bool Gamma, bool Bessel> void check_native_special()
{
  static_assert(std::invocable<decltype(m::lgamma), R> == Gamma);
  static_assert(std::invocable<decltype(m::lgamma_sign), R> == Gamma);
  static_assert(std::invocable<decltype(m::bessel_j), int, R> == Bessel);
  static_assert(std::invocable<decltype(m::bessel_y), int, R> == Bessel);
  R zero(0), one(1), inf = numeric_limits<R>::infinity(), nan = numeric_limits<R>::quiet_NaN();
  if constexpr (Gamma)
  {
    static_assert(std::same_as<decltype(m::lgamma(one)), R>);
    static_assert(std::same_as<decltype(m::lgamma_sign(one)), uni20::lgamma_result<R>>);
    EXPECT_EQ(m::lgamma(one), zero);
    EXPECT_EQ(m::lgamma(R(2)), zero);
    for (R x : {R(-0.5), R(-1.5), R(-2.5), R(0.5), R(3.5)})
    {
      auto result = m::lgamma_sign(x);
      auto next = m::lgamma_sign(x + one);
      R error = m::abs(next.value - result.value - m::log(m::abs(x)));
      EXPECT_TRUE(error <= R(16) * numeric_limits<R>::epsilon());
      EXPECT_EQ(next.sign, x < zero ? -result.sign : result.sign);
    }
    EXPECT_EQ(m::lgamma_sign(R(-0.5)).sign, -1);
    EXPECT_EQ(m::lgamma_sign(R(-1.5)).sign, 1);
    EXPECT_EQ(m::lgamma_sign(zero).sign, 1);
    EXPECT_EQ(m::lgamma_sign(-zero).sign, -1);
    EXPECT_EQ(m::lgamma_sign(inf).sign, 1);
    for (R x : {R(-1), R(-2), -inf})
    {
      EXPECT_EQ(m::lgamma_sign(x).sign, 0);
      EXPECT_TRUE(m::isinf(m::lgamma(x)));
    }
    EXPECT_EQ(m::lgamma_sign(nan).sign, 0);
    EXPECT_TRUE(m::isnan(m::lgamma(nan)));
#if defined(__GLIBC__)
    // A wrapper around ordinary lgamma would modify this global even if its
    // caller only requested the magnitude, creating races in async programs.
    int saved = ::signgam;
    ::signgam = 73;
    (void)m::lgamma(R(-0.5));
    (void)m::lgamma_sign(R(0.5));
    EXPECT_EQ(::signgam, 73);
    ::signgam = saved;
#endif
  }
  if constexpr (Bessel)
  {
    static_assert(std::same_as<decltype(m::bessel_j(1, one)), R>);
    static_assert(std::same_as<decltype(m::bessel_y(1, one)), R>);
    EXPECT_EQ(m::bessel_j(0, zero), one);
    EXPECT_EQ(m::bessel_j(1, zero), zero);
    EXPECT_TRUE(m::signbit(m::bessel_j(1, -zero)));
    EXPECT_FALSE(m::signbit(m::bessel_j(-1, -zero)));
    for (int n : {1, 2, 3})
    {
      R sign = n % 2 ? R(-1) : one;
      EXPECT_EQ(m::bessel_j(-n, one), sign * m::bessel_j(n, one));
      EXPECT_EQ(m::bessel_j(n, -one), sign * m::bessel_j(n, one));
      EXPECT_EQ(m::bessel_y(-n, one), sign * m::bessel_y(n, one));
      R recurrence = m::bessel_j(n - 1, one) + m::bessel_j(n + 1, one) - R(2 * n) * m::bessel_j(n, one);
      EXPECT_TRUE(m::abs(recurrence) <= R(16) * numeric_limits<R>::epsilon());
    }
    EXPECT_TRUE(m::isnan(m::bessel_j(1, nan)));
    EXPECT_TRUE(m::isnan(m::bessel_y(1, nan)));
    EXPECT_TRUE(m::isnan(m::bessel_y(1, -one)));
    EXPECT_EQ(m::bessel_j(1, inf), zero);
    EXPECT_EQ(m::bessel_y(1, inf), zero);
    EXPECT_EQ(m::bessel_y(1, zero), -inf);
    constexpr auto largest = numeric_limits<int>::max();
    EXPECT_EQ(m::bessel_j(largest, zero), zero);
    EXPECT_EQ(m::bessel_j(-largest, zero), zero);
    EXPECT_THROW(m::bessel_j(std::int64_t(largest) + 1, zero), std::out_of_range);
    EXPECT_THROW(m::bessel_y(-std::int64_t(largest) - 1, zero), std::out_of_range);
    EXPECT_THROW(m::bessel_j(numeric_limits<std::uint64_t>::max(), zero), std::out_of_range);
  }
}

TEST(NativeSpecialMath, Float) { check_native_special<float, UNI20_HAS_LGAMMA_R_FLOAT, UNI20_HAS_BESSEL_FLOAT>(); }
TEST(NativeSpecialMath, Double) { check_native_special<double, UNI20_HAS_LGAMMA_R_DOUBLE, UNI20_HAS_BESSEL_DOUBLE>(); }
TEST(NativeSpecialMath, LongDouble)
{
  check_native_special<long double, UNI20_HAS_LGAMMA_R_LONG_DOUBLE, UNI20_HAS_BESSEL_LONG_DOUBLE>();
}
#if UNI20_HAS_FLOAT128
TEST(NativeSpecialMath, Float128)
{
  check_native_special<uni20::float128, UNI20_HAS_LGAMMA_R_FLOAT128, UNI20_HAS_BESSEL_FLOAT128>();
}
#endif
} // namespace
