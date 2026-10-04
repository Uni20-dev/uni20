#include <cstdint>
#include <gtest/gtest.h>
#include <type_traits>
#include <uni20/core/math.hpp>

namespace
{
namespace m = uni20::math;
using uni20::numeric_limits;
struct Convertible
{
    operator double() const { return 1; }
};

template <class... F> consteval bool unary_utilities(F...)
{
  return ((std::invocable<F, float> && std::invocable<F, double> && std::invocable<F, long double> &&
           !std::invocable<F, Convertible> && !std::invocable<F, double, double>) &&
          ...);
}
static_assert(unary_utilities(m::floor, m::ceil, m::trunc, m::round, m::round_even, m::frexp, m::modf, m::ilogb,
                              m::sincos, m::sinhcosh, m::next_up, m::next_down, m::isfinite, m::isnan, m::isinf,
                              m::signbit));
template <class... F> consteval bool integer_utilities(F...)
{
  return ((std::invocable<F, double, int> && std::invocable<F, float, std::int64_t> &&
           !std::invocable<F, double, bool> && !std::invocable<F, double, double> &&
           !std::invocable<F, Convertible, int>) &&
          ...);
}
static_assert(integer_utilities(m::pown, m::rootn, m::ldexp, m::scalbn));
static_assert(std::same_as<decltype(m::frexp(1.0)), uni20::frexp_result<double>>);
static_assert(std::same_as<decltype(m::modf(1.0)), uni20::modf_result<double>>);
static_assert(std::same_as<decltype(m::remquo(1.0, 1.0)), uni20::remquo_result<double>>);
static_assert(std::same_as<decltype(m::nextafter(1.0f, 1.0L)), float>);
static_assert(std::same_as<decltype(m::copysign(1.0f, 1.0L)), float>);
static_assert(!std::invocable<decltype(m::fma), Convertible, Convertible, Convertible>);
static_assert(!std::invocable<decltype(m::remquo), Convertible, Convertible>);

TEST(ScalarMathUtilities, NativeTiesSignsAndExceptionalValues)
{
  for (double sign : {-1.0, 1.0})
  {
    EXPECT_EQ(m::round_even(sign * 2.5), sign * 2);
    EXPECT_EQ(m::round_even(sign * 3.5), sign * 4);
    EXPECT_EQ(m::round(sign * 2.5), sign * 3);
  }
  EXPECT_TRUE(m::signbit(m::round_even(-0.5)));
  EXPECT_TRUE(m::signbit(m::nextafter(0.0, -0.0L)));
  EXPECT_TRUE(m::signbit(m::fmin(0.0, -0.0)));
  EXPECT_TRUE(m::signbit(m::fmin(-0.0, 0.0)));
  EXPECT_FALSE(m::signbit(m::fmax(0.0, -0.0)));
  EXPECT_FALSE(m::signbit(m::fmax(-0.0, 0.0)));
  auto inf = numeric_limits<double>::infinity(), nan = numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(m::round_even(inf), inf);
  EXPECT_TRUE(m::isnan(m::round_even(nan)));
  EXPECT_EQ(m::frexp(inf).exponent, 0);
  EXPECT_EQ(m::frexp(inf).fraction, inf);
  EXPECT_EQ(m::remquo(inf, 1.0).quotient, 0);
  EXPECT_TRUE(m::isnan(m::remquo(inf, 1.0).remainder));
  EXPECT_EQ(m::remquo(-23.0, 2.0).quotient, -4);
  EXPECT_EQ(m::remquo(-23.0, 2.0).remainder, 1);
  EXPECT_THROW(m::ilogb(0.0), std::domain_error);
  EXPECT_THROW(m::ilogb(nan), std::domain_error);
  EXPECT_THROW(m::ilogb(inf), std::domain_error);
  EXPECT_EQ(m::next_up(0.0), numeric_limits<double>::denorm_min());
  EXPECT_EQ(m::next_up(numeric_limits<double>::max()), inf);
  EXPECT_EQ(m::next_down(inf), numeric_limits<double>::max());
}

TEST(ScalarMathUtilities, NativeIntegerDomainAndNegativeRoots)
{
  auto huge = numeric_limits<std::int64_t>::max(), small = numeric_limits<std::int64_t>::min();
  EXPECT_EQ(m::pown(-1.0, huge), -1);
  EXPECT_EQ(m::pown(-1.0, small), 1);
  EXPECT_EQ(m::rootn(-1.0, huge), -1);
  EXPECT_TRUE(m::isnan(m::rootn(-1.0, small)));
  EXPECT_EQ(m::rootn(-8.0, -3), -0.5);
  EXPECT_TRUE(m::isnan(m::rootn(-8.0, 2)));
  EXPECT_TRUE(m::isnan(m::rootn(1.0, 0)));
  EXPECT_TRUE(m::isinf(m::ldexp(1.0, huge)));
  EXPECT_EQ(m::ldexp(-1.0, small), 0);
  EXPECT_TRUE(m::signbit(m::ldexp(-1.0, small)));
  EXPECT_EQ(m::ldexp(1.0, char(3)), 8);
  auto too_large = numeric_limits<std::uint64_t>::max();
  EXPECT_THROW(m::ldexp(1.0, too_large), std::out_of_range);
  EXPECT_THROW(m::scalbn(1.0, too_large), std::out_of_range);
  EXPECT_THROW(m::pown(1.0, too_large), std::out_of_range);
  EXPECT_THROW(m::rootn(1.0, too_large), std::out_of_range);
}
} // namespace
