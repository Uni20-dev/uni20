#include <gtest/gtest.h>
#include <uni20/core/math.hpp>

namespace
{
namespace m = uni20::math;
struct ConversionOnly
{
    operator double() const;
};
template <class R>
constexpr bool signature = std::same_as<std::invoke_result_t<decltype(m::zeta), R>, R> &&
                           !std::invocable<decltype(m::zeta), R, uni20::Precision>;
static_assert(signature<float> && signature<double> && signature<long double>);
static_assert(!std::invocable<decltype(m::zeta), int>);
static_assert(!std::invocable<decltype(m::zeta), ConversionOnly>);
static_assert(!std::invocable<decltype(m::zeta), uni20::complex<double>>);
#if UNI20_HAS_FLOAT128
static_assert(signature<uni20::float128>);
#endif

template <class R> class NativeZeta : public testing::Test {};
using NativeReals = testing::Types<float, double, long double
#if UNI20_HAS_FLOAT128
                                   ,
                                   uni20::float128
#endif
                                   >;
TYPED_TEST_SUITE(NativeZeta, NativeReals);
TYPED_TEST(NativeZeta, ExceptionalValuesAndExactZeros)
{
  using R = TypeParam;
  using L = uni20::numeric_limits<R>;
  EXPECT_EQ(m::zeta(R(0)), -R(.5));
  EXPECT_EQ(m::zeta(-R(0)), -R(.5));
  EXPECT_EQ(m::zeta(R(1)), L::infinity());
  EXPECT_EQ(m::zeta(L::infinity()), R(1));
  EXPECT_TRUE(m::isnan(m::zeta(-L::infinity())));
  EXPECT_TRUE(m::isnan(m::zeta(L::quiet_NaN())));
  for (R x : {R(-2), R(-16), R(-34), -L::max()})
  {
    EXPECT_EQ(m::zeta(x), R(0));
    EXPECT_FALSE(m::signbit(m::zeta(x)));
  }
  EXPECT_EQ(m::zeta(L::max()), R(1));
  EXPECT_EQ(m::zeta(L::denorm_min()), -R(.5));
  EXPECT_EQ(m::zeta(-L::denorm_min()), -R(.5));
}
TYPED_TEST(NativeZeta, PoleAndReflection)
{
  using R = TypeParam;
  for (R x : {R(-2), R(-16), R(-32)})
  {
    auto below = m::zeta(m::next_down(x)), above = m::zeta(m::next_up(x));
    EXPECT_NE(below, R(0));
    EXPECT_NE(above, R(0));
    EXPECT_NE(m::signbit(below), m::signbit(above));
  }
  EXPECT_LT(m::zeta(m::next_down(R(1))), R(0));
  EXPECT_GT(m::zeta(m::next_up(R(1))), R(0));
  // The former GCC13 float provider overflowed at this finite input.
  EXPECT_TRUE(m::isfinite(m::zeta(R(-35))));
  EXPECT_GT(m::zeta(R(-35)), R(0));
}
} // namespace
