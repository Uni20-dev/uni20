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
constexpr bool signature = std::same_as<std::invoke_result_t<decltype(m::digamma), R>, R> &&
                           !std::invocable<decltype(m::digamma), R, uni20::Precision>;
static_assert(signature<float> && signature<double> && signature<long double>);
static_assert(!std::invocable<decltype(m::digamma), int> && !std::invocable<decltype(m::digamma), ConversionOnly>);
static_assert(!std::invocable<decltype(m::digamma), uni20::complex<double>>);
#if UNI20_HAS_FLOAT128
static_assert(signature<uni20::float128>);
#endif

template <class R> class NativeDigamma : public testing::Test {};
using NativeReals = testing::Types<float, double, long double
#if UNI20_HAS_FLOAT128
                                   ,
                                   uni20::float128
#endif
                                   >;
TYPED_TEST_SUITE(NativeDigamma, NativeReals);
TYPED_TEST(NativeDigamma, PolesAndExceptionalValues)
{
  using R = TypeParam;
  using L = uni20::numeric_limits<R>;
  EXPECT_EQ(m::digamma(R(0)), -L::infinity());
  EXPECT_EQ(m::digamma(-R(0)), L::infinity());
  EXPECT_EQ(m::digamma(L::infinity()), L::infinity());
  EXPECT_TRUE(m::isnan(m::digamma(-L::infinity())));
  EXPECT_TRUE(m::isnan(m::digamma(L::quiet_NaN())));
  for (R x : {R(-1), R(-2), R(-1024), -L::max()})
    EXPECT_TRUE(m::isnan(m::digamma(x)));
  EXPECT_LT(m::digamma(m::next_up(R(-1))), R(0));
  EXPECT_GT(m::digamma(m::next_down(R(-1))), R(0));
}
TYPED_TEST(NativeDigamma, ExtremeRangeAndHalfIntegers)
{
  using R = TypeParam;
  using L = uni20::numeric_limits<R>;
  R x = R(1) / L::max();
  for (int i = 0; i < 16; ++i)
  {
    R reciprocal = R(1) / x;
    EXPECT_EQ(m::isinf(m::digamma(x)), m::isinf(reciprocal));
    EXPECT_EQ(m::isinf(m::digamma(-x)), m::isinf(reciprocal));
    EXPECT_LT(m::digamma(x), R(0));
    EXPECT_GT(m::digamma(-x), R(0));
    x = m::next_up(x);
  }
  EXPECT_EQ(m::digamma(L::denorm_min()), -L::infinity());
  EXPECT_EQ(m::digamma(-L::denorm_min()), L::infinity());
  EXPECT_TRUE(m::isfinite(m::digamma(L::max())));
  R expected = -uni20::euler_gamma<R> - 2 * uni20::log_two<R>;
  EXPECT_LE(m::abs(m::digamma(R(.5)) - expected), R(8) * L::epsilon() * m::abs(expected));
  EXPECT_LE(m::abs(m::digamma(R(-.5)) - (expected + 2)), R(8) * L::epsilon());
}
} // namespace
