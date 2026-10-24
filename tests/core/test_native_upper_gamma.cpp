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
constexpr bool signature =
    std::same_as<std::invoke_result_t<decltype(m::upper_gamma), R, R>, R> &&
    !std::invocable<decltype(m::upper_gamma), R, R, uni20::Precision> &&
    !std::invocable<decltype(m::upper_gamma), R, int> && !std::invocable<decltype(m::upper_gamma), int, R>;
static_assert(signature<float> && signature<double> && signature<long double>);
static_assert(!std::invocable<decltype(m::upper_gamma), double, float>);
static_assert(!std::invocable<decltype(m::upper_gamma), ConversionOnly, ConversionOnly>);
static_assert(!std::invocable<decltype(m::upper_gamma), uni20::complex<double>, uni20::complex<double>>);
#if UNI20_HAS_FLOAT128
static_assert(signature<uni20::float128>);
#endif

template <class R> class NativeUpperGamma : public testing::Test {};
using NativeReals = testing::Types<float, double, long double
#if UNI20_HAS_FLOAT128
                                   ,
                                   uni20::float128
#endif
                                   >;
TYPED_TEST_SUITE(NativeUpperGamma, NativeReals);
TYPED_TEST(NativeUpperGamma, EndpointAndExceptionalValues)
{
  using R = TypeParam;
  using L = uni20::numeric_limits<R>;
  auto inf = L::infinity(), nan = L::quiet_NaN();
  for (R a : {R(-3), R(-.5), R(0), R(.5), R(2)})
  {
    EXPECT_TRUE(m::isnan(m::upper_gamma(a, R(-1))));
    EXPECT_TRUE(m::isnan(m::upper_gamma(a, nan)));
    EXPECT_TRUE(m::isnan(m::upper_gamma(nan, a)));
    auto zero = m::upper_gamma(a, inf);
    EXPECT_EQ(zero, R(0));
    EXPECT_FALSE(m::signbit(zero));
  }
  for (R x : {R(0), -R(0)})
  {
    EXPECT_EQ(m::upper_gamma(R(5), x), R(24));
    // A compiler-folded tgamma constant and the runtime libm call can differ
    // by one rounding unit. Check the analytic endpoint at native precision.
    auto expected = -2 * m::sqrt(uni20::pi<R>);
    EXPECT_LE(m::abs(m::upper_gamma(R(-.5), x) - expected), 8 * L::epsilon() * m::abs(expected));
    EXPECT_TRUE(m::isnan(m::upper_gamma(R(-3), x)));
    EXPECT_EQ(m::upper_gamma(R(0), x), inf);
    EXPECT_EQ(m::upper_gamma(-R(0), x), -inf);
  }
  EXPECT_EQ(m::upper_gamma(inf, R(-1)), inf);
  EXPECT_TRUE(m::isnan(m::upper_gamma(inf, inf)));
  EXPECT_TRUE(m::isnan(m::upper_gamma(inf, -inf)));
  EXPECT_EQ(m::upper_gamma(-inf, R(.5)), inf);
  EXPECT_EQ(m::upper_gamma(-inf, R(1)), R(0));
  EXPECT_EQ(m::upper_gamma(-inf, inf), R(0));
  EXPECT_TRUE(m::isnan(m::upper_gamma(-inf, R(0))));
}
TYPED_TEST(NativeUpperGamma, ExtremeFiniteParametersHaveBoundedRangeBehavior)
{
  using R = TypeParam;
  using L = uni20::numeric_limits<R>;
  R huge = m::ldexp(R(1), L::digits());
  EXPECT_EQ(m::upper_gamma(huge, m::next_up(huge)), L::infinity());
  EXPECT_EQ(m::upper_gamma(L::max(), L::max()), L::infinity());
  // The logarithmic difference is finite although its unscaled product
  // overflows. The integral over [x,x+1] already exceeds the native range.
  R large_x = m::ldexp(R(7), L::max_exponent - 3);
  R large_a = (L::max() / m::log(large_x)) * R(2.5);
  EXPECT_EQ(m::upper_gamma(large_a, large_x), L::infinity());
  for (R a : {R(-100), R(0), R(2), R(10000)})
    EXPECT_EQ(m::upper_gamma(a, L::max()), R(0));
  EXPECT_GT(m::upper_gamma(R(-1), R(1)), R(0));
  EXPECT_GT(m::upper_gamma(R(-32), R(.5)), R(0));
}
TEST(NativeUpperGammaRegression, GuardedSmallEndpointRemainsFinite)
{
  // Exact represented arguments from the range-guard convergence regression.
  // MPFR256 rounds the finite reference to this float32 value.
  float result = m::upper_gamma(-3.5f, 0x1.5fd7fep-37f);
  ASSERT_TRUE(m::isfinite(result));
  EXPECT_EQ(result, 0x1.0fe3b8p+126f);
}
TEST(NativeUpperGammaRegression, FiniteLogEstimateWithOverflowingProduct)
{
  float result = m::upper_gamma(0x1.9d4d4ap+122f, 0x1.cf67fep+126f);
  EXPECT_EQ(result, uni20::numeric_limits<float>::infinity());
}
} // namespace
