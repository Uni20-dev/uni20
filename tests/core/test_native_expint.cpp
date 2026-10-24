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
constexpr bool signature = std::same_as<std::invoke_result_t<decltype(m::expint), R>, R> &&
                           !std::invocable<decltype(m::expint), R, uni20::Precision>;
static_assert(signature<float> && signature<double> && signature<long double>);
static_assert(!std::invocable<decltype(m::expint), int>);
static_assert(!std::invocable<decltype(m::expint), ConversionOnly>);
static_assert(!std::invocable<decltype(m::expint), uni20::complex<double>>);
#if UNI20_HAS_FLOAT128
static_assert(signature<uni20::float128>);
#endif

template <class R> class NativeExpint : public testing::Test {};
using NativeReals = testing::Types<float, double, long double
#if UNI20_HAS_FLOAT128
                                   ,
                                   uni20::float128
#endif
                                   >;
TYPED_TEST_SUITE(NativeExpint, NativeReals);
TYPED_TEST(NativeExpint, ExceptionalValues)
{
  using R = TypeParam;
  using L = uni20::numeric_limits<R>;
  EXPECT_EQ(m::expint(R(0)), -L::infinity());
  EXPECT_EQ(m::expint(-R(0)), -L::infinity());
  EXPECT_EQ(m::expint(L::infinity()), L::infinity());
  EXPECT_EQ(m::expint(-L::infinity()), R(0));
  EXPECT_TRUE(m::signbit(m::expint(-L::infinity())));
  EXPECT_TRUE(m::isnan(m::expint(L::quiet_NaN())));
  EXPECT_EQ(m::expint(L::max()), L::infinity());
  EXPECT_EQ(m::expint(-L::max()), R(0));
  EXPECT_TRUE(m::signbit(m::expint(-L::max())));
}
TYPED_TEST(NativeExpint, ScaledRangeAndDispatch)
{
  using R = TypeParam;
  using L = uni20::numeric_limits<R>;
  R logarithm = m::log(L::max()), x = logarithm + m::log(logarithm) / 2;
  EXPECT_TRUE(m::isinf(m::exp(x)));
  EXPECT_TRUE(m::isfinite(m::expint(x)));
  EXPECT_GT(m::expint(x), R(0));
  R tail = m::expint(m::log(L::min()));
  EXPECT_LT(tail, R(0));
  EXPECT_LT(m::abs(tail), L::min());
  EXPECT_LT(m::expint(R(.25)), R(0));
  EXPECT_GT(m::expint(R(.5)), R(0));
}
} // namespace
