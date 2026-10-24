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
constexpr bool signature = std::same_as<std::invoke_result_t<decltype(m::airy_ai), R>, R> &&
                           !std::invocable<decltype(m::airy_ai), R, uni20::Precision>;
static_assert(signature<float> && signature<double> && signature<long double>);
static_assert(!std::invocable<decltype(m::airy_ai), int>);
static_assert(!std::invocable<decltype(m::airy_ai), ConversionOnly>);
static_assert(!std::invocable<decltype(m::airy_ai), uni20::complex<double>>);
#if UNI20_HAS_FLOAT128
static_assert(signature<uni20::float128>);
#endif

template <class R> class NativeAiry : public testing::Test {};
using NativeReals = testing::Types<float, double, long double
#if UNI20_HAS_FLOAT128
                                   ,
                                   uni20::float128
#endif
                                   >;
TYPED_TEST_SUITE(NativeAiry, NativeReals);
TYPED_TEST(NativeAiry, DomainsAndSigns)
{
  using R = TypeParam;
  using L = uni20::numeric_limits<R>;
  EXPECT_TRUE(m::isnan(m::airy_ai(L::quiet_NaN())));
  for (R x : {L::infinity(), -L::infinity(), L::max()})
  {
    auto value = m::airy_ai(x);
    EXPECT_EQ(value, R(0));
    EXPECT_FALSE(m::signbit(value));
  }
  EXPECT_EQ(m::airy_ai(R(0)), m::airy_ai(-R(0)));
  EXPECT_GT(m::airy_ai(R(-1)), R(0));
  EXPECT_LT(m::airy_ai(R(-3)), R(0));
  EXPECT_GT(m::airy_ai(R(1)), R(0));
  EXPECT_TRUE(m::isfinite(m::airy_ai(-L::max())));
}
TYPED_TEST(NativeAiry, PhaseNeverFormsAnOverflowingPower)
{
  using R = TypeParam;
  using L = uni20::numeric_limits<R>;
  for (int exponent : {L::digits(), 2 * L::digits(), L::max_exponent - 1})
  {
    R x = -m::ldexp(R(.875), exponent);
    auto value = m::airy_ai(x);
    EXPECT_TRUE(m::isfinite(value));
    auto amplitude = 1 / (m::sqrt(uni20::pi<R>) * m::sqrt(m::sqrt(-x)));
    EXPECT_LE(m::abs(value), amplitude * (1 + 8 * L::epsilon()));
  }
}
} // namespace
