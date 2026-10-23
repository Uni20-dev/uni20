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
constexpr bool agm_signature =
    std::same_as<std::invoke_result_t<decltype(m::agm), R, R>, R> && !std::invocable<decltype(m::agm), R, int> &&
    !std::invocable<decltype(m::agm), R, uni20::Precision> && !std::invocable<decltype(m::agm), R, R, uni20::Precision>;
static_assert(agm_signature<float> && agm_signature<double> && agm_signature<long double>);
static_assert(!std::invocable<decltype(m::agm), float, double>);
static_assert(!std::invocable<decltype(m::agm), ConversionOnly, ConversionOnly>);
static_assert(!std::invocable<decltype(m::agm), uni20::complex<double>, uni20::complex<double>>);
#if UNI20_HAS_FLOAT128
static_assert(agm_signature<uni20::float128>);
#endif

template <class R> class NativeAgm : public testing::Test {};
using NativeReals = testing::Types<float, double, long double
#if UNI20_HAS_FLOAT128
                                   ,
                                   uni20::float128
#endif
                                   >;
TYPED_TEST_SUITE(NativeAgm, NativeReals);
TYPED_TEST(NativeAgm, ExceptionalValuesAndZeroPrecedence)
{
  using R = TypeParam;
  using L = uni20::numeric_limits<R>;
  R inf = L::infinity(), nan = L::quiet_NaN();
  for (R zero : {R(0), -R(0)})
  {
    for (R other : {R(0), -R(0), R(2), R(-2), L::max(), -L::max()})
    {
      EXPECT_EQ(m::agm(zero, other), R(0));
      EXPECT_FALSE(m::signbit(m::agm(zero, other)));
      EXPECT_EQ(m::agm(other, zero), R(0));
      EXPECT_FALSE(m::signbit(m::agm(other, zero)));
    }
    for (R other : {inf, -inf, nan})
    {
      EXPECT_TRUE(m::isnan(m::agm(zero, other)));
      EXPECT_TRUE(m::isnan(m::agm(other, zero)));
    }
  }
  for (R x : {R(-1), -inf, nan})
  {
    for (R y : {R(1), R(-1), inf, -inf, nan})
    {
      EXPECT_TRUE(m::isnan(m::agm(x, y)));
      EXPECT_TRUE(m::isnan(m::agm(y, x)));
    }
  }
  EXPECT_EQ(m::agm(inf, R(1)), inf);
  EXPECT_EQ(m::agm(R(1), inf), inf);
  EXPECT_EQ(m::agm(inf, inf), inf);
}
TYPED_TEST(NativeAgm, ExtremeRangeSymmetryAndEquality)
{
  using R = TypeParam;
  using L = uni20::numeric_limits<R>;
  R tiny = L::denorm_min(), big = L::max();
  for (R x : {tiny, L::min(), R(1), big})
  {
    EXPECT_EQ(m::agm(x, x), x);
    for (R y : {tiny, L::min(), R(1), big})
    {
      R result = m::agm(x, y);
      EXPECT_TRUE(m::isfinite(result));
      EXPECT_GT(result, R(0));
      EXPECT_GE(result, std::min(x, y));
      EXPECT_LE(result, std::max(x, y));
      EXPECT_EQ(result, m::agm(y, x));
    }
  }
  // Normalizing both original operands by the larger exponent erases tiny.
  int exponent = static_cast<int>(m::frexp(big).exponent);
  EXPECT_EQ(m::ldexp(tiny, -exponent), R(0));
  EXPECT_GT(m::agm(tiny, big), big / R(100000));
}
} // namespace
