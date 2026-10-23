#include <gtest/gtest.h>
#include <uni20/core/math.hpp>

namespace
{
namespace m = uni20::math;
struct ConversionOnly
{
    operator double() const;
};
template <class R, class... F> consteval bool signatures(F...)
{
  return ((std::same_as<std::invoke_result_t<F, R>, R> && !std::invocable<F, int> &&
           !std::invocable<F, ConversionOnly> && !std::invocable<F, R, uni20::Precision>) &&
          ...);
}
template <class R>
constexpr bool reciprocal_signatures = signatures<R>(m::sec, m::csc, m::cot, m::sech, m::csch, m::coth);
static_assert(reciprocal_signatures<float> && reciprocal_signatures<double> && reciprocal_signatures<long double>);
#if UNI20_HAS_FLOAT128
static_assert(reciprocal_signatures<uni20::float128>);
#endif

template <class R> class NativeReciprocalMath : public testing::Test {};
using NativeReals = testing::Types<float, double, long double
#if UNI20_HAS_FLOAT128
                                   ,
                                   uni20::float128
#endif
                                   >;
TYPED_TEST_SUITE(NativeReciprocalMath, NativeReals);
TYPED_TEST(NativeReciprocalMath, ExceptionalValuesAndSignedZeros)
{
  using R = TypeParam;
  using L = uni20::numeric_limits<R>;
  auto inf = L::infinity(), nan = L::quiet_NaN();
  for (R x : {R(0), -R(0)})
  {
    EXPECT_EQ(m::sec(x), R(1));
    EXPECT_EQ(m::sech(x), R(1));
    for (R result : {m::csc(x), m::cot(x), m::csch(x), m::coth(x)})
    {
      EXPECT_TRUE(m::isinf(result));
      EXPECT_EQ(m::signbit(result), m::signbit(x));
    }
  }
  for (R x : {inf, -inf})
  {
    EXPECT_TRUE(m::isnan(m::sec(x)));
    EXPECT_TRUE(m::isnan(m::csc(x)));
    EXPECT_TRUE(m::isnan(m::cot(x)));
    EXPECT_EQ(m::sech(x), R(0));
    EXPECT_FALSE(m::signbit(m::sech(x)));
    EXPECT_EQ(m::csch(x), R(0));
    EXPECT_EQ(m::signbit(m::csch(x)), m::signbit(x));
    EXPECT_EQ(m::coth(x), m::copysign(R(1), x));
  }
  for (R result : {m::sec(nan), m::csc(nan), m::cot(nan), m::sech(nan), m::csch(nan), m::coth(nan)})
    EXPECT_TRUE(m::isnan(result));
}
TYPED_TEST(NativeReciprocalMath, RetainsTailsBeyondHyperbolicOverflow)
{
  using R = TypeParam;
  using L = uni20::numeric_limits<R>;
  R x = R(L::max_exponent) * uni20::log_two<R> + R(2);
  EXPECT_TRUE(m::isinf(m::cosh(x)));
  EXPECT_TRUE(m::isinf(m::sinh(x)));
  for (R result : {m::sech(x), m::csch(x), -m::csch(-x)})
  {
    EXPECT_GT(result, R(0));
    EXPECT_LT(result, L::min());
  }
  EXPECT_EQ(m::sech(-x), m::sech(x));
  EXPECT_EQ(m::coth(x), R(1));
  EXPECT_EQ(m::coth(-x), R(-1));
}
} // namespace
