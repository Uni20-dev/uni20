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
constexpr bool native_signatures = signatures<R>(m::exp10, m::exp2m1, m::exp10m1, m::log2p1, m::log10p1);
static_assert(native_signatures<float> && native_signatures<double> && native_signatures<long double>);
#if UNI20_HAS_FLOAT128
static_assert(native_signatures<uni20::float128>);
#endif

struct Facade
{
    static constexpr auto exp10 = m::exp10;
    static constexpr auto exp2m1 = m::exp2m1;
    static constexpr auto exp10m1 = m::exp10m1;
    static constexpr auto log2p1 = m::log2p1;
    static constexpr auto log10p1 = m::log10p1;
};
template <class R> class NativeExponentialMath : public testing::Test {};
using NativeReals = testing::Types<float, double, long double
#if UNI20_HAS_FLOAT128
                                   ,
                                   uni20::float128
#endif
                                   >;
TYPED_TEST_SUITE(NativeExponentialMath, NativeReals);

template <class R, class F> void boundaries()
{
  using limits = uni20::numeric_limits<R>;
  R inf = limits::infinity(), nan = limits::quiet_NaN();
  for (R x : {R(0), -R(0)})
  {
    EXPECT_EQ(F::exp10(x), R(1));
    for (R y : {F::exp2m1(x), F::exp10m1(x), F::log2p1(x), F::log10p1(x)})
    {
      EXPECT_EQ(y, R(0));
      EXPECT_EQ(m::signbit(y), m::signbit(x));
    }
  }
  for (R y : {F::exp10(nan), F::exp2m1(nan), F::exp10m1(nan), F::log2p1(nan), F::log10p1(nan)})
    EXPECT_TRUE(m::isnan(y));
  for (R y : {F::exp10(inf), F::exp2m1(inf), F::exp10m1(inf), F::log2p1(inf), F::log10p1(inf)})
    EXPECT_EQ(y, inf);
  EXPECT_EQ(F::exp10(-inf), R(0));
  EXPECT_FALSE(m::signbit(F::exp10(-inf)));
  EXPECT_EQ(F::exp2m1(-inf), R(-1));
  EXPECT_EQ(F::exp10m1(-inf), R(-1));
  EXPECT_EQ(F::log2p1(R(-1)), -inf);
  EXPECT_EQ(F::log10p1(R(-1)), -inf);
  for (R x : {m::next_down(R(-1)), R(-2), -inf})
  {
    EXPECT_TRUE(m::isnan(F::log2p1(x)));
    EXPECT_TRUE(m::isnan(F::log10p1(x)));
  }
  EXPECT_TRUE(m::isfinite(F::log2p1(m::next_up(R(-1)))));
  EXPECT_TRUE(m::isfinite(F::log10p1(m::next_up(R(-1)))));
  EXPECT_EQ(F::log2p1(R(1)), R(1));
  EXPECT_EQ(F::log10p1(R(9)), R(1));
  for (int n = -8; n <= 8; ++n)
    EXPECT_EQ(F::exp2m1(R(n)), m::ldexp(R(1), n) - R(1));

  R tiny = m::ldexp(R(1), -limits::digits() - 8);
  EXPECT_EQ(R(1) + tiny, R(1)); // A naive log(1+x) would lose the input.
  for (R x : {tiny, -tiny})
    for (R y : {F::exp2m1(x), F::exp10m1(x), F::log2p1(x), F::log10p1(x)})
    {
      EXPECT_NE(y, R(0));
      EXPECT_EQ(m::signbit(y), m::signbit(x));
    }
  R subnormal = limits::denorm_min();
  EXPECT_EQ(F::exp2m1(subnormal), subnormal);
  EXPECT_EQ(F::exp10m1(subnormal), R(2) * subnormal);
  EXPECT_EQ(F::log2p1(subnormal), subnormal);
  EXPECT_EQ(F::log10p1(subnormal), R(0));
  EXPECT_EQ(F::log10p1(-subnormal), R(0));
  EXPECT_TRUE(m::signbit(F::log10p1(-subnormal)));
}
TYPED_TEST(NativeExponentialMath, ValueContractsAndCancellation)
{
  boundaries<TypeParam, Facade>();
  boundaries<TypeParam, uni20::detail::native_math::portable_exponential_math<TypeParam>>();
}
} // namespace
