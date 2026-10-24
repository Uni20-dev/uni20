#include <gtest/gtest.h>
#include <uni20/core/math.hpp>

namespace
{
namespace m = uni20::math;

struct ConversionOnly
{
    operator double() const;
};
template <class R, class... F> consteval bool unary_signatures(F...)
{
  return ((std::same_as<std::invoke_result_t<F, R>, R> && !std::invocable<F, int> &&
           !std::invocable<F, ConversionOnly> && !std::invocable<F, R, uni20::Precision>) &&
          ...);
}
template <class R>
constexpr bool signatures = unary_signatures<R>(m::sinpi, m::cospi, m::tanpi, m::asinpi, m::acospi, m::atanpi) &&
                            std::same_as<std::invoke_result_t<decltype(m::atan2pi), R, R>, R>;
static_assert(signatures<float> && signatures<double> && signatures<long double>);
#if UNI20_HAS_FLOAT128
static_assert(signatures<uni20::float128>);
#endif
static_assert(std::same_as<std::invoke_result_t<decltype(m::atan2pi), float, double>, double>);
static_assert(std::same_as<std::invoke_result_t<decltype(m::atan2pi), double, long double>, long double>);
static_assert(!std::invocable<decltype(m::atan2pi), int, double>);
static_assert(!std::invocable<decltype(m::atan2pi), ConversionOnly, ConversionOnly>);

// Exercise the public dispatch and the portable path independently, including
// on platforms whose libc supplies every pi-scaled function.
struct Facade
{
    static constexpr auto sinpi = m::sinpi;
    static constexpr auto cospi = m::cospi;
    static constexpr auto tanpi = m::tanpi;
    static constexpr auto asinpi = m::asinpi;
    static constexpr auto acospi = m::acospi;
    static constexpr auto atanpi = m::atanpi;
    static constexpr auto atan2pi = m::atan2pi;
};
template <class R> class NativePiMath : public testing::Test {};
using NativeReals = testing::Types<float, double, long double
#if UNI20_HAS_FLOAT128
                                   ,
                                   uni20::float128
#endif
                                   >;
TYPED_TEST_SUITE(NativePiMath, NativeReals);

TEST(NativePiMath, MixedNativeArgumentsRetainTheWiderOperand)
{
  double x = 1 + std::ldexp(1.0, -40);
  auto result = m::atan2pi(1.0f, x);
  EXPECT_EQ(result, m::atan2pi(1.0, x));
  EXPECT_LT(result, 0.25); // Rounding x to float would give exactly 1/4.
}

template <class R, class F> void exact_grid()
{
  for (int n = -8; n <= 8; ++n)
  {
    R integer(n), half = integer + R(0.5);
    EXPECT_EQ(F::sinpi(integer), R(0));
    EXPECT_EQ(F::tanpi(integer), R(0));
    EXPECT_EQ(m::signbit(F::sinpi(integer)), n < 0);
    EXPECT_EQ(m::signbit(F::tanpi(integer)), n < 0);
    EXPECT_EQ(F::cospi(integer), n % 2 == 0 ? R(1) : R(-1));
    EXPECT_EQ(F::sinpi(half), n % 2 == 0 ? R(1) : R(-1));
    EXPECT_EQ(F::cospi(half), R(0));
    EXPECT_FALSE(m::signbit(F::cospi(half)));
    EXPECT_TRUE(m::isinf(F::tanpi(half)));
    EXPECT_EQ(m::signbit(F::tanpi(half)), n % 2 != 0);
    EXPECT_EQ(F::tanpi(integer + R(0.25)), R(1));
    EXPECT_EQ(F::tanpi(integer - R(0.25)), R(-1));
  }
  R huge = m::ldexp(R(1), uni20::numeric_limits<R>::digits() - 3);
  for (R x : {huge, -huge})
  {
    EXPECT_EQ(F::sinpi(x + R(0.5)), R(1));
    EXPECT_EQ(F::cospi(x + R(1)), R(-1));
    EXPECT_EQ(F::tanpi(x + R(0.25)), R(1));
  }
  for (R x : {uni20::numeric_limits<R>::max(), -uni20::numeric_limits<R>::max()})
  {
    EXPECT_EQ(F::sinpi(x), R(0));
    EXPECT_EQ(F::tanpi(x), R(0));
    EXPECT_EQ(m::signbit(F::sinpi(x)), m::signbit(x));
    EXPECT_EQ(F::cospi(x), R(1));
  }
}
TYPED_TEST(NativePiMath, ExactGridAndLargeArguments)
{
  exact_grid<TypeParam, Facade>();
  exact_grid<TypeParam, uni20::detail::native_math::portable_pi_math<TypeParam>>();
}

template <class R, class F> void boundaries()
{
  using limits = uni20::numeric_limits<R>;
  R zero(0), negzero = -zero, inf = limits::infinity(), nan = limits::quiet_NaN();
  for (R x : {zero, negzero})
  {
    for (auto value : {F::sinpi(x), F::tanpi(x), F::asinpi(x), F::atanpi(x)})
    {
      EXPECT_EQ(value, R(0));
      EXPECT_EQ(m::signbit(value), m::signbit(x));
    }
    EXPECT_EQ(F::cospi(x), R(1));
    EXPECT_EQ(F::acospi(x), R(0.5));
    for (R y : {R(1), R(-1), inf, -inf})
      EXPECT_EQ(F::atan2pi(y, x), m::copysign(R(0.5), y));
    for (R y : {zero, negzero})
    {
      R result = F::atan2pi(y, x);
      EXPECT_EQ(result, m::signbit(x) ? m::copysign(R(1), y) : y);
      EXPECT_EQ(m::signbit(result), m::signbit(y));
    }
  }
  for (R x : {inf, -inf, nan})
  {
    EXPECT_TRUE(m::isnan(F::sinpi(x)));
    EXPECT_TRUE(m::isnan(F::cospi(x)));
    EXPECT_TRUE(m::isnan(F::tanpi(x)));
    EXPECT_TRUE(m::isnan(F::asinpi(x)));
    EXPECT_TRUE(m::isnan(F::acospi(x)));
  }
  EXPECT_TRUE(m::isnan(F::atanpi(nan)));
  EXPECT_TRUE(m::isnan(F::atan2pi(nan, zero)));
  EXPECT_TRUE(m::isnan(F::atan2pi(zero, nan)));
  EXPECT_TRUE(m::isnan(F::asinpi(m::next_up(R(1)))));
  EXPECT_TRUE(m::isnan(F::acospi(m::next_down(R(-1)))));
  for (R sign : {R(1), R(-1)})
  {
    EXPECT_EQ(F::asinpi(sign), sign / 2);
    EXPECT_EQ(F::acospi(sign), sign == R(1) ? R(0) : R(1));
    EXPECT_EQ(F::atanpi(sign), sign / 4);
    EXPECT_EQ(F::atanpi(sign * inf), sign / 2);
    EXPECT_EQ(F::atan2pi(sign * inf, inf), sign / 4);
    EXPECT_EQ(F::atan2pi(sign * inf, -inf), sign * R(0.75));
    EXPECT_EQ(F::atan2pi(sign, inf), m::copysign(R(0), sign));
    EXPECT_EQ(m::signbit(F::atan2pi(sign, inf)), sign < R(0));
    EXPECT_EQ(F::atan2pi(sign, -inf), sign);
    EXPECT_EQ(F::atan2pi(sign * inf, R(1)), sign / 2);
  }
  R tiny = limits::denorm_min();
  EXPECT_EQ(F::sinpi(tiny), R(3) * tiny);
  EXPECT_EQ(F::tanpi(tiny), R(3) * tiny);
  EXPECT_EQ(F::asinpi(R(2) * tiny), tiny);
  EXPECT_EQ(F::atanpi(R(2) * tiny), tiny);
  EXPECT_EQ(F::atan2pi(R(2) * tiny, R(1)), tiny);
  // A direct y/x would overflow here; the angle is still well defined.
  EXPECT_EQ(F::atan2pi(limits::max(), tiny), R(0.5));
}
TYPED_TEST(NativePiMath, SignedZerosPolesDomainsAndSubnormals)
{
  boundaries<TypeParam, Facade>();
  boundaries<TypeParam, uni20::detail::native_math::portable_pi_math<TypeParam>>();
}
} // namespace
