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
constexpr bool compound_signature =
    std::same_as<std::invoke_result_t<decltype(m::compound), R, int>, R> &&
    std::same_as<std::invoke_result_t<decltype(m::compound), R, std::int64_t>, R> &&
    !std::invocable<decltype(m::compound), R, bool> && !std::invocable<decltype(m::compound), R, R> &&
    !std::invocable<decltype(m::compound), R, int, uni20::Precision>;
static_assert(compound_signature<float> && compound_signature<double> && compound_signature<long double>);
static_assert(!std::invocable<decltype(m::compound), int, int>);
static_assert(!std::invocable<decltype(m::compound), ConversionOnly, int>);
#if UNI20_HAS_FLOAT128
static_assert(compound_signature<uni20::float128>);
#endif

template <class R> class NativeCompound : public testing::Test {};
using NativeReals = testing::Types<float, double, long double
#if UNI20_HAS_FLOAT128
                                   ,
                                   uni20::float128
#endif
                                   >;
TYPED_TEST_SUITE(NativeCompound, NativeReals);

template <class R, class F> void boundaries(F compound)
{
  using L = uni20::numeric_limits<R>;
  R inf = L::infinity(), nan = L::quiet_NaN();
  for (R x : {R(-2), m::next_down(R(-1)), -inf})
    for (int n : {-2, 0, 2})
      EXPECT_TRUE(m::isnan(compound(x, n)));
  for (R x : {R(-1), -R(0), R(0), R(2), inf, nan})
    EXPECT_EQ(compound(x, 0), R(1));
  for (int n : {-3, -2, -1, 1, 2, 3})
  {
    EXPECT_TRUE(m::isnan(compound(nan, n)));
    EXPECT_EQ(compound(R(-1), n), n < 0 ? inf : R(0));
    EXPECT_FALSE(m::signbit(compound(R(-1), n)));
    EXPECT_EQ(compound(inf, n), n > 0 ? inf : R(0));
    EXPECT_FALSE(m::signbit(compound(inf, n)));
    EXPECT_EQ(compound(R(0), n), R(1));
    EXPECT_EQ(compound(-R(0), n), R(1));
  }
  for (auto n : {uni20::numeric_limits<std::int64_t>::min(), uni20::numeric_limits<std::int64_t>::max()})
  {
    EXPECT_EQ(compound(R(0), n), R(1));
    EXPECT_EQ(compound(R(1), n), n > 0 ? inf : R(0));
    EXPECT_EQ(compound(R(-0.5), n), n < 0 ? inf : R(0));
  }
  // The exponent stays integer, and the original tiny increment is retained.
  R tiny = m::ldexp(R(1), -60);
  EXPECT_GT(compound(tiny, std::int64_t(1) << 60), R(2.7));
  EXPECT_LT(compound(tiny, std::int64_t(1) << 60), R(2.8));
  EXPECT_GT(compound(tiny, -(std::int64_t(1) << 60)), R(0.36));
  EXPECT_LT(compound(tiny, -(std::int64_t(1) << 60)), R(0.37));
  EXPECT_GT(compound(-tiny, uni20::numeric_limits<std::int64_t>::min()), R(2000));

  for (int e : {L::min_exponent - L::digits(), L::min_exponent, -8, 0, 8, L::max_exponent - 1})
  {
    EXPECT_EQ(compound(R(1), e), m::ldexp(R(1), e));
    EXPECT_EQ(compound(R(-0.5), -e), m::ldexp(R(1), e));
  }
  // Inverting an already rounded positive power would lose this subnormal.
  EXPECT_EQ(compound(R(1), -L::max_exponent), m::ldexp(R(1), -L::max_exponent));
  EXPECT_EQ(compound(R(-0.5), L::max_exponent), m::ldexp(R(1), -L::max_exponent));
  EXPECT_EQ(compound(m::next_up(R(-1)), -2), m::ldexp(R(1), 2 * L::digits()));
}
TYPED_TEST(NativeCompound, ValueContractsAndRetainedIncrement)
{
  boundaries<TypeParam>(m::compound);
  boundaries<TypeParam>(uni20::detail::native_math::portable_compound<TypeParam>);
}
TYPED_TEST(NativeCompound, CheckedIntegerExponent)
{
  auto out_of_range = std::uint64_t(1) << 63;
  EXPECT_THROW(m::compound(TypeParam(0), out_of_range), std::out_of_range);
  EXPECT_EQ(m::compound(TypeParam(0), out_of_range - 1), TypeParam(1));
}
} // namespace
