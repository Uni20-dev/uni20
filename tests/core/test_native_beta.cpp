#include <gtest/gtest.h>
#include <uni20/core/math.hpp>

namespace
{
namespace m = uni20::math;
namespace native = uni20::detail::native_math;
struct ConversionOnly
{
    operator double() const;
};
template <class R>
constexpr bool signature = std::same_as<std::invoke_result_t<decltype(m::beta), R, R>, R> &&
                           !std::invocable<decltype(m::beta), R, R, uni20::Precision>;
static_assert(signature<float> && signature<double> && signature<long double>);
static_assert(!std::invocable<decltype(m::beta), int, int>);
static_assert(!std::invocable<decltype(m::beta), float, double>);
static_assert(!std::invocable<decltype(m::beta), double, int>);
static_assert(!std::invocable<decltype(m::beta), ConversionOnly, ConversionOnly>);
static_assert(!std::invocable<decltype(m::beta), uni20::complex<double>, uni20::complex<double>>);
#if UNI20_HAS_FLOAT128
static_assert(signature<uni20::float128>);
#endif
template <class R> class NativeBeta : public testing::Test {};
using NativeReals = testing::Types<float, double, long double
#if UNI20_HAS_FLOAT128
                                   ,
                                   uni20::float128
#endif
                                   >;
TYPED_TEST_SUITE(NativeBeta, NativeReals);
TYPED_TEST(NativeBeta, PortableValuesAndSymmetry)
{
  using R = TypeParam;
  using L = uni20::numeric_limits<R>;
  for (R x : {R(0), R(-1), -L::infinity(), L::quiet_NaN()})
  {
    EXPECT_TRUE(m::isnan(native::portable_beta(x, R(2))));
    EXPECT_TRUE(m::isnan(native::portable_beta(R(2), x)));
  }
  EXPECT_EQ(native::portable_beta(L::infinity(), R(2)), R(0));
  EXPECT_FALSE(m::signbit(native::portable_beta(L::infinity(), L::infinity())));
  for (R a : {R(.125), R(.5), R(8), R(16), R(32), L::max()})
    for (R b : {R(.25), R(1), R(16), L::max()})
      EXPECT_EQ(native::portable_beta(a, b), native::portable_beta(b, a));
  EXPECT_EQ(native::portable_beta(L::max(), L::max()), R(0));
}
TYPED_TEST(NativeBeta, ReciprocalsAndExtremeRatios)
{
  using R = TypeParam;
  using L = uni20::numeric_limits<R>;
  for (R x : {L::denorm_min(), L::min(), R(.5), R(2), L::max()})
    EXPECT_EQ(native::portable_beta(x, R(1)), R(1) / x);
  R tiny = R(1) / L::max();
  for (int k = 0; k < 16; ++k)
  {
    // B(tiny,b) differs from 1/tiny by O(log(b)); the difference is far
    // below one native spacing at the overflow boundary.
    for (R b : {R(8), R(16), R(32), L::max()})
      EXPECT_EQ(native::portable_beta(tiny, b), R(1) / tiny);
    tiny = m::next_up(tiny);
  }
  R result = native::portable_beta(R(.5), L::max());
  EXPECT_GT(result, R(0));
  EXPECT_TRUE(m::isfinite(result));
}
TEST(NativeBetaRegression, CompensatedReciprocalSumAvoidsPrematureOverflow)
{
  // Both arguments are exactly representable subnormals. Separately rounded
  // reciprocals sum to infinity although Beta rounds to a finite float32.
  float a = 0x1.14326cp-127f, b = 0x1.dd1c78p-128f;
  EXPECT_TRUE(m::isinf(1.f / a + 1.f / b));
  EXPECT_TRUE(m::isfinite(native::portable_beta(a, b)));
}
TYPED_TEST(NativeBeta, PublicSelection)
{
  using R = TypeParam;
  R a = R(.5), b = R(.5);
  if constexpr (requires { native::standard_special_provider<R>::beta(a, b); })
    EXPECT_EQ(m::beta(a, b), native::standard_special_provider<R>::beta(a, b));
  else
    EXPECT_EQ(m::beta(a, b), native::portable_beta(a, b));
  EXPECT_LE(m::abs(native::portable_beta(a, b) - uni20::pi<R>),
            R(32) * uni20::numeric_limits<R>::epsilon() * uni20::pi<R>);
}
} // namespace
