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
constexpr bool signature = std::same_as<std::invoke_result_t<decltype(m::dilog_real), R>, R> &&
                           !std::invocable<decltype(m::dilog_real), R, uni20::Precision>;
static_assert(signature<float> && signature<double> && signature<long double>);
static_assert(!std::invocable<decltype(m::dilog_real), int>);
static_assert(!std::invocable<decltype(m::dilog_real), ConversionOnly>);
static_assert(!std::invocable<decltype(m::dilog_real), uni20::complex<double>>);
#if UNI20_HAS_FLOAT128
static_assert(signature<uni20::float128>);
#endif

template <class R> class NativeDilog : public testing::Test {};
using NativeReals = testing::Types<float, double, long double
#if UNI20_HAS_FLOAT128
                                   ,
                                   uni20::float128
#endif
                                   >;
TYPED_TEST_SUITE(NativeDilog, NativeReals);
TYPED_TEST(NativeDilog, ExceptionalValuesAndTinyInputs)
{
  using R = TypeParam;
  using L = uni20::numeric_limits<R>;
  for (R sign : {R(1), R(-1)})
  {
    R zero = sign * R(0);
    EXPECT_EQ(m::dilog_real(zero), R(0));
    EXPECT_EQ(m::signbit(m::dilog_real(zero)), m::signbit(zero));
    EXPECT_EQ(m::dilog_real(sign * L::infinity()), -L::infinity());
    for (R tiny : {L::denorm_min(), L::min(), L::epsilon() / 8})
      EXPECT_EQ(m::dilog_real(sign * tiny), sign * tiny);
    EXPECT_TRUE(m::isfinite(m::dilog_real(sign * L::max())));
    EXPECT_LT(m::dilog_real(sign * L::max()), R(0));
  }
  EXPECT_TRUE(m::isnan(m::dilog_real(L::quiet_NaN())));
}
TYPED_TEST(NativeDilog, SpecialValuesAndRealContinuation)
{
  using R = TypeParam;
  using L = uni20::numeric_limits<R>;
  R sixth = uni20::pi<R> * uni20::pi<R> / 6, ln2 = uni20::log_two<R>;
  auto check = [&](R x, R reference) {
    EXPECT_LE(m::abs(m::dilog_real(x) - reference), R(16) * L::epsilon() * m::abs(reference));
  };
  check(R(-1), -sixth / 2);
  check(R(.5), sixth / 2 - ln2 * ln2 / 2);
  check(R(1), sixth);
  check(R(2), R(3) * sixth / 2);
  EXPECT_GT(m::dilog_real(R(12)), R(0));
  EXPECT_LT(m::dilog_real(R(13)), R(0));
  EXPECT_GT(m::dilog_real(m::next_down(R(1))), R(0));
  EXPECT_GT(m::dilog_real(m::next_up(R(1))), R(0));
}
} // namespace
