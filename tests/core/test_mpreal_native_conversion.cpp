// Exercise the include order where a consumer includes MPFR before Uni20.
#include <gtest/gtest.h>
#include <mpfr.h>
#include <uni20/core/mpreal.hpp>

namespace
{
using namespace uni20;

enum IntegerEnum
{
  one = 1
};
static_assert(!std::constructible_from<mpreal, bool, Precision>);
static_assert(!std::constructible_from<mpreal, IntegerEnum, Precision>);
static_assert(std::constructible_from<mpreal, int, Precision>);

#if !UNI20_HAS_FLOAT128 && defined(__STDCPP_FLOAT128_T__)
// A compiler-provided type must not bypass Uni20's configured support and
// silently enter the long-double constructor when its precision is wider.
static_assert(std::same_as<_Float128, long double> || !std::constructible_from<mpreal, _Float128, Precision>);
#endif

template <class R> struct NativeCase
{
    using real_type = R;
    static constexpr int digits = numeric_limits<R>::digits;
    static constexpr int min_exponent = numeric_limits<R>::min_exponent;
    static constexpr int max_exponent = numeric_limits<R>::max_exponent;
};
#if UNI20_HAS_FLOAT128
// These are the binary128 format parameters, independent of whether the
// standard library specializes numeric_limits for the provider's spelling.
struct Binary128Case
{
    using real_type = float128;
    static constexpr int digits = 113;
    static constexpr int min_exponent = -16381;
    static constexpr int max_exponent = 16384;
};
#endif
using NativeCases = testing::Types<NativeCase<float>, NativeCase<double>, NativeCase<long double>
#if UNI20_HAS_FLOAT128
                                   ,
                                   Binary128Case
#endif
                                   >;
template <class Case> class MpRealNativeConversion : public testing::Test {};
TYPED_TEST_SUITE(MpRealNativeConversion, NativeCases);

template <class R> R native_power(int exponent)
{
  R result(1);
  for (int i = 0; i < exponent; ++i)
    result *= R(2);
  for (int i = 0; i > exponent; --i)
    result /= R(2);
  return result;
}
mpreal exact_power(int exponent) { return pow(mpreal{2}, mpreal{exponent}); }

TYPED_TEST(MpRealNativeConversion, ImportRetainsEverySignificandBit)
{
  using R = typename TypeParam::real_type;
  static_assert(std::constructible_from<mpreal, R, Precision>);
  static_assert(!std::convertible_to<R, mpreal>);
  static_assert(!std::convertible_to<mpreal, R>);
  for (int bits : {128, 256})
  {
    auto p = Precision::bits(bits);
    for (int bit = 1; bit < TypeParam::digits; ++bit)
    {
      SCOPED_TRACE(bit);
      auto native = R(1) + native_power<R>(-bit);
      auto expected = mpreal{1} + exact_power(-bit);
      auto imported = mpreal(native, p);
      EXPECT_EQ(imported.precision(), p);
      EXPECT_EQ(imported, expected);
      EXPECT_TRUE(static_cast<R>(imported) == native);
      EXPECT_EQ(mpreal(-native, p), -expected);
    }
  }
  EXPECT_THROW((void)mpreal(R(1), Precision::exact()), std::logic_error);
  EXPECT_THROW((void)static_cast<R>(mpreal(uninitialized)), std::logic_error);
}

TYPED_TEST(MpRealNativeConversion, ImportRoundsOnceToSelectedPrecision)
{
  using R = typename TypeParam::real_type;
  constexpr int bits = TypeParam::digits - 2;
  auto p = Precision::bits(bits);
  R half = native_power<R>(-bits);
  R fine = native_power<R>(1 - TypeParam::digits);
  auto one = mpreal(1, p);
  auto next = (mpreal{1} + exact_power(1 - bits)).at(p);
  EXPECT_EQ(mpreal(R(1) + half, p), one);
  EXPECT_EQ(mpreal(R(1) + half - fine, p), one);
  EXPECT_EQ(mpreal(R(1) + half + fine, p), next);
  EXPECT_EQ(mpreal(R(1) + R(3) * half, p), (mpreal{1} + exact_power(2 - bits)).at(p));
}

TYPED_TEST(MpRealNativeConversion, ExactAndApproximateExportsAvoidDoubleRounding)
{
  using R = typename TypeParam::real_type;
  auto midpoint = mpreal{1} + exact_power(-TypeParam::digits);
  auto tiny = exact_power(-4 * TypeParam::digits) / 3;
  auto next = R(1) + native_power<R>(1 - TypeParam::digits);
  EXPECT_TRUE(static_cast<R>(mpreal("1/3", Precision::exact())) == R(1) / R(3));
  for (auto p : {Precision::exact(), Precision::bits(8 * TypeParam::digits)})
  {
    EXPECT_TRUE(static_cast<R>((midpoint - tiny).at(p)) == R(1));
    EXPECT_TRUE(static_cast<R>(midpoint.at(p)) == R(1));
    EXPECT_TRUE(static_cast<R>((midpoint + tiny).at(p)) == next);
    EXPECT_TRUE(static_cast<R>((-midpoint - tiny).at(p)) == -next);
    auto odd_midpoint = mpreal{1} + 3 * exact_power(-TypeParam::digits);
    EXPECT_TRUE(static_cast<R>(odd_midpoint.at(p)) == R(1) + native_power<R>(2 - TypeParam::digits));
  }
}

TYPED_TEST(MpRealNativeConversion, SubnormalNormalAndOverflowBoundaries)
{
  using R = typename TypeParam::real_type;
  constexpr int sub_exp = TypeParam::min_exponent - TypeParam::digits;
  auto denorm = native_power<R>(sub_exp);
  auto normal = native_power<R>(TypeParam::min_exponent - 1);
  auto max = (R(2) - native_power<R>(1 - TypeParam::digits)) * native_power<R>(TypeParam::max_exponent - 1);
  auto p = Precision::bits(256);
  EXPECT_EQ(mpreal(denorm, p), exact_power(sub_exp));
  EXPECT_EQ(mpreal(normal, p), exact_power(TypeParam::min_exponent - 1));
  auto exact_max = exact_power(TypeParam::max_exponent) - exact_power(TypeParam::max_exponent - TypeParam::digits);
  EXPECT_EQ(mpreal(max, p), exact_max);
  EXPECT_TRUE(static_cast<R>(exact_max) == max);
  auto half = exact_power(sub_exp - 1);
  auto tiny = exact_power(sub_exp - 4 * TypeParam::digits);
  for (auto precision : {Precision::exact(), Precision::bits(8 * TypeParam::digits)})
  {
    EXPECT_TRUE(static_cast<R>(half.at(precision)) == R(0));
    EXPECT_TRUE(__builtin_signbit(static_cast<R>((-half).at(precision))));
    EXPECT_TRUE(static_cast<R>((half + tiny).at(precision)) == denorm);
    EXPECT_TRUE(static_cast<R>((3 * half).at(precision)) == 2 * denorm);
    auto normal_midpoint = exact_power(TypeParam::min_exponent - 1) - half;
    EXPECT_TRUE(static_cast<R>(normal_midpoint.at(precision)) == normal);
    EXPECT_TRUE(static_cast<R>((normal_midpoint - tiny).at(precision)) == normal - denorm);
    auto overflow_midpoint = exact_max + exact_power(TypeParam::max_exponent - TypeParam::digits - 1);
    EXPECT_TRUE(__builtin_isinf(static_cast<R>(overflow_midpoint.at(precision))));
    auto below_overflow = overflow_midpoint - exact_power(TypeParam::max_exponent - 4 * TypeParam::digits);
    EXPECT_TRUE(static_cast<R>(below_overflow.at(precision)) == max);
  }
}

TYPED_TEST(MpRealNativeConversion, PreservesSignedZeroAndSpecialValueClasses)
{
  using R = typename TypeParam::real_type;
  auto p = Precision::bits(128);
  for (auto negative : {false, true})
  {
    R zero = negative ? -R(0) : R(0);
    auto imported = mpreal(zero, p);
    EXPECT_EQ(signbit(imported), negative);
    EXPECT_EQ(bool(__builtin_signbit(static_cast<R>(imported))), negative);
    R inf = R(std::numeric_limits<double>::infinity());
    if (negative) inf = -inf;
    auto infinity = mpreal(inf, p);
    EXPECT_TRUE(isinf(infinity));
    EXPECT_EQ(signbit(infinity), negative);
    EXPECT_TRUE(static_cast<R>(infinity) == inf);
  }
  auto nan = mpreal(R(std::numeric_limits<double>::quiet_NaN()), p);
  EXPECT_TRUE(isnan(nan));
  EXPECT_TRUE(__builtin_isnan(static_cast<R>(nan)));
}
} // namespace
