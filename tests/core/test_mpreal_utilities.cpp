#include <cstdint>
#include <gtest/gtest.h>
#include <uni20/core/math.hpp>

namespace
{
using namespace uni20;
namespace m = uni20::math;
mpreal q(char const* text) { return mpreal(text, Precision::exact()); }

template <class... F> consteval bool unary_signatures(F...)
{
  return (
      (std::invocable<F, mpreal> && std::invocable<F, mpreal, Precision> && !std::invocable<F, double, Precision>) &&
      ...);
}
static_assert(unary_signatures(m::floor, m::ceil, m::trunc, m::round, m::round_even, m::frexp, m::modf, m::next_up,
                               m::next_down, m::sincos, m::sinhcosh));
template <class... F> consteval bool integer_signatures(F...)
{
  return ((std::invocable<F, mpreal, int> && std::invocable<F, mpreal, std::int64_t, Precision> &&
           !std::invocable<F, mpreal, double> && !std::invocable<F, mpreal, bool> &&
           !std::invocable<F, double, int, Precision>) &&
          ...);
}
static_assert(integer_signatures(m::ldexp, m::scalbn, m::pown, m::rootn));

TEST(MprealUtilities, ExactRoundingAndResultPrecision)
{
  auto p = Precision::bits(2);
  EXPECT_EQ(m::floor(q("-1/3")), -1);
  EXPECT_EQ(m::ceil(q("-1/3")), 0);
  EXPECT_EQ(m::trunc(q("-4/3")), -1);
  EXPECT_EQ(m::round(q("-5/2")), -3);
  EXPECT_EQ(m::round_even(q("-5/2")), -2);
  EXPECT_EQ(m::round_even(q("-7/2")), -4);
  EXPECT_EQ(m::round_even(q("5/2")), 2);
  EXPECT_EQ(m::round_even(q("7/2")), 4);
  EXPECT_TRUE(m::floor(q("1/3")).is_exact());
  EXPECT_EQ(m::floor(q("1023/1024"), p), 0);
  EXPECT_EQ(m::ceil(q("9217/1024"), p), 8); // ceil is 10; ties-to-even at two bits gives 8.
  auto wide = q("1023/1024").at(Precision::bits(128));
  EXPECT_EQ(m::floor(wide, p), 0);
  EXPECT_EQ(m::floor(wide.at(p)), 1); // Regression control: premature input rounding changes the result.
  EXPECT_EQ(m::floor(wide, p).precision(), p);
  EXPECT_THROW(m::floor(q("1/3"), Precision::exact()), std::logic_error);
  EXPECT_THROW(m::round_even(mpreal(uninitialized)), std::logic_error);
  auto negative_zero = mpreal("-0", p);
  EXPECT_TRUE(m::signbit(m::round_even(negative_zero)));
  EXPECT_TRUE(m::signbit(m::trunc(mpreal("-0.25", p))));
}

TEST(MprealUtilities, FusedMixedRationalsAndExtremeExponents)
{
  auto p = Precision::bits(3);
  auto result = m::fma(q("1/3"), mpreal(3, p), mpreal(-1, p));
  EXPECT_EQ(result, 0);
  EXPECT_EQ(result.precision(), p);
  EXPECT_NE(m::fma(q("1/3").at(p), mpreal(3, p), mpreal(-1, p)), 0);
  EXPECT_EQ(m::fma(q("1/3"), mpreal{3}, mpreal{-1}), 0);
  EXPECT_TRUE(m::fma(q("1/3"), mpreal{3}, mpreal{-1}).is_exact());
  EXPECT_EQ(m::fma(q("9/8"), mpreal{1}, mpreal{-1}, p), q("1/8"));
  auto wide = Precision::bits(128);
  auto huge = m::ldexp(mpreal(1, wide), 1'000'000'000);
  EXPECT_EQ(m::fma(q("1/3"), huge * 3, -huge), 0);
  auto tiny = m::ldexp(mpreal(1, wide), -1'000'000'000);
  EXPECT_EQ(m::fma(q("1/3"), tiny * 3, -tiny), 0);
  EXPECT_THROW(m::fma(mpreal(1, p), mpreal(1, wide), mpreal{}), std::invalid_argument);
  EXPECT_EQ(m::fma(mpreal(1, p), mpreal(1, wide), mpreal{}, p), 1);
  auto nz = mpreal("-0", p);
  EXPECT_TRUE(m::signbit(m::fma(mpreal{1}, nz, nz)));
  EXPECT_FALSE(m::signbit(m::fma(mpreal{1}, nz, mpreal{})));
  EXPECT_TRUE(m::isnan(m::fma(mpreal{}, mpreal("inf", p), mpreal{1})));
  EXPECT_EQ(m::fma(q("1e10000"), mpreal{1}, mpreal("-inf", p)), mpreal("-inf", p));
}

TEST(MprealUtilities, ExactRemaindersAndLowQuotientBits)
{
  auto p = Precision::bits(128);
  EXPECT_EQ(m::fmod(q("7/3"), q("2/3")), q("1/3"));
  EXPECT_EQ(m::remainder(q("7/3"), q("2/3")), q("-1/3"));
  auto parts = m::remquo(q("-7/3"), q("2/3"));
  EXPECT_EQ(parts.remainder, q("1/3"));
  EXPECT_EQ(parts.quotient, -4);
  EXPECT_TRUE(parts.remainder.is_exact());
  EXPECT_EQ(m::fmod(mpreal(1, p), q("1/3")), 0);
  auto huge = m::ldexp(mpreal(1, p), 1'000'000'000);
  auto tiny = m::ldexp(mpreal(1, p), -1'000'000'000);
  EXPECT_EQ(m::fmod(huge, q("3/7")), q("1/7").at(p));
  EXPECT_EQ(m::remquo(huge, q("3/7")).quotient, 5);
  EXPECT_EQ(m::fmod(-huge, q("3/7")), -q("1/7").at(p));
  EXPECT_EQ(m::fmod(tiny, q("1/3")), tiny);
  EXPECT_EQ(m::fmod(q("1/3"), tiny), m::ldexp(q("1/3").at(p), -1'000'000'000));
  EXPECT_EQ(m::remquo(q("1/3"), tiny).quotient, 5);
  EXPECT_EQ(m::fmod(q("1/3"), huge), q("1/3").at(p));
  EXPECT_THROW(m::fmod(mpreal{1}, mpreal{}), std::domain_error);
  EXPECT_TRUE(m::isnan(m::fmod(mpreal(1, p), mpreal{})));
  EXPECT_TRUE(m::signbit(m::fmod(mpreal(-1, p), q("1/3"))));
  EXPECT_EQ(m::remainder(q("1/3"), mpreal("inf", p)), q("1/3").at(p));
}

TEST(MprealUtilities, MixedRemaindersAgreeWithExactQuotientRules)
{
  auto p = Precision::bits(80);
  for (int numerator = -15; numerator <= 15; ++numerator)
    for (int denominator : {-7, -3, 3, 7})
      for (int shift : {-8, 0, 8})
      {
        mpreal a = m::ldexp(mpreal{numerator}, shift), b = mpreal{denominator} / 5;
        auto ratio = a / b;
        auto expected = a - m::round_even(ratio) * b;
        auto truncated = a - m::trunc(ratio) * b;
        auto quotient = static_cast<int>(static_cast<double>(m::round_even(ratio))) % 8;
        auto mixed = m::remquo(a.at(p), b);
        EXPECT_EQ(mixed.remainder, expected.at(p));
        EXPECT_EQ(mixed.quotient, quotient);
        EXPECT_EQ(m::fmod(a.at(p), b), truncated.at(p));
        auto reverse = m::remquo(b, a.at(p));
        if (numerator == 0)
          EXPECT_TRUE(m::isnan(reverse.remainder));
        else
          EXPECT_EQ(reverse.remainder, (b - m::round_even(b / a) * a).at(p));
      }
}

TEST(MprealUtilities, DecompositionAndScaling)
{
  auto p = Precision::bits(128);
  auto parts = m::frexp(q("-7/3"));
  EXPECT_EQ(parts.fraction, q("-7/12"));
  EXPECT_EQ(parts.exponent, 2);
  EXPECT_EQ(m::ldexp(parts.fraction, parts.exponent), q("-7/3"));
  EXPECT_TRUE(parts.fraction.is_exact());
  auto fractional = m::modf(q("-7/3"));
  EXPECT_EQ(fractional.fraction, q("-1/3"));
  EXPECT_EQ(fractional.integer, -2);
  EXPECT_EQ(m::ilogb(q("7/3")), 1);
  EXPECT_EQ(m::ilogb(q("1/3")), -2);
  auto huge = m::ldexp(mpreal(1, p), 1'000'000'000);
  EXPECT_EQ(m::frexp(huge).exponent, 1'000'000'001);
  EXPECT_EQ(m::frexp(huge).fraction, q("1/2"));
  EXPECT_EQ(m::ilogb(huge), 1'000'000'000);
  auto rounded = m::frexp(q("15/16"), Precision::bits(2));
  EXPECT_EQ(rounded.fraction, q("1/2"));
  EXPECT_EQ(rounded.exponent, 1);
  auto split = m::modf(q("1023/1024"), Precision::bits(2));
  EXPECT_EQ(split.integer, 0);
  EXPECT_EQ(split.fraction, 1); // Components round independently after the split.
  EXPECT_EQ(m::scalbn(q("1/3"), 3), q("8/3"));
  EXPECT_EQ(m::ldexp(q("1/3"), 3, p), q("8/3").at(p));
  EXPECT_THROW(m::ilogb(mpreal{}), std::domain_error);
  EXPECT_THROW(m::ilogb(mpreal("inf", p)), std::domain_error);
  EXPECT_EQ(m::frexp(mpreal("inf", p)).exponent, 0);
  EXPECT_EQ(m::modf(mpreal("-inf", p)).integer, mpreal("-inf", p));
  EXPECT_TRUE(m::signbit(m::modf(mpreal("-inf", p)).fraction));
}

TEST(MprealUtilities, IntegerPowersRootsAndPairedFunctions)
{
  auto p = Precision::bits(128);
  auto too_large = numeric_limits<std::uint64_t>::max();
  EXPECT_THROW(m::ldexp(mpreal{1}, too_large, p), std::out_of_range);
  EXPECT_THROW(m::scalbn(mpreal{1}, too_large, p), std::out_of_range);
  EXPECT_THROW(m::pown(mpreal{1}, too_large, p), std::out_of_range);
  EXPECT_THROW(m::rootn(mpreal{1}, too_large, p), std::out_of_range);
  EXPECT_EQ(m::pown(q("2/3"), -3), q("27/8"));
  EXPECT_EQ(m::rootn(q("-8/27"), -3), q("-3/2"));
  EXPECT_THROW(m::rootn(mpreal{2}, 3), std::logic_error);
  EXPECT_THROW(m::rootn(mpreal{2}, 0), std::domain_error);
  EXPECT_TRUE(m::isnan(m::rootn(mpreal{-1}, 2, p)));
  EXPECT_TRUE(m::isinf(m::rootn(mpreal{}, -3, p)));
  auto low = Precision::bits(3);
  EXPECT_EQ(m::pown(q("9/8"), 2, low), q("5/4"));
  EXPECT_EQ(m::pown(q("2/3"), -2, low), q("2")); // 9/4 ties to even at three bits.
  auto ref = Precision::bits(512);
  EXPECT_EQ(m::rootn(q("2/3"), -3, p), m::rootn(q("2/3").at(ref), -3).at(p));
  EXPECT_EQ(m::pown(q("2/3"), 7, p), q("128/2187").at(p));
  EXPECT_EQ(m::pown(mpreal(-1, p), numeric_limits<std::int64_t>::min()), 1);
  EXPECT_EQ(m::rootn(mpreal(1, p), numeric_limits<std::int64_t>::min()), 1);
  EXPECT_EQ(m::sincos(mpreal{}).cos, 1);
  EXPECT_TRUE(m::sincos(mpreal{}).sin.is_exact());
  EXPECT_EQ(m::sinhcosh(mpreal{}).cosh, 1);
  auto x = q("1/3").at(p);
  auto trig = m::sincos(x);
  EXPECT_EQ(trig.sin, m::sin(x));
  EXPECT_EQ(trig.cos, m::cos(x));
  auto hyperbolic = m::sinhcosh(x);
  EXPECT_EQ(hyperbolic.sinh, m::sinh(x));
  EXPECT_EQ(hyperbolic.cosh, m::cosh(x));
}

TEST(MprealUtilities, GridAndSignOperandsDoNotSupplyPrecision)
{
  auto p = Precision::bits(80), wider = Precision::bits(256);
  auto one = mpreal(1, p), direction = one.at(wider) + epsilon(wider);
  EXPECT_EQ(m::nextafter(one, direction), one + epsilon(p));
  EXPECT_EQ(m::next_down(one), one - epsilon(p) / 2);
  EXPECT_EQ(m::next_up(mpreal{1}, p).precision(), p);
  EXPECT_THROW(m::next_up(mpreal{1}), std::logic_error);
  EXPECT_THROW(m::nextafter(mpreal{1}, direction), std::logic_error);
  EXPECT_TRUE(m::signbit(m::nextafter(mpreal(0, p), mpreal("-0", wider))));
  EXPECT_EQ(m::copysign(one, mpreal(-1, wider)).precision(), p);
  EXPECT_EQ(m::copysign(q("1/3"), mpreal(-1, wider)), q("-1/3"));
  EXPECT_TRUE(m::copysign(q("1/3"), mpreal(-1, wider)).is_exact());
  EXPECT_TRUE(m::signbit(m::fmin(mpreal(0, p), mpreal("-0", p))));
  EXPECT_FALSE(m::signbit(m::fmax(mpreal(0, p), mpreal("-0", p))));
  EXPECT_EQ(m::fmin(mpreal("nan", p), q("1/3")), q("1/3").at(p));
  EXPECT_EQ(m::fmax(mpreal("nan", p), q("1/3")), q("1/3").at(p));
  EXPECT_EQ(m::fdim(q("9/8"), mpreal{1}, Precision::bits(3)), q("1/8"));
  EXPECT_EQ(m::fdim(mpreal("inf", p), mpreal("inf", p)), 0);
  EXPECT_TRUE(m::isfinite(q("1/3")));
  EXPECT_TRUE(m::isinf(mpreal("inf", p)));
  EXPECT_TRUE(m::isnan(mpreal("nan", p)));
  EXPECT_THROW(m::isfinite(mpreal(uninitialized)), std::logic_error);
}
} // namespace
