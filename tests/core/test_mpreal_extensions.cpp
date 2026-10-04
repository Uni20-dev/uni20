#include <uni20/core/math.hpp>

#include <cstdint>
#include <gtest/gtest.h>

namespace
{
using namespace uni20;
namespace m = uni20::math;
mpreal q(char const* text) { return mpreal(text, Precision::exact()); }

TEST(MprealExtensions, ExactPiReductionAndInverseValues)
{
  auto check = [](mpreal const& value, char const* expected) {
    EXPECT_TRUE(value.is_exact());
    EXPECT_EQ(value, q(expected));
  };
  check(m::sinpi(q("1/6")), "1/2");
  check(m::sinpi(q("-5/6")), "-1/2");
  check(m::sinpi(q("3/2")), "-1");
  check(m::cospi(q("1/3")), "1/2");
  check(m::cospi(q("-2/3")), "-1/2");
  check(m::cospi(q("1/2")), "0");
  check(m::tanpi(q("1/4")), "1");
  check(m::tanpi(q("3/4")), "-1");
  check(m::tanpi(q("-1/4")), "-1");
  auto huge = m::pown(mpreal{10}, 100);
  check(m::sinpi(huge + q("1/6")), "1/2");
  check(m::cospi(huge + q("1/3")), "1/2");
  check(m::tanpi(huge + q("1/4")), "1");
  check(m::asinpi(q("-1/2")), "-1/6");
  check(m::asinpi(mpreal{1}), "1/2");
  check(m::acospi(q("-1/2")), "2/3");
  check(m::acospi(mpreal{1}), "0");
  check(m::atanpi(mpreal{-1}), "-1/4");
  check(m::atan2pi(mpreal{1}, mpreal{-1}), "3/4");
  check(m::atan2pi(mpreal{-1}, mpreal{-1}), "-3/4");
  check(m::atan2pi(mpreal{1}, mpreal{}), "1/2");
  check(m::atan2pi(mpreal{}, mpreal{-1}), "1");
  check(m::atan2pi(mpreal{}, mpreal{1}), "0");
  EXPECT_THROW(m::atan2pi(mpreal{}, mpreal{}), std::domain_error);
  EXPECT_THROW(m::sinpi(q("1/4")), std::logic_error);
  EXPECT_THROW(m::cospi(q("1/6")), std::logic_error);
  EXPECT_THROW(m::tanpi(q("1/6")), std::logic_error);
  EXPECT_THROW(m::tanpi(q("1/2")), std::domain_error);
  EXPECT_THROW(m::asinpi(q("1/3")), std::logic_error);
  EXPECT_THROW(m::acospi(q("1/3")), std::logic_error);
  EXPECT_THROW(m::atanpi(q("1/3")), std::logic_error);
  EXPECT_THROW(m::atan2pi(mpreal{1}, mpreal{2}), std::logic_error);
}

TEST(MprealExtensions, ExactExponentialsAndReciprocals)
{
  auto check = [](mpreal const& value, char const* expected) {
    EXPECT_TRUE(value.is_exact());
    EXPECT_EQ(value, q(expected));
  };
  check(m::exp10(mpreal{-3}), "1/1000");
  check(m::exp2m1(mpreal{-3}), "-7/8");
  check(m::exp10m1(mpreal{-3}), "-999/1000");
  check(m::log2p1(mpreal{7}), "3");
  check(m::log10p1(mpreal{99}), "2");
  check(m::log2p1(q("-7/8")), "-3");
  check(m::log10p1(q("-999/1000")), "-3");
  check(m::sec(mpreal{}), "1");
  check(m::sech(mpreal{}), "1");
  EXPECT_THROW(m::exp10(q("1/2")), std::logic_error);
  EXPECT_THROW(m::exp2m1(q("1/2")), std::logic_error);
  EXPECT_THROW(m::exp10m1(q("1/2")), std::logic_error);
  EXPECT_THROW(m::log2p1(mpreal{2}), std::logic_error);
  EXPECT_THROW(m::log10p1(mpreal{2}), std::logic_error);
  EXPECT_THROW(m::csc(mpreal{1}), std::logic_error);
  EXPECT_THROW(m::cot(mpreal{1}), std::logic_error);
  EXPECT_THROW(m::csch(mpreal{1}), std::logic_error);
  EXPECT_THROW(m::coth(mpreal{1}), std::logic_error);
}

TEST(MprealExtensions, CompoundRoundsResultAndChecksIntegerDomain)
{
  auto exact = m::compound(q("1/3"), -3);
  EXPECT_TRUE(exact.is_exact());
  EXPECT_EQ(exact, q("27/64"));
  auto low = Precision::bits(3);
  // (1+1/16)^8 = (17/16)^8 rounds to 1.5, but rounding the
  // sum to three bits before exponentiation gives 1.
  auto x = mpreal("1/16", Precision::bits(128));
  auto expected = m::pown(q("17/16"), 8).at(low);
  EXPECT_EQ(expected, mpreal("1.5", low));
  EXPECT_EQ(m::compound(x, 8, low), expected);
  EXPECT_EQ(m::compound(q("1/16"), 8, low), expected);
  EXPECT_EQ(m::pown((mpreal{1} + x).at(low), 8), 1);
  // This case also distinguishes result rounding from rounding x first.
  EXPECT_EQ(m::compound(q("2/3"), 2, low), 3);
  EXPECT_EQ(m::compound(q("2/3").at(low), 2), mpreal("2.5", low));
  for (auto p : {Precision::bits(128), Precision::bits(256)})
  {
    auto input = mpreal("1/4", p);
    for (int n : {-7, 0, 3})
    {
      auto expected_value = detail::mpreal_access::finite_result(
          p, [&](mpfr_ptr out) { mpfr_compound_si(out, input.native_handle(), n, MPFR_RNDN); });
      EXPECT_EQ(m::compound(input, n), expected_value);
      EXPECT_EQ(m::compound(input, n).precision(), p);
    }
    EXPECT_TRUE(isnan(m::compound(mpreal(-2, p), 0)));
    EXPECT_TRUE(isnan(m::compound(mpreal{-2}, 0, p)));
    EXPECT_EQ(m::compound(mpreal("nan", p), 0), 1);
    EXPECT_TRUE(isinf(m::compound(mpreal{-1}, -1, p)));
    EXPECT_EQ(m::compound(mpreal{-1}, 0, p), 1);
    EXPECT_EQ(m::compound(mpreal{-1}, 1, p), 0);
  }
  EXPECT_THROW(m::compound(mpreal{-2}, 0), std::domain_error);
  EXPECT_THROW(m::compound(mpreal{-1}, -1), std::domain_error);
  EXPECT_THROW(m::compound(x, numeric_limits<std::uint64_t>::max()), std::out_of_range);
  EXPECT_THROW(m::compound(mpreal(uninitialized), 0), std::logic_error);
  EXPECT_THROW(m::compound(mpreal(uninitialized), 0, low), std::logic_error);
  EXPECT_THROW(m::compound(x, 0, Precision::exact()), std::logic_error);
}

TEST(MprealExtensions, PiScaledSignedZerosAndLargeArguments)
{
  auto p = Precision::bits(128);
  auto negzero = mpreal("-0", p), zero = mpreal(0, p), one = mpreal(1, p);
  EXPECT_EQ(m::atan2pi(negzero, -one), -1);
  EXPECT_EQ(m::atan2pi(zero, -one), 1);
  EXPECT_TRUE(signbit(m::atan2pi(negzero, one)));
  EXPECT_EQ(m::atan2pi(one, zero), mpreal("1/2", p));
  EXPECT_EQ(m::cospi(mpreal("-1/2", p)), 0);
  EXPECT_FALSE(signbit(m::cospi(mpreal("-1/2", p))));
  EXPECT_EQ(m::sinpi(m::ldexp(one, 100)), 0);
  EXPECT_EQ(m::cospi(m::ldexp(one, 100) + mpreal("1/2", p)), 0);
  EXPECT_EQ(m::tanpi(m::ldexp(one, 100) + mpreal("1/4", p)), 1);
  EXPECT_TRUE(isinf(m::csc(negzero)));
  EXPECT_TRUE(signbit(m::csc(negzero)));
  EXPECT_TRUE(signbit(m::csch(negzero)));
  EXPECT_TRUE(signbit(m::coth(negzero)));
}
} // namespace
