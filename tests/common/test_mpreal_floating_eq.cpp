#include <uni20/common/gtest.hpp>
#include <uni20/core/math.hpp>

#include <gtest/gtest-spi.h>

#if UNI20_ENABLE_MPFR
namespace
{
using namespace uni20;
using check::float_abs_distance;
using check::float_distance;
using Ulp = check::FloatingULP<mpreal>;
static_assert(check::UlpComparable<mpreal>);
#if UNI20_ENABLE_MPC
static_assert(!check::UlpComparable<complex<mpreal>>);
#endif

TEST(MprealFloatingEq, AdjacentValuesAcrossBinadesAndSigns)
{
  for (auto p : {Precision::bits(128), Precision::bits(256)})
    for (int exponent : {-1'000'000'000, -1000, -1, 0, 1, 1000, 1'000'000'000})
    {
      auto value = math::ldexp(mpreal(1, p), exponent);
      auto previous = math::next_down(value), next = math::next_up(value);
      EXPECT_EQ(float_distance(previous, value), 1);
      EXPECT_EQ(float_distance(value, next), 1);
      EXPECT_EQ(float_distance(previous, next), 2);
      EXPECT_EQ(float_distance(next, previous), -2);
      EXPECT_EQ(float_distance(-next, -previous), 2);
      EXPECT_FLOATING_EQ(previous, next, 2);
      EXPECT_FALSE(Ulp::eq(previous, next, 1));
      auto step = value;
      for (int i = 1; i <= 5; ++i)
      {
        step = math::next_up(step);
        EXPECT_EQ(float_abs_distance(value, step), i);
        EXPECT_TRUE(Ulp::eq(value, step, i));
        EXPECT_FALSE(Ulp::eq(value, step, i - 1));
        EXPECT_EQ(Ulp::eq(value, step), i <= 4);
      }
      // These neighbors disappear if the comparison narrows through double.
      if (exponent == 0)
      {
        EXPECT_EQ(static_cast<double>(previous), static_cast<double>(next));
      }
    }
}

TEST(MprealFloatingEq, CountsStepsOnSmallPrecisionGrid)
{
  // Independently enumerate a small grid across multiple binades. This checks
  // the rank formula without constructing expectations through that formula.
  for (auto p : {Precision::bits(1), Precision::bits(3), Precision::bits(8)})
    for (int sign : {-1, 1})
      for (int start = 0; start != 12; ++start)
      {
        mpreal first(1, p);
        for (int i = 0; i != start; ++i)
          first = math::next_up(first);
        auto last = first;
        for (int steps = 0; steps != 20; ++steps)
        {
          EXPECT_EQ(float_distance(first * sign, last * sign), steps * sign);
          EXPECT_TRUE(Ulp::eq(first * sign, last * sign, steps));
          if (steps)
          {
            EXPECT_FALSE(Ulp::eq(first * sign, last * sign, steps - 1));
          }
          last = math::next_up(last);
        }
      }
}

TEST(MprealFloatingEq, SignedZeroAndExponentEndpoints)
{
  for (auto p : {Precision::bits(128), Precision::bits(256)})
  {
    mpreal zero(0, p), negative_zero("-0", p), infinity("inf", p);
    auto smallest = math::next_up(zero), next = math::next_up(smallest);
    EXPECT_EQ(float_distance(zero, negative_zero), 0);
    EXPECT_EQ(float_distance(zero, smallest), 1);
    EXPECT_EQ(float_distance(-smallest, smallest), 2);
    EXPECT_EQ(float_distance(-next, next), 4);
    EXPECT_FLOATING_EQ(-next, next);
    EXPECT_FALSE(Ulp::eq(-next, next, 3));
    auto largest = math::next_down(infinity);
    EXPECT_EQ(float_distance(math::next_down(largest), largest), 1);
    EXPECT_FALSE(Ulp::eq(largest, infinity));
  }
}

TEST(MprealFloatingEq, SaturatesDiagnosticsWithoutRelaxingTolerance)
{
  constexpr auto limit = numeric_limits<std::int64_t>::max();
  for (auto p : {Precision::bits(128), Precision::bits(256)})
  {
    mpreal one(1, p), two(2, p);
    EXPECT_EQ(float_distance(one, two), limit);
    EXPECT_EQ(float_distance(two, one), -limit);
    EXPECT_EQ(float_abs_distance(two, one), limit);
    EXPECT_FALSE(Ulp::eq(one, two, limit));
    auto at_limit = one + math::ldexp(mpreal(limit, p), 1 - p.bit_count());
    EXPECT_TRUE(Ulp::eq(one, at_limit, limit));
    EXPECT_FALSE(Ulp::eq(one, at_limit, limit - 1));
    EXPECT_FALSE(Ulp::eq(one, math::next_up(at_limit), limit));
    EXPECT_EQ(float_distance(one, at_limit), limit);
    EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(one, two), "exceeds diagnostic range");
  }
}

TEST(MprealFloatingEq, UsesCurrentExponentRangeWithoutChangingIt)
{
  struct RestoreRange
  {
      mpfr_exp_t emin = mpfr_get_emin(), emax = mpfr_get_emax();
      ~RestoreRange()
      {
        mpfr_set_emin(emin);
        mpfr_set_emax(emax);
      }
  } restore_range;
  check::detail::mpfr_ulp_flags restore_flags;
  auto p = Precision::bits(3);
  auto outside = mpreal(256, p);
  ASSERT_EQ(mpfr_set_emin(-4), 0);
  ASSERT_EQ(mpfr_set_emax(4), 0);
  mpreal zero(0, p), infinity("inf", p);
  auto smallest = math::next_up(zero), largest = math::next_down(infinity);
  // Nine binades, four significands per binade: 36 positive finite values.
  EXPECT_EQ(float_distance(zero, smallest), 1);
  EXPECT_EQ(float_distance(zero, largest), 36);
  EXPECT_EQ(float_distance(-largest, largest), 72);
  EXPECT_FLOATING_EQ(mpreal("1/1024", Precision::exact()), zero, 0);
  EXPECT_FLOATING_EQ(mpreal{32}, infinity, 0);
  EXPECT_FALSE(Ulp::eq(outside, outside));
  EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(outside, outside), "outside the current MPFR exponent range");
  EXPECT_EQ(mpfr_get_emin(), -4);
  EXPECT_EQ(mpfr_get_emax(), 4);
}

TEST(MprealFloatingEq, ExceptionalValuesAndPrecisionMismatch)
{
  auto p = Precision::bits(128);
  mpreal nan("nan", p), infinity("inf", p), one(1, p);
  EXPECT_FALSE(Ulp::eq(nan, nan));
  EXPECT_TRUE(Ulp::eq(infinity, infinity));
  EXPECT_FALSE(Ulp::eq(infinity, -infinity));
  EXPECT_FALSE(Ulp::eq(one, infinity));
  EXPECT_FALSE(Ulp::eq(one, one, -1));
  EXPECT_EQ(float_distance(infinity, infinity), 0);
  EXPECT_EQ(float_distance(nan, one), numeric_limits<long long>::max());
  EXPECT_FALSE(Ulp::eq(one, one.at(Precision::bits(256))));
  EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(one, one.at(Precision::bits(256))), "working precisions differ");
  EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(nan, one), "NaN");
  EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(one, one, -1), "non-negative ULP tolerance");
}

TEST(MprealFloatingEq, DiagnosticsAndSingleEvaluation)
{
  for (auto p : {Precision::bits(128), Precision::bits(256)})
  {
    auto a = mpreal(1, p), b = math::next_up(a);
    EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(a, b, 0), "actual distance: 1");
    EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(a, b, 0), "allowed tolerance: 0 ULP");
    EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(a, b, 0), b.to_string());
    auto precision = std::to_string(p.bit_count()) + " bits";
    EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(a, b, 0), precision);
    int first = 0, second = 0, tolerance = 0;
    EXPECT_FLOATING_EQ((++first, a), (++second, b), (++tolerance, 1));
    EXPECT_EQ(first, 1);
    EXPECT_EQ(second, 1);
    EXPECT_EQ(tolerance, 1);
  }
}

void assert_outside_tolerance()
{
  auto a = mpreal(1, Precision::bits(256));
  ASSERT_FLOATING_EQ(a, math::next_up(a), 0);
  ADD_FAILURE() << "fatal assertion did not return";
}

TEST(MprealFloatingEq, FatalAssertionIntegration)
{
  auto a = mpreal(1, Precision::bits(128));
  ASSERT_FLOATING_EQ(a, math::next_up(a), 1);
  EXPECT_FATAL_FAILURE(assert_outside_tolerance(), "actual distance: 1");
}

TEST(MprealFloatingEq, ExactOperandsSelectNoGrid)
{
  auto third = mpreal("1/3", Precision::exact());
  EXPECT_FLOATING_EQ(third, mpreal("2/6", Precision::exact()), 0);
  EXPECT_FALSE(Ulp::eq(third, mpreal{1}, numeric_limits<std::int64_t>::max()));
  EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(third, mpreal{1}), "unequal exact values have no ULP grid");
  for (auto p : {Precision::bits(128), Precision::bits(256)})
  {
    auto rounded = third.at(p), next = math::next_up(rounded);
    EXPECT_FLOATING_EQ(third, rounded, 0);
    EXPECT_FLOATING_EQ(rounded, third, 0);
    EXPECT_EQ(float_distance(third, next), 1);
    EXPECT_EQ(float_distance(next, third), -1);
    EXPECT_FALSE(Ulp::eq(third, next, 0));
    EXPECT_FLOATING_EQ(mpreal(0, p), mpreal{}, 0);
    EXPECT_FLOATING_EQ(mpreal{1}, mpreal(1, p), 0);
    EXPECT_TRUE(third.is_exact());
    EXPECT_EQ(rounded.precision(), p);
  }
  // Rounding is nearest with ties to even, including a binade boundary.
  auto p = Precision::bits(3);
  EXPECT_FLOATING_EQ(mpreal("9/8", Precision::exact()), mpreal(1, p), 0);
  EXPECT_FLOATING_EQ(mpreal("15/16", Precision::exact()), mpreal(1, p), 0);
}

TEST(MprealFloatingEq, UnsetValuesFailWithDiagnostics)
{
  mpreal unset(uninitialized), one(1, Precision::bits(128));
  EXPECT_FALSE(Ulp::eq(unset, unset));
  EXPECT_FALSE(Ulp::eq(unset, one));
  EXPECT_FALSE(Ulp::eq(one, unset));
  EXPECT_EQ(float_distance(unset, one), numeric_limits<long long>::max());
  EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(unset, one), "unset operand");
  EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(one, unset), "<unset>");
}

TEST(MprealFloatingEq, PreservesMpfrStateAndOperands)
{
  check::detail::mpfr_ulp_flags restore_flags;
  auto third = mpreal("1/3", Precision::exact());
  auto p = Precision::bits(128);
  auto rounded = third.at(p);
  auto next = math::next_up(rounded);
  auto const default_precision = mpfr_get_default_prec();
  auto const emin = mpfr_get_emin(), emax = mpfr_get_emax();
  mpfr_clear_flags();
  mpfr_set_divby0();
  auto const flags = mpfr_flags_save();
  EXPECT_TRUE(Ulp::eq(third, rounded, 0));
  EXPECT_FALSE(Ulp::eq(third, next, 0));
  EXPECT_EQ(float_distance(third, next), 1);
  EXPECT_EQ(mpfr_flags_save(), flags);
  EXPECT_EQ(mpfr_get_default_prec(), default_precision);
  EXPECT_EQ(mpfr_get_emin(), emin);
  EXPECT_EQ(mpfr_get_emax(), emax);
  EXPECT_TRUE(third.is_exact());
  EXPECT_EQ(rounded.precision(), p);
}

TEST(MprealFloatingEqDeathTest, RuntimeCheckUsesSameComparator)
{
  GTEST_FLAG_SET(death_test_style, "fast");
  auto one = mpreal(1, Precision::bits(128));
  CHECK_FLOATING_EQ(one, math::next_up(one), 1);
  EXPECT_DEATH(CHECK_FLOATING_EQ(one, mpreal(2, one.precision())), "CHECK_FLOATING_EQ");
}
} // namespace
#endif
