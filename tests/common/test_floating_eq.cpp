#include <uni20/common/gtest.hpp>

#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>

#include <gtest/gtest-spi.h>

TEST(FloatDistance, CrossesSignedZeroWithoutCountingZeroTwice)
{
  float const gap = std::numeric_limits<float>::denorm_min();
  EXPECT_EQ(uni20::check::float_distance(-gap, gap), 2);
  EXPECT_EQ(uni20::check::float_distance(gap, -gap), -2);
}

TEST(FloatDistance, ClearlyDifferentSignsAreNotClose)
{
  EXPECT_FALSE(uni20::check::FloatingULP<float>::eq(-1.0f, 1.0f));
}

// --- EXPECT_FLOATING_EQ ---

TEST(FloatingEqGTest, ExpectPassesWithinTolerance)
{
  float a = 1.0f;
  float b = std::nextafter(a, 2.0f); // 1 ULP away
  EXPECT_FLOATING_EQ(a, b, 1);       // should pass
}

TEST(FloatingEqGTest, ExpectFailsOutsideTolerance)
{
  float a = 1.0f;
  float b = std::nextafter(a, 0.0f); // 1 ULP away in the opposite direction
  EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(a, b, 0), "EXPECT_FLOATING_EQ failed");
}

TEST(FloatingEqGTest, ExpectDefaultToleranceIsFour)
{
  float a = 1.0f;
  float b = std::bit_cast<float>(std::bit_cast<std::uint32_t>(a) + 4);
  EXPECT_FLOATING_EQ(a, b); // default tolerance is 4 ULP
}

TEST(FloatingEqGTest, ExpectReportsToleranceAndDistance)
{
  float a = 1.0f;
  float b = std::nextafter(a, 0.0f); // 1 ULP away
  EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(a, b, 0), "allowed tolerance: 0 ULP");
  EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(a, b, 0), "actual distance: 1");
}

TEST(FloatingEqGTest, ExpectRejectsNegativeTolerance)
{
  EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(1.0f, 1.0f, -1), "non-negative ULP tolerance");
}

// --- ASSERT_FLOATING_EQ ---

TEST(FloatingEqGTest, AssertPassesWithinTolerance)
{
  double a = 1.0;
  double b = std::nextafter(a, 2.0);
  ASSERT_FLOATING_EQ(a, b, 1); // should pass
  SUCCEED();
}

TEST(FloatingEqGTest, AssertFailsOutsideTolerance)
{
  static double const a = 1.0; // these need to be static because EXPECT_FATAL_FAILURE uses a lambda with no capture
  static double const b = std::nextafter(a, 0.0);
  EXPECT_FATAL_FAILURE(ASSERT_FLOATING_EQ(a, b, 0), "ASSERT_FLOATING_EQ failed");
}

TEST(FloatingEqGTest, AssertDefaultToleranceIsFour)
{
  double a = 1.0;
  double b = std::bit_cast<double>(std::bit_cast<std::uint64_t>(a) + 4);
  ASSERT_FLOATING_EQ(a, b); // default tolerance is 4 ULP
  SUCCEED();
}

TEST(FloatingEqGTest, AssertRejectsNegativeTolerance)
{
  EXPECT_FATAL_FAILURE(ASSERT_FLOATING_EQ(1.0, 1.0, -1), "non-negative ULP tolerance");
}

// --- Complex numbers ---

TEST(FloatingEqGTest, ComplexPasses)
{
  uni20::complex<float> a{1.0f, 2.0f};
  uni20::complex<float> b{std::nextafter(1.0f, 2.0f), 2.0f};
  EXPECT_FLOATING_EQ(a, b, 1); // real differs by 1 ULP
}

TEST(FloatingEqGTest, ComplexFails)
{
  uni20::complex<double> a{1.0, 2.0};
  uni20::complex<double> b{1.0, 2.1};
  EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(a, b, 1), "EXPECT_FLOATING_EQ failed");
}

// --- NaN and infinity behavior ---

TEST(FloatingEqGTest, NaNFails)
{
  float nan = std::numeric_limits<float>::quiet_NaN();
  EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(nan, nan), "unrepresentable");
}

TEST(FloatingEqGTest, SameInfinityPasses)
{
  double inf = std::numeric_limits<double>::infinity();
  EXPECT_FLOATING_EQ(inf, inf);
}

TEST(FloatingEqGTest, OppositeInfinityFails)
{
  static double const pos_inf = std::numeric_limits<double>::infinity();
  static double const neg_inf = -std::numeric_limits<double>::infinity();
  EXPECT_FATAL_FAILURE(ASSERT_FLOATING_EQ(pos_inf, neg_inf), "unrepresentable");
}

#if UNI20_HAS_FLOAT128 && UNI20_FLOAT128_PROVIDER_MPLAPACK && defined(MPLAPACK_BINARY128_MODE) &&                      \
    (MPLAPACK_BINARY128_MODE != MPLAPACK_BINARY128_MODE_LDBL)

TEST(Float128FloatingEq, DistinguishesValuesWhoseLowWordsMatch)
{
  using Real = uni20::float128;
  Real const one = Real{1};
  Real const two = Real{2};

  static_assert(uni20::check::IeeeBinaryReal<Real>);
  static_assert(uni20::check::UlpComparable<Real>);
  EXPECT_FALSE(uni20::check::FloatingULP<Real>::eq(one, two));
  EXPECT_FALSE(uni20::check::FloatingULP<Real>::eq(one, two, std::numeric_limits<std::int64_t>::max()));
  EXPECT_EQ(uni20::check::float_distance(one, two), std::numeric_limits<long long>::max());
  EXPECT_EQ(uni20::check::float_distance(two, one), -std::numeric_limits<long long>::max());
  EXPECT_EQ(uni20::check::float_abs_distance(one, two), std::numeric_limits<long long>::max());
}

TEST(Float128FloatingEq, AcceptsAdjacentRealAndComplexValues)
{
  using Real = uni20::float128;
  using Complex = uni20::complex<Real>;
  Real const one = Real{1};
  auto const one_bits = std::bit_cast<__uint128_t>(one);
  Real const next = std::bit_cast<Real>(one_bits + 1);
  Complex const expected{one, Real{2}};
  Complex const actual{next, Real{2}};

  EXPECT_EQ(uni20::check::float_distance(one, next), 1);
  EXPECT_FLOATING_EQ(one, next, 1);
  EXPECT_FLOATING_EQ(expected, actual, 1);
  CHECK_FLOATING_EQ(one, next, 1);
  CHECK_FLOATING_EQ(expected, actual, 1);
}

TEST(Float128FloatingEqDeathTest, CheckRejectsClearlyDifferentRealAndComplexValues)
{
  GTEST_FLAG_SET(death_test_style, "fast");
  using Real = uni20::float128;
  using Complex = uni20::complex<Real>;
  Real const one = Real{1};
  Real const two = Real{2};
  Complex const expected{one, one};
  Complex const actual{one, two};

  EXPECT_DEATH({ CHECK_FLOATING_EQ(one, two); }, "CHECK_FLOATING_EQ");
  EXPECT_DEATH({ CHECK_FLOATING_EQ(expected, actual); }, "CHECK_FLOATING_EQ");
}

#endif

#if UNI20_HAS_FLOAT80
namespace
{
using Real80 = uni20::float80;
using Limits80 = uni20::numeric_limits<Real80>;
using Ulp80 = uni20::check::FloatingULP<Real80>;
using uni20::check::float_abs_distance;
using uni20::check::float_distance;
static_assert(!uni20::check::IeeeBinaryReal<Real80>);
static_assert(uni20::check::UlpOrderedReal<Real80>);
static_assert(uni20::check::UlpComparable<Real80>);
static_assert(uni20::check::UlpComparable<uni20::complex<Real80>>);
} // namespace

TEST(Float80FloatingEq, AdjacentValuesAcrossEveryBinade)
{
  // Includes every subnormal power of two, the normal/subnormal boundary,
  // and the change in spacing on either side of every normal power of two.
  for (int exponent = Limits80::min_exponent - Limits80::digits; exponent < Limits80::max_exponent; ++exponent)
  {
    SCOPED_TRACE(exponent);
    Real80 const value = std::ldexp(Real80{1}, exponent);
    Real80 const previous = std::nextafter(value, Real80{0});
    Real80 const next = std::nextafter(value, Limits80::infinity());
    EXPECT_EQ(float_distance(previous, value), 1);
    EXPECT_EQ(float_distance(value, next), 1);
    EXPECT_EQ(float_distance(previous, next), 2);
    EXPECT_EQ(float_distance(next, previous), -2);
    EXPECT_EQ(float_distance(-next, -value), 1);
    EXPECT_EQ(float_distance(-value, -previous), 1);
    EXPECT_TRUE(Ulp80::eq(previous, next, 2));
    EXPECT_FALSE(Ulp80::eq(previous, next, 1));
  }
}

TEST(Float80FloatingEq, InteriorValuesAndToleranceBoundaries)
{
  for (int exponent : {-16000, -20, 0, 20, 16000})
    for (Real80 sign : {Real80{-1}, Real80{1}})
    {
      SCOPED_TRACE(exponent);
      SCOPED_TRACE(uni20::format_real(sign));
      Real80 const value = sign * std::ldexp(Real80{1.5L}, exponent);
      Real80 next = value;
      for (int steps = 1; steps <= 16; ++steps)
      {
        next = std::nextafter(next, Limits80::infinity());
        EXPECT_EQ(float_distance(value, next), steps);
        EXPECT_EQ(float_abs_distance(next, value), steps);
        EXPECT_TRUE(Ulp80::eq(value, next, steps));
        EXPECT_FALSE(Ulp80::eq(value, next, steps - 1));
      }
    }
}

TEST(Float80FloatingEq, SignedZeroAndSubnormals)
{
  Real80 const tiny = Limits80::denorm_min();
  EXPECT_EQ(float_distance(Real80{-0.0L}, Real80{0}), 0);
  EXPECT_EQ(float_distance(-tiny, tiny), 2);
  EXPECT_EQ(float_distance(tiny, -tiny), -2);
  EXPECT_EQ(float_distance(-tiny, Real80{0}), 1);
  EXPECT_TRUE(Ulp80::eq(Real80{-0.0L}, Real80{0}, 0));
  EXPECT_TRUE(Ulp80::eq(-tiny, tiny, 2));
  EXPECT_FALSE(Ulp80::eq(-tiny, tiny, 1));
}

TEST(Float80FloatingEq, LargeDistancesSaturateOnlyTheDiagnostic)
{
  constexpr auto limit = std::numeric_limits<long long>::max();
  Real80 const one = 1, two = 2;
  // The [1,2) binade has 2^63 representable fp80 values.
  EXPECT_EQ(float_distance(one, two), limit);
  EXPECT_EQ(float_distance(two, one), -limit);
  EXPECT_EQ(float_abs_distance(two, one), limit);
  EXPECT_FALSE(Ulp80::eq(one, two, limit));
  EXPECT_TRUE(Ulp80::eq(one, std::nextafter(two, one), limit));
  EXPECT_FALSE(Ulp80::eq(-one, one, limit));
  EXPECT_FALSE(Ulp80::eq(-Limits80::max(), Limits80::max(), limit));
  EXPECT_EQ(float_distance(-Limits80::max(), Limits80::max()), limit);
}

TEST(Float80FloatingEq, NaNInfinityAndInvalidTolerance)
{
  Real80 const infinity = Limits80::infinity(), nan = Limits80::quiet_NaN(), one = 1;
  constexpr auto limit = std::numeric_limits<long long>::max();
  EXPECT_FALSE(Ulp80::eq(nan, nan));
  EXPECT_FALSE(Ulp80::eq(one, nan));
  EXPECT_FALSE(Ulp80::eq(nan, one));
  EXPECT_TRUE(Ulp80::eq(infinity, infinity, 0));
  EXPECT_TRUE(Ulp80::eq(-infinity, -infinity, 0));
  EXPECT_FALSE(Ulp80::eq(infinity, -infinity, limit));
  EXPECT_FALSE(Ulp80::eq(Limits80::max(), infinity, limit));
  EXPECT_FALSE(Ulp80::eq(one, one, -1));
  EXPECT_EQ(float_distance(nan, one), limit);
  EXPECT_EQ(float_distance(one, infinity), limit);
  EXPECT_EQ(float_distance(infinity, infinity), 0);
}

TEST(Float80FloatingEq, RealAndComplexAssertionIntegration)
{
  Real80 const one = 1, next = std::nextafter(one, Real80{2});
  EXPECT_FLOATING_EQ(one, next, 1);
  ASSERT_FLOATING_EQ(one, next, 1);
  CHECK_FLOATING_EQ(one, next, 1);
  PRECONDITION_FLOATING_EQ(one, next, 1);
  EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(one, next, 0), "actual distance: 1");
  EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(one, next, -1), "non-negative ULP tolerance");
  static Real80 const fixed_one = 1, fixed_next = std::nextafter(fixed_one, Real80{2});
  EXPECT_FATAL_FAILURE(ASSERT_FLOATING_EQ(fixed_one, fixed_next, 0), "actual distance: 1");
  uni20::complex<Real80> const a{one, -next}, b{next, -one};
  EXPECT_FLOATING_EQ(a, b, 1);
  ASSERT_FLOATING_EQ(a, b, 1);
  CHECK_FLOATING_EQ(a, b, 1);
  EXPECT_EQ(float_abs_distance(a, b), 1);
  EXPECT_NONFATAL_FAILURE(EXPECT_FLOATING_EQ(a, b, 0), "actual distance: 1");
}

TEST(Float80FloatingEqDeathTest, CheckRejectsDifferentValues)
{
  GTEST_FLAG_SET(death_test_style, "fast");
  EXPECT_DEATH({ CHECK_FLOATING_EQ(Real80{1}, Real80{2}); }, "CHECK_FLOATING_EQ");
  EXPECT_DEATH({ PRECONDITION_FLOATING_EQ(Real80{1}, Real80{2}); }, "PRECONDITION_FLOATING_EQ");
}
#endif
