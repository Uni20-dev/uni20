#include <uni20/core/scalar_concepts.hpp>
#include <uni20/core/scalar_io.hpp>
#include <uni20/core/scalar_precision.hpp>

#include <gtest/gtest.h>

#include <cmath>
#include <sstream>

TEST(Float80Test, AvailabilityMatchesNativeFormat)
{
  using Limits = uni20::numeric_limits<long double>;
  constexpr bool native = Limits::is_iec559 && Limits::radix == 2 && Limits::digits == 64 &&
                          Limits::min_exponent == -16381 && Limits::max_exponent == 16384 &&
                          Limits::has_denorm == std::denorm_present;
  EXPECT_EQ(uni20::has_float80, native);
}

#if UNI20_HAS_FLOAT80
using Real80 = uni20::float80;
static_assert(std::same_as<Real80, long double>);
static_assert(uni20::Real<Real80>);
static_assert(uni20::Complex<uni20::complex<Real80>>);
static_assert(!uni20::BlasReal<Real80> && !uni20::LapackReal<Real80>);
static_assert(!uni20::BlasComplex<uni20::complex<Real80>> && !uni20::LapackComplex<uni20::complex<Real80>>);

TEST(Float80Test, ArithmeticAndMathRetainNativePrecision)
{
  Real80 const one = 1;
  Real80 const next = std::nextafter(one, Real80{2});
  EXPECT_NE(next, one);
  EXPECT_EQ(next - one, uni20::numeric_limits<Real80>::epsilon());
  EXPECT_EQ(static_cast<double>(next), 1.0);
  Real80 const root = std::sqrt(Real80{2});
  EXPECT_LT(std::abs(root * root - Real80{2}), Real80{4} * uni20::numeric_limits<Real80>::epsilon());
  EXPECT_NE(root, Real80(static_cast<double>(root)));
}

TEST(Float80Test, ScalarIoRoundTripsBeyondDouble)
{
  for (Real80 value : {std::nextafter(Real80{1}, Real80{2}), -std::sqrt(Real80{2}),
                       uni20::numeric_limits<Real80>::min(), uni20::numeric_limits<Real80>::max()})
  {
    std::string const text = uni20::format_real(value);
    EXPECT_EQ(uni20::parse_real<Real80>(text), value) << text;
    std::istringstream stream(text);
    Real80 parsed = 0;
    uni20::read_real(stream, parsed);
    ASSERT_TRUE(stream);
    EXPECT_EQ(parsed, value);
  }
  Real80 const value = uni20::parse_real<Real80>("1.000000000000000001");
  EXPECT_NE(value, Real80(static_cast<double>(value)));
  EXPECT_FALSE(uni20::format_complex(uni20::complex<Real80>{value, -value}).empty());
}
#endif
