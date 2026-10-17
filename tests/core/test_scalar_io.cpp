#include <uni20/core/scalar_io.hpp>

#include <gtest/gtest.h>

#include <sstream>

TEST(ScalarIoTest, FormatsBuiltInRealScalars)
{
  uni20::scalar_format_options options;
  options.precision = 4;

  EXPECT_EQ(uni20::format_real(1.0 / 3.0, options), "0.3333");

  options.notation = uni20::real_format_notation::scientific;
  options.precision = 2;
  EXPECT_EQ(uni20::format_real(12.5, options), "1.25e+01");
}

TEST(ScalarIoTest, FormatsBuiltInComplexScalars)
{
  uni20::scalar_format_options options;
  options.precision = 3;

  EXPECT_EQ(uni20::format_complex(uni20::complex<double>{1.25, -3.5}, options), "1.25-3.5i");

  options.notation = uni20::real_format_notation::fixed;
  options.precision = 2;
  EXPECT_EQ(uni20::format_complex(uni20::complex<double>{1.25, -0.0}, options), "1.25+0.00i");
}

TEST(ScalarIoTest, ParsesBuiltInRealScalars)
{
  EXPECT_FLOAT_EQ(uni20::parse_real<float>("1.25"), 1.25f);
  EXPECT_DOUBLE_EQ(uni20::parse_real<double>("-2.5e1"), -25.0);
  EXPECT_THROW((void)uni20::parse_real<double>("1.0x"), std::invalid_argument);
}

TEST(ScalarIoTest, ReadsBuiltInRealScalarsFromStreamToken)
{
  std::istringstream input("3.5");
  double value = 0.0;

  uni20::read_real(input, value);

  EXPECT_TRUE(input);
  EXPECT_DOUBLE_EQ(value, 3.5);
}

#if UNI20_HAS_MPREAL
TEST(ScalarIoTest, ExactRationalsRoundDirectlyToDecimalTiesToEven)
{
  using namespace uni20;
  auto format = [](char const* input, real_format_notation notation, int precision) {
    return format_real(mpreal(input, Precision::exact()),
                       {.precision = precision, .notation = notation, .exact_fractions = false});
  };
  auto fixed = real_format_notation::fixed;
  auto scientific = real_format_notation::scientific;
  auto general = real_format_notation::general;
  EXPECT_EQ(format("1.005", fixed, 2), "1.00");
  EXPECT_EQ(format("1.015", fixed, 2), "1.02");
  EXPECT_EQ(format("-1.005", fixed, 2), "-1.00");
  EXPECT_EQ(format("-1.015", fixed, 2), "-1.02");
  EXPECT_EQ(format("0.0001", fixed, 2), "0.00");
  EXPECT_EQ(format("-0.0001", fixed, 2), "-0.00");
  EXPECT_EQ(format("1/3", fixed, 6), "0.333333");
  EXPECT_EQ(format("9.995", scientific, 2), "1.00e+01");
  EXPECT_EQ(format("1.005", scientific, 2), "1.00e+00");
  EXPECT_EQ(format("999.5", general, 3), "1e+03");
  EXPECT_EQ(format("999.4", general, 3), "999");
  EXPECT_EQ(format("0.00009995", general, 3), "0.0001");
  EXPECT_EQ(format("0.00009994", general, 3), "9.99e-05");
  EXPECT_EQ(format("0", general, 5), "0");
  EXPECT_EQ(format("0", scientific, 3), "0.000e+00");
  EXPECT_EQ(format("1000", general, 6), "1000");
  EXPECT_EQ(format("1e100", scientific, 3), "1.000e+100");
  EXPECT_EQ(format("1e-100", scientific, 3), "1.000e-100");
  EXPECT_EQ(format("42", general, -1), "42");
  EXPECT_EQ(format_real(mpreal("1/3", Precision::exact())), "1/3");
  EXPECT_THROW((void)format("1/3", fixed, -1), std::invalid_argument);
  EXPECT_THROW((void)format("1/3", general, 0), std::invalid_argument);
}

TEST(ScalarIoTest, ExactDecimalFormattingMatchesIndependentDyadicOracle)
{
  using namespace uni20;
  // These rationals are exactly representable in double, so fmt's independent
  // native conversion is an oracle for decimal rounding and notation selection.
  for (int numerator = -257; numerator <= 257; numerator += 7)
    for (int shift : {-20, 0, 20})
      for (auto notation :
           {real_format_notation::general, real_format_notation::fixed, real_format_notation::scientific})
        for (int digits : {1, 2, 6})
        {
          auto exact = mpreal{numerator} / mpreal{8};
          if (shift < 0)
            exact /= mpreal{1 << -shift};
          else
            exact *= mpreal{1 << shift};
          scalar_format_options options{.precision = digits, .notation = notation, .exact_fractions = false};
          EXPECT_EQ(format_real(exact, options), format_real(std::ldexp(numerator / 8.0, shift), options))
              << numerator << " " << shift << " " << digits;
        }
}
#endif
