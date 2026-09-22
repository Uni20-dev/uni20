#include <uni20/core/numeric_limits.hpp>
#include <uni20/core/scalar_precision.hpp>

#include <gtest/gtest.h>
#include <string>

namespace
{

TEST(ScalarPrecisionTest, ParsesStablePrecisionNames)
{
  EXPECT_EQ(uni20::parse_scalar_precision("fp32"), uni20::ScalarPrecision::fp32);
  EXPECT_EQ(uni20::parse_scalar_precision("fp64"), uni20::ScalarPrecision::fp64);
  EXPECT_EQ(uni20::parse_scalar_precision("fp80"), uni20::ScalarPrecision::fp80);
  EXPECT_EQ(uni20::parse_scalar_precision("fp128"), uni20::ScalarPrecision::fp128);
  EXPECT_EQ(uni20::parse_scalar_precision("double"), std::nullopt);
}

TEST(ScalarPrecisionTest, ReportsConfiguredPrecisions)
{
  auto const configured = uni20::configured_scalar_precisions();
  ASSERT_GE(configured.size(), 2);
  EXPECT_EQ(configured[0], uni20::ScalarPrecision::fp32);
  EXPECT_EQ(configured[1], uni20::ScalarPrecision::fp64);
  EXPECT_EQ(configured.size(), 2 + uni20::has_float80 + uni20::has_float128);
  EXPECT_EQ(uni20::scalar_precision_is_available(uni20::ScalarPrecision::fp80), uni20::has_float80);
  EXPECT_EQ(uni20::scalar_precision_is_available(uni20::ScalarPrecision::fp128), uni20::has_float128);
  std::string choices;
  for (auto precision : configured)
  {
    EXPECT_TRUE(uni20::scalar_precision_is_available(precision));
    auto const name = uni20::scalar_precision_name(precision);
    EXPECT_EQ(uni20::parse_scalar_precision(name), precision);
    if (!choices.empty()) choices += '|';
    choices += name;
  }
  EXPECT_EQ(choices, uni20::configured_scalar_precision_choices());
  if (uni20::has_float80)
  {
    EXPECT_EQ(configured[2], uni20::ScalarPrecision::fp80);
  }
  if (uni20::has_float128)
  {
    EXPECT_EQ(configured.back(), uni20::ScalarPrecision::fp128);
  }
}

TEST(ScalarPrecisionTest, VisitsConfiguredConcreteTypes)
{
  auto type_size = []<typename Scalar>() { return sizeof(Scalar); };
  EXPECT_EQ(uni20::visit_scalar_precision(uni20::ScalarPrecision::fp32, type_size), sizeof(uni20::float32));
  EXPECT_EQ(uni20::visit_scalar_precision(uni20::ScalarPrecision::fp64, type_size), sizeof(uni20::float64));

  if (uni20::has_float80)
  {
    auto significand_bits = []<typename Scalar>() { return uni20::numeric_limits<Scalar>::digits; };
    EXPECT_EQ(uni20::visit_scalar_precision(uni20::ScalarPrecision::fp80, significand_bits), 64);
    EXPECT_EQ(uni20::visit_scalar_precision(uni20::ScalarPrecision::fp80, type_size), sizeof(long double));
  }
  else
  {
    EXPECT_THROW(uni20::visit_scalar_precision(uni20::ScalarPrecision::fp80, type_size), std::invalid_argument);
  }
  if (uni20::has_float128)
  {
    EXPECT_GT(uni20::visit_scalar_precision(uni20::ScalarPrecision::fp128, type_size), sizeof(uni20::float64));
  }
  else
  {
    EXPECT_THROW(uni20::visit_scalar_precision(uni20::ScalarPrecision::fp128, type_size), std::invalid_argument);
  }
}

TEST(ScalarPrecisionTest, RejectsInvalidIdentifier)
{
  auto const invalid = static_cast<uni20::ScalarPrecision>(-1);
  EXPECT_EQ(uni20::scalar_precision_name(invalid), "unknown");
  EXPECT_FALSE(uni20::scalar_precision_is_available(invalid));
  EXPECT_THROW(uni20::visit_scalar_precision(invalid, []<typename Scalar>() { return sizeof(Scalar); }),
               std::invalid_argument);
}

} // namespace
