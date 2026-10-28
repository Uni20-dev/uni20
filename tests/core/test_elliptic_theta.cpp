#include <gtest/gtest.h>
#include <uni20/core/elliptic_theta.hpp>

namespace
{
using namespace uni20;
TEST(ThetaConstant, DomainAndBudget)
{
  for (unsigned kind : {0u, 1u, 5u})
    EXPECT_THROW(elliptic_theta_constant(kind, 1.0), std::invalid_argument);
  for (double t : {0.0, -1.0, numeric_limits<double>::infinity(), numeric_limits<double>::quiet_NaN()})
    EXPECT_THROW(elliptic_theta_constant(3, t), std::invalid_argument);
  EXPECT_THROW(elliptic_theta_constant(3, 1.0, 0), std::invalid_argument);
  EXPECT_THROW(elliptic_theta_constant(3, 1.0, 1), std::runtime_error);
  EXPECT_THROW(elliptic_theta_constant(3, 1.0, Precision::bits(64)), std::invalid_argument);
  EXPECT_THROW(elliptic_theta_constant(3, 1.0, Precision::exact()), std::invalid_argument);
  auto result = elliptic_theta_constant(3, 1.0, Precision::bits(53));
  EXPECT_EQ(result.scaled_value, elliptic_theta_constant(3, 1.0).scaled_value);
}

template <class R> class ThetaNative : public testing::Test {};
using NativeReals = testing::Types<float, double, long double
#if UNI20_HAS_FLOAT128
                                   ,
                                   float128
#endif
                                   >;
TYPED_TEST_SUITE(ThetaNative, NativeReals);
TYPED_TEST(ThetaNative, ScaledRangeWithoutIntermediateOverflow)
{
  using R = TypeParam;
  using L = numeric_limits<R>;
  auto large = elliptic_theta_constant(2, L::max());
  EXPECT_EQ(large.scaled_value, R(2));
  EXPECT_TRUE(uni20::isfinite(large.log_scale));
  EXPECT_LT(large.log_scale, R(0));
  EXPECT_EQ(math::exp(large.log_scale), R(0));
  for (unsigned kind : {2u, 3u})
  {
    auto small = elliptic_theta_constant(kind, L::denorm_min());
    EXPECT_EQ(small.scaled_value, R(1));
    EXPECT_TRUE(uni20::isfinite(small.log_scale));
    EXPECT_GT(small.log_scale, R(0));
  }
  EXPECT_THROW(elliptic_theta_constant(4, L::denorm_min()), std::overflow_error);
  // 1/t overflows here, but pi/(4*t) remains representable.
  R const t = R(1) / L::max();
  auto tiny = elliptic_theta_constant(4, t);
  EXPECT_EQ(tiny.scaled_value, R(2));
  EXPECT_TRUE(uni20::isfinite(tiny.log_scale));
}

#if UNI20_HAS_MPREAL
TEST(ThetaConstant, ExplicitMpfrPrecisionAndRange)
{
  auto const p = Precision::bits(256);
  EXPECT_THROW(elliptic_theta_constant(3, mpreal(1)), std::invalid_argument);
  EXPECT_THROW(elliptic_theta_constant(3, mpreal{}), std::logic_error);
  for (int bits : {2, 4, 128, 256, 333})
  {
    auto const requested = Precision::bits(bits);
    auto result = elliptic_theta_constant(3, mpreal(1), requested);
    EXPECT_EQ(result.scaled_value.precision(), requested);
    EXPECT_EQ(result.log_scale.precision(), requested);
    EXPECT_GT(result.scaled_value, 0);
  }
  for (auto const& t : {mpreal("1e-7000", p), mpreal("1e7000", p)})
    for (unsigned kind : {2u, 3u, 4u})
    {
      auto result = elliptic_theta_constant(kind, t);
      EXPECT_EQ(result.scaled_value.precision(), p);
      EXPECT_EQ(result.log_scale.precision(), p);
      EXPECT_TRUE(uni20::isfinite(result.log_scale));
      EXPECT_GT(result.scaled_value, 0);
    }
}

TEST(ThetaConstant, MpfrAccuracyBeyondBinary128)
{
  auto const reference_precision = Precision::bits(1536);
  auto one = mpreal(1, reference_precision);
  auto reference =
      math::tgamma(one / 4) / (math::sqrt(2 * one) * math::pow(pi<mpreal>.at(reference_precision), 3 * one / 4));
  auto previous_error = mpreal(1, reference_precision);
  auto const default_precision = mpfr_get_default_prec();
  auto const emin = mpfr_get_emin(), emax = mpfr_get_emax();
  for (int bits : {128, 256, 333, 1024})
  {
    auto const p = Precision::bits(bits);
    auto result = elliptic_theta_constant(3, mpreal(1, p));
    auto error = math::abs(result.scaled_value.at(reference_precision) - reference);
    EXPECT_LE(error, numeric_limits<mpreal>::epsilon(mpreal(1, p)).at(reference_precision));
    EXPECT_LT(error, previous_error / 1000);
    previous_error = error;
    EXPECT_EQ(result.scaled_value.precision(), p);
    EXPECT_EQ(result.log_scale.precision(), p);
  }
  EXPECT_EQ(mpfr_get_default_prec(), default_precision);
  EXPECT_EQ(mpfr_get_emin(), emin);
  EXPECT_EQ(mpfr_get_emax(), emax);
}
#endif
} // namespace
