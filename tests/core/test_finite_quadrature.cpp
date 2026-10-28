#include <gtest/gtest.h>
#include <uni20/core/finite_quadrature.hpp>

namespace
{
using namespace uni20;
using Status = quadrature_status;
auto constant = [](quadrature_point<double> const&) { return std::array{1.0}; };

TEST(FiniteQuadrature, InvalidRequestsDoNotEvaluate)
{
  tanh_sinh_options<double> options;
  auto never = [](auto const&) -> std::array<double, 1> {
    ADD_FAILURE();
    return {1};
  };
  EXPECT_THROW(tanh_sinh_integrate(never, 2.0, 1.0, options), std::invalid_argument);
  EXPECT_THROW(tanh_sinh_integrate(never, 0.0, numeric_limits<double>::infinity(), options), std::invalid_argument);
  EXPECT_THROW(tanh_sinh_integrate(never, numeric_limits<double>::quiet_NaN(), 1.0, options), std::invalid_argument);
  options.relative_tolerance = -1;
  EXPECT_THROW(tanh_sinh_integrate(never, 0.0, 1.0, options), std::invalid_argument);
  options.relative_tolerance = 0;
  EXPECT_THROW(tanh_sinh_integrate(never, 0.0, 1.0, options), std::invalid_argument);
  options.absolute_tolerance[0] = 1e-10;
  options.endpoint_power = 1;
  EXPECT_THROW(tanh_sinh_integrate(never, 0.0, 1.0, options), std::invalid_argument);
  EXPECT_THROW((void)tanh_sinh_options<double>(Precision::bits(128)), std::invalid_argument);
}

TEST(FiniteQuadrature, BudgetsRetainLastCompleteMesh)
{
  tanh_sinh_options<double> options;
  options.max_evaluations = 0;
  auto a = tanh_sinh_integrate(constant, 0.0, 1.0, options);
  EXPECT_EQ(a.status, Status::evaluation_limit);
  EXPECT_EQ(a.evaluations, 0u);
  EXPECT_EQ(a.levels, 0u);
  EXPECT_EQ(a.value[0], 0);
  EXPECT_EQ(a.error[0], numeric_limits<double>::infinity());
  options.max_evaluations = 200000;
  options.max_levels = 0;
  a = tanh_sinh_integrate(constant, 0.0, 1.0, options);
  EXPECT_EQ(a.status, Status::refinement_limit);
  EXPECT_EQ(a.evaluations, 0u);
  options.max_levels = 1;
  a = tanh_sinh_integrate(constant, 0.0, 1.0, options);
  EXPECT_EQ(a.status, Status::refinement_limit);
  EXPECT_EQ(a.levels, 1u);
  EXPECT_EQ(a.error[0], numeric_limits<double>::infinity());
  options.max_levels = 12;
  options.max_evaluations = a.evaluations + 2;
  auto b = tanh_sinh_integrate(constant, 0.0, 1.0, options);
  EXPECT_EQ(b.status, Status::evaluation_limit);
  EXPECT_EQ(b.evaluations, options.max_evaluations);
  EXPECT_EQ(b.levels, a.levels);
  EXPECT_EQ(b.value, a.value);
  EXPECT_EQ(b.absolute_integral, a.absolute_integral);
  EXPECT_EQ(b.endpoint_indicator, a.endpoint_indicator);
  options.max_evaluations = 200000;
  std::size_t calls = 0;
  auto interrupted = [&](auto const&) {
    return std::array{++calls > a.evaluations + 1 ? numeric_limits<double>::infinity() : 1.0};
  };
  b = tanh_sinh_integrate(interrupted, 0.0, 1.0, options);
  EXPECT_EQ(b.status, Status::nonfinite_integrand);
  EXPECT_EQ(b.levels, a.levels);
  EXPECT_EQ(b.evaluations, a.evaluations + 2);
  EXPECT_EQ(b.value, a.value);
  EXPECT_EQ(b.absolute_integral, a.absolute_integral);
  EXPECT_EQ(b.endpoint_indicator, a.endpoint_indicator);
}

TEST(FiniteQuadrature, CallbackAndArithmeticFailures)
{
  tanh_sinh_options<double> options;
  auto bad = [](auto const&) { return std::array{numeric_limits<double>::quiet_NaN()}; };
  auto result = tanh_sinh_integrate(bad, 0.0, 1.0, options);
  EXPECT_EQ(result.status, Status::nonfinite_integrand);
  EXPECT_EQ(result.evaluations, 1u);
  auto throws = [](auto const&) -> std::array<double, 1> { throw std::runtime_error("callback"); };
  EXPECT_THROW(tanh_sinh_integrate(throws, 0.0, 1.0, options), std::runtime_error);
  auto huge = [](auto const&) { return std::array{numeric_limits<double>::max()}; };
  EXPECT_EQ(tanh_sinh_integrate(huge, 0.0, 4.0, options).status, Status::arithmetic_range);
  double const maximum = numeric_limits<double>::max();
  EXPECT_EQ(tanh_sinh_integrate(constant, -maximum, maximum, options).status, Status::arithmetic_range);
  EXPECT_EQ(tanh_sinh_integrate(constant, 0.0, numeric_limits<double>::denorm_min(), options).status,
            Status::arithmetic_range);
}

TEST(FiniteQuadrature, RoundedEndpointRequiresDistanceInput)
{
  tanh_sinh_options<double> options;
  bool rounded = false;
  auto stable = [&](auto const& point) {
    rounded |= point.value == 2.0;
    EXPECT_GT(point.upper_distance, 0);
    return std::array{1.0 / math::sqrt(point.upper_distance)};
  };
  auto result = tanh_sinh_integrate(stable, 1.0, 2.0, options);
  EXPECT_TRUE(rounded);
  EXPECT_EQ(result.status, Status::converged);
  EXPECT_LE(math::abs(result.value[0] - 2), options.relative_tolerance * 2);
  auto unstable = [](auto const& point) { return std::array{1.0 / math::sqrt(2 - point.value)}; };
  EXPECT_EQ(tanh_sinh_integrate(unstable, 1.0, 2.0, options).status, Status::nonfinite_integrand);
}

TEST(FiniteQuadrature, SubnormalIntegralCannotClaimFullRelativeAccuracy)
{
  tanh_sinh_options<double> options;
  options.max_levels = 12;
  double const tiny = numeric_limits<double>::denorm_min() * 32;
  auto f = [tiny](auto const&) { return std::array{tiny}; };
  auto result = tanh_sinh_integrate(f, 0.0, 1.0, options);
  EXPECT_EQ(result.status, Status::refinement_limit);
  EXPECT_GT(result.error[0], 0);
  EXPECT_GT(result.error[0], options.relative_tolerance * result.value[0]);
}

TEST(FiniteQuadrature, StrongSingularityNeedsDeeperEndpoints)
{
  tanh_sinh_options<double> options;
  options.max_levels = 16;
  options.endpoint_power = 2;
  auto f = [](auto const& x) { return std::array{1.0 / math::sqrt(x.lower_distance * math::sqrt(x.lower_distance))}; };
  auto truncated = tanh_sinh_integrate(f, 0.0, 1.0, options);
  EXPECT_EQ(truncated.status, Status::endpoint_limit);
  EXPECT_GT(truncated.error[0], options.relative_tolerance * math::abs(truncated.value[0]));
  options.endpoint_power = 6;
  auto deep = tanh_sinh_integrate(f, 0.0, 1.0, options);
  EXPECT_EQ(deep.status, Status::converged);
  EXPECT_LE(math::abs(deep.value[0] - 4), options.relative_tolerance * 4);
}

#if UNI20_HAS_MPREAL
TEST(FiniteQuadratureMpfr, LowPrecisionCannotStallTheMesh)
{
  for (int bits : {1, 2, 3, 4})
  {
    SCOPED_TRACE(bits);
    auto p = Precision::bits(bits);
    tanh_sinh_options<mpreal> options(p);
    options.relative_tolerance = numeric_limits<mpreal>::epsilon(p) / 64;
    options.max_evaluations = 1000;
    auto f = [p](auto const&) { return std::array{mpreal(1, p)}; };
    auto result = tanh_sinh_integrate(f, mpreal(0, p), mpreal(1, p), options);
    EXPECT_EQ(result.status, Status::arithmetic_range);
    EXPECT_LT(result.evaluations, options.max_evaluations);
    EXPECT_EQ(result.value[0].precision(), p);
  }
}

TEST(FiniteQuadratureMpfr, PrecisionFailuresAndTypedEmptyResults)
{
  auto p = Precision::bits(128), q = Precision::bits(256);
  using R = mpreal;
  tanh_sinh_options<R> options(p);
  auto f = [p](auto const&) { return std::array{R(1, p)}; };
  EXPECT_THROW((void)tanh_sinh_options<R>(Precision::exact()), std::invalid_argument);
  EXPECT_THROW(tanh_sinh_integrate(f, R(0), R(1, p), options), std::invalid_argument);
  EXPECT_THROW(tanh_sinh_integrate(f, R(0, p), R(1, q), options), std::invalid_argument);
  EXPECT_THROW(tanh_sinh_integrate(f, R(uninitialized), R(1, p), options), std::logic_error);
  options.absolute_tolerance[0] = R(0);
  EXPECT_THROW(tanh_sinh_integrate(f, R(0, p), R(1, p), options), std::invalid_argument);
  options.absolute_tolerance[0] = R(0, p);
  auto exact = [](auto const&) { return std::array{R(1)}; };
  EXPECT_THROW(tanh_sinh_integrate(exact, R(0, p), R(1, p), options), std::invalid_argument);
  auto wrong = [q](auto const&) { return std::array{R(1, q)}; };
  EXPECT_THROW(tanh_sinh_integrate(wrong, R(0, p), R(1, p), options), std::invalid_argument);
  options.max_evaluations = 0;
  auto empty = tanh_sinh_integrate(f, R(2, p), R(2, p), options);
  EXPECT_EQ(empty.status, Status::converged);
  EXPECT_EQ(empty.evaluations, 0u);
  EXPECT_EQ(empty.value[0], 0);
  EXPECT_EQ(empty.error[0], 0);
  auto failed = tanh_sinh_integrate(f, R(0, p), R(1, p), options);
  EXPECT_EQ(failed.status, Status::evaluation_limit);
  for (auto const& r : {empty, failed})
    for (auto const& a : {r.value, r.error, r.absolute_integral, r.endpoint_indicator})
      EXPECT_EQ(a[0].precision(), p);
}

TEST(FiniteQuadratureMpfr, PrecisionGrowthAndExtendedExponents)
{
  auto const oracle_p = Precision::bits(1536);
  auto reference = math::log(mpreal(2, oracle_p));
  auto last_error = mpreal(1, oracle_p);
  auto const ambient = mpfr_get_default_prec();
  for (int bits : {128, 256, 333, 1024})
  {
    auto p = Precision::bits(bits);
    tanh_sinh_options<mpreal> options(p);
    options.max_levels = 16;
    for (char const* text : {"1", "1e-7000", "1e7000"})
    {
      auto scale = mpreal(text, p), one = mpreal(1, p);
      auto f = [&](auto const& x) { return std::array{one / (scale + x.value)}; };
      auto result = tanh_sinh_integrate(f, mpreal(0, p), scale, options);
      ASSERT_EQ(result.status, Status::converged) << "bits=" << bits << " scale=" << text;
      auto error = math::abs(result.value[0].at(oracle_p) - reference);
      EXPECT_LE(error, numeric_limits<mpreal>::epsilon(p).at(oracle_p) * 16);
      if (text == std::string_view("1"))
      {
        EXPECT_LT(error * 1024, last_error);
        last_error = error;
      }
      EXPECT_EQ(result.value[0].precision(), p);
      EXPECT_EQ(result.error[0].precision(), p);
    }
  }
  EXPECT_EQ(mpfr_get_default_prec(), ambient);
}
#endif
} // namespace
