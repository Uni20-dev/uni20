#include <uni20/krylov/dense_host_vector.hpp>
#include <uni20/krylov/gmres.hpp>

#include <gtest/gtest.h>

namespace
{
using namespace uni20;
using namespace uni20::krylov;
using V = DenseHostVector<double>;
using E = GmresOperatorEstimate<double>;

TEST(GmresDiagnostics, OperatorInformationProducesIndependentErrorDiagnostics)
{
  DenseHostVectorOps<double> ops(2, {4, 0, 0, 2});
  V x{{0, 0.25}}, b{{0, 1}};
  auto info = gmres_with_info<double>(
      ops, ops, x, b,
      {.max_iterations = 0,
       .operator_estimates = {.norm = E{4, GmresEstimateKind::upper_bound}, .inverse_norm = E{0.5}}});
  EXPECT_EQ(info.status, GmresStatus::iteration_limit);
  EXPECT_EQ(info.operator_applications, 1);
  ASSERT_TRUE(info.errors);
  EXPECT_EQ(info.residual->solution_norm, 0.25);
  EXPECT_EQ(info.errors->normwise_backward_error, 0.25);
  EXPECT_EQ(info.errors->absolute_solution_error, 0.25); // Exact error: x* = (0, 0.5).
  EXPECT_EQ(info.errors->relative_solution_error, 1);    // Worst-case kappa*r/b, not the actual error 0.5.
  EXPECT_EQ(info.errors->operator_estimates.norm->kind, GmresEstimateKind::upper_bound);
  EXPECT_EQ(info.errors->operator_estimates.inverse_norm->kind, GmresEstimateKind::estimate);
}

TEST(GmresDiagnostics, OnlyAvailableQuantitiesAreReported)
{
  DenseHostVectorOps<double> ops(1, {2});
  V x{{0}}, b{{1}};
  auto info = gmres_with_info<double>(ops, ops, x, b, {.max_iterations = 0});
  EXPECT_FALSE(info.errors);
  info =
      gmres_with_info<double>(ops, ops, x, b, {.max_iterations = 0, .operator_estimates = {.condition_number = E{1}}});
  ASSERT_TRUE(info.errors);
  EXPECT_FALSE(info.errors->normwise_backward_error);
  EXPECT_FALSE(info.errors->absolute_solution_error);
  EXPECT_EQ(info.errors->relative_solution_error, 1);
  info = gmres_with_info<double>(ops, ops, x, b, {.max_iterations = 0, .operator_estimates = {.inverse_norm = E{0.5}}});
  EXPECT_EQ(info.errors->absolute_solution_error, 0.5);
  EXPECT_FALSE(info.errors->relative_solution_error);
  b.values = {0};
  info = gmres_with_info<double>(ops, ops, x, b, {.operator_estimates = {.norm = E{2}, .inverse_norm = E{0.5}}});
  ASSERT_TRUE(info.succeeded());
  EXPECT_EQ(info.errors->normwise_backward_error, 0);
  EXPECT_EQ(info.errors->absolute_solution_error, 0);
  EXPECT_FALSE(info.errors->relative_solution_error);
}

TEST(GmresDiagnostics, ExplicitConditionEstimateTakesPrecedence)
{
  DenseHostVectorOps<double> ops(1, {2});
  V x{{0}}, b{{1}};
  auto info = gmres_with_info<double>(
      ops, ops, x, b,
      {.max_iterations = 0, .operator_estimates = {.norm = E{4}, .inverse_norm = E{1}, .condition_number = E{1}}});
  EXPECT_EQ(info.errors->relative_solution_error, 1);
  EXPECT_EQ(info.errors->absolute_solution_error, 1);
}

TEST(GmresDiagnostics, DiagnosticScaleAvoidsPrematureOverflowAndUnderflow)
{
  // A=diag(1e300,1): r=(0,1e-300), b=(1e300,1e-300).
  DenseHostVectorOps<double> ops(2, {1e300, 0, 0, 1});
  V x{{1, 0}}, b{{1e300, 1e-300}};
  auto info = gmres_with_info<double>(
      ops, ops, x, b,
      {.relative_tolerance = 0, .max_iterations = 0, .operator_estimates = {.norm = E{1e300}, .inverse_norm = E{1}}});
  ASSERT_TRUE(info.errors);
  EXPECT_NEAR(*info.errors->relative_solution_error / 1e-300, 1, 1e-15);
  EXPECT_EQ(info.errors->absolute_solution_error, 1e-300);

  // The iterate lies in the small-singular-value direction, so A*x is finite
  // even though the diagnostic denominator ||A||*||x|| exceeds double's range.
  DenseHostVectorOps<double> large(2, {1e200, 0, 0, 1});
  x.values = {0, 1e200};
  b.values = {0, 0};
  info = gmres_with_info<double>(large, large, x, b, {.max_iterations = 0, .operator_estimates = {.norm = E{1e200}}});
  EXPECT_NEAR(*info.errors->normwise_backward_error / 1e-200, 1, 1e-15);
  DenseHostVectorOps<double> ill_conditioned(2, {1e200, 0, 0, 1e-200});
  x.values = {1, 0};
  b.values = {1e200, 1};
  info = gmres_with_info<double>(ill_conditioned, ill_conditioned, x, b,
                                 {.relative_tolerance = 0,
                                  .max_iterations = 0,
                                  .operator_estimates = {.norm = E{1e200}, .inverse_norm = E{1e200}}});
  EXPECT_NEAR(*info.errors->relative_solution_error / 1e200, 1, 1e-15);
}

TEST(GmresDiagnostics, OverflowingErrorEstimateDoesNotInvalidateSolve)
{
  DenseHostVectorOps<double> ops(1, {1});
  V x{{0}}, b{{2}};
  auto info =
      gmres_with_info<double>(ops, ops, x, b, {.max_iterations = 0, .operator_estimates = {.inverse_norm = E{1e308}}});
  EXPECT_EQ(info.status, GmresStatus::iteration_limit);
  EXPECT_TRUE(math::isinf(*info.errors->absolute_solution_error));
  EXPECT_EQ(info.residual->norm, 2);
}

TEST(GmresDiagnostics, InvalidEstimatesFailBeforeOperatorApplication)
{
  DenseHostVectorOps<double> ops(1, {2});
  V x{{0.5}}, b{{1}};
  for (auto properties :
       {GmresOperatorEstimates<double>{.norm = E{-1}}, GmresOperatorEstimates<double>{.inverse_norm = E{0}},
        GmresOperatorEstimates<double>{.condition_number = E{0.5}},
        GmresOperatorEstimates<double>{.norm = E{numeric_limits<double>::infinity()}}})
    EXPECT_THROW(gmres_with_info<double>(ops, ops, x, b, {.operator_estimates = properties}), std::invalid_argument);
  EXPECT_EQ(ops.matvec_count(), 0);
}

#if UNI20_HAS_MPREAL
TEST(GmresDiagnostics, ConvertingAKnownUpperBoundDoesNotRoundItDown)
{
  auto p = Precision::bits(80);
  mpreal value("3/2", Precision::bits(256));
  value += ldexp(mpreal(1, value.precision()), -100);
  EXPECT_LT(value.at(p), value);
  auto properties = gmres_detail::operator_estimates_at_precision(
      GmresOperatorEstimates<mpreal>{.norm = GmresOperatorEstimate<mpreal>{value, GmresEstimateKind::upper_bound}}, p);
  EXPECT_GE(properties.norm->value, value);
  EXPECT_EQ(properties.norm->value.precision(), p);
}
#endif
} // namespace
