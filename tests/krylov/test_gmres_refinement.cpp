#include <uni20/krylov/dense_host_vector.hpp>
#include <uni20/krylov/gmres_refinement.hpp>

#include <gtest/gtest.h>

namespace
{
using namespace uni20;
using namespace uni20::krylov;
using V = DenseHostVector<double>;
using IV = DenseHostVector<float>;

TEST(GmresRefinement, FloatCorrectionsRecoverDoubleAccuracy)
{
  DenseHostVectorOps<double> ops(2, {3, 1, 1, 2});
  DenseHostVectorOps<float> inner_ops(2, {3, 1, 1, 2});
  V x{{0, 0}}, b{{1, 0}};
  IV inner{{0, 0}};
  IdentityPreconditioner preconditioner{inner_ops};
  DenseHostVectorTransfer transfer;
  GmresRefinementContext context{.working = krylov_space<double>(ops, x),
                                 .inner = krylov_space<float>(inner_ops, inner),
                                 .residual = krylov_space<double>(ops, b),
                                 .inner_operator = inner_ops,
                                 .residual_operator = ops,
                                 .preconditioner = preconditioner,
                                 .transfer = transfer};
  auto info = gmres_refine_with_info(context, x, b);
  ASSERT_TRUE(info.succeeded());
  EXPECT_GT(info.corrections, 1);
  EXPECT_NEAR(x.values[0], 0.4, 1e-16);
  EXPECT_NEAR(x.values[1], -0.2, 1e-16);
  EXPECT_LE(info.residual->norm, numeric_limits<double>::epsilon());
}

TEST(GmresRefinement, NormalizesBeforeNarrowingTinyResidual)
{
  DenseHostVectorOps<double> ops(2, {1, 0, 0, 1});
  DenseHostVectorOps<float> inner_ops(2, {1, 0, 0, 1});
  V x{{0, 0}}, b{{1e-300, 2e-300}};
  IV inner{{0, 0}};
  IdentityPreconditioner preconditioner{inner_ops};
  DenseHostVectorTransfer transfer;
  GmresRefinementContext context{.working = krylov_space<double>(ops, x),
                                 .inner = krylov_space<float>(inner_ops, inner),
                                 .residual = krylov_space<double>(ops, b),
                                 .inner_operator = inner_ops,
                                 .residual_operator = ops,
                                 .preconditioner = preconditioner,
                                 .transfer = transfer};
  auto info = gmres_refine_with_info(context, x, b);
  ASSERT_TRUE(info.succeeded());
  EXPECT_DOUBLE_EQ(x.values[0], b.values[0]);
  EXPECT_DOUBLE_EQ(x.values[1], b.values[1]);
  EXPECT_GT(info.corrections, 0);
}

TEST(GmresRefinement, InitialGuessAndInnerFailureAreDistinct)
{
  DenseHostVectorOps<double> ops(2, {2, 0, 0, 4});
  DenseHostVectorOps<float> inner_ops(2, {2, 0, 0, 4});
  V x{{0.5, 0.25}}, b{{1, 1}};
  IV inner{{0, 0}};
  IdentityPreconditioner preconditioner{inner_ops};
  DenseHostVectorTransfer transfer;
  GmresRefinementContext context{.working = krylov_space<double>(ops, x),
                                 .inner = krylov_space<float>(inner_ops, inner),
                                 .residual = krylov_space<double>(ops, b),
                                 .inner_operator = inner_ops,
                                 .residual_operator = ops,
                                 .preconditioner = preconditioner,
                                 .transfer = transfer};
  auto info = gmres_refine_with_info(context, x, b);
  EXPECT_TRUE(info.succeeded());
  EXPECT_EQ(info.corrections, 0);
  EXPECT_FALSE(info.last_inner);
  x.values = {0.4, 0.25};
  info = gmres_refine_with_info(context, x, b, {.inner = {.max_iterations = 0}});
  EXPECT_EQ(info.status, GmresRefinementStatus::inner_failed);
  ASSERT_TRUE(info.last_inner);
  EXPECT_EQ(info.last_inner->status, GmresStatus::iteration_limit);
  EXPECT_EQ(x.values, (std::vector<double>{0.4, 0.25}));
  EXPECT_NEAR(info.residual->norm, 0.2, 1e-15);
}

TEST(GmresRefinement, ResidualFailureDoesNotPublishCandidate)
{
  DenseHostVectorOps<double> ops(2, {2, 0, 0, 4});
  DenseHostVectorOps<float> inner_ops(2, {2, 0, 0, 4});
  struct FailingOperator
  {
      DenseHostVectorOps<double>& ops;
      int calls = 0;
      void matvec(V& out, V const& in)
      {
        if (++calls == 2)
          out.values = {numeric_limits<double>::quiet_NaN(), 0};
        else
          ops.matvec(out, in);
      }
  } residual_op{ops};
  V x{{0.4, 0.25}}, b{{1, 1}};
  IV inner{{0, 0}};
  IdentityPreconditioner preconditioner{inner_ops};
  DenseHostVectorTransfer transfer;
  GmresRefinementContext context{.working = krylov_space<double>(ops, x),
                                 .inner = krylov_space<float>(inner_ops, inner),
                                 .residual = krylov_space<double>(ops, b),
                                 .inner_operator = inner_ops,
                                 .residual_operator = residual_op,
                                 .preconditioner = preconditioner,
                                 .transfer = transfer};
  auto info = gmres_refine_with_info(context, x, b,
                                     {.operator_estimates = {.norm = GmresOperatorEstimate<double>{4},
                                                             .inverse_norm = GmresOperatorEstimate<double>{0.5}}});
  EXPECT_EQ(info.status, GmresRefinementStatus::nonfinite_result);
  ASSERT_TRUE(info.last_inner->succeeded());
  EXPECT_EQ(x.values, (std::vector<double>{0.4, 0.25}));
  EXPECT_NEAR(info.residual->norm, 0.2, 1e-15);
  ASSERT_TRUE(info.errors);
  EXPECT_NEAR(*info.errors->absolute_solution_error, 0.1, 1e-15);
  EXPECT_NEAR(info.residual->solution_norm, std::hypot(0.4, 0.25), 1e-15);
}

TEST(GmresRefinement, InvalidInnerOptionsFailBeforeAnAlreadySolvedGuess)
{
  DenseHostVectorOps<double> ops(1, {2});
  V x{{0.5}}, b{{1}};
  IdentityPreconditioner preconditioner{ops};
  DenseHostVectorTransfer transfer;
  auto space = krylov_space<double>(ops, x);
  GmresRefinementContext context{space, space, space, ops, ops, preconditioner, transfer};
  EXPECT_THROW(gmres_refine_with_info(context, x, b, {.inner = {.restart = 0}}), std::invalid_argument);
  EXPECT_THROW(gmres_refine_with_info(context, x, b, {.inner = {.relative_tolerance = -1}}), std::invalid_argument);
  EXPECT_THROW(
      gmres_refine_with_info(
          context, x, b, {.inner = {.operator_estimates = {.condition_number = GmresOperatorEstimate<double>{0.5}}}}),
      std::invalid_argument);
  EXPECT_EQ(ops.matvec_count(), 0);
  EXPECT_EQ(x.values[0], 0.5);
}

TEST(GmresRefinement, OuterLimitRetainsItsOwnResidualAndLastIterate)
{
  DenseHostVectorOps<double> ops(2, {3, 1, 1, 2});
  DenseHostVectorOps<float> inner_ops(2, {3, 1, 1, 2});
  V x{{0, 0}}, b{{1, 0}};
  IV inner{{0, 0}};
  IdentityPreconditioner preconditioner{inner_ops};
  DenseHostVectorTransfer transfer;
  GmresRefinementContext context{.working = krylov_space<double>(ops, x),
                                 .inner = krylov_space<float>(inner_ops, inner),
                                 .residual = krylov_space<double>(ops, b),
                                 .inner_operator = inner_ops,
                                 .residual_operator = ops,
                                 .preconditioner = preconditioner,
                                 .transfer = transfer};
  auto info = gmres_refine_with_info(context, x, b, {.max_corrections = 0});
  EXPECT_EQ(info.status, GmresRefinementStatus::iteration_limit);
  EXPECT_FALSE(info.last_inner);
  EXPECT_EQ(info.residual->norm, 1);
  EXPECT_EQ(x.values, (std::vector<double>{0, 0}));
  info = gmres_refine_with_info(context, x, b, {.max_corrections = 1});
  EXPECT_EQ(info.status, GmresRefinementStatus::iteration_limit);
  ASSERT_TRUE(info.last_inner);
  EXPECT_TRUE(info.last_inner->succeeded());
  EXPECT_GT(info.residual->norm, numeric_limits<double>::epsilon());
  EXPECT_LT(info.residual->norm, 1e-6);
  EXPECT_NEAR(x.values[0], 0.4, 1e-7);
}

#if UNI20_HAS_MPREAL
TEST(GmresRefinement, RuntimeSpacesUseIndependentPrecisions)
{
  using M = DenseHostVector<mpreal>;
  auto pw = Precision::bits(128), pi = Precision::bits(40), pr = Precision::bits(256);
  auto matrix = [](Precision p) { return std::vector<mpreal>{mpreal(3, p), mpreal(1, p), mpreal(1, p), mpreal(2, p)}; };
  DenseHostVectorOps<mpreal> wops(2, matrix(pw)), iops(2, matrix(pi)), rops(2, matrix(pr));
  M x{{mpreal(0, pw), mpreal(0, pw)}}, inner{{mpreal(0, pi), mpreal(0, pi)}}, b{{mpreal(1, pr), mpreal(0, pr)}};
  IdentityPreconditioner preconditioner{iops};
  DenseHostVectorTransfer transfer;
  GmresRefinementContext context{.working = krylov_space<mpreal>(wops, x, pw),
                                 .inner = krylov_space<mpreal>(iops, inner, pi),
                                 .residual = krylov_space<mpreal>(rops, b, pr),
                                 .inner_operator = iops,
                                 .residual_operator = rops,
                                 .preconditioner = preconditioner,
                                 .transfer = transfer};
  auto info =
      gmres_refine_with_info(context, x, b,
                             {.operator_estimates = {.norm = GmresOperatorEstimate<mpreal>{mpreal(4)},
                                                     .inverse_norm = GmresOperatorEstimate<mpreal>{mpreal(1)}}});
  ASSERT_TRUE(info.succeeded());
  EXPECT_EQ(x.values[0].precision(), pw);
  EXPECT_EQ(info.last_inner->residual->norm.precision(), pi);
  EXPECT_EQ(info.residual->norm.precision(), pr);
  ASSERT_TRUE(info.errors);
  EXPECT_EQ(info.errors->normwise_backward_error->precision(), pr);
  EXPECT_EQ(info.errors->absolute_solution_error->precision(), pr);
  EXPECT_EQ(info.errors->relative_solution_error->precision(), pr);
  EXPECT_TRUE(abs(x.values[0] - mpreal("2/5", pw)) <= numeric_limits<mpreal>::epsilon(pw));
}
#endif
} // namespace
