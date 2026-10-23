#include <uni20/krylov/dense_host_vector.hpp>
#include <uni20/krylov/gmres.hpp>

#include <gtest/gtest.h>

namespace
{
using namespace uni20;
using namespace uni20::krylov;
using V = DenseHostVector<double>;

TEST(Gmres, SolvesZeroLeadingRotationWithoutHanging)
{
  DenseHostVectorOps<double> ops(2, {0, -1, 1, 0});
  V x{{0, 0}}, b{{1, 0}};
  auto info = gmres_with_info<double>(ops, ops, x, b, {.relative_tolerance = 1e-13, .restart = 2, .max_iterations = 4});
  ASSERT_TRUE(info.succeeded());
  EXPECT_EQ(info.iterations, 2);
  EXPECT_NEAR(x.values[0], 0, 1e-14);
  EXPECT_NEAR(x.values[1], -1, 1e-14);
  EXPECT_LE(info.residual->norm, 1e-13);
}

TEST(Gmres, AlreadySolvedGuessNeedsNoArnoldiSteps)
{
  DenseHostVectorOps<double> ops(2, {2, 0, 0, 4});
  V x{{0.5, 0.25}}, b{{1, 1}};
  auto info = gmres_with_info<double>(ops, ops, x, b, {.relative_tolerance = 1e-13});
  ASSERT_TRUE(info.succeeded());
  EXPECT_EQ(info.iterations, 0);
  EXPECT_EQ(info.cycles, 0);
  EXPECT_EQ(info.operator_applications, 1);
  EXPECT_EQ(x.values, (std::vector<double>{0.5, 0.25}));
}

TEST(Gmres, PreservesInitialGuessAtZeroIterationBudget)
{
  DenseHostVectorOps<double> ops(2, {2, 0, 0, 4});
  V x{{0.4, 0.25}}, b{{1, 1}};
  auto info = gmres_with_info<double>(ops, ops, x, b, {.relative_tolerance = 1e-13, .max_iterations = 0});
  EXPECT_EQ(info.status, GmresStatus::iteration_limit);
  EXPECT_EQ(info.iterations, 0);
  EXPECT_EQ(x.values, (std::vector<double>{0.4, 0.25}));
  ASSERT_TRUE(info.residual);
  EXPECT_NEAR(info.residual->norm, 0.2, 1e-15);
}

TEST(Gmres, RestartsAndCountsArnoldiStepsExactly)
{
  DenseHostVectorOps<double> ops(2, {4, 1, 0, 2});
  V x{{0, 0}}, b{{1, 1}};
  auto info =
      gmres_with_info<double>(ops, ops, x, b, {.relative_tolerance = 1e-12, .restart = 1, .max_iterations = 50});
  ASSERT_TRUE(info.succeeded());
  EXPECT_GT(info.cycles, 1);
  EXPECT_EQ(info.cycles, info.iterations);
  EXPECT_EQ(info.operator_applications, 1 + 2 * info.iterations);
  EXPECT_NEAR(x.values[0], 0.125, 1e-12);
  EXPECT_NEAR(x.values[1], 0.5, 1e-12);
  double residual = std::hypot(1 - 4 * x.values[0] - x.values[1], 1 - 2 * x.values[1]);
  EXPECT_NEAR(info.residual->norm, residual, 1e-15);

  x.values = {0, 0};
  auto limited =
      gmres_with_info<double>(ops, ops, x, b, {.relative_tolerance = 1e-14, .restart = 2, .max_iterations = 1});
  EXPECT_EQ(limited.status, GmresStatus::iteration_limit);
  EXPECT_EQ(limited.iterations, 1);
}

TEST(Gmres, HappyBreakdownAndZeroRhs)
{
  DenseHostVectorOps<double> ops(2, {1, 0, 0, 1});
  V x{{0, 0}}, b{{1, 2}};
  auto info = gmres_with_info<double>(ops, ops, x, b, {.relative_tolerance = 1e-14});
  ASSERT_TRUE(info.succeeded());
  EXPECT_EQ(info.iterations, 1);
  EXPECT_NEAR(x.values[0], 1, 1e-14);
  EXPECT_NEAR(x.values[1], 2, 1e-14);

  b.values = {0, 0};
  info = gmres_with_info<double>(ops, ops, x, b, {.relative_tolerance = 0, .absolute_tolerance = 1e-14});
  ASSERT_TRUE(info.succeeded());
  EXPECT_LE(info.residual->norm, 1e-14);
  x.values = {0, 0};
  info = gmres_with_info<double>(ops, ops, x, b, {.relative_tolerance = 0});
  EXPECT_TRUE(info.succeeded());
  EXPECT_EQ(info.iterations, 0);
}

TEST(Gmres, SingularOperatorReturnsBreakdown)
{
  DenseHostVectorOps<double> ops(2, {0, 0, 0, 0});
  V x{{0, 0}}, b{{1, 2}};
  auto info = gmres_with_info<double>(ops, ops, x, b, {.relative_tolerance = 1e-12, .max_iterations = 5});
  EXPECT_EQ(info.status, GmresStatus::breakdown);
  EXPECT_EQ(info.iterations, 1);
  EXPECT_EQ(x.values, (std::vector<double>{0, 0}));
  ASSERT_TRUE(info.residual);
  EXPECT_NEAR(info.residual->norm, std::sqrt(5.), 1e-15);
}

TEST(Gmres, FixedRightPreconditionerUsesOriginalResidual)
{
  DenseHostVectorOps<double> ops(2, {2, 0, 0, 8});
  struct InverseDiagonal
  {
      void apply(V& out, V const& in) { out.values = {in.values[0] / 2, in.values[1] / 8}; }
  } preconditioner;
  V x{{0, 0}}, b{{1, 2}};
  auto info = gmres_with_info<double>(ops, ops, preconditioner, x, b, {.relative_tolerance = 1e-14});
  ASSERT_TRUE(info.succeeded());
  EXPECT_EQ(info.iterations, 1);
  EXPECT_EQ(info.preconditioner_applications, 2);
  EXPECT_NEAR(x.values[0], 0.5, 1e-14);
  EXPECT_NEAR(x.values[1], 0.25, 1e-14);
}

TEST(Gmres, FailedCandidateKeepsLastDiagnosedIterate)
{
  DenseHostVectorOps<double> ops(2, {2, 0, 0, 3});
  struct FailingOperator
  {
      DenseHostVectorOps<double>& ops;
      int calls = 0;
      void matvec(V& out, V const& in)
      {
        if (++calls == 3)
          out.values = {numeric_limits<double>::quiet_NaN(), 0};
        else
          ops.matvec(out, in);
      }
  } op{ops};
  V x{{0.25, 0}}, b{{1, 1}};
  auto info = gmres_with_info<double>(ops, op, x, b, {.relative_tolerance = 1e-14, .restart = 1});
  EXPECT_EQ(info.status, GmresStatus::nonfinite_result);
  EXPECT_EQ(info.operator_applications, 3);
  EXPECT_EQ(x.values, (std::vector<double>{0.25, 0}));
  ASSERT_TRUE(info.residual);
  EXPECT_NEAR(info.residual->norm, std::sqrt(1.25), 1e-15);
}

TEST(Gmres, NaNInnerProductDoesNotMasqueradeAsZeroNorm)
{
  struct Ops
  {
      std::size_t problem_dimension() { return 1; }
      std::size_t vector_dimension(V const& v) { return v.values.size(); }
      V allocate_like(V const&) { return V{{0}}; }
      void copy(V& out, V const& in) { out = in; }
      void set_zero(V& x) { x.values[0] = 0; }
      void scal(V& x, double a) { x.values[0] *= a; }
      void axpy(V& y, double a, V const& x) { y.values[0] += a * x.values[0]; }
      double inner_product(V const&, V const&) { return numeric_limits<double>::quiet_NaN(); }
      void matvec(V& out, V const& in) { out = in; }
  } ops;
  V x{{0}}, b{{1}};
  auto info = gmres_with_info<double>(ops, ops, x, b, {.relative_tolerance = 1e-12});
  EXPECT_EQ(info.status, GmresStatus::nonfinite_input);
  EXPECT_FALSE(info.residual);
  EXPECT_EQ(info.operator_applications, 0);
}

TEST(Gmres, RejectsInvalidParametersBeforeMutation)
{
  DenseHostVectorOps<double> ops(2, {1, 0, 0, 1});
  V x{{3, 4}}, b{{1, 2}};
  EXPECT_THROW(gmres_with_info<double>(ops, ops, x, b, {.relative_tolerance = -1}), std::invalid_argument);
  EXPECT_THROW(gmres_with_info<double>(ops, ops, x, b, {.relative_tolerance = 1e-12, .restart = 0}),
               std::invalid_argument);
  EXPECT_THROW(gmres_with_info<double>(ops, ops, x, b, {.relative_tolerance = 1e-12}, Precision::bits(24)),
               std::invalid_argument);
  EXPECT_EQ(x.values, (std::vector<double>{3, 4}));
}

TEST(Gmres, SubnormalResidualDoesNotOverflowNormalization)
{
  DenseHostVectorOps<double> ops(1, {1});
  V x{{0}}, b{{numeric_limits<double>::denorm_min()}};
  auto info = gmres_with_info<double>(ops, ops, x, b, {.relative_tolerance = 0});
  ASSERT_TRUE(info.succeeded());
  EXPECT_EQ(x.values, b.values);
}

TEST(Gmres, ComplexRotationAvoidsSquaringLargeComponents)
{
  using S = uni20::complex<double>;
  S f(0, 3e200), g(4e200, 0);
  auto rotation = gmres_detail::rotation(f, g, native_precision<S>());
  ASSERT_TRUE(uni20::isfinite(rotation.r));
  EXPECT_NEAR(rotation.c, 0.6, 1e-15);
  EXPECT_NEAR(rotation.s.imag(), 0.8, 1e-15);
  gmres_detail::rotate(rotation, f, g);
  EXPECT_LT(math::abs(g) / 5e200, 1e-15);
  EXPECT_NEAR(f.imag() / 5e200, 1, 1e-15);
}

TEST(Gmres, ZeroToleranceDoesNotAcceptAnUnderflowedResidualRatio)
{
  DenseHostVectorOps<double> ops(2, {1e300, 0, 0, 1});
  V x{{1, 0}}, b{{1e300, 1e-300}};
  auto info = gmres_with_info<double>(ops, ops, x, b, {.relative_tolerance = 0, .max_iterations = 0});
  EXPECT_EQ(info.status, GmresStatus::iteration_limit);
  ASSERT_TRUE(info.residual);
  EXPECT_EQ(info.residual->norm, 1e-300);
}
} // namespace
