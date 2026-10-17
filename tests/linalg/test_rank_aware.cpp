#include <uni20/common/gtest.hpp>
#include <uni20/linalg/ops/lu_rcond.hpp>
#include <uni20/linalg/ops/svd_least_squares.hpp>
#include <uni20/tensor/conjugate.hpp>

using namespace uni20;
using namespace uni20::linalg;

TEST(SvdLeastSquares, WideMinimumNormAndResiduals)
{
  DenseMatrix<double, RowMajor> a(1, 2), b(1, 2);
  a[0, 0] = 1;
  a[0, 1] = 2;
  b[0, 0] = 3;
  b[0, 1] = 6;
  auto r = svd_least_squares_with_info(a, b, {.return_singular_values = true});
  ASSERT_TRUE(r.info.succeeded());
  ASSERT_TRUE(r.solution);
  ASSERT_TRUE(r.diagnostics);
  EXPECT_EQ(r.diagnostics->rank, 1);
  EXPECT_EQ(r.diagnostics->relative_cutoff, numeric_limits<double>::epsilon());
  ASSERT_EQ(r.diagnostics->singular_values.size(), 1);
  EXPECT_NEAR(r.diagnostics->singular_values[0], math::sqrt(5.0), 1e-14);
  EXPECT_NEAR(((*r.solution)[0, 0]), 0.6, 1e-14);
  EXPECT_NEAR(((*r.solution)[1, 0]), 1.2, 1e-14);
  EXPECT_NEAR(((*r.solution)[1, 1]), 2.4, 1e-14);
  // Orthogonality to the nullspace distinguishes the minimum-norm solution.
  EXPECT_NEAR((2 * (*r.solution)[0, 0] - (*r.solution)[1, 0]), 0, 1e-14);
  EXPECT_LT(r.residual_norms[0], 1e-14);
  EXPECT_EQ((a[0, 1]), 2);
  EXPECT_EQ((b[0, 1]), 6);
}

TEST(SvdLeastSquares, RankDeficientInconsistentAndZeroMatrix)
{
  DenseMatrix<double> a(3, 2), b(3, 1);
  a[0, 0] = a[0, 1] = 1;
  b[0, 0] = 2;
  b[1, 0] = 3;
  auto r = svd_least_squares_with_info(a, b);
  ASSERT_TRUE(r.info.succeeded());
  EXPECT_EQ(r.diagnostics->rank, 1);
  EXPECT_TRUE(r.diagnostics->singular_values.empty());
  EXPECT_NEAR(((*r.solution)[0, 0]), 1, 1e-14);
  EXPECT_NEAR(((*r.solution)[1, 0]), 1, 1e-14);
  EXPECT_NEAR(r.residual_norms[0], 3, 1e-14);
  a[0, 0] = a[0, 1] = 0;
  auto zero = svd_least_squares_with_info(a, b);
  ASSERT_TRUE(zero.info.succeeded());
  EXPECT_EQ(zero.diagnostics->rank, 0);
  EXPECT_EQ(((*zero.solution)[0, 0]), 0);
  EXPECT_NEAR(zero.residual_norms[0], math::sqrt(13.0), 1e-14);
}

TEST(SvdLeastSquares, CutoffIsExplicitAndNotDimensionScaled)
{
  DenseMatrix<double> a(8, 2), b(8, 1);
  double eps = numeric_limits<double>::epsilon();
  a[0, 0] = 1;
  a[1, 1] = 2 * eps;
  b[0, 0] = 1;
  b[1, 0] = 2 * eps;
  auto full = svd_least_squares_with_info(a, b);
  ASSERT_EQ(full.diagnostics->rank, 2);
  EXPECT_NEAR(((*full.solution)[1, 0]), 1, 1e-14);
  auto truncated = svd_least_squares_with_info(a, b, {.relative_cutoff = 4 * eps});
  ASSERT_TRUE(truncated.info.succeeded());
  EXPECT_EQ(truncated.diagnostics->rank, 1);
  EXPECT_EQ(((*truncated.solution)[1, 0]), 0);
  EXPECT_EQ(truncated.residual_norms[0], 2 * eps);
  a[1, 1] = b[1, 0] = eps / 4;
  EXPECT_EQ(svd_least_squares_with_info(a, b).diagnostics->rank, 1);
  EXPECT_EQ(svd_least_squares_with_info(a, b, {.relative_cutoff = 0.0}).diagnostics->rank, 2);
}

TEST(SvdLeastSquares, ComplexLogicalViews)
{
  using S = complex<double>;
  DenseMatrix<S, RowMajor> a(1, 2), b(1, 1);
  a[0, 0] = S(1, 1);
  a[0, 1] = S(2, -1);
  b[0, 0] = S(3, 2);
  auto r = svd_least_squares_with_info(uni20::conj(a), b);
  ASSERT_TRUE(r.info.succeeded());
  // For a row v, v^H b / ||v||^2 is the independent minimum-norm oracle.
  EXPECT_LT(math::abs((*r.solution)[0, 0] - a[0, 0] * b[0, 0] / 7.0), 1e-14);
  EXPECT_LT(math::abs((*r.solution)[1, 0] - a[0, 1] * b[0, 0] / 7.0), 1e-14);
  EXPECT_LT(r.residual_norms[0], 1e-14);
}

TEST(SvdLeastSquares, EmptyShapesAndRankWithoutRhs)
{
  DenseMatrix<double> a(3, 0), b(3, 1);
  b[0, 0] = 3;
  b[1, 0] = 4;
  auto r = svd_least_squares_with_info(a, b);
  EXPECT_EQ(r.diagnostics->rank, 0);
  EXPECT_EQ(r.residual_norms[0], 5);
  DenseMatrix<double> wide(0, 3), empty(0, 2);
  auto w = svd_least_squares_with_info(wide, empty);
  ASSERT_TRUE(w.info.succeeded());
  EXPECT_EQ(w.solution->extent(0), 3);
  EXPECT_EQ(((*w.solution)[2, 1]), 0);
  EXPECT_EQ(w.residual_norms, (std::vector<double>{0, 0}));
  DenseMatrix<double> diagonal(2, 2), none(2, 0);
  diagonal[0, 0] = 1;
  diagonal[1, 1] = 2;
  auto rank = svd_least_squares_with_info(diagonal, none);
  ASSERT_TRUE(rank.info.succeeded());
  EXPECT_EQ(rank.diagnostics->rank, 2);
  diagonal[0, 0] = numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(svd_least_squares_with_info(diagonal, none).info.status, LeastSquaresStatus::nonfinite_input);
}

TEST(SvdLeastSquares, FailuresPreserveInputsAndFixedOutputs)
{
  DenseMatrix<double> a(1, 1), b(1, 1), x(1, 1);
  a[0, 0] = numeric_limits<double>::infinity();
  b[0, 0] = 1;
  auto r = svd_least_squares_with_info(a, b);
  EXPECT_EQ(r.info.status, LeastSquaresStatus::nonfinite_input);
  EXPECT_FALSE(r.solution);
  EXPECT_FALSE(r.diagnostics);
  EXPECT_TRUE(r.residual_norms.empty());
  auto ad = mdspec_of(std::as_const(a)), bd = mdspec_of(std::as_const(b));
  auto xd = mdspec_of(x);
  x[0, 0] = 17;
  std::vector<double> residuals{23};
  LeastSquaresInfo info;
  SvdLeastSquaresDiagnostics<double> diagnostics{.rank = 42};
  SvdLeastSquaresOptions<double> options;
  EXPECT_TRUE(try_dispatch_kernel(LapackBackend{}, svd_least_squares_op{}, xd, ad, bd, std::span(residuals), info,
                                  diagnostics, options));
  EXPECT_EQ((x[0, 0]), 17);
  EXPECT_EQ(residuals[0], 23);
  EXPECT_EQ(diagnostics.rank, 42);
  a[0, 0] = numeric_limits<double>::min();
  b[0, 0] = numeric_limits<double>::max();
  EXPECT_EQ(svd_least_squares_with_info(a, b).info.status, LeastSquaresStatus::nonfinite_result);
  EXPECT_THROW((void)svd_least_squares_with_info(a, b, Precision::bits(24)), std::invalid_argument);
  EXPECT_DEATH((void)svd_least_squares_with_info(a, b, {.relative_cutoff = -1.0}), "nonnegative relative cutoff");
}

TEST(LuCondition, RetainedNormsAndKnownInverse)
{
  DenseMatrix<double, RowMajor> a(3, 3);
  // A cyclic row permutation of an upper triangular matrix exercises pivoting.
  // The unpermuted matrix is [1 2 0; 0 3 4; 0 0 5].
  a[0, 1] = 3;
  a[0, 2] = 4;
  a[1, 2] = 5;
  a[2, 0] = 1;
  a[2, 1] = 2;
  for (auto backend : {false, true})
  {
    auto f = backend ? lu_factor(LapackBackend{}, a) : lu_factor(CpuReferenceBackend{}, a);
    ASSERT_TRUE(f.norms());
    EXPECT_EQ(f.norms()->one, 9);
    EXPECT_EQ(f.norms()->infinity, 7);
    auto one = lu_rcond_with_info(f);
    auto inf = lu_rcond_with_info(f, MatrixNorm::Infinity);
    ASSERT_TRUE(one.estimate);
    ASSERT_TRUE(inf.estimate);
    EXPECT_EQ(one.norm, MatrixNorm::One);
    EXPECT_EQ(inf.norm, MatrixNorm::Infinity);
    // inv(A), up to column permutation, is [1 -2/3 8/15; 0 1/3 -4/15; 0 0 1/5].
    EXPECT_NEAR(*one.estimate, 1.0 / 9, 1e-14);
    EXPECT_NEAR(*inf.estimate, 5.0 / 77, 1e-14);
    DenseMatrix<double> rhs(3, 1);
    rhs[0, 0] = 7;
    rhs[1, 0] = 5;
    rhs[2, 0] = 3;
    auto x = lu_solve(f, rhs);
    for (int i = 0; i < 3; ++i)
      EXPECT_NEAR((x[i, 0]), 1, 1e-14);
  }
}

TEST(LuCondition, OptOutAndEmptyFactors)
{
  DenseMatrix<double> a(2, 2);
  a[0, 0] = 1;
  a[1, 1] = 4;
  auto f = lu_factor(a, LuOptions<double>{.retain_norms = false});
  EXPECT_FALSE(f.norms());
  EXPECT_DEATH((void)lu_rcond(f), "norms were not retained");
  EXPECT_EQ(lu_rcond(f, MatrixNorm::One, 4.0), 0.25);
  EXPECT_DEATH((void)lu_rcond(f, MatrixNorm::Frobenius, 4.0), "one- or infinity-norm");
  EXPECT_DEATH((void)lu_rcond(f, MatrixNorm::One, -1.0), "nonnegative");
  DenseMatrix<double> empty(0, 0);
  auto e = lu_factor(empty);
  EXPECT_EQ(e.norms()->one, 0);
  EXPECT_EQ(lu_rcond(e), 1);
}

TEST(LuCondition, OverflowingNormDoesNotInvalidateFactors)
{
  DenseMatrix<double> a(2, 2), b(2, 1);
  a[0, 0] = a[0, 1] = 1e308;
  a[1, 1] = 1e308;
  b[0, 0] = b[1, 0] = 1e308;
  auto f = lu_factor(a);
  EXPECT_TRUE(math::isinf(f.norms()->one));
  auto c = lu_rcond_with_info(f);
  EXPECT_EQ(c.info.status, SolveStatus::nonfinite_result);
  EXPECT_FALSE(c.estimate);
  auto x = lu_solve(f, b);
  EXPECT_EQ((x[0, 0]), 0);
  EXPECT_EQ((x[1, 0]), 1);
}

#if UNI20_HAS_MPLAPACK_MPREAL
TEST(RankAwarePrecision, ExactInputsAndConversionBoundary)
{
  auto p = Precision::bits(128), q = Precision::bits(256);
  DenseMatrix<mpreal> a(1, 2, Precision::exact()), b(1, 1, Precision::exact());
  a[0, 0] = mpreal(1);
  a[0, 1] = mpreal(2);
  b[0, 0] = mpreal(3);
  EXPECT_THROW((void)svd_least_squares_with_info(a, b), std::logic_error);
  auto r = svd_least_squares_with_info(a, b, p);
  ASSERT_TRUE(r.info.succeeded());
  EXPECT_FLOATING_EQ(((*r.solution)[0, 0]), mpreal("0.6", p), 32);
  EXPECT_EQ(r.diagnostics->relative_cutoff.precision(), p);
  EXPECT_TRUE((a[0, 0].is_exact()));
  DenseMatrix<mpreal> diagonal(2, 2, q);
  diagonal[0, 0] = mpreal(1, q) + math::ldexp(mpreal(1, q), -200);
  diagonal[1, 1] = mpreal(2, q);
  auto f = lu_factor(diagonal, p);
  EXPECT_EQ(f.norms()->one.precision(), p);
  EXPECT_EQ(f.norms()->one, mpreal(2));
  EXPECT_EQ(lu_rcond(f), mpreal("0.5", p));
  EXPECT_EQ(lu_rcond(f).precision(), p);
  EXPECT_EQ(precision_of(diagonal), q);
}
#endif

namespace
{
struct NonconvergingSvdBackend
{
    static constexpr std::string_view name = "nonconverging_svd";
    bool* called;
};
template <class... Args> consteval auto kernel_accepts_types(NonconvergingSvdBackend, svd_least_squares_op, Args&&...)
{
  return kernel_types_yes;
}
template <class X, class A, class B>
KernelAttempt try_kernel(NonconvergingSvdBackend backend, svd_least_squares_op, X&, A&, B&, std::span<double>,
                         LeastSquaresInfo& info, SvdLeastSquaresDiagnostics<double>&,
                         SvdLeastSquaresOptions<double> const&)
{
  *backend.called = true;
  info = {.status = LeastSquaresStatus::nonconvergence, .unconverged = 2};
  return KernelAttempt::success;
}
} // namespace

TEST(SvdLeastSquares, NonconvergencePublishesOnlyDiagnosticsAndNeverFallsBack)
{
  DenseMatrix<double> a(3, 3), b(3, 1);
  bool first = false, fallback = false;
  auto r = svd_least_squares_with_info(
      backend_list{NonconvergingSvdBackend{&first}, NonconvergingSvdBackend{&fallback}}, a, b);
  EXPECT_TRUE(first);
  EXPECT_FALSE(fallback);
  EXPECT_EQ(r.info.status, LeastSquaresStatus::nonconvergence);
  EXPECT_EQ(r.info.unconverged, 2);
  EXPECT_FALSE(r.solution);
  EXPECT_FALSE(r.diagnostics);
  EXPECT_TRUE(r.residual_norms.empty());
}
