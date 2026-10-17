#include <uni20/common/gtest.hpp>
#include <uni20/linalg/ops/least_squares.hpp>
#include <uni20/tensor/conjugate.hpp>

using namespace uni20;
using namespace uni20::linalg;

template <class Backend> class LeastSquares : public ::testing::Test {};
using Backends = ::testing::Types<CpuReferenceBackend, LapackBackend>;
TYPED_TEST_SUITE(LeastSquares, Backends);

TYPED_TEST(LeastSquares, RecoverableFailuresPreserveInputs)
{
  DenseMatrix<double, RowMajor> a(3, 2), b(3, 1);
  b[0, 0] = 1;
  auto zero = least_squares_with_info(TypeParam{}, a, b);
  EXPECT_EQ(zero.info.status, LeastSquaresStatus::zero_diagonal);
  EXPECT_FALSE(zero.solution);
  EXPECT_TRUE(zero.residual_norms.empty());
  a[0, 0] = 1;
  a[1, 1] = 1e-20;
  b[1, 0] = 1e-20;
  auto rejected = least_squares_with_info(TypeParam{}, a, b, LeastSquaresOptions<double>{1e-10});
  EXPECT_EQ(rejected.info.status, LeastSquaresStatus::small_diagonal);
  EXPECT_EQ(rejected.info.column, 1);
  EXPECT_FALSE(rejected.solution);
  auto accepted = least_squares_with_info(TypeParam{}, a, b);
  ASSERT_TRUE(accepted.solution);
  EXPECT_FLOATING_EQ(((*accepted.solution)[1, 0]), 1.0);
  EXPECT_EQ((a[1, 1]), 1e-20);
  EXPECT_EQ(least_squares_with_info(a, b, {.relative_diagonal_tolerance = 1e-10}).info.status,
            LeastSquaresStatus::small_diagonal);
  b[0, 0] = numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(least_squares_with_info(TypeParam{}, a, b).info.status, LeastSquaresStatus::nonfinite_input);
  EXPECT_TRUE(math::isnan(b[0, 0]));
  b[0, 0] = 1;
  a[0, 0] = numeric_limits<double>::infinity();
  EXPECT_EQ(least_squares_with_info(TypeParam{}, a, b).info.status, LeastSquaresStatus::nonfinite_input);
}

TYPED_TEST(LeastSquares, EmptyShapes)
{
  DenseMatrix<double> a(3, 0), b(3, 1);
  b[0, 0] = 3;
  b[1, 0] = 4;
  auto result = least_squares_with_info(TypeParam{}, a, b);
  ASSERT_TRUE(result.info.succeeded());
  EXPECT_EQ(result.solution->extent(0), 0);
  EXPECT_EQ(result.residual_norms[0], 5);
  DenseMatrix<double> c(0, 0), d(0, 2);
  auto empty = least_squares_with_info(TypeParam{}, c, d);
  EXPECT_EQ(empty.residual_norms, (std::vector<double>{0, 0}));
  DenseMatrix<double> bad(2, 2), none(2, 0);
  bad[0, 0] = numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(least_squares_with_info(TypeParam{}, bad, none).info.succeeded());
}

TYPED_TEST(LeastSquares, LogicalConjugationAndRowMajorInputs)
{
  using S = complex<double>;
  DenseMatrix<S, RowMajor> a(3, 2), b(3, 1);
  a[0, 0] = S(1, 1);
  a[1, 1] = S(0, 2);
  a[2, 0] = S(0, 1);
  a[2, 1] = S(1, -1);
  S x(2, 1), y(-1, 2);
  for (int i = 0; i < 3; ++i)
    b[i, 0] = uni20::conj(a[i, 0]) * x + uni20::conj(a[i, 1]) * y;
  auto result = least_squares_with_info(TypeParam{}, uni20::conj(a), b);
  ASSERT_TRUE(result.info.succeeded());
  EXPECT_LT(math::abs((*result.solution)[0, 0] - x), 1e-14);
  EXPECT_LT(math::abs((*result.solution)[1, 0] - y), 1e-14);
  EXPECT_LT(result.residual_norms[0], 1e-14);
}

TYPED_TEST(LeastSquares, NonfiniteArithmeticIsRecoverable)
{
  DenseMatrix<double> a(2, 1), b(2, 1);
  a[0, 0] = numeric_limits<double>::min();
  b[0, 0] = numeric_limits<double>::max();
  auto result = least_squares_with_info(TypeParam{}, a, b);
  EXPECT_EQ(result.info.status, LeastSquaresStatus::nonfinite_result);
  EXPECT_FALSE(result.solution);
  EXPECT_TRUE(result.residual_norms.empty());
}

TYPED_TEST(LeastSquares, ZeroLeadingEntryComplexRotation)
{
  using S = complex<double>;
  DenseMatrix<S> a(3, 1), b(3, 1);
  a[1, 0] = S(0, 2);
  a[2, 0] = S(3, -1);
  S expected(2, 3);
  for (int i = 0; i < 3; ++i)
    b[i, 0] = a[i, 0] * expected;
  auto x = least_squares(TypeParam{}, a, b);
  EXPECT_LT(math::abs(x[0, 0] - expected), 1e-14);
}

TEST(LeastSquaresContract, RejectsInvalidPrecisionBeforeReadingElements)
{
  DenseMatrix<double> a(uninitialized, 3, 2), b(uninitialized, 3, 1);
  EXPECT_THROW((void)least_squares_with_info(a, b, Precision::bits(24)), std::invalid_argument);
  EXPECT_EQ(a.extent(0), 3);
}

TEST(LeastSquaresContract, ShapeOptionsAndStrictFailure)
{
  DenseMatrix<double> wide(2, 3), a(3, 2), b(3, 1), short_b(2, 1);
  EXPECT_DEATH((void)least_squares_with_info(wide, short_b), "least squares requires m >= n");
  EXPECT_DEATH((void)least_squares_with_info(a, short_b), "matching RHS rows");
  EXPECT_DEATH((void)least_squares_with_info(a, b, LeastSquaresOptions<double>{-1}), "nonnegative diagonal");
  EXPECT_DEATH((void)least_squares(a, b), "least squares failed");
}

TEST(LeastSquaresContract, DirectDispatchFailurePreservesOutputs)
{
  DenseMatrix<double> a(3, 2), b(3, 1), x(2, 1);
  x[0, 0] = 17;
  auto ad = mdspec_of(std::as_const(a));
  auto bd = mdspec_of(std::as_const(b));
  auto xd = mdspec_of(x);
  std::vector<double> residuals{23};
  LeastSquaresInfo info{.status = LeastSquaresStatus::small_diagonal, .column = 42};
  LeastSquaresOptions<double> options;
  static_assert(
      probe_dispatch_kernel_types<BlasBackend, least_squares_op, decltype(xd)&, decltype(ad)&, decltype(bd)&,
                                  std::span<double>, LeastSquaresInfo&, LeastSquaresOptions<double> const&>() ==
      KernelTypeAcceptance::no);
  EXPECT_TRUE(
      try_dispatch_kernel(CpuReferenceBackend{}, least_squares_op{}, xd, ad, bd, std::span(residuals), info, options));
  EXPECT_EQ(info.status, LeastSquaresStatus::zero_diagonal);
  EXPECT_EQ((x[0, 0]), 17);
  EXPECT_EQ(residuals[0], 23);
}

#if UNI20_HAS_MPREAL
TEST(LeastSquaresPrecision, ExplicitConversionAndExactInput)
{
  auto p = Precision::bits(128), q = Precision::bits(256);
  DenseMatrix<mpreal> a(3, 1, p), b(3, 1, q);
  a[0, 0] = mpreal(1, p);
  b[0, 0] = mpreal("1.125", q);
  EXPECT_THROW((void)least_squares_with_info(a, b), std::invalid_argument);
  auto result = least_squares_with_info(a, b, p);
  ASSERT_TRUE(result.info.succeeded());
  EXPECT_EQ(((*result.solution)[0, 0].precision()), p);
  EXPECT_EQ(result.residual_norms[0].precision(), p);
  EXPECT_EQ(precision_of(b), q);
  DenseMatrix<mpreal> exact(2, 1, Precision::exact()), rhs(2, 1, Precision::exact());
  exact[0, 0] = exact[1, 0] = mpreal(1);
  rhs[0, 0] = mpreal(1);
  rhs[1, 0] = mpreal(2);
  EXPECT_THROW((void)least_squares_with_info(exact, rhs), std::logic_error);
  auto finite = least_squares_with_info(CpuReferenceBackend{}, exact, rhs, q);
  ASSERT_TRUE(finite.solution);
  EXPECT_LE(abs((*finite.solution)[0, 0] - mpreal("1.5", q)), numeric_limits<mpreal>::epsilon(q) * 8);
  EXPECT_TRUE((exact[0, 0].is_exact()));
}
#endif
