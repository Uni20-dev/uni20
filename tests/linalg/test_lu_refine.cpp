#include <gtest/gtest.h>
#include <uni20/linalg/ops/lu_refine.hpp>
#include <uni20/tensor/conjugate.hpp>

using namespace uni20;
using namespace uni20::linalg;

template <class Backend> class LuRefinement : public testing::Test {};
using RefinementBackends = testing::Types<CpuReferenceBackend, LapackBackend>;
TYPED_TEST_SUITE(LuRefinement, RefinementBackends);

TYPED_TEST(LuRefinement, ThreeNativePrecisionsAndIndependentRhs)
{
  DenseMatrix<double, RowMajor> a(2, 2), b(2, 3);
  a[0, 0] = 3;
  a[0, 1] = a[1, 0] = 1;
  a[1, 1] = 2;
  b[0, 0] = 1;
  b[0, 2] = 1e-200; // Below float's range: residual scaling must preserve it.
  auto p = LuRefinementPrecision<float, double, long double>::native();
  auto initial = lu_refine_with_info(TypeParam{}, a, b, p, {.max_iterations = 0});
  ASSERT_EQ(initial.info.status, RefinementStatus::iteration_limit);
  ASSERT_TRUE(initial.solution);
  EXPECT_GT(std::abs((*initial.solution)[0, 0] - 0.4), 1e-10);
  auto result = lu_refine_with_info(TypeParam{}, a, b, p);
  ASSERT_TRUE(result.info.succeeded());
  ASSERT_TRUE(result.solution);
  EXPECT_NEAR(((*result.solution)[0, 0]), 0.4, 2e-16);
  EXPECT_NEAR(((*result.solution)[1, 0]), -0.2, 1e-16);
  EXPECT_NEAR(((*result.solution)[0, 2] / 1e-200), 0.4, 2e-16);
  EXPECT_GT(result.columns[0].corrections, 0);
  EXPECT_EQ(result.columns[1].corrections, 0);
  EXPECT_EQ(((*result.solution)[0, 1]), 0);
  for (auto const& column : result.columns)
  {
    EXPECT_TRUE(column.converged);
    EXPECT_LE(column.error.componentwise_backward_error, numeric_limits<double>::epsilon());
  }
  EXPECT_EQ((a[0, 0]), 3);
  EXPECT_EQ((a[1, 0]), 1);
  EXPECT_EQ((b[0, 2]), 1e-200);
}

TYPED_TEST(LuRefinement, ComplexLogicalViewAndReusableFactors)
{
  using F = complex<float>;
  using W = complex<double>;
  using R = complex<long double>;
  DenseMatrix<W, RowMajor> a(2, 2), b(2, 1);
  W phase(1, 1);
  a[0, 0] = 3.0 * phase;
  a[0, 1] = a[1, 0] = phase;
  a[1, 1] = 2.0 * phase;
  b[0, 0] = uni20::conj(phase) * W(1, -1);
  auto view = uni20::conj(a);
  auto p = LuRefinementPrecision<F, W, R>::native();
  DenseMatrix<F> low(2, 2);
  low[0, 0] = F(3, -3);
  low[0, 1] = low[1, 0] = F(1, -1);
  low[1, 1] = F(2, -2);
  auto factor = lu_factor(TypeParam{}, low);
  auto before = factor.packed()[1, 1];
  auto result = lu_refine_with_info(TypeParam{}, factor, view, b, p);
  ASSERT_TRUE(result.info.succeeded());
  EXPECT_LT(math::abs((*result.solution)[0, 0] - W(0.4, -0.4)), 4e-16);
  EXPECT_LT(math::abs((*result.solution)[1, 0] - W(-0.2, 0.2)), 2e-16);
  EXPECT_EQ((factor.packed()[1, 1]), before);
  EXPECT_EQ((a[0, 0]), W(3, 3));
}

TYPED_TEST(LuRefinement, FactorPrecisionCanLoseNonsingularity)
{
  DenseMatrix<double> a(2, 2), b(2, 1);
  a[0, 0] = a[0, 1] = a[1, 0] = 1;
  a[1, 1] = 1 + 0x1p-30;
  b[1, 0] = 0x1p-30;
  auto p = LuRefinementPrecision<float, double, long double>::native();
  auto result = lu_refine_with_info(TypeParam{}, a, b, p);
  EXPECT_EQ(result.info.status, RefinementStatus::factorization_failed);
  ASSERT_TRUE(result.info.solve_info);
  EXPECT_EQ(result.info.solve_info->status, SolveStatus::singular);
  EXPECT_FALSE(result.solution);
  EXPECT_TRUE(result.columns.empty());
  EXPECT_TRUE(lu_factor_with_info(TypeParam{}, a).info.succeeded());
}

TYPED_TEST(LuRefinement, ResidualUsesOriginalMatrixBeforeFactorRounding)
{
  DenseMatrix<double> a(1, 1), b(1, 1);
  a[0, 0] = 1 + 0x1p-30;
  b[0, 0] = 1;
  auto p = LuRefinementPrecision<float, double, long double>::native();
  auto initial = lu_refine_with_info(TypeParam{}, a, b, p, {.max_iterations = 0});
  ASSERT_TRUE(initial.solution);
  EXPECT_EQ(initial.info.status, RefinementStatus::iteration_limit);
  EXPECT_EQ(((*initial.solution)[0, 0]), 1);
  EXPECT_EQ(initial.columns[0].error.residual_infinity_norm, 0x1p-30L);
  auto result = lu_refine_with_info(TypeParam{}, a, b, p);
  ASSERT_TRUE(result.info.succeeded());
  // Rounding A to float loses the perturbation. Refining that rounded problem
  // would incorrectly keep x = 1 with a zero residual and no correction.
  EXPECT_NEAR(((*result.solution)[0, 0]), 1.0 / (1 + 0x1p-30), 2e-16);
  EXPECT_GT(result.columns[0].corrections, 0);
}

TYPED_TEST(LuRefinement, SmallMatrixScaleDoesNotOverflowNormalizedSolve)
{
  DenseMatrix<double> a(1, 1), b(1, 2);
  a[0, 0] = b[0, 0] = 1e-40;
  b[0, 1] = 1e-200;
  auto p = LuRefinementPrecision<float, double, long double>::native();
  auto result = lu_refine_with_info(TypeParam{}, a, b, p);
  if constexpr (std::same_as<TypeParam, LapackBackend>)
  {
    // Some BLAS triangular solves form 1/U_ii internally, overflowing for a
    // subnormal diagonal even when RHS/U_ii is finite. Verify that this is a
    // primitive-provider limitation and that refinement reports it honestly.
    DenseMatrix<float> low_a(1, 1), low_b(1, 2);
    low_a[0, 0] = low_b[0, 0] = low_b[0, 1] = float(a[0, 0]);
    auto low_factor = lu_factor(TypeParam{}, low_a);
    auto direct = lu_solve_inplace_with_info(TypeParam{}, low_factor, low_b);
    if (!direct.succeeded())
    {
      EXPECT_EQ(direct.status, SolveStatus::nonfinite_result);
      EXPECT_EQ(result.info.status, RefinementStatus::correction_failed);
      ASSERT_TRUE(result.info.solve_info);
      EXPECT_EQ(result.info.solve_info->status, SolveStatus::nonfinite_result);
      EXPECT_FALSE(result.solution);
      return;
    }
  }
  ASSERT_TRUE(result.info.succeeded()) << static_cast<int>(result.info.status);
  EXPECT_NEAR(((*result.solution)[0, 0]), 1.0, 1e-15);
  EXPECT_NEAR(((*result.solution)[0, 1] / 1e-160), 1.0, 1e-15);
}

TYPED_TEST(LuRefinement, ScalingSurvivesNarrowerResidualExponentRange)
{
  // Exact dyadic solutions must survive the initial solve; further refinement
  // could otherwise conceal a component lost during a workspace conversion.
  auto check = []<class F, class W>() {
    W phase = [] {
      if constexpr (Complex<W>)
        return W(1, -1);
      else
        return W(1);
    }();
    auto p = LuRefinementPrecision<F, W, W>::native();
    DenseMatrix<W> a(2, 2), b(2, 1);
    a[0, 0] = 0x1p-140F;
    a[1, 1] = 1;
    b[0, 0] = 0x1p-140F * phase;
    auto small_rhs = lu_refine_with_info(TypeParam{}, a, b, p, {.max_iterations = 0});
    ASSERT_TRUE(small_rhs.info.succeeded());
    ASSERT_TRUE(small_rhs.solution);
    EXPECT_EQ(((*small_rhs.solution)[0, 0]), phase);
    EXPECT_EQ(((*small_rhs.solution)[1, 0]), W(0));

    // Downward RHS normalization produces a solved component below float's
    // range. Undoing that scale must precede narrowing its exponent range.
    a[0, 0] = a[1, 1] = 0x1p120F;
    b[0, 0] = 0x1p100F * phase;
    b[1, 0] = 0x1p-20F * phase;
    auto large_rhs = lu_refine_with_info(TypeParam{}, a, b, p, {.max_iterations = 0});
    ASSERT_TRUE(large_rhs.info.succeeded());
    ASSERT_TRUE(large_rhs.solution);
    EXPECT_EQ(((*large_rhs.solution)[0, 0]), 0x1p-20F * phase);
    EXPECT_EQ(((*large_rhs.solution)[1, 0]), 0x1p-140F * phase);

    // The incoming transfer must likewise avoid underflow while scaling in
    // float before conversion to the wider factor scalar.
    a[0, 0] = 0x1p100F;
    a[1, 1] = 0x1p-100F;
    b[1, 0] = 0x1p-140F * phase;
    auto mixed_scale = lu_refine_with_info(TypeParam{}, a, b, p, {.max_iterations = 0});
    ASSERT_TRUE(mixed_scale.info.succeeded());
    ASSERT_TRUE(mixed_scale.solution);
    EXPECT_EQ(((*mixed_scale.solution)[0, 0]), phase);
    EXPECT_EQ(((*mixed_scale.solution)[1, 0]), 0x1p-40F * phase);
  };
  check.template operator()<double, float>();
  check.template operator()<complex<double>, complex<float>>();
}

TYPED_TEST(LuRefinement, ScalingSurvivesWiderResidualExponentRange)
{
  auto check = []<class F, class W>() {
    W phase = [] {
      if constexpr (Complex<W>)
        return W(1, -1);
      else
        return W(1);
    }();
    DenseMatrix<W> a(1, 1), b(1, 2);
    a[0, 0] = 1;
    b[0, 0] = 0x1p-200 * phase;
    b[0, 1] = 0x1p200 * phase;
    auto p = LuRefinementPrecision<F, W, W>::native();
    auto result = lu_refine_with_info(TypeParam{}, a, b, p, {.max_iterations = 0});
    ASSERT_TRUE(result.info.succeeded());
    ASSERT_TRUE(result.solution);
    // Unscaling in factor arithmetic would underflow one RHS and overflow
    // the other, although both exact solutions fit the working scalar.
    EXPECT_EQ(((*result.solution)[0, 0]), (b[0, 0]));
    EXPECT_EQ(((*result.solution)[0, 1]), (b[0, 1]));

    // A representable RHS can still need upward scaling when the factors
    // are large: the unscaled float solve would underflow its tiny solution.
    a[0, 0] = 0x1p100;
    b[0, 0] = 0x1p-100 * phase;
    b[0, 1] = 0x1p100 * phase;
    auto large_factor = lu_refine_with_info(TypeParam{}, a, b, p, {.max_iterations = 0});
    ASSERT_TRUE(large_factor.info.succeeded());
    ASSERT_TRUE(large_factor.solution);
    EXPECT_EQ(((*large_factor.solution)[0, 0]), 0x1p-200 * phase);
    EXPECT_EQ(((*large_factor.solution)[0, 1]), phase);
  };
  check.template operator()<float, double>();
  check.template operator()<complex<float>, complex<double>>();
}

TEST(LuRefinementApi, SmallRhsScalingDoesNotOverflowWithDisparatePivots)
{
  DenseMatrix<double> a(2, 2), b(2, 1);
  a[0, 0] = 1;
  a[1, 1] = b[1, 0] = 0x1p-130;
  auto p = LuRefinementPrecision<float, double, double>::native();
  auto result = lu_refine_with_info(CpuReferenceBackend{}, a, b, p);
  ASSERT_TRUE(result.info.succeeded());
  ASSERT_TRUE(result.solution);
  EXPECT_EQ(((*result.solution)[0, 0]), 0);
  EXPECT_EQ(((*result.solution)[1, 0]), 1);
}

namespace
{
struct FailSecondCorrection
{
    static constexpr std::string_view name = "fail_second_correction";
    int* calls;
};
template <class A, class B>
consteval auto kernel_accepts_types(FailSecondCorrection, lu_solve_op const&, A&, std::span<std::size_t const>, B&,
                                    SolveInfo&)
{
  return kernel_types_yes;
}
template <class A, class B>
KernelAttempt try_kernel(FailSecondCorrection backend, lu_solve_op const& op, A& a, std::span<std::size_t const> pivots,
                         B& b, SolveInfo& info)
{
  if (++*backend.calls == 2)
  {
    info.status = SolveStatus::nonfinite_result;
    return KernelAttempt::success;
  }
  return try_kernel(CpuReferenceBackend{}, op, a, pivots, b, info);
}
} // namespace

TEST(LuRefinementApi, LaterSolveFailurePreservesPublishedIterate)
{
  DenseMatrix<float> a(1, 1), b(1, 1);
  a[0, 0] = 3;
  b[0, 0] = 1;
  auto factor = lu_factor(CpuReferenceBackend{}, a);
  auto p = LuRefinementPrecision<float, double, long double>::native();
  auto initial = lu_refine_with_info(factor, a, b, p, {.max_iterations = 0});
  int calls = 0;
  auto failed = lu_refine_with_info(FailSecondCorrection{&calls}, factor, a, b, p);
  ASSERT_EQ(failed.info.status, RefinementStatus::correction_failed);
  ASSERT_TRUE(failed.solution);
  EXPECT_EQ(calls, 2);
  EXPECT_EQ(((*failed.solution)[0, 0]), ((*initial.solution)[0, 0]));
  EXPECT_EQ(failed.columns[0].error.componentwise_backward_error,
            initial.columns[0].error.componentwise_backward_error);
  EXPECT_EQ(failed.columns[0].corrections, 0);
}

TYPED_TEST(LuRefinement, StagnationRetainsDiagnosedIterate)
{
  DenseMatrix<double> a(1, 1), b(1, 1);
  a[0, 0] = 3;
  b[0, 0] = 1;
  auto q = LuRefinementPrecision<float, float, double>::native();
  auto result = lu_refine_with_info(TypeParam{}, a, b, q, {.backward_error_tolerance = 0.0});
  ASSERT_EQ(result.info.status, RefinementStatus::stagnated);
  ASSERT_TRUE(result.solution);
  EXPECT_EQ(((*result.solution)[0, 0]), 1.0F / 3.0F);
  EXPECT_TRUE(result.columns[0].stagnated);
  EXPECT_FALSE(result.columns[0].converged);
  auto checked = solve_diagnostics_with_info<double>(a, b, *result.solution, native_precision<double>());
  ASSERT_TRUE(checked.info.succeeded());
  EXPECT_EQ(result.columns[0].error.componentwise_backward_error, checked.columns[0].componentwise_backward_error);
  EXPECT_GT(checked.columns[0].componentwise_backward_error, 0);
}

TEST(SolveDiagnostics, ResidualPrecisionControlsCancellation)
{
  DenseMatrix<double> a(1, 1), b(1, 1), x(1, 1);
  a[0, 0] = 3;
  b[0, 0] = 1;
  x[0, 0] = double(1.0F / 3.0F);
  auto low = solve_diagnostics_with_info<float>(a, b, x, native_precision<float>());
  auto high = solve_diagnostics_with_info<double>(a, b, x, native_precision<double>());
  ASSERT_TRUE(low.info.succeeded());
  ASSERT_TRUE(high.info.succeeded());
  EXPECT_EQ(low.columns[0].residual_infinity_norm, 0);
  EXPECT_GT(high.columns[0].residual_infinity_norm, 0);
  EXPECT_EQ(high.columns[0].residual_infinity_norm, 0x1p-25);
}

TEST(SolveDiagnostics, ExactDefinitionsAndZeroRows)
{
  DenseMatrix<double> a(2, 2), b(2, 1), x(2, 1);
  a[0, 0] = 2;
  b[0, 0] = 5;
  x[0, 0] = 2;
  auto result = solve_diagnostics_with_info<double>(a, b, x, native_precision<double>());
  ASSERT_TRUE(result.info.succeeded());
  EXPECT_EQ(((*result.residual)[0, 0]), 1);
  EXPECT_EQ(((*result.residual)[1, 0]), 0);
  EXPECT_EQ(result.columns[0].residual_infinity_norm, 1);
  EXPECT_EQ(result.columns[0].normwise_backward_error, 1.0 / 9.0);
  EXPECT_EQ(result.columns[0].componentwise_backward_error, 1.0 / 9.0);
}

TEST(SolveDiagnostics, LaterDiagnosticOverflowPublishesNoPartialColumns)
{
  DenseMatrix<double> a(1, 1), b(1, 2), x(1, 2);
  a[0, 0] = 2;
  b[0, 0] = b[0, 1] = 1;
  x[0, 0] = 0.5;
  x[0, 1] = numeric_limits<double>::max();
  auto result = solve_diagnostics_with_info<double>(a, b, x, native_precision<double>());
  EXPECT_EQ(result.info.status, SolveStatus::nonfinite_result);
  EXPECT_FALSE(result.residual);
  EXPECT_TRUE(result.columns.empty());
}

TEST(LuRefinementApi, EmptyInputsAndInvalidArguments)
{
  auto p = LuRefinementPrecision<float, double, long double>::native();
  DenseMatrix<double> empty_a(0, 0), empty_b(0, 3);
  auto empty = lu_refine_with_info(empty_a, empty_b, p);
  ASSERT_TRUE(empty.info.succeeded());
  EXPECT_EQ(empty.columns.size(), 3);
  DenseMatrix<double> a(1, 1), b(1, 0);
  a[0, 0] = numeric_limits<double>::quiet_NaN();
  EXPECT_TRUE(lu_refine_with_info(a, b, p).info.succeeded());
  p.factor = native_precision<double>();
  EXPECT_THROW((void)lu_refine_with_info(a, b, p), std::invalid_argument);
  p.factor = native_precision<float>();
  p.residual = Precision::exact();
  EXPECT_THROW((void)lu_refine_with_info(a, b, p), std::invalid_argument);
}

TEST(LuRefinementApi, NonfiniteValuesAndFiniteConversionOverflow)
{
  auto p = LuRefinementPrecision<float, double, long double>::native();
  DenseMatrix<double> a(1, 1), b(1, 1);
  a[0, 0] = numeric_limits<double>::infinity();
  EXPECT_EQ(lu_refine_with_info(a, b, p).info.status, RefinementStatus::nonfinite_input);
  a[0, 0] = 1e100;
  auto result = lu_refine_with_info(a, b, p);
  EXPECT_EQ(result.info.status, RefinementStatus::nonfinite_result);
  EXPECT_FALSE(result.solution);
  a[0, 0] = 1;
  b[0, 0] = numeric_limits<double>::quiet_NaN();
  EXPECT_EQ(lu_refine_with_info(a, b, p).info.status, RefinementStatus::nonfinite_input);
}

TEST(LuRefinementApi, StrictAndOptionErrors)
{
  auto p = LuRefinementPrecision<float, double, long double>::native();
  DenseMatrix<double> a(1, 1), b(1, 1);
  a[0, 0] = 3;
  b[0, 0] = 1;
  EXPECT_DEATH((void)lu_refine(a, b, p, RefinementOptions<long double>{.max_iterations = 0}), "LU refinement failed");
  EXPECT_DEATH((void)lu_refine_with_info(a, b, p, {.backward_error_tolerance = -1.0L}), "nonnegative");
  DenseMatrix<double> bad(2, 1);
  EXPECT_DEATH((void)lu_refine_with_info(a, bad, p), "matching RHS");
}

#if UNI20_HAS_MPLAPACK_MPREAL
template <class S> void check_runtime_factors()
{
  auto pf = Precision::bits(64), pw = Precision::bits(128), pr = Precision::bits(256);
  LuRefinementPrecision<S> p{pf, pw, pr};
  DenseMatrix<S> a(2, 2, Precision::exact()), b(2, 1, Precision::exact());
  auto phase = [] {
    if constexpr (Complex<S>)
      return S(mpreal{1}, mpreal{1});
    else
      return mpreal{1};
  }();
  a[0, 0] = phase * 3;
  a[0, 1] = a[1, 0] = phase;
  a[1, 1] = phase * 2;
  b[0, 0] = phase;
  auto factor = lu_factor(LapackBackend{}, a, pf);
  auto result = lu_refine_with_info(LapackBackend{}, factor, a, b, p);
  ASSERT_TRUE(result.info.succeeded());
  EXPECT_EQ(precision_of(*result.solution), pw);
  EXPECT_EQ(((*result.solution)[0, 0].precision()), pw);
  EXPECT_EQ(result.columns[0].error.componentwise_backward_error.precision(), pr);
  EXPECT_EQ(factor.precision(), pf);
  EXPECT_TRUE((a[0, 0].is_exact()));
  EXPECT_GT(result.columns[0].corrections, 0);
  auto reference = mpreal("2/5", Precision::bits(512));
  EXPECT_LT(abs(uni20::real((*result.solution)[0, 0]).at(reference.precision()) - reference),
            mpreal("1e-38", reference.precision()));
  p.factor = Precision::exact();
  EXPECT_THROW((void)lu_refine_with_info(a, b, p), std::logic_error);
}
TEST(LuRefinementApi, ThreeRuntimePrecisionsAndExactInput) { check_runtime_factors<mpreal>(); }
TEST(LuRefinementApi, ThreeRuntimeComplexPrecisionsAndExactInput) { check_runtime_factors<complex<mpreal>>(); }
#endif
