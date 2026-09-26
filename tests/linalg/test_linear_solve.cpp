#include <uni20/common/gtest.hpp>
#include <uni20/core/numeric_limits.hpp>
#include <uni20/linalg/ops/linear_solve.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include "deferred_host_tensor.hpp"

#include <cmath>
#include <concepts>

namespace
{

template <class Matrix> void initialize_coefficients(Matrix& matrix)
{
  matrix[0, 0] = 3.0;
  matrix[0, 1] = 1.0;
  matrix[1, 0] = 1.0;
  matrix[1, 1] = 2.0;
}

template <class Matrix> void initialize_rhs(Matrix& rhs)
{
  rhs[0, 0] = 9.0;
  rhs[1, 0] = 8.0;
  rhs[0, 1] = 1.0;
  rhs[1, 1] = 0.0;
}

template <class Matrix> void check_solution(Matrix const& solution)
{
  EXPECT_NEAR((solution[0, 0]), 2.0, 1.0e-14);
  EXPECT_NEAR((solution[1, 0]), 3.0, 1.0e-14);
  EXPECT_NEAR((solution[0, 1]), 0.4, 1.0e-14);
  EXPECT_NEAR((solution[1, 1]), -0.2, 1.0e-14);
}

} // namespace

TEST(LinearSolveTest, ValueApiPreservesInputsAndSolvesMultipleRightHandSides)
{
  uni20::DenseMatrix<double, uni20::RowMajor> coefficients(2, 2);
  uni20::DenseMatrix<double, uni20::RowMajor> rhs(2, 2);
  initialize_coefficients(coefficients);
  initialize_rhs(rhs);

  auto solution = uni20::linalg::solve(coefficients, rhs);

  static_assert(std::same_as<typename decltype(solution)::layout_type, uni20::ColumnMajor>);
  check_solution(solution);
  EXPECT_DOUBLE_EQ((coefficients[0, 0]), 3.0);
  EXPECT_DOUBLE_EQ((coefficients[1, 1]), 2.0);
  EXPECT_DOUBLE_EQ((rhs[0, 0]), 9.0);
  EXPECT_DOUBLE_EQ((rhs[1, 1]), 0.0);
}

TEST(LinearSolveTest, LapackInplaceApiOverwritesWorkspaces)
{
  uni20::DenseMatrix<double> coefficients(2, 2);
  uni20::DenseMatrix<double> rhs(2, 2);
  initialize_coefficients(coefficients);
  initialize_rhs(rhs);

  uni20::linalg::solve_inplace(uni20::linalg::LapackBackend{}, coefficients, rhs);

  check_solution(rhs);
  EXPECT_NE((coefficients[1, 0]), 1.0);
}

TEST(LinearSolveTest, CpuReferencePerformsPartialPivoting)
{
  uni20::DenseMatrix<double> coefficients(2, 2);
  coefficients[0, 0] = 0.0;
  coefficients[0, 1] = 2.0;
  coefficients[1, 0] = 1.0;
  coefficients[1, 1] = 3.0;
  uni20::DenseMatrix<double> rhs(2, 1);
  rhs[0, 0] = 4.0;
  rhs[1, 0] = 7.0;

  uni20::linalg::solve_inplace(uni20::linalg::CpuReferenceBackend{}, coefficients, rhs);

  EXPECT_NEAR((rhs[0, 0]), 1.0, 1.0e-14);
  EXPECT_NEAR((rhs[1, 0]), 2.0, 1.0e-14);
}

TEST(LinearSolveTest, CpuReferenceSupportsComplexSystems)
{
  using complex_type = uni20::complex<double>;
  uni20::DenseMatrix<complex_type> coefficients(2, 2);
  coefficients[0, 0] = complex_type{1.0, 1.0};
  coefficients[0, 1] = complex_type{};
  coefficients[1, 0] = complex_type{};
  coefficients[1, 1] = complex_type{2.0, -1.0};
  uni20::DenseMatrix<complex_type> rhs(2, 1);
  rhs[0, 0] = complex_type{2.0, 2.0};
  rhs[1, 0] = complex_type{5.0, 0.0};

  auto solution = uni20::linalg::solve(uni20::linalg::CpuReferenceBackend{}, coefficients, rhs);

  EXPECT_NEAR(std::abs(solution[0, 0] - complex_type{2.0, 0.0}), 0.0, 1.0e-14);
  EXPECT_NEAR(std::abs(solution[1, 0] - complex_type{2.0, 1.0}), 0.0, 1.0e-14);
}

TEST(LinearSolveTest, LapackDeclinesRowMajorWorkspacesWithoutMutation)
{
  uni20::DenseMatrix<double, uni20::RowMajor> coefficients(2, 2);
  uni20::DenseMatrix<double, uni20::RowMajor> rhs(2, 2);
  initialize_coefficients(coefficients);
  initialize_rhs(rhs);
  auto coefficient_descriptor = uni20::mdspec_of(coefficients);
  auto rhs_descriptor = uni20::mdspec_of(rhs);
  uni20::linalg::SolveInfo info{.status = uni20::linalg::SolveStatus::small_pivot, .pivot = 17};
  uni20::linalg::SolveOptions<double> options;

  bool const accepted =
      uni20::linalg::try_dispatch_kernel(uni20::linalg::LapackBackend{}, uni20::linalg::linear_solve_op{},
                                         coefficient_descriptor, rhs_descriptor, info, options);

  EXPECT_FALSE(accepted);
  EXPECT_EQ(info.status, uni20::linalg::SolveStatus::small_pivot);
  EXPECT_EQ(info.pivot, 17);
  EXPECT_DOUBLE_EQ((coefficients[0, 0]), 3.0);
  EXPECT_DOUBLE_EQ((coefficients[1, 0]), 1.0);
  EXPECT_DOUBLE_EQ((rhs[0, 0]), 9.0);
  EXPECT_DOUBLE_EQ((rhs[1, 1]), 0.0);
}

TEST(LinearSolveTest, DefaultInplaceApiFallsBackForRowMajorWorkspaces)
{
  uni20::DenseMatrix<double, uni20::RowMajor> coefficients(2, 2);
  uni20::DenseMatrix<double, uni20::RowMajor> rhs(2, 2);
  initialize_coefficients(coefficients);
  initialize_rhs(rhs);

  uni20::linalg::solve_inplace(coefficients, rhs);

  check_solution(rhs);
}

TEST(LinearSolveTest, AcquiresDeferredHostWorkspaces)
{
  uni20::test::DeferredHostTensor<double, 2> coefficients(2, 2);
  uni20::test::DeferredHostTensor<double, 2> rhs(2, 2);
  {
    auto coefficient_access = uni20::test::acquire_host_write_access_sync(coefficients);
    auto rhs_access = uni20::test::acquire_host_write_access_sync(rhs);
    auto coefficient_span = coefficient_access.mdspan();
    auto rhs_span = rhs_access.mdspan();
    initialize_coefficients(coefficient_span);
    initialize_rhs(rhs_span);
  }

  uni20::linalg::solve_inplace(coefficients, rhs);

  auto rhs_access = uni20::test::acquire_host_read_access_sync(rhs);
  check_solution(rhs_access.mdspan());
}

TEST(LinearSolveTest, SingularSystemIsTerminal)
{
  uni20::DenseMatrix<double> coefficients(2, 2);
  coefficients[0, 0] = 1.0;
  coefficients[0, 1] = 2.0;
  coefficients[1, 0] = 2.0;
  coefficients[1, 1] = 4.0;
  uni20::DenseMatrix<double> rhs(2, 1);
  rhs[0, 0] = 1.0;
  rhs[1, 0] = 2.0;

  EXPECT_DEATH(
      { uni20::linalg::solve_inplace(uni20::linalg::CpuReferenceBackend{}, coefficients, rhs); },
      "singular matrix in solve");
}

TEST(LinearSolveTest, LapackSingularSystemIsTerminal)
{
  uni20::DenseMatrix<double> coefficients(2, 2);
  coefficients[0, 0] = 1.0;
  coefficients[0, 1] = 2.0;
  coefficients[1, 0] = 2.0;
  coefficients[1, 1] = 4.0;
  uni20::DenseMatrix<double> rhs(2, 1);
  rhs[0, 0] = 1.0;
  rhs[1, 0] = 2.0;

  EXPECT_DEATH(
      { uni20::linalg::solve_inplace(uni20::linalg::LapackBackend{}, coefficients, rhs); }, "found a singular matrix");
}

TEST(LinearSolveTest, EmptyRightHandSideIsAVacuousCpuNoOp)
{
  uni20::DenseMatrix<double> coefficients(2, 2);
  coefficients[0, 0] = 1.0;
  coefficients[0, 1] = 2.0;
  coefficients[1, 0] = 2.0;
  coefficients[1, 1] = 4.0;
  uni20::DenseMatrix<double> rhs(2, 0);

  uni20::linalg::solve_inplace(uni20::linalg::CpuReferenceBackend{}, coefficients, rhs);

  EXPECT_DOUBLE_EQ((coefficients[0, 0]), 1.0);
  EXPECT_DOUBLE_EQ((coefficients[0, 1]), 2.0);
  EXPECT_DOUBLE_EQ((coefficients[1, 0]), 2.0);
  EXPECT_DOUBLE_EQ((coefficients[1, 1]), 4.0);
}

TEST(LinearSolveTest, EmptyRightHandSideIsAVacuousLapackNoOp)
{
  uni20::DenseMatrix<double> coefficients(2, 2);
  coefficients[0, 0] = 1.0;
  coefficients[0, 1] = 2.0;
  coefficients[1, 0] = 2.0;
  coefficients[1, 1] = 4.0;
  uni20::DenseMatrix<double> rhs(2, 0);

  uni20::linalg::solve_inplace(uni20::linalg::LapackBackend{}, coefficients, rhs);

  EXPECT_DOUBLE_EQ((coefficients[0, 0]), 1.0);
  EXPECT_DOUBLE_EQ((coefficients[0, 1]), 2.0);
  EXPECT_DOUBLE_EQ((coefficients[1, 0]), 2.0);
  EXPECT_DOUBLE_EQ((coefficients[1, 1]), 4.0);
}

TEST(LinearSolveTest, RejectsMismatchedShapesBeforeDispatch)
{
  uni20::DenseMatrix<double> coefficients(2, 3);
  uni20::DenseMatrix<double> rhs(2, 1);

  EXPECT_DEATH({ (void)uni20::linalg::solve(coefficients, rhs); }, "solve requires a square coefficient matrix");
}

TEST(LinearSolveTest, RejectsMismatchedRightHandSideRowsBeforeDispatch)
{
  uni20::DenseMatrix<double> coefficients(2, 2);
  uni20::DenseMatrix<double> rhs(3, 1);

  EXPECT_DEATH(
      { (void)uni20::linalg::solve(coefficients, rhs); },
      "solve coefficient and right-hand-side row counts do not agree");
}

namespace
{
using namespace uni20::linalg;

template <class Scalar, class Backend> struct SolveCase
{
    using scalar = Scalar;
    using backend = Backend;
};

template <class Case> class RecoverableSolveTest : public testing::Test {};

using SolveCases =
    testing::Types<SolveCase<float, CpuReferenceBackend>, SolveCase<float, LapackBackend>,
                   SolveCase<double, CpuReferenceBackend>, SolveCase<double, LapackBackend>,
                   SolveCase<long double, CpuReferenceBackend>, SolveCase<uni20::complex<double>, CpuReferenceBackend>,
                   SolveCase<uni20::complex<double>, LapackBackend>
#if UNI20_HAS_FLOAT128 && UNI20_FLOAT128_PROVIDER_MPLAPACK
                   ,
                   SolveCase<uni20::float128, CpuReferenceBackend>, SolveCase<uni20::float128, LapackBackend>,
                   SolveCase<uni20::complex<uni20::float128>, CpuReferenceBackend>,
                   SolveCase<uni20::complex<uni20::float128>, LapackBackend>
#endif
                   >;
TYPED_TEST_SUITE(RecoverableSolveTest, SolveCases);

TYPED_TEST(RecoverableSolveTest, SuccessAndNativePrecision)
{
  using Scalar = typename TypeParam::scalar;
  using Real = uni20::make_real_t<Scalar>;
  Real const gap = Real{16} * uni20::numeric_limits<Real>::epsilon();
  uni20::DenseMatrix<Scalar> a(2, 2), b(2, 1);
  a[0, 0] = a[0, 1] = a[1, 0] = Scalar{1};
  a[1, 1] = Scalar{Real{1} + gap};
  b[0, 0] = Scalar{2};
  b[1, 0] = Scalar{Real{2} + gap};
  auto const info = solve_inplace_with_info(typename TypeParam::backend{}, a, b);
  ASSERT_TRUE(info.succeeded());
  EXPECT_FALSE(info.pivot.has_value());
  EXPECT_FLOATING_EQ((b[0, 0]), Scalar{1});
  EXPECT_FLOATING_EQ((b[1, 0]), Scalar{1});
}

TYPED_TEST(RecoverableSolveTest, ReportsSingularColumn)
{
  using Scalar = typename TypeParam::scalar;
  uni20::DenseMatrix<Scalar> a(2, 2), b(2, 1);
  a[0, 0] = Scalar{1};
  a[0, 1] = Scalar{2};
  a[1, 0] = Scalar{2};
  a[1, 1] = Scalar{4};
  b[0, 0] = Scalar{1};
  b[1, 0] = Scalar{2};
  auto const info = solve_inplace_with_info(typename TypeParam::backend{}, a, b);
  EXPECT_EQ(info.status, SolveStatus::singular);
  EXPECT_EQ(info.pivot, 1);
}

TYPED_TEST(RecoverableSolveTest, RelativeThresholdIsScaleInvariantAndOptional)
{
  using Scalar = typename TypeParam::scalar;
  using Real = uni20::make_real_t<Scalar>;
  Real const gap = Real{16} * uni20::numeric_limits<Real>::epsilon();
  for (Real scale : {Real{1}, uni20::numeric_limits<Real>::max() / Real{4},
                     uni20::numeric_limits<Real>::min() / uni20::numeric_limits<Real>::epsilon() * Real{1024}})
  {
    uni20::DenseMatrix<Scalar> a(2, 2), b(2, 1);
    a[0, 0] = Scalar{scale};
    a[0, 1] = a[1, 0] = Scalar{};
    a[1, 1] = Scalar{scale * gap};
    b[0, 0] = Scalar{scale};
    b[1, 0] = Scalar{scale * gap};
    auto const rejected =
        solve_inplace_with_info(typename TypeParam::backend{}, a, b, {.relative_pivot_tolerance = gap});
    EXPECT_EQ(rejected.status, SolveStatus::small_pivot);
    EXPECT_EQ(rejected.pivot, 1);
    // Reinitialize both destructive workspaces before retrying.
    a[0, 0] = Scalar{scale};
    a[0, 1] = a[1, 0] = Scalar{};
    a[1, 1] = Scalar{scale * gap};
    b[0, 0] = Scalar{scale};
    b[1, 0] = Scalar{scale * gap};
    ASSERT_TRUE(solve_inplace_with_info(typename TypeParam::backend{}, a, b).succeeded());
    EXPECT_FLOATING_EQ((b[0, 0]), Scalar{1});
    EXPECT_FLOATING_EQ((b[1, 0]), Scalar{1});
  }
}

TYPED_TEST(RecoverableSolveTest, ToleranceDoesNotNarrowToDouble)
{
  using Scalar = typename TypeParam::scalar;
  using Real = uni20::make_real_t<Scalar>;
  Real const epsilon = uni20::numeric_limits<Real>::epsilon();
  uni20::DenseMatrix<Scalar> a(2, 2), b(2, 1);
  a[0, 0] = Scalar{1};
  a[0, 1] = a[1, 0] = Scalar{};
  a[1, 1] = Scalar{Real{1} - Real{8} * epsilon};
  b[0, 0] = b[1, 0] = Scalar{1};
  auto const info = solve_inplace_with_info(typename TypeParam::backend{}, a, b,
                                            {.relative_pivot_tolerance = Real{1} - Real{4} * epsilon});
  EXPECT_EQ(info.status, SolveStatus::small_pivot);
  EXPECT_EQ(info.pivot, 1);
}

TYPED_TEST(RecoverableSolveTest, NonfiniteInputDoesNotMutateWorkspaces)
{
  using Scalar = typename TypeParam::scalar;
  using Real = uni20::make_real_t<Scalar>;
  for (Real invalid : {uni20::numeric_limits<Real>::infinity(), uni20::numeric_limits<Real>::quiet_NaN()})
    for (bool invalid_rhs : {false, true})
    {
      uni20::DenseMatrix<Scalar> a(1, 1), b(1, 1);
      Scalar nonfinite{invalid};
      if constexpr (uni20::Complex<Scalar>) nonfinite = Scalar{Real{}, invalid};
      a[0, 0] = invalid_rhs ? Scalar{2} : nonfinite;
      b[0, 0] = invalid_rhs ? nonfinite : Scalar{3};
      auto const info = solve_inplace_with_info(typename TypeParam::backend{}, a, b);
      EXPECT_EQ(info.status, SolveStatus::nonfinite_input);
      EXPECT_FALSE(info.pivot.has_value());
      if (invalid_rhs)
        EXPECT_TRUE((a[0, 0] == Scalar{2}));
      else
        EXPECT_TRUE((b[0, 0] == Scalar{3}));
    }
}

TYPED_TEST(RecoverableSolveTest, ReportsOverflowInSolutionAndFactorization)
{
  using Scalar = typename TypeParam::scalar;
  using Real = uni20::make_real_t<Scalar>;
  uni20::DenseMatrix<Scalar> a(1, 1), b(1, 1);
  a[0, 0] = Scalar{Real{1} / Real{4}};
  b[0, 0] = Scalar{uni20::numeric_limits<Real>::max() / Real{2}};
  EXPECT_EQ(solve_inplace_with_info(typename TypeParam::backend{}, a, b).status, SolveStatus::nonfinite_result);
  uni20::DenseMatrix<Scalar> c(2, 2), d(2, 1);
  Real const large = uni20::numeric_limits<Real>::max();
  c[0, 0] = c[0, 1] = c[1, 1] = Scalar{large};
  c[1, 0] = Scalar{-large};
  d[0, 0] = d[1, 0] = Scalar{};
  EXPECT_EQ(solve_inplace_with_info(typename TypeParam::backend{}, c, d).status, SolveStatus::nonfinite_result);
}

TYPED_TEST(RecoverableSolveTest, EmptyProblemsSucceedWithoutInspectingValues)
{
  using Scalar = typename TypeParam::scalar;
  using Real = uni20::make_real_t<Scalar>;
  uni20::DenseMatrix<Scalar> a(1, 1), b(1, 0), c(0, 0), d(0, 3);
  a[0, 0] = Scalar{uni20::numeric_limits<Real>::infinity()};
  EXPECT_TRUE(solve_inplace_with_info(typename TypeParam::backend{}, a, b).succeeded());
  EXPECT_TRUE(solve_inplace_with_info(typename TypeParam::backend{}, c, d).succeeded());
  EXPECT_FALSE(uni20::isfinite(a[0, 0]));
}

TEST(RecoverableSolveDispatchTest, ComplexMagnitudesUseAbsoluteValueNotComponentSum)
{
  using Scalar = uni20::complex<double>;
  for (bool lapack : {false, true})
  {
    uni20::DenseMatrix<Scalar> a(2, 2), b(2, 1);
    a[0, 0] = Scalar{3, 4}; // magnitude 5, component sum 7
    a[0, 1] = a[1, 0] = Scalar{};
    a[1, 1] = Scalar{4};
    b[0, 0] = Scalar{3, 4};
    b[1, 0] = Scalar{4};
    SolveOptions<double> options{.relative_pivot_tolerance = 0.7};
    auto const info = lapack ? solve_inplace_with_info(LapackBackend{}, a, b, options)
                             : solve_inplace_with_info(CpuReferenceBackend{}, a, b, options);
    EXPECT_TRUE(info.succeeded());
    EXPECT_FLOATING_EQ((b[0, 0]), Scalar{1});
    EXPECT_FLOATING_EQ((b[1, 0]), Scalar{1});
    double const large = uni20::numeric_limits<double>::max();
    a[0, 0] = Scalar{large, large};
    auto const overflow =
        lapack ? solve_inplace_with_info(LapackBackend{}, a, b) : solve_inplace_with_info(CpuReferenceBackend{}, a, b);
    EXPECT_EQ(overflow.status, SolveStatus::nonfinite_result);
  }
}

struct NegatingReference
{
    double* value;
    operator double() const { return -*value; }
    void operator=(double x) const { *value = -x; }
};

struct NegatingAccessor
{
    using element_type = double;
    using data_handle_type = double*;
    using reference = NegatingReference;
    using offset_policy = NegatingAccessor;
    reference access(double* data, std::size_t offset) const { return {data + offset}; }
    double* offset(double* data, std::size_t offset) const { return data + offset; }
};

} // namespace

template <> inline constexpr bool uni20::enable_accessor_in_domain<NegatingAccessor, uni20::host_access_domain> = true;

namespace
{

TEST(RecoverableSolveDispatchTest, MutableProxyAccessorIsRespected)
{
  using Span = stdex::mdspan<double, stdex::dextents<uni20::index_type, 2>, stdex::layout_left, NegatingAccessor>;
  double a_storage[]{3, 1, 1, 2}, b_storage[]{9, 8};
  Span a(a_storage, 2, 2), b(b_storage, 2, 1);
  SolveInfo info{.status = SolveStatus::singular, .pivot = 99};
  SolveOptions<double> options{.relative_pivot_tolerance = 0.01};
  dispatch_kernel(backend_list{LapackBackend{}, CpuReferenceBackend{}}, linear_solve_op{}, a, b, info, options);
  EXPECT_TRUE(info.succeeded());
  EXPECT_FALSE(info.pivot.has_value());
  EXPECT_DOUBLE_EQ(b_storage[0], -2);
  EXPECT_DOUBLE_EQ(b_storage[1], -3);
}

struct UnexpectedFallback
{
    static constexpr std::string_view name = "unexpected_fallback";
    bool* called;
};
template <class... Args>
consteval auto kernel_accepts_types(UnexpectedFallback const&, linear_solve_op const&, Args&...)
{
  return kernel_types_yes;
}
template <class... Args> KernelAttempt try_kernel(UnexpectedFallback backend, linear_solve_op const&, Args&...)
{
  *backend.called = true;
  return KernelAttempt::success;
}

TEST(RecoverableSolveDispatchTest, NumericalFailureNeverAttemptsFallback)
{
  for (bool lapack : {false, true})
  {
    uni20::DenseMatrix<double> a(2, 2), b(2, 1);
    a[0, 0] = 1;
    a[0, 1] = 2;
    a[1, 0] = 2;
    a[1, 1] = 4;
    b[0, 0] = 1;
    b[1, 0] = 2;
    bool called = false;
    auto const info =
        lapack ? solve_inplace_with_info(backend_list{LapackBackend{}, UnexpectedFallback{&called}}, a, b)
               : solve_inplace_with_info(backend_list{CpuReferenceBackend{}, UnexpectedFallback{&called}}, a, b);
    EXPECT_EQ(info.status, SolveStatus::singular);
    EXPECT_FALSE(called);
  }
}

TEST(RecoverableSolveDispatchTest, InvalidToleranceIsTerminal)
{
  uni20::DenseMatrix<double> a(1, 1), b(1, 1);
  a[0, 0] = b[0, 0] = 1;
  for (double invalid : {-1.0, uni20::numeric_limits<double>::infinity(), uni20::numeric_limits<double>::quiet_NaN()})
    EXPECT_DEATH(
        { (void)solve_inplace_with_info(a, b, {.relative_pivot_tolerance = invalid}); },
        "finite nonnegative relative pivot tolerance");
}
} // namespace
