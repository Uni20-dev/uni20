#include <cmath>
#include <gtest/gtest.h>
#include <limits>
#include <uni20/linalg/ops/linear_solve.hpp>
#include <uni20/linalg/ops/matrix_product.hpp>
#include <uni20/linalg/ops/slogdet.hpp>
#include <uni20/tensor/conjugate.hpp>
#include <uni20/tensor/reshape.hpp>

using namespace uni20;
using namespace uni20::linalg;

TEST(Lu, ReconstructAndReuse)
{
  DenseMatrix<double, RowMajor> a(3, 3), b(3, 1);
  double values[3][3] = {{1, 1, 1}, {2, 2, 3}, {3, 4, 5}};
  for (std::size_t i = 0; i < 3; ++i)
  {
    b[i, 0] = 0;
    for (std::size_t j = 0; j < 3; ++j)
    {
      a[i, j] = values[i][j];
      b[i, 0] += values[i][j] * (j + 1);
    }
  }
  auto check = [&](auto backend) {
    auto f = lu_factor(backend, a);
    auto pa = make_tensor<ColumnMajor>(a);
    for (std::size_t k = 0; k < 3; ++k)
      for (std::size_t j = 0; j < 3; ++j)
        std::swap(pa[k, j], pa[f.pivots()[k], j]);
    auto const& packed = f.packed();
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 3; ++j)
      {
        double product = 0;
        for (std::size_t k = 0; k < 3; ++k)
          product += (i == k ? 1 : (i > k ? packed[i, k] : 0)) * (k <= j ? packed[k, j] : 0);
        EXPECT_NEAR(product, (pa[i, j]), 1e-14);
      }
    auto saved = make_tensor<ColumnMajor>(packed);
    for (int repeat = 0; repeat < 2; ++repeat)
    {
      auto x = lu_solve(backend, f, b);
      for (std::size_t i = 0; i < 3; ++i)
        EXPECT_NEAR((x[i, 0]), i + 1, 1e-14);
    }
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 3; ++j)
        EXPECT_EQ((saved[i, j]), (packed[i, j]));
    auto determinant = slogdet(f);
    EXPECT_EQ(determinant.phase, -1);
    EXPECT_NEAR(determinant.log_absolute, 0, 1e-14);
  };
  check(CpuReferenceBackend{});
  check(LapackBackend{});
}

TEST(Lu, ComplexMagnitudeOverflowAndPhase)
{
  DenseMatrix<complex<double>> a(1, 1);
  a[0, 0] = {1.7e308, 1.7e308};
  auto check = [&](auto backend) {
    auto d = slogdet(backend, a);
    EXPECT_NEAR(d.log_absolute, std::log(1.7e308) + std::log(2.0) / 2, 1e-12);
    EXPECT_NEAR(d.phase.real(), std::sqrt(0.5), 1e-15);
    EXPECT_NEAR(d.phase.imag(), std::sqrt(0.5), 1e-15);
    auto rejected = slogdet_with_info(backend, a, SolveOptions<double>{.relative_pivot_tolerance = 1});
    EXPECT_EQ(rejected.info.status, SolveStatus::small_pivot);
    EXPECT_FALSE(rejected.value);
  };
  check(CpuReferenceBackend{});
  check(LapackBackend{});
}

TEST(Lu, SingularThresholdEmptyAndNonfinite)
{
  DenseMatrix<double> a(2, 2);
  a[0, 0] = 1;
  a[1, 1] = 0;
  auto check = [&](auto backend) {
    auto f = lu_factor_with_info(backend, a);
    EXPECT_EQ(f.info.status, SolveStatus::singular);
    EXPECT_EQ(f.info.pivot, 1);
    EXPECT_FALSE(f.factor);
    auto d = slogdet_with_info(backend, a);
    ASSERT_TRUE(d.value);
    EXPECT_EQ(d.info.status, SolveStatus::singular);
    EXPECT_EQ(d.value->phase, 0);
    EXPECT_EQ(d.value->log_absolute, -numeric_limits<double>::infinity());
    EXPECT_EQ(slogdet(backend, a).phase, 0);
  };
  check(CpuReferenceBackend{});
  check(LapackBackend{});
  a[1, 1] = 1e-20;
  auto small = slogdet_with_info(a, SolveOptions<double>{.relative_pivot_tolerance = 1e-10});
  EXPECT_EQ(small.info.status, SolveStatus::small_pivot);
  EXPECT_FALSE(small.value);
  EXPECT_TRUE(lu_factor_with_info(a).factor);
  DenseMatrix<double> empty(0, 0);
  auto d = slogdet(empty);
  EXPECT_EQ(d.phase, 1);
  EXPECT_EQ(d.log_absolute, 0);
  a[0, 1] = numeric_limits<double>::infinity();
  EXPECT_EQ(lu_factor_with_info(a).info.status, SolveStatus::nonfinite_input);
}

TEST(Lu, FailedRhsDoesNotInvalidateFactors)
{
  DenseMatrix<double> a(1, 1), b(1, 1);
  a[0, 0] = 0.5;
  auto check = [&](auto backend) {
    auto f = lu_factor(backend, a);
    b[0, 0] = numeric_limits<double>::infinity();
    EXPECT_EQ(lu_solve_inplace_with_info(backend, f, b).status, SolveStatus::nonfinite_input);
    EXPECT_EQ((b[0, 0]), numeric_limits<double>::infinity());
    b[0, 0] = numeric_limits<double>::max();
    EXPECT_EQ(lu_solve_inplace_with_info(backend, f, b).status, SolveStatus::nonfinite_result);
    b[0, 0] = 3;
    auto x = lu_solve(backend, f, b);
    EXPECT_EQ((x[0, 0]), 6);
    EXPECT_EQ((f.packed()[0, 0]), 0.5);
  };
  check(CpuReferenceBackend{});
  check(LapackBackend{});
}

#if UNI20_ENABLE_MPLAPACK_MPFR
TEST(Lu, RuntimePrecisionAndComplex)
{
  auto p = Precision::bits(512);
  DenseMatrix<mpreal> a(2, 2, Precision::exact()), b(2, 1, Precision::bits(256));
  a[0, 0] = a[0, 1] = a[1, 0] = mpreal{1};
  a[1, 1] = mpreal(exact_constant{1} + exact_constant("1e-100"));
  b[0, 0] = mpreal{2};
  b[1, 0] = mpreal(exact_constant{2} + exact_constant("1e-100"));
  auto f = lu_factor(a, p);
  EXPECT_EQ(f.precision(), p);
  auto x = lu_solve(f, b);
  EXPECT_EQ(x.default_precision(), p);
  EXPECT_LT(abs(x[0, 0] - 1), mpreal("1e-50", p));
  EXPECT_LT(abs(x[1, 0] - 1), mpreal("1e-50", p));
  auto q = Precision::bits(768);
  for (std::size_t i = 0; i < 2; ++i)
  {
    auto residual = a[i, 0].at(q) * x[0, 0].at(q) + a[i, 1].at(q) * x[1, 0].at(q) - b[i, 0].at(q);
    EXPECT_LT(abs(residual), mpreal("1e-152", q));
  }
  auto d = slogdet(f);
  auto expected = log(mpreal("1e-100", p));
  EXPECT_LT(abs(d.log_absolute - expected), mpreal("1e-50", p));
  EXPECT_EQ(d.phase, 1);
  EXPECT_EQ(d.log_absolute.precision(), p);
  auto low = lu_factor_with_info(a, Precision::bits(113));
  EXPECT_EQ(low.info.status, SolveStatus::singular);
  EXPECT_FALSE(low.factor);
  DenseMatrix<complex<mpreal>> c(1, 1, p);
  c[0, 0] = complex<mpreal>(mpreal{3}, mpreal{4});
  auto cd = slogdet(c);
  EXPECT_EQ(cd.phase.precision(), p);
  EXPECT_LT(abs(cd.phase.real() - mpreal("0.6", p)), mpreal("1e-150", p));
  EXPECT_LT(abs(cd.log_absolute - log(mpreal{5}, p)), mpreal("1e-150", p));
}

TEST(Lu, SingularComplexRuntimeDeterminantRetainsPrecision)
{
  auto p = Precision::bits(256);
  DenseMatrix<complex<mpreal>> a(2, 2, p);
  a[0, 0] = complex<mpreal>(mpreal{1}, mpreal{1}, p);
  auto d = slogdet_with_info(MplapackMpfrBackend{}, a);
  ASSERT_TRUE(d.value);
  EXPECT_EQ(d.info.status, SolveStatus::singular);
  EXPECT_EQ(d.value->phase, complex<mpreal>(0, p));
  EXPECT_EQ(d.value->phase.precision(), p);
  EXPECT_EQ(d.value->log_absolute.precision(), p);
  EXPECT_EQ(d.value->log_absolute, mpreal("-inf", p));
}
#endif

TEST(Lu, NativePrecisionsExtremeProductsAndConjugation)
{
  auto check = []<class R>() {
    DenseMatrix<R> a(4, 4);
    using std::log;
    using std::exp;
    auto big = exp(R{50});
    auto small = exp(R{-50});
    a[0, 0] = big;
    a[1, 1] = big;
    a[2, 2] = small;
    a[3, 3] = small;
    auto d = slogdet(CpuReferenceBackend{}, a);
    EXPECT_EQ(d.phase, R{1});
    EXPECT_LT(abs(d.log_absolute), R{128} * numeric_limits<R>::epsilon());
    // The product overflows float32; every logarithm and the result stay finite.
    a[2, 2] = big;
    a[3, 3] = big;
    EXPECT_LT(abs(slogdet(CpuReferenceBackend{}, a).log_absolute - R{200}), R{1024} * numeric_limits<R>::epsilon());
    DenseMatrix<complex<R>, RowMajor> c(2, 2);
    c[0, 0] = complex<R>{1, 1};
    c[0, 1] = complex<R>{2};
    c[1, 0] = complex<R>{3};
    c[1, 1] = complex<R>{4, -1};
    auto d1 = slogdet(CpuReferenceBackend{}, c);
    auto d2 = slogdet(CpuReferenceBackend{}, uni20::conj(c));
    auto exact = complex<R>{-1, 3};
    auto magnitude = abs(exact);
    EXPECT_LT(abs(d1.phase - exact / magnitude), R{128} * numeric_limits<R>::epsilon());
    EXPECT_LT(abs(d2.phase - uni20::conj(d1.phase)), R{128} * numeric_limits<R>::epsilon());
    EXPECT_LT(abs(d1.log_absolute - log(magnitude)), R{128} * numeric_limits<R>::epsilon());
  };
  check.template operator()<float>();
  check.template operator()<double>();
  check.template operator()<long double>();
#if UNI20_HAS_FLOAT128
  check.template operator()<float128>();
#endif
}

TEST(Lu, DirectDeclineAndStridedCpuFactors)
{
  using E = stdex::dextents<std::size_t, 2>;
  double storage[12] = {0};
  stdex::mdspan<double, E, stdex::layout_stride> a(
      storage, stdex::layout_stride::mapping<E>(E(2, 2), std::array<std::size_t, 2>{5, 2}));
  a[0, 0] = 0;
  a[0, 1] = 2;
  a[1, 0] = 1;
  a[1, 1] = 3;
  std::size_t pivots[2] = {71, 72};
  SolveInfo info{.status = SolveStatus::small_pivot, .pivot = 17};
  EXPECT_EQ(try_kernel(LapackBackend{}, lu_factor_op{}, a, std::span(pivots), info, SolveOptions<double>{}),
            KernelAttempt::unsupported_layout);
  EXPECT_EQ(info.status, SolveStatus::small_pivot);
  EXPECT_EQ(info.pivot, 17);
  EXPECT_EQ(pivots[0], 71);
  EXPECT_EQ(pivots[1], 72);
  EXPECT_EQ((a[1, 0]), 1);
  EXPECT_EQ(try_kernel(CpuReferenceBackend{}, lu_factor_op{}, a, std::span(pivots), info, SolveOptions<double>{}),
            KernelAttempt::success);
  EXPECT_TRUE(info.succeeded());
  double rhs_data[4] = {4, 0, 7, 0};
  stdex::mdspan<double, E, stdex::layout_stride> rhs(
      rhs_data, stdex::layout_stride::mapping<E>(E(2, 1), std::array<std::size_t, 2>{2, 1}));
  auto readonly = make_const_mdspan(a);
  EXPECT_EQ(try_kernel(CpuReferenceBackend{}, lu_solve_op{}, readonly, std::span<std::size_t const>(pivots), rhs, info),
            KernelAttempt::success);
  EXPECT_NEAR((rhs[0, 0]), 1, 1e-14);
  EXPECT_NEAR((rhs[1, 0]), 2, 1e-14);
}

TEST(Lu, DirectCpuFactorizationAcrossStorageOrders)
{
  auto check = [](auto a) {
    using S = typename decltype(a)::value_type;
    using R = make_real_t<S>;
    S original[3][3];
    double values[3][3] = {{1, 1, 1}, {2, 2, 3}, {3, 4, 5}};
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 3; ++j)
      {
        S value = S(values[i][j]);
        if constexpr (Complex<S>) value.imag(double(i) - double(j));
        original[i][j] = a[i, j] = value;
      }
    std::size_t pivots[3];
    SolveInfo info;
    ASSERT_EQ(try_kernel(CpuReferenceBackend{}, lu_factor_op{}, a, std::span(pivots), info, SolveOptions<R>{}),
              KernelAttempt::success);
    ASSERT_TRUE(info.succeeded());
    for (std::size_t k = 0; k < 3; ++k)
      for (std::size_t j = 0; j < 3; ++j)
        std::swap(original[k][j], original[pivots[k]][j]);
    // Reconstruct P*A from L and U rather than comparing two loop implementations.
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 3; ++j)
      {
        S product{};
        for (std::size_t k = 0; k < 3; ++k)
          product += (i == k ? S{1} : (i > k ? a[i, k] : S{})) * (k <= j ? a[k, j] : S{});
        EXPECT_LE(math::abs(product - original[i][j]), R{256} * numeric_limits<R>::epsilon());
      }

    // A finite input whose first trailing update overflows must report failure
    // in either traversal, including complex arithmetic and padded mappings.
    for (std::size_t i = 0; i < 3; ++i)
      for (std::size_t j = 0; j < 3; ++j)
        a[i, j] = S(i == j ? 1 : 0);
    a[1, 0] = S{1};
    a[0, 1] = S(numeric_limits<R>::max());
    a[1, 1] = S(-numeric_limits<R>::max());
    EXPECT_EQ(try_kernel(CpuReferenceBackend{}, lu_factor_op{}, a, std::span(pivots), info, SolveOptions<R>{}),
              KernelAttempt::success);
    EXPECT_EQ(info.status, SolveStatus::nonfinite_result);
  };
  auto layouts = [&]<class S>() {
    DenseMatrix<S, ColumnMajor> column_major(3, 3);
    DenseMatrix<S, RowMajor> row_major(3, 3);
    check(column_major.mdspan());
    check(row_major.mdspan());
    using E = stdex::dextents<std::size_t, 2>;
    S storage[27]{};
    for (auto strides : {std::array<std::size_t, 2>{2, 11}, std::array<std::size_t, 2>{11, 2}})
    {
      stdex::mdspan<S, E, stdex::layout_stride> padded(
          storage, stdex::layout_stride::mapping<E>(E(3, 3), strides));
      check(padded);
    }
  };
  layouts.template operator()<double>();
  layouts.template operator()<complex<double>>();
}

namespace
{
struct UnexpectedLuBackend
{
    static constexpr std::string_view name = "unexpected_lu";
    bool* called;
};
template <class... Args> consteval auto kernel_accepts_types(UnexpectedLuBackend, lu_factor_op, Args&&...)
{
  return kernel_types_yes;
}
template <class... Args> KernelAttempt try_kernel(UnexpectedLuBackend b, lu_factor_op, Args&&...)
{
  *b.called = true;
  return KernelAttempt::unsupported_instance;
}
} // namespace
TEST(Lu, NumericalFailureNeverFallsBack)
{
  DenseMatrix<double> a(2, 2);
  a[0, 0] = 1;
  bool called = false;
  auto result = lu_factor_with_info(backend_list{CpuReferenceBackend{}, UnexpectedLuBackend{&called}}, a);
  EXPECT_EQ(result.info.status, SolveStatus::singular);
  EXPECT_FALSE(called);
}

#if UNI20_ENABLE_MPLAPACK_MPFR
TEST(Lu, FactorPrecisionOwnsReusePolicy)
{
  auto p = Precision::bits(256), q = Precision::bits(128);
  DenseMatrix<mpreal> a(1, 1, q), b(1, 1, q);
  a[0, 0] = mpreal("1.5", Precision::bits(512));
  b[0, 0] = mpreal{3};
  auto factor = lu_factor(a, p);
  EXPECT_EQ(a.default_precision(), q);
  EXPECT_EQ((a[0, 0].precision()), Precision::bits(512));
  auto view = reshape_view(b, 1, 1);
  EXPECT_TRUE(lu_solve_inplace_with_info(factor, view).succeeded());
  EXPECT_EQ((b[0, 0]), 2);
  EXPECT_EQ((b[0, 0].precision()), p);
  EXPECT_EQ(b.default_precision(), q);
  b[0, 0] = mpreal("inf", q);
  EXPECT_EQ(lu_solve_inplace_with_info(factor, b).status, SolveStatus::nonfinite_input);
  EXPECT_EQ(b.default_precision(), q);
  b[0, 0] = mpreal{6};
  EXPECT_TRUE(lu_solve_inplace_with_info(factor, b).succeeded());
  EXPECT_EQ((b[0, 0]), 4);
  EXPECT_EQ(b.default_precision(), p);
  EXPECT_EQ((factor.packed()[0, 0]), mpreal("1.5", p));

  DenseMatrix<mpreal> empty(0, 0, Precision::exact()), empty_rhs(0, 2, q);
  auto f0 = lu_factor(empty, p);
  EXPECT_EQ(f0.precision(), p);
  auto d0 = slogdet(f0);
  EXPECT_EQ(d0.phase, 1);
  EXPECT_EQ(d0.log_absolute, 0);
  EXPECT_EQ(d0.phase.precision(), p);
  EXPECT_EQ(d0.log_absolute.precision(), p);
  EXPECT_TRUE(lu_solve_inplace_with_info(f0, empty_rhs).succeeded());
  EXPECT_EQ(empty_rhs.default_precision(), q);
  auto x0 = lu_solve(f0, empty_rhs);
  EXPECT_EQ(x0.default_precision(), p);
}
#endif

TEST(Lu, FactorsArePortableBetweenHostBackends)
{
  auto check = []<class R>() {
    DenseMatrix<complex<R>> a(2, 2);
    a[0, 0] = {0, 1};
    a[0, 1] = {2, 1};
    a[1, 0] = {3, -2};
    a[1, 1] = {4, 3};
    auto f = lu_factor(CpuReferenceBackend{}, a);
    auto copy = f;
    auto x = lu_solve(LapackBackend{}, copy, a);
    auto lapack = lu_factor(LapackBackend{}, a);
    auto y = lu_solve(CpuReferenceBackend{}, lapack, a);
    for (std::size_t i = 0; i < 2; ++i)
      for (std::size_t j = 0; j < 2; ++j)
      {
        auto expected = complex<R>(i == j ? 1 : 0);
        EXPECT_LT(abs(x[i, j] - expected), R{128} * numeric_limits<R>::epsilon());
        EXPECT_LT(abs(y[i, j] - expected), R{128} * numeric_limits<R>::epsilon());
      }
    EXPECT_NE(f.packed().storage().data(), copy.packed().storage().data());
  };
  check.template operator()<float>();
  check.template operator()<double>();
#if UNI20_ENABLE_MPLAPACK
  check.template operator()<float128>();
#endif
}
