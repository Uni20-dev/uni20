#include "precision_cases.hpp"
#include <uni20/linalg/ops/linear_solve.hpp>
#include <uni20/linalg/ops/matrix_norm.hpp>
#include <uni20/linalg/ops/matrix_product.hpp>
#include <uni20/tensor/reductions.hpp>

namespace uni20::test
{
template <class C> using NumericalLinalg = PrecisionTest<C>;
TYPED_TEST_SUITE(NumericalLinalg, PrecisionCases, PrecisionCaseNames);

TYPED_TEST(NumericalLinalg, CpuMatrixOneNormRetainsIncrement)
{
  using C = TypeParam;
  this->RecordProperty("backend", "cpu_reference");
  if constexpr (!C::available)
    GTEST_SKIP() << "unavailable: scalar dependency or native format not configured";
  else
  {
    auto a = C::matrix(2, 2);
    // Imaginary entries exercise complex magnitude without irrational oracles.
    a[0, 0] = C::scalar(C::real(1) + C::gap(), C::real(0));
    a[1, 0] = C::is_complex ? C::scalar(0, 1) : C::scalar(1);
    auto norm = linalg::matrix_norm_host(linalg::CpuReferenceBackend{}, a, linalg::MatrixNorm::One);
    EXPECT_EQ(norm, C::real(2) + C::gap());
    C::expect_precision(norm);
  }
}

TYPED_TEST(NumericalLinalg, CpuReductionsRetainIncrement)
{
  using C = TypeParam;
  this->RecordProperty("backend", "cpu_reference");
  if constexpr (!C::available)
    GTEST_SKIP() << "unavailable: scalar dependency or native format not configured";
  else
  {
    auto a = C::matrix(2, 1), b = C::matrix(2, 1);
    a[0, 0] = C::scalar(C::real(1) + C::gap(), C::real(1));
    a[1, 0] = C::scalar(1, -1);
    b[0, 0] = C::scalar(1);
    b[1, 0] = C::scalar(2);
    auto sum = sum_host(linalg::CpuReferenceBackend{}, a);
    auto inner = inner_product_host(linalg::CpuReferenceBackend{}, a, b);
    expect_equal(sum, C::scalar(C::real(2) + C::gap(), C::real(0)));
    expect_equal(inner, C::scalar(C::real(3) + C::gap(), C::real(1)));
    auto norm = norm_host(linalg::CpuReferenceBackend{}, a);
    auto squared = (C::real(1) + C::gap()) * (C::real(1) + C::gap()) + C::real(C::is_complex ? 3 : 1);
    expect_error_at_most(norm * norm, squared, C::real(16) * C::epsilon());
    C::expect_precision(sum);
    C::expect_precision(inner);
    C::expect_precision(norm);
  }
}

template <class C, class Backend> void check_product(Backend backend)
{
  testing::Test::RecordProperty("backend", std::string(Backend::name));
  auto a = C::matrix(2, 2), b = C::matrix(2, 1), out = C::matrix(2, 1);
  a[0, 0] = C::scalar(C::real(1) + C::gap(), C::real(1));
  a[0, 1] = C::scalar(1);
  a[1, 0] = C::scalar(1, 1);
  a[1, 1] = C::scalar(1, -1);
  b[0, 0] = b[1, 0] = C::scalar(1, 1);
  linalg::assign_product(backend, out, a, b);
  auto expected = C::scalar(C::real(C::is_complex ? 1 : 2) + C::gap(), C::real(3) + C::gap());
  expect_equal(typename C::scalar_type(out[0, 0]), expected);
  expect_equal(typename C::scalar_type(out[1, 0]), C::scalar(2, 2));
  C::expect_precision(out[0, 0]);
}

TYPED_TEST(NumericalLinalg, CpuGemmRetainsIncrement)
{
  using C = TypeParam;
  this->RecordProperty("backend", "cpu_reference");
  if constexpr (!C::available)
    GTEST_SKIP() << "unavailable: scalar dependency or native format not configured";
  else if constexpr (C::runtime)
    GTEST_SKIP() << "unsupported: CPU GEMM handles exact mode only for runtime scalars";
  else
    check_product<C>(linalg::CpuReferenceBackend{});
}

TYPED_TEST(NumericalLinalg, ProviderGemmRetainsIncrement)
{
  using C = TypeParam;
  this->RecordProperty("backend", "blas_or_mplapack");
  if constexpr (!C::available)
    GTEST_SKIP() << "unavailable: scalar dependency or native format not configured";
  else if constexpr (C::runtime)
  {
#if UNI20_ENABLE_MPLAPACK_MPFR
    check_product<C>(linalg::MplapackMpfrBackend{});
#else
    GTEST_SKIP() << "unavailable: configure UNI20_ENABLE_MPLAPACK_MPFR";
#endif
  }
  else if constexpr (C::native_dense_provider)
    check_product<C>(linalg::BlasBackend{});
  else
    GTEST_SKIP() << "unsupported: float80 provider not wired on this branch";
}

template <class C, class Backend> void check_solve(Backend backend, bool exponent_range)
{
  testing::Test::RecordProperty("backend", std::string(Backend::name));
  auto a = C::matrix(2, 2), b = C::matrix(2, 1);
  auto scale = exponent_range ? C::power_of_two(-1200) : C::real(1);
  auto const phase = C::scalar(1, 1);
  // Exact dyadic data, analytic x=(1,2). Rounding away the small gap makes
  // the system singular; residuals are checked without calling provider GEMM.
  a[0, 0] = a[0, 1] = a[1, 0] = C::scalar(scale, C::real(0)) * phase;
  a[1, 1] = C::scalar(scale * (C::real(1) + C::gap()), C::real(0)) * phase;
  b[0, 0] = C::scalar(C::real(3) * scale, C::real(0)) * phase;
  b[1, 0] = C::scalar((C::real(3) + C::real(2) * C::gap()) * scale, C::real(0)) * phase;
  auto x = linalg::solve(backend, a, b);
  expect_error_at_most(typename C::scalar_type(x[0, 0]), C::scalar(1), C::real(16) * C::epsilon());
  expect_error_at_most(typename C::scalar_type(x[1, 0]), C::scalar(2), C::real(16) * C::epsilon());
  C::expect_precision(x[0, 0]);
  for (index_type i = 0; i < 2; ++i)
    expect_error_at_most((a[i, 0] * x[0, 0] + a[i, 1] * x[1, 0]) / C::scalar(scale, C::real(0)),
                         typename C::scalar_type(b[i, 0] / C::scalar(scale, C::real(0))), C::real(64) * C::epsilon());
}

TYPED_TEST(NumericalLinalg, CpuSolveResolvesSmallGap)
{
  using C = TypeParam;
  this->RecordProperty("backend", "cpu_reference");
  if constexpr (!C::available)
    GTEST_SKIP() << "unavailable: scalar dependency or native format not configured";
  else if constexpr (C::runtime)
    GTEST_SKIP() << "unsupported: CPU solve declines runtime-precision scalars";
  else
    check_solve<C>(linalg::CpuReferenceBackend{}, false);
}

TYPED_TEST(NumericalLinalg, ProviderSolveResolvesSmallGap)
{
  using C = TypeParam;
  this->RecordProperty("backend", "lapack_or_mplapack");
  if constexpr (!C::available)
    GTEST_SKIP() << "unavailable: scalar dependency or native format not configured";
  else if constexpr (C::runtime)
  {
#if UNI20_ENABLE_MPLAPACK_MPFR
    check_solve<C>(linalg::MplapackMpfrBackend{}, false);
#else
    GTEST_SKIP() << "unavailable: configure UNI20_ENABLE_MPLAPACK_MPFR";
#endif
  }
  else if constexpr (C::native_dense_provider)
    check_solve<C>(linalg::LapackBackend{}, false);
  else
    GTEST_SKIP() << "unsupported: float80 provider not wired on this branch";
}

TYPED_TEST(NumericalLinalg, SolvePreservesExtendedExponentRange)
{
  using C = TypeParam;
  this->RecordProperty("backend", "cpu_or_mplapack_mpfr");
  if constexpr (!C::available)
    GTEST_SKIP() << "unavailable: scalar dependency or native format not configured";
  else if constexpr (!C::runtime && C::digits() <= 53)
    GTEST_SKIP() << "not_applicable: this probe requires a wider exponent range than double";
  else if constexpr (C::runtime)
  {
#if UNI20_ENABLE_MPLAPACK_MPFR
    check_solve<C>(linalg::MplapackMpfrBackend{}, true);
#else
    GTEST_SKIP() << "unavailable: configure UNI20_ENABLE_MPLAPACK_MPFR";
#endif
  }
  else
    check_solve<C>(linalg::CpuReferenceBackend{}, true);
}

#if UNI20_ENABLE_MPFR
template <class C, class Backend> void check_solve_accuracy(Backend backend)
{
  testing::Test::RecordProperty("backend", std::string(Backend::name));
  auto a = C::matrix(2, 2), b = C::matrix(2, 1);
  a[0, 0] = C::scalar(2);
  a[0, 1] = a[1, 0] = C::scalar(1);
  a[1, 1] = C::scalar(3);
  b[0, 0] = C::scalar(1, 1);
  auto x = linalg::solve(backend, a, b);
  for (index_type i = 0; i < 2; ++i)
  {
    int const numerator = i == 0 ? 3 : -1;
    if constexpr (C::is_complex)
    {
      expect_rational_accuracy<C>(x[i, 0].real(), numerator, 5);
      expect_rational_accuracy<C>(x[i, 0].imag(), numerator, 5);
    }
    else
      expect_rational_accuracy<C>(x[i, 0], numerator, 5);
    C::expect_precision(x[i, 0]);
  }
}
#endif

TYPED_TEST(NumericalLinalg, SolveAccuracyImprovesWithPrecision)
{
  using C = TypeParam;
  this->RecordProperty("backend", "cpu_or_mplapack_mpfr");
  if constexpr (!C::available) GTEST_SKIP() << "unavailable: scalar dependency or native format not configured";
#if !UNI20_ENABLE_MPFR
  else
    GTEST_SKIP() << "unavailable: MPFR required for independent 512-bit error measurement";
#else
  else if constexpr (C::runtime)
  {
#if UNI20_ENABLE_MPLAPACK_MPFR
    check_solve_accuracy<C>(linalg::MplapackMpfrBackend{});
#else
    GTEST_SKIP() << "unavailable: configure UNI20_ENABLE_MPLAPACK_MPFR";
#endif
  }
  else
    check_solve_accuracy<C>(linalg::CpuReferenceBackend{});
#endif
}
} // namespace uni20::test
