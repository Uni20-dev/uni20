#include "precision_cases.hpp"
#include <uni20/krylov/dense_host_vector.hpp>
#include <uni20/krylov/dense_subspace.hpp>
#include <uni20/krylov/krylov_exponential.hpp>
#include <uni20/krylov/nonsymmetric_arnoldi.hpp>
#include <uni20/krylov/symmetric_lanczos.hpp>

namespace uni20::test
{
template <class C> using NumericalKrylov = PrecisionTest<C>;
TYPED_TEST_SUITE(NumericalKrylov, PrecisionCases, PrecisionCaseNames);

TYPED_TEST(NumericalKrylov, ProjectedTridiagonalResolvesGap)
{
  using C = TypeParam;
  this->RecordProperty("backend", "projected_lapack");
  if constexpr (!C::available)
    GTEST_SKIP() << "unavailable: scalar dependency or native format not configured";
  else if constexpr (C::is_complex)
    GTEST_SKIP() << "not_applicable: Hermitian tridiagonal projection is real";
  else if constexpr (!C::projected_lapack)
    GTEST_SKIP() << "unsupported: projected LAPACK wrappers do not accept this scalar";
  else
  {
    using R = typename C::real_type;
    auto d = C::gap();
    std::vector<R> diagonal{C::real(1), C::real(1) + d, C::real(2)};
    std::vector<R> offdiagonal{d / C::real(4), C::real(0)};
    auto result = krylov::symmetric_tridiagonal_eigensystem(diagonal, offdiagonal, false);
    using std::sqrt;
    auto center = C::real(1) + d / C::real(2);
    auto radius = d * sqrt(C::real(5)) / C::real(4);
    ASSERT_EQ(result.eigenvalues.size(), 3);
    EXPECT_LT(result.eigenvalues[0], result.eigenvalues[1]);
    expect_error_at_most(result.eigenvalues[0], center - radius, C::real(16) * C::epsilon());
    expect_error_at_most(result.eigenvalues[1], center + radius, C::real(16) * C::epsilon());
  }
}

TYPED_TEST(NumericalKrylov, LanczosResolvesGapAndResidual)
{
  using C = TypeParam;
  this->RecordProperty("backend", "native_krylov_projected_lapack");
  if constexpr (!C::available)
    GTEST_SKIP() << "unavailable: scalar dependency or native format not configured";
  else if constexpr (!C::projected_lapack)
    GTEST_SKIP() << "unsupported: Lanczos requires projected LAPACK support and scalar-generic tolerances";
  else
  {
    using S = typename C::scalar_type;
    using R = typename C::real_type;
    std::vector<S> matrix(16, C::scalar(0));
    std::vector<R> diagonal{C::real(1), C::real(1) + C::gap(), C::real(2), C::real(3)};
    for (std::size_t i = 0; i < 4; ++i)
      matrix[i * 4 + i] = C::scalar(diagonal[i], C::real(0));
    krylov::DenseHostVectorOps<S> ops(4, matrix);
    krylov::DenseHostVector<S> initial{{C::scalar(1, 1), C::scalar(1, -1), C::scalar(1), C::scalar(-1, 1)}};
    krylov::SymmetricEigenParams<R> params;
    params.eigenvalue_count = 2;
    params.krylov_dimension = 4;
    params.tolerance = C::real(64) * C::epsilon();
    params.spectrum = krylov::SpectrumPart::SmallestAlgebraic;
    params.compute_eigenvectors = true;
    auto result = krylov::symmetric_lanczos_standard<S>(ops, initial, params);
    ASSERT_EQ(result.status, 0);
    ASSERT_EQ(result.converged_count, 2);
    ASSERT_EQ(result.eigenvectors.size(), 2);
    ASSERT_EQ(result.eigenvalues.size(), 2);
    auto sorted = result.eigenvalues;
    std::ranges::sort(sorted);
    EXPECT_LT(sorted[0], sorted[1]);
    expect_error_at_most(sorted[0], diagonal[0], params.tolerance);
    expect_error_at_most(sorted[1], diagonal[1], params.tolerance);
    // Analytic componentwise residual: no tested matvec/reduction as oracle.
    for (std::size_t j = 0; j < 2; ++j)
    {
      R squared_norm = C::real(0);
      for (std::size_t i = 0; i < 4; ++i)
      {
        expect_error_at_most((diagonal[i] - result.eigenvalues[j]) * result.eigenvectors[j].values[i], C::scalar(0),
                             params.tolerance);
        auto const& value = result.eigenvectors[j].values[i];
        if constexpr (C::is_complex)
          squared_norm += value.real() * value.real() + value.imag() * value.imag();
        else
          squared_norm += value * value;
      }
      expect_error_at_most(squared_norm, C::real(1), params.tolerance);
    }
  }
}

TYPED_TEST(NumericalKrylov, ArnoldiResolvesGap)
{
  using C = TypeParam;
  this->RecordProperty("backend", "native_krylov_projected_lapack");
  if constexpr (!C::available)
    GTEST_SKIP() << "unavailable: scalar dependency or native format not configured";
  else if constexpr (!C::projected_lapack)
    GTEST_SKIP() << "unsupported: Arnoldi requires projected LAPACK support and scalar-generic tolerances";
  else
  {
    using S = typename C::scalar_type;
    using R = typename C::real_type;
    auto d = C::gap();
    krylov::DenseHostVectorOps<S> ops(
        2, {C::scalar(C::real(1), d), C::scalar(0), C::scalar(0), C::scalar(C::real(1) + d, C::real(2) * d)});
    krylov::DenseHostVector<S> initial{{C::scalar(1), C::scalar(1)}};
    krylov::NonsymmetricEigenParams<R> params;
    params.eigenvalue_count = 2;
    params.krylov_dimension = 2;
    params.tolerance = C::real(32) * C::epsilon();
    params.complex_pair_tolerance = params.tolerance;
    params.spectrum = krylov::SpectrumPart::SmallestReal;
    params.compute_eigenvectors = false;
    auto result = [&] {
      if constexpr (C::is_complex)
        return krylov::complex_nonsymmetric_arnoldi_standard<R>(ops, initial, params);
      else
        return krylov::real_nonsymmetric_arnoldi_standard<R>(ops, initial, params);
    }();
    ASSERT_EQ(result.status, krylov::NonsymmetricStatus::Converged);
    ASSERT_EQ(result.eigenvalues.size(), 2);
    std::ranges::sort(result.eigenvalues, {}, [](auto z) { return z.real(); });
    EXPECT_LT(result.eigenvalues[0].real(), result.eigenvalues[1].real());
    expect_error_at_most(result.eigenvalues[0].real(), C::real(1), params.tolerance);
    expect_error_at_most(result.eigenvalues[1].real(), C::real(1) + d, params.tolerance);
    expect_error_at_most(result.eigenvalues[0].imag(), C::is_complex ? d : C::real(0), params.tolerance);
    expect_error_at_most(result.eigenvalues[1].imag(), C::is_complex ? C::real(2) * d : C::real(0), params.tolerance);
  }
}

TYPED_TEST(NumericalKrylov, ExponentialActionRetainsIncrement)
{
  using C = TypeParam;
  this->RecordProperty("backend", "native_krylov_projected_lapack");
  if constexpr (!C::available)
    GTEST_SKIP() << "unavailable: scalar dependency or native format not configured";
  else if constexpr (!C::projected_lapack)
    GTEST_SKIP() << "unsupported: Hermitian exponential action requires projected LAPACK support";
  else
  {
    using S = typename C::scalar_type;
    auto d = C::gap();
    krylov::DenseHostVectorOps<S> ops(
        2, {C::scalar(d, C::real(0)), C::scalar(0), C::scalar(0), C::scalar(C::real(2) * d, C::real(0))});
    krylov::DenseHostVector<S> initial{{C::scalar(1, 1), C::scalar(-1, 1)}};
    krylov::KrylovExponentialParams<typename C::real_type> params;
    params.krylov_dimension = 2;
    auto result = krylov::hermitian_krylov_exponential_action<S>(ops, initial, C::scalar(1), params);
    ASSERT_EQ(result.projected_dimension, 2);
    using std::exp;
    expect_error_at_most(result.action.values[0], initial.values[0] * exp(d), C::real(32) * C::epsilon());
    expect_error_at_most(result.action.values[1], initial.values[1] * exp(C::real(2) * d), C::real(32) * C::epsilon());
  }
}
} // namespace uni20::test
