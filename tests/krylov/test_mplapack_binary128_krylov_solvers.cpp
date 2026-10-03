#include "../numerics/precision_cases.hpp"
#include <mplapack_config.h>
#include <uni20/krylov/dense_host_vector.hpp>
#include <uni20/krylov/dense_subspace.hpp>
#include <uni20/krylov/krylov_exponential.hpp>
#include <uni20/krylov/nonsymmetric_arnoldi.hpp>
#include <uni20/krylov/symmetric_lanczos.hpp>
#include <uni20/linalg/ops/matrix_set.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <utility>
#include <vector>

#if defined(MPLAPACK_BINARY128_MODE) && (MPLAPACK_BINARY128_MODE == MPLAPACK_BINARY128_MODE_LDBL)

TEST(MplapackBinary128KrylovSolversTest, SkipsLongDoubleAliasMode)
{
  GTEST_SKIP() << "configured MPLAPACK binary128 mode aliases long double";
}

#else

namespace
{

using Binary128 = mplapack_binary128_t;
using ComplexBinary128 = uni20::complex<Binary128>;
using Vector = uni20::krylov::DenseHostVector<Binary128>;
using Ops = uni20::krylov::DenseHostVectorOps<Binary128>;

Binary128 abs_error(Binary128 actual, Binary128 expected) { return std::abs(actual - expected); }

Binary128 tolerance() { return static_cast<Binary128>(1.0e-25L); }

Binary128 binary_power_of_two(int exponent)
{
  return uni20::test::PrecisionContext<Binary128, false>::power_of_two(exponent);
}

Binary128 below_double_resolution_gap() { return binary_power_of_two(-80); }

void expect_gap_is_binary128_only(Binary128 gap)
{
  EXPECT_TRUE(Binary128{1} + gap > Binary128{1});
  EXPECT_EQ(static_cast<double>(Binary128{1} + gap), 1.0);
}

} // namespace

TEST(MplapackBinary128KrylovSolversTest, RealSchurAndReorderUseBinary128ProjectedLAPACK)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  uni20::DenseMatrix<Binary128> matrix(3, 3);
  uni20::linalg::set_matrix(matrix, Binary128{}, Binary128{});
  matrix[0, 0] = Binary128{1};
  matrix[1, 1] = Binary128{1} + delta;
  matrix[2, 2] = Binary128{3};
  matrix[0, 2] = Binary128{0.125};
  matrix[1, 2] = Binary128{0.25};

  auto schur = uni20::krylov::real_schur(matrix, true);
  ASSERT_EQ(schur.eigenvalues.size(), 3);
  auto reordered = uni20::krylov::reorder_real_schur(std::move(schur), std::vector<std::size_t>{2});

  ASSERT_EQ(reordered.eigenvalues.size(), 3);
  EXPECT_TRUE(abs_error(reordered.eigenvalues[0].real(), Binary128{3}) <= tolerance());
  EXPECT_TRUE(abs_error(reordered.eigenvalues[0].imag(), Binary128{}) <= tolerance());
}

TEST(MplapackBinary128KrylovSolversTest, RealHessenbergSchurUsesBinary128ProjectedLAPACK)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  uni20::DenseMatrix<Binary128> hessenberg(3, 3);
  uni20::linalg::set_matrix(hessenberg, Binary128{}, Binary128{});
  hessenberg[0, 0] = Binary128{1};
  hessenberg[1, 1] = Binary128{1} + delta;
  hessenberg[2, 2] = Binary128{2};
  hessenberg[0, 2] = Binary128{0.125};
  hessenberg[1, 2] = Binary128{0.25};

  auto schur = uni20::krylov::real_hessenberg_schur(hessenberg, true);
  ASSERT_EQ(schur.eigenvalues.size(), 3);
  std::ranges::sort(schur.eigenvalues, [](auto const& lhs, auto const& rhs) { return lhs.real() < rhs.real(); });
  EXPECT_TRUE(abs_error(schur.eigenvalues[0].real(), Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(schur.eigenvalues[1].real(), Binary128{1} + delta) <= tolerance());
}

TEST(MplapackBinary128KrylovSolversTest, RealArnoldiResolvesTriangularGapBelowDoublePrecision)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  std::vector<Binary128> matrix{
      Binary128{1}, Binary128{0},         Binary128{0.125}, Binary128{0},    //
      Binary128{0}, Binary128{1} + delta, Binary128{0},     Binary128{0},    //
      Binary128{0}, Binary128{0},         Binary128{2},     Binary128{0.25}, //
      Binary128{0}, Binary128{0},         Binary128{0},     Binary128{3},
  };
  Ops ops(4, matrix);
  Vector initial{{Binary128{1}, Binary128{1}, Binary128{1}, Binary128{1}}};

  uni20::krylov::NonsymmetricEigenParams<Binary128> params;
  params.eigenvalue_count = 2;
  params.krylov_dimension = 4;
  params.tolerance = tolerance();
  params.complex_pair_tolerance = tolerance();
  params.spectrum = uni20::krylov::SpectrumPart::SmallestReal;
  params.compute_eigenvectors = true;

  auto result = uni20::krylov::real_nonsymmetric_arnoldi_standard(ops, initial, params);

  ASSERT_EQ(result.status, uni20::krylov::NonsymmetricStatus::Converged);
  ASSERT_EQ(result.converged_count, 2);
  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.right_eigenvectors.size(), 2);

  auto eigenvalues = result.eigenvalues;
  std::ranges::sort(eigenvalues, [](auto const& lhs, auto const& rhs) { return lhs.real() < rhs.real(); });
  EXPECT_EQ(static_cast<double>(eigenvalues[0].real()), 1.0);
  EXPECT_EQ(static_cast<double>(eigenvalues[1].real()), 1.0);
  EXPECT_TRUE(eigenvalues[1].real() > eigenvalues[0].real());
  EXPECT_TRUE(abs_error(eigenvalues[0].real(), Binary128{1}) <= tolerance());
  EXPECT_TRUE(abs_error(eigenvalues[1].real(), Binary128{1} + delta) <= tolerance());
}

TEST(MplapackBinary128KrylovSolversTest, ComplexSchurAndReorderUseBinary128ProjectedLAPACK)
{
  Binary128 const delta = below_double_resolution_gap();
  expect_gap_is_binary128_only(delta);

  uni20::DenseMatrix<ComplexBinary128> matrix(3, 3);
  uni20::linalg::set_matrix(matrix, ComplexBinary128{}, ComplexBinary128{});
  matrix[0, 0] = ComplexBinary128{Binary128{1}, delta};
  matrix[1, 1] = ComplexBinary128{Binary128{1} + delta, Binary128{2} * delta};
  matrix[2, 2] = ComplexBinary128{Binary128{3}, Binary128{}};
  matrix[0, 2] = ComplexBinary128{Binary128{0.125}, Binary128{}};
  matrix[1, 2] = ComplexBinary128{Binary128{0.25}, Binary128{}};

  auto schur = uni20::krylov::complex_schur<Binary128>(matrix, true);
  ASSERT_EQ(schur.eigenvalues.size(), 3);
  auto reordered = uni20::krylov::reorder_complex_schur(std::move(schur), std::vector<std::size_t>{2});

  ASSERT_EQ(reordered.eigenvalues.size(), 3);
  EXPECT_TRUE(abs_error(reordered.eigenvalues[0].real(), Binary128{3}) <= tolerance());
  EXPECT_TRUE(abs_error(reordered.eigenvalues[0].imag(), Binary128{}) <= tolerance());
}

#endif
