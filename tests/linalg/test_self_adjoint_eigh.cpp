#include <uni20/config.hpp>
#include <uni20/core/types.hpp>
#include <uni20/linalg/ops/self_adjoint_eigh.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include "deferred_host_tensor.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <type_traits>
#include <utility>

namespace
{
template <class Matrix, class MatrixElement>
void check_eigenvector_residuals(MatrixElement&& matrix_element, Matrix const& eigenvectors,
                                 uni20::Tensor<uni20::make_real_t<typename Matrix::value_type>, 1> const& values,
                                 double tolerance)
{
  using scalar_type = typename Matrix::value_type;
  for (uni20::index_type col = 0; col < eigenvectors.cols(); ++col)
  {
    for (uni20::index_type row = 0; row < eigenvectors.rows(); ++row)
    {
      scalar_type residual{};
      for (uni20::index_type k = 0; k < eigenvectors.rows(); ++k)
        residual += matrix_element(row, k) * eigenvectors[k, col];
      residual -= values[col] * eigenvectors[row, col];
      EXPECT_LT(static_cast<double>(std::abs(residual)), tolerance);
    }
  }
}
} // namespace

TEST(SelfAdjointEighTest, ValueApiPreservesRealInputAndReturnsEigenvectors)
{
  uni20::DenseMatrix<double, uni20::RowMajor> matrix(2, 2);
  matrix[0, 0] = 2.0;
  matrix[0, 1] = 1.0;
  matrix[1, 0] = 1.0;
  matrix[1, 1] = 2.0;

  auto result = uni20::linalg::eigh(matrix);

  static_assert(std::same_as<typename decltype(result.eigenvectors)::layout_type, uni20::ColumnMajor>);
  EXPECT_NEAR(result.eigenvalues[0], 1.0, 1e-13);
  EXPECT_NEAR(result.eigenvalues[1], 3.0, 1e-13);
  EXPECT_DOUBLE_EQ((matrix[0, 1]), 1.0);
  check_eigenvector_residuals([](uni20::index_type row, uni20::index_type col) { return row == col ? 2.0 : 1.0; },
                              result.eigenvectors, result.eigenvalues, 1e-12);
}

TEST(SelfAdjointEighTest, ConsumingValueApiReusesCompatibleOwningStorage)
{
  uni20::DenseMatrix<double> matrix(2, 2);
  matrix[0, 0] = 2.0;
  matrix[0, 1] = 1.0;
  matrix[1, 0] = 1.0;
  matrix[1, 1] = 2.0;
  double* original_storage = matrix.mutable_handle();

  auto result = uni20::linalg::eigh(std::move(matrix));

  static_assert(std::same_as<decltype(result.eigenvectors), uni20::DenseMatrix<double>>);
  EXPECT_EQ(result.eigenvectors.mutable_handle(), original_storage);
  EXPECT_NEAR(result.eigenvalues[0], 1.0, 1e-13);
  EXPECT_NEAR(result.eigenvalues[1], 3.0, 1e-13);
  check_eigenvector_residuals([](uni20::index_type row, uni20::index_type col) { return row == col ? 2.0 : 1.0; },
                              result.eigenvectors, result.eigenvalues, 1e-12);
}

TEST(SelfAdjointEighTest, ConsumingValueApiReusesRowMajorOwningStorage)
{
  uni20::DenseMatrix<double, uni20::RowMajor> matrix(2, 2);
  matrix[0, 0] = 2.0;
  matrix[0, 1] = 1.0;
  matrix[1, 0] = 1.0;
  matrix[1, 1] = 2.0;
  double* original_storage = matrix.mutable_handle();

  auto result = uni20::linalg::eigh(std::move(matrix));

  static_assert(std::same_as<typename decltype(result.eigenvectors)::layout_type, uni20::layout_stride>);
  EXPECT_EQ(result.eigenvectors.mutable_handle(), original_storage);
  EXPECT_EQ(result.eigenvectors.mapping().stride(0), 1);
  EXPECT_EQ(result.eigenvectors.mapping().stride(1), 2);
  EXPECT_NEAR(result.eigenvalues[0], 1.0, 1e-13);
  EXPECT_NEAR(result.eigenvalues[1], 3.0, 1e-13);
  check_eigenvector_residuals([](uni20::index_type row, uni20::index_type col) { return row == col ? 2.0 : 1.0; },
                              result.eigenvectors, result.eigenvalues, 1e-12);
}

TEST(SelfAdjointEighTest, ConsumingValueApiPreservesPaddedLdaAndComplexRowMajorSemantics)
{
  using scalar_type = uni20::complex<double>;
  using matrix_type = uni20::StridedTensor<scalar_type, 2>;
  matrix_type matrix(matrix_type::extents_type{2, 2}, std::array<uni20::index_type, 2>{4, 1});
  matrix.storage().resize(matrix.storage().size() + 3);
  matrix[0, 0] = scalar_type{2.0, 0.0};
  matrix[0, 1] = scalar_type{1.0, 1.0};
  matrix[1, 0] = scalar_type{99.0, 17.0};
  matrix[1, 1] = scalar_type{3.0, 0.0};
  scalar_type* original_storage = matrix.mutable_handle();
  auto const original_storage_size = matrix.storage().size();

  auto result = uni20::linalg::eigh(std::move(matrix), uni20::linalg::MatrixTriangle::Upper);

  static_assert(std::same_as<decltype(result.eigenvectors), matrix_type>);
  EXPECT_EQ(result.eigenvectors.mutable_handle(), original_storage);
  EXPECT_EQ(result.eigenvectors.storage().size(), original_storage_size);
  EXPECT_EQ(result.eigenvectors.mapping().stride(0), 1);
  EXPECT_EQ(result.eigenvectors.mapping().stride(1), 4);
  EXPECT_NEAR(result.eigenvalues[0], 1.0, 1e-12);
  EXPECT_NEAR(result.eigenvalues[1], 4.0, 1e-12);
  auto matrix_element = [](uni20::index_type row, uni20::index_type col) {
    if (row == 0 && col == 0) return scalar_type{2.0, 0.0};
    if (row == 1 && col == 1) return scalar_type{3.0, 0.0};
    if (row == 0) return scalar_type{1.0, 1.0};
    return scalar_type{1.0, -1.0};
  };
  check_eigenvector_residuals(matrix_element, result.eigenvectors, result.eigenvalues, 1e-11);
}

TEST(SelfAdjointEighTest, ConsumingValueApiMaterializesMappingWithoutUnitStride)
{
  using matrix_type = uni20::StridedTensor<double, 2>;
  matrix_type matrix(matrix_type::extents_type{2, 2}, std::array<uni20::index_type, 2>{3, 2});
  matrix[0, 0] = 2.0;
  matrix[0, 1] = 1.0;
  matrix[1, 0] = 1.0;
  matrix[1, 1] = 2.0;
  double* original_storage = matrix.mutable_handle();

  auto result = uni20::linalg::eigh(std::move(matrix));

  EXPECT_NE(result.eigenvectors.mutable_handle(), original_storage);
  EXPECT_EQ(result.eigenvectors.mapping().stride(0), 1);
  EXPECT_EQ(result.eigenvectors.mapping().stride(1), 2);
  EXPECT_NEAR(result.eigenvalues[0], 1.0, 1e-13);
  EXPECT_NEAR(result.eigenvalues[1], 3.0, 1e-13);
}

TEST(SelfAdjointEighTest, ComplexValueApiUsesSelectedLowerTriangle)
{
  using scalar_type = uni20::complex<double>;
  uni20::DenseMatrix<scalar_type> matrix(2, 2);
  matrix[0, 0] = scalar_type{2.0, 0.0};
  matrix[0, 1] = scalar_type{99.0, 17.0};
  matrix[1, 0] = scalar_type{1.0, -1.0};
  matrix[1, 1] = scalar_type{3.0, 0.0};

  auto result = uni20::linalg::eigh(matrix, uni20::linalg::MatrixTriangle::Lower);

  EXPECT_NEAR(result.eigenvalues[0], 1.0, 1e-12);
  EXPECT_NEAR(result.eigenvalues[1], 4.0, 1e-12);
  auto matrix_element = [](uni20::index_type row, uni20::index_type col) {
    if (row == 0 && col == 0) return scalar_type{2.0, 0.0};
    if (row == 1 && col == 1) return scalar_type{3.0, 0.0};
    if (row == 0) return scalar_type{1.0, 1.0};
    return scalar_type{1.0, -1.0};
  };
  check_eigenvector_residuals(matrix_element, result.eigenvectors, result.eigenvalues, 1e-11);
}

TEST(SelfAdjointEighTest, InPlaceValueOnlySolveResizesEigenvalueOutput)
{
  uni20::DenseMatrix<double> matrix(2, 2);
  matrix[0, 0] = 4.0;
  matrix[0, 1] = 0.0;
  matrix[1, 0] = 0.0;
  matrix[1, 1] = 2.0;
  uni20::Tensor<double, 1> eigenvalues;

  uni20::linalg::self_adjoint_eigh(eigenvalues, matrix,
                                   uni20::linalg::SelfAdjointEighOptions{
                                       .compute_vectors = false, .triangle = uni20::linalg::MatrixTriangle::Upper});

  EXPECT_EQ(eigenvalues.extent(0), 2);
  EXPECT_NEAR(eigenvalues[0], 2.0, 1e-13);
  EXPECT_NEAR(eigenvalues[1], 4.0, 1e-13);
}

TEST(SelfAdjointEighTest, InPlaceSolveAcquiresDeferredWritableOperands)
{
  uni20::test::DeferredHostTensor<double, 2> matrix(2, 2);
  uni20::test::DeferredHostTensor<double, 1> eigenvalues(2);
  matrix.storage() = {2.0, 1.0, 1.0, 2.0};

  uni20::linalg::self_adjoint_eigh(uni20::linalg::LapackBackend{}, eigenvalues, matrix,
                                   uni20::linalg::SelfAdjointEighOptions{
                                       .compute_vectors = false, .triangle = uni20::linalg::MatrixTriangle::Upper});

  EXPECT_NEAR(eigenvalues.storage()[0], 1.0, 1e-13);
  EXPECT_NEAR(eigenvalues.storage()[1], 3.0, 1e-13);
}

TEST(SelfAdjointEighTest, DirectLapackBackendDeclinesRowMajorBeforeMutation)
{
  uni20::DenseMatrix<double, uni20::RowMajor> matrix(2, 2);
  matrix[0, 0] = 2.0;
  matrix[0, 1] = 1.0;
  matrix[1, 0] = 1.0;
  matrix[1, 1] = 2.0;
  uni20::Tensor<double, 1> eigenvalues(2);
  eigenvalues[0] = 8.0;
  eigenvalues[1] = 9.0;
  auto matrix_span = matrix.mdspan();
  auto eigenvalue_span = eigenvalues.mdspan();

  EXPECT_FALSE(uni20::linalg::try_dispatch_kernel(uni20::linalg::LapackBackend{}, uni20::linalg::self_adjoint_eigh_op{},
                                                  eigenvalue_span, matrix_span));
  EXPECT_DOUBLE_EQ((matrix[0, 1]), 1.0);
  EXPECT_DOUBLE_EQ(eigenvalues[0], 8.0);
  EXPECT_DOUBLE_EQ(eigenvalues[1], 9.0);
}

TEST(SelfAdjointEighTest, EmptyMatrixHasEmptyEigensystem)
{
  uni20::DenseMatrix<double> matrix(0, 0);

  auto result = uni20::linalg::eigh(matrix);

  EXPECT_EQ(result.eigenvalues.extent(0), 0);
  EXPECT_EQ(result.eigenvectors.rows(), 0);
  EXPECT_EQ(result.eigenvectors.cols(), 0);
}

#if UNI20_HAS_MPLAPACK_FLOAT128
TEST(SelfAdjointEighTest, MplapackFloat128RealAndComplexPaths)
{
  using real_type = uni20::float128;
  using complex_type = uni20::complex<real_type>;

  uni20::DenseMatrix<real_type> real_matrix(2, 2);
  real_matrix[0, 0] = real_type{2};
  real_matrix[0, 1] = real_type{};
  real_matrix[1, 0] = real_type{};
  real_matrix[1, 1] = real_type{3};
  auto real_result = uni20::linalg::eigh(real_matrix);
  EXPECT_TRUE(real_result.eigenvalues[0] == real_type{2});
  EXPECT_TRUE(real_result.eigenvalues[1] == real_type{3});

  uni20::DenseMatrix<complex_type> complex_matrix(2, 2);
  complex_matrix[0, 0] = complex_type{real_type{4}, real_type{}};
  complex_matrix[0, 1] = complex_type{};
  complex_matrix[1, 0] = complex_type{};
  complex_matrix[1, 1] = complex_type{real_type{5}, real_type{}};
  auto complex_result = uni20::linalg::eigh(complex_matrix);
  EXPECT_TRUE(complex_result.eigenvalues[0] == real_type{4});
  EXPECT_TRUE(complex_result.eigenvalues[1] == real_type{5});

  uni20::DenseMatrix<complex_type, uni20::RowMajor> row_major_matrix(2, 2);
  row_major_matrix[0, 0] = complex_type{real_type{2}, real_type{}};
  row_major_matrix[0, 1] = complex_type{real_type{1}, real_type{1}};
  row_major_matrix[1, 0] = complex_type{real_type{99}, real_type{17}};
  row_major_matrix[1, 1] = complex_type{real_type{3}, real_type{}};
  complex_type* original_storage = row_major_matrix.mutable_handle();
  auto row_major_result = uni20::linalg::eigh(std::move(row_major_matrix), uni20::linalg::MatrixTriangle::Upper);
  EXPECT_EQ(row_major_result.eigenvectors.mutable_handle(), original_storage);
  EXPECT_TRUE(row_major_result.eigenvalues[0] == real_type{1});
  EXPECT_TRUE(row_major_result.eigenvalues[1] == real_type{4});
  auto matrix_element = [](uni20::index_type row, uni20::index_type col) {
    if (row == 0 && col == 0) return complex_type{real_type{2}, real_type{}};
    if (row == 1 && col == 1) return complex_type{real_type{3}, real_type{}};
    if (row == 0) return complex_type{real_type{1}, real_type{1}};
    return complex_type{real_type{1}, real_type{-1}};
  };
  check_eigenvector_residuals(matrix_element, row_major_result.eigenvectors, row_major_result.eigenvalues, 1e-20);
}
#endif

TEST(SelfAdjointEighTest, SelectedComplexEigenpairsUseAscendingHalfOpenIndices)
{
  using namespace uni20::linalg;
  using Scalar = uni20::complex<double>;
  uni20::DenseMatrix<Scalar, uni20::RowMajor> matrix(3, 3);
  matrix[0, 0] = 3;
  matrix[1, 1] = 3;
  matrix[0, 1] = Scalar{0, 1};
  matrix[1, 0] = Scalar{0, -1};

  auto const result = eigh(matrix, EigenvalueIndexRange{1, 3});
  ASSERT_EQ(result.eigenvalues.size(), 2);
  ASSERT_EQ(result.eigenvectors.extent(0), 3);
  ASSERT_EQ(result.eigenvectors.extent(1), 2);
  EXPECT_NEAR(result.eigenvalues[0], 2, 1e-13);
  EXPECT_NEAR(result.eigenvalues[1], 4, 1e-13);
  check_eigenvector_residuals([&](auto row, auto col) { return matrix[row, col]; }, result.eigenvectors,
                              result.eigenvalues, 1e-13);

  auto const empty = eigh(matrix, EigenvalueIndexRange{2, 2});
  EXPECT_EQ(empty.eigenvalues.size(), 0);
  EXPECT_EQ(empty.eigenvectors.extent(0), 3);
  EXPECT_EQ(empty.eigenvectors.extent(1), 0);
  EXPECT_THROW((void)eigh(matrix, EigenvalueIndexRange{-1, 3}), std::invalid_argument);
  EXPECT_THROW((void)eigh(matrix, EigenvalueIndexRange{2, 4}), std::invalid_argument);
}

#if UNI20_HAS_MPLAPACK_FLOAT128
TEST(SelfAdjointEighTest, SelectedBinary128EigenpairsUseTheProviderWorkspace)
{
  using namespace uni20::linalg;
  using Real = uni20::float128;
  uni20::DenseMatrix<Real> matrix(3, 3);
  matrix[0, 0] = Real{1};
  matrix[1, 1] = Real{2};
  matrix[2, 2] = Real{4};
  auto const result = eigh(matrix, EigenvalueIndexRange{2, 3});
  ASSERT_EQ(result.eigenvalues.size(), 1);
  EXPECT_TRUE(result.eigenvalues[0] == Real{4});
  EXPECT_EQ(result.eigenvectors.extent(1), 1);
}
#endif

TEST(SelfAdjointEighTest, SelectedRealEigenpairsCanKeepAnInteriorRange)
{
  using namespace uni20::linalg;
  uni20::DenseMatrix<double> matrix(3, 3);
  matrix[0, 0] = 4;
  matrix[1, 1] = 1;
  matrix[2, 2] = 3;
  auto const result = eigh(matrix, EigenvalueIndexRange{1, 2});
  ASSERT_EQ(result.eigenvalues.size(), 1);
  EXPECT_NEAR(result.eigenvalues[0], 3, 1e-13);
  EXPECT_NEAR(std::abs(result.eigenvectors[2, 0]), 1, 1e-13);
}

TEST(SelfAdjointEighTest, DivideAndConquerChecksWorkspaceBeforeQuery)
{
  using uni20::linalg::lapack_detail::divide_and_conquer_eigh_workspace_fits;
  constexpr auto limit = std::size_t{1} << (std::numeric_limits<uni20::blas_int>::digits / 2);
  EXPECT_TRUE(divide_and_conquer_eigh_workspace_fits<float>(true, limit - 2));
  EXPECT_FALSE(divide_and_conquer_eigh_workspace_fits<float>(true, limit - 1));
  EXPECT_TRUE(divide_and_conquer_eigh_workspace_fits<float>(false, limit - 1));

  using Complex = uni20::complex<float>;
  EXPECT_TRUE(divide_and_conquer_eigh_workspace_fits<Complex>(true, limit - 2));
  EXPECT_FALSE(divide_and_conquer_eigh_workspace_fits<Complex>(true, limit - 1));
  EXPECT_TRUE(divide_and_conquer_eigh_workspace_fits<Complex>(false, limit - 1));
}

TEST(SelfAdjointEighTest, WorkspaceQueryCoversRoundedSinglePrecisionLengths)
{
  using uni20::linalg::lapack_detail::workspace_size;
  // CHEEVD with vectors at n=3000 needs 18,015,001 real entries.
  constexpr uni20::blas_int lrwork = 1 + 5 * 3000 + 2 * 3000 * 3000;
  EXPECT_GE(workspace_size(static_cast<float>(lrwork)), lrwork);
  // Complex WORK queries encode LWORK in their real component. At this
  // magnitude the spacing is four, so adding just one entry is insufficient.
  constexpr uni20::blas_int lwork = 33'554'434;
  EXPECT_GE(workspace_size(uni20::complex<float>{static_cast<float>(lwork), 0}), lwork);
  EXPECT_EQ(workspace_size(16'777'215.0f), 16'777'215);
  EXPECT_EQ(workspace_size(static_cast<double>(lrwork)), lrwork);
}

TEST(SelfAdjointEighTest, DivideAndConquerReturnsHermitianEigenpairs)
{
  using namespace uni20::linalg;
  using Scalar = uni20::complex<double>;
  uni20::DenseMatrix<Scalar, uni20::RowMajor> matrix(2, 2);
  matrix[0, 0] = 3;
  matrix[1, 1] = 3;
  matrix[0, 1] = Scalar{0, 1};
  matrix[1, 0] = Scalar{0, -1};
  auto result = eigh(matrix, EighDriver::DivideAndConquer);
  EXPECT_NEAR(result.eigenvalues[0], 2, 1e-13);
  EXPECT_NEAR(result.eigenvalues[1], 4, 1e-13);
  check_eigenvector_residuals([&](auto row, auto col) { return matrix[row, col]; }, result.eigenvectors,
                              result.eigenvalues, 1e-13);
}

#if UNI20_HAS_MPLAPACK_FLOAT128
TEST(SelfAdjointEighTest, DivideAndConquerSupportsBinary128)
{
  using namespace uni20::linalg;
  using Real = uni20::float128;
  uni20::DenseMatrix<Real> matrix(2, 2);
  matrix[0, 0] = Real{2};
  matrix[1, 1] = Real{4};
  auto result = eigh(matrix, EighDriver::DivideAndConquer);
  EXPECT_TRUE(result.eigenvalues[0] == Real{2});
  EXPECT_TRUE(result.eigenvalues[1] == Real{4});
}
#endif
