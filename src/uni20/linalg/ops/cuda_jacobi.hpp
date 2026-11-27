#pragma once

/**
 * \file cuda_jacobi.hpp
 * \ingroup linalg
 * \brief Preserving, device-resident CUDA Jacobi decompositions.
 */

#include <uni20/config.hpp>

#if UNI20_BACKEND_CUSOLVER

#include <uni20/linalg/backends/cusolver/jacobi.hpp>
#include <uni20/linalg/ops/self_adjoint_eigh.hpp>
#include <uni20/linalg/ops/svd.hpp>

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <utility>

namespace uni20::linalg
{
namespace detail
{

template <class Matrix>
concept CudaJacobiMatrix = uni20::RankedTensorView<Matrix, 2> && cuda_resident_eigh<Matrix>();

inline void validate_cuda_jacobi_options(CudaJacobiOptions options)
{
  if (options.tolerance && (!std::isfinite(*options.tolerance) || *options.tolerance <= 0))
    throw std::invalid_argument("CUDA Jacobi tolerance must be finite and positive");
  if (options.maximum_sweeps && *options.maximum_sweeps <= 0)
    throw std::invalid_argument("CUDA Jacobi maximum sweeps must be positive");
}

template <class Index> [[nodiscard]] int cuda_jacobi_dimension(Index value)
{
  if (!std::in_range<int>(value)) throw std::invalid_argument("CUDA Jacobi dimension exceeds cuSOLVER integer range");
  return static_cast<int>(value);
}

} // namespace detail

/// \brief Compute a real symmetric eigensystem with cuSOLVER Jacobi.
/// \details The chosen input triangle defines the symmetric matrix. The result holds
///          ascending eigenvalues and column eigenvectors on the original CUDA device.
///          The call blocks for convergence diagnostics and preserves the input.
///          At the sweep limit the approximate factors are returned with \c converged false.
template <detail::CudaJacobiMatrix Matrix>
[[nodiscard]] auto eigh_jacobi(Matrix const& matrix, CudaJacobiOptions options = {},
                               MatrixTriangle triangle = MatrixTriangle::Upper)
{
  using Scalar = uni20::tensor_element_t<Matrix>;
  using Values = uni20::CudaTensor<Scalar, 1>;
  using Vectors = uni20::CudaMatrix<Scalar, uni20::ColumnMajor>;
  using Factors = SelfAdjointEighResult<Values, Vectors>;

  detail::validate_cuda_jacobi_options(options);
  if (matrix.extent(0) != matrix.extent(1))
    throw std::invalid_argument("CUDA Jacobi eigensystem requires a square matrix");
  auto const n = detail::cuda_jacobi_dimension(matrix.extent(0));
  auto& resources = detail::cuda_matrix_resources(matrix);
  Values values(uni20::uninitialized, resources, n);
  Vectors vectors(uni20::uninitialized, resources, n, n);
  JacobiConvergence convergence;
  if (n != 0)
  {
    uni20::copy(vectors, matrix);
    convergence = detail::cusolver_backend::execute_syevj(n, triangle, options, vectors.storage(), values.storage());
  }
  return CudaJacobiResult<Factors>{{std::move(values), std::move(vectors)}, convergence};
}

/// \brief Compute a reduced real SVD with cuSOLVER Jacobi.
/// \details The factors contain the left vectors, descending singular values, and the
///          right adjoint on the original CUDA device. The call blocks for convergence
///          diagnostics and preserves the input. At the sweep limit the approximate
///          factors are returned with \c converged false.
template <detail::CudaJacobiMatrix Matrix>
[[nodiscard]] auto svd_jacobi(Matrix const& matrix, CudaJacobiOptions options = {})
{
  using Scalar = uni20::tensor_element_t<Matrix>;
  using Values = uni20::CudaTensor<Scalar, 1>;
  using Vectors = uni20::CudaMatrix<Scalar, uni20::ColumnMajor>;
  using RightAdjoint = uni20::CudaMatrix<Scalar, uni20::RowMajor>;
  using Factors = SvdResult<Vectors, Values, RightAdjoint>;

  detail::validate_cuda_jacobi_options(options);
  auto const rows = detail::cuda_jacobi_dimension(matrix.extent(0));
  auto const cols = detail::cuda_jacobi_dimension(matrix.extent(1));
  auto const k = std::min(rows, cols);
  auto& resources = detail::cuda_matrix_resources(matrix);
  Values values(uni20::uninitialized, resources, k);
  Vectors left(uni20::uninitialized, resources, rows, k);
  Vectors right(uni20::uninitialized, resources, cols, k);
  JacobiConvergence convergence;
  if (k != 0)
  {
    Vectors work(uni20::uninitialized, resources, rows, cols);
    uni20::copy(work, matrix);
    convergence = detail::cusolver_backend::execute_gesvdj(rows, cols, options, work.storage(), values.storage(),
                                                           left.storage(), right.storage());
  }

  // For real scalars, column-major V and row-major V^T have the same storage.
  auto right_adjoint =
      RightAdjoint::adopt_storage(typename RightAdjoint::mapping_type{typename RightAdjoint::extents_type{k, cols}},
                                  std::move(right).release_storage());
  return CudaJacobiResult<Factors>{{std::move(left), std::move(values), std::move(right_adjoint)}, convergence};
}

} // namespace uni20::linalg

#endif
