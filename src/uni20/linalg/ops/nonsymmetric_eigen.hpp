#pragma once

/**
 * \file nonsymmetric_eigen.hpp
 * \ingroup linalg
 * \brief Fixed-output dense nonsymmetric eigensystem operation.
 */

#include <uni20/linalg/backends/lapack/nonsymmetric_eigen.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/concepts.hpp>

#include <span>
#include <utility>

namespace uni20::linalg
{

/// \brief Compute eigenvalues and optional right eigenvectors, destroying the input matrix.
/// \details For the original matrix `A`, column `j` of `right_eigenvectors` satisfies
///          `A * v = eigenvalues[j] * v` and has unit Euclidean norm. Eigenvalues are
///          not sorted. For real input, conjugate pairs are consecutive with the
///          positive-imaginary value first; Uni20 returns actual complex vector columns
///          rather than LAPACK's packed real representation. Left vectors are not computed.
/// \pre `matrix_work` is `n x n`, and `eigenvalues` has length `n`. Eigenvalues and
///      vectors use `uni20::complex<make_real_t<matrix_scalar>>`, even for real input.
///      Requested vector output is `n x n`. Work and output storage must not overlap;
///      no output is resized.
/// \note LAPACK convergence failure uses Uni20's error policy; modified workspaces
///       are not rolled back and no partial-result API is provided.
/// \note `compute_right_vectors`: If false, vector elements are unchanged and their dimensions
///       are ignored; a tensor of a supported type is still required.
template <class BackendSelector, uni20::MutableRankedTensorView<2> MatrixTensor, class EigenScalar,
          uni20::MutableRankedTensorView<2> RightEigenvectorTensor>
void nonsymmetric_eigen(BackendSelector&& selector, MatrixTensor&& matrix_work, std::span<EigenScalar> eigenvalues,
                        RightEigenvectorTensor&& right_eigenvectors, bool compute_right_vectors)
{
  auto matrix_descriptor = uni20::mdspec_of(matrix_work);
  auto right_eigenvector_descriptor = uni20::mdspec_of(right_eigenvectors);
  dispatch_kernel(std::forward<BackendSelector>(selector),
                  nonsymmetric_eigen_op{.compute_right_vectors = compute_right_vectors}, matrix_descriptor, eigenvalues,
                  right_eigenvector_descriptor);
}

/// \brief Apply the destructive `nonsymmetric_eigen` contract using tensor storage policy.
template <uni20::MutableRankedTensorView<2> MatrixTensor, class EigenScalar,
          uni20::MutableRankedTensorView<2> RightEigenvectorTensor>
void nonsymmetric_eigen(MatrixTensor&& matrix_work, std::span<EigenScalar> eigenvalues,
                        RightEigenvectorTensor&& right_eigenvectors, bool compute_right_vectors)
{
  auto operation = nonsymmetric_eigen_op{.compute_right_vectors = compute_right_vectors};
  auto selector = select_backend(operation, matrix_work, right_eigenvectors);
  nonsymmetric_eigen(selector, std::forward<MatrixTensor>(matrix_work), eigenvalues,
                     std::forward<RightEigenvectorTensor>(right_eigenvectors), compute_right_vectors);
}

} // namespace uni20::linalg
