#pragma once

/**
 * \file tridiagonal_eigen.hpp
 * \ingroup linalg
 * \brief Real symmetric tridiagonal eigensystem operation.
 */

#include <uni20/linalg/backends/lapack/tridiagonal_eigen.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/concepts.hpp>

#include <span>
#include <utility>

namespace uni20::linalg
{

/// \brief Overwrite a real symmetric tridiagonal matrix with its eigensystem.
/// \details On success, `diagonal` contains eigenvalues in ascending order and
///          `subdiagonal` is destroyed. With `compute_vectors == true`, column `j` of
///          `eigenvectors` is the orthonormal eigenvector of the original tridiagonal
///          matrix associated with `diagonal[j]`. Existing vector entries are ignored;
///          this does not accumulate a prior reduction from a dense matrix.
/// \pre For `n = diagonal.size()`, subdiagonal length is `n - 1` (zero when `n == 0`).
///      Requested vectors have shape `n x n` and the same real scalar type. The spans
///      and requested vector storage must not overlap. No output is resized.
/// \note Empty input is a no-op. LAPACK convergence failure uses Uni20's error policy;
///       modified workspaces are not rolled back and no partial-result API is provided.
/// \note `compute_vectors`: If false, vector elements are unchanged and their dimensions
///       are ignored; a tensor of a supported type must still be supplied.
template <class BackendSelector, uni20::LapackReal Scalar, uni20::MutableRankedTensorView<2> EigenvectorTensor>
void symmetric_tridiagonal_eigen(BackendSelector&& selector, std::span<Scalar> diagonal, std::span<Scalar> subdiagonal,
                                 EigenvectorTensor&& eigenvectors, bool compute_vectors)
{
  auto eigenvector_descriptor = uni20::mdspec_of(eigenvectors);
  dispatch_kernel(std::forward<BackendSelector>(selector),
                  symmetric_tridiagonal_eigen_op{.compute_vectors = compute_vectors}, diagonal, subdiagonal,
                  eigenvector_descriptor);
}

/// \brief Apply the destructive `symmetric_tridiagonal_eigen` contract using host tensor backend policy.
template <uni20::LapackReal Scalar, uni20::MutableRankedTensorView<2> EigenvectorTensor>
void symmetric_tridiagonal_eigen(std::span<Scalar> diagonal, std::span<Scalar> subdiagonal,
                                 EigenvectorTensor&& eigenvectors, bool compute_vectors)
{
  auto selector = select_backend(symmetric_tridiagonal_eigen_op{.compute_vectors = compute_vectors}, eigenvectors);
  symmetric_tridiagonal_eigen(selector, diagonal, subdiagonal, std::forward<EigenvectorTensor>(eigenvectors),
                              compute_vectors);
}

} // namespace uni20::linalg
