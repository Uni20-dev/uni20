#pragma once

/**
 * \file schur.hpp
 * \ingroup linalg
 * \brief Fixed-output dense Schur decomposition and reordering operations.
 */

#include <uni20/linalg/backends/lapack/schur.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/concepts.hpp>

#include <span>
#include <utility>

namespace uni20::linalg
{

/// \brief Replace a square matrix by its Schur form and optionally return Schur vectors.
/// \details On success, `matrix_work` holds `T` and requested `schur_vectors` hold `Q`,
///          with `A = Q * T * adjoint(Q)` for the original matrix `A`. Real input gives
///          real orthogonal `Q` and upper quasi-triangular `T` with 1x1 or 2x2 diagonal
///          blocks. Complex input gives unitary `Q` and upper triangular `T`.
///          Schur vectors are not in general eigenvectors. Eigenvalues follow the
///          diagonal block order; no sorting is requested.
/// \pre `matrix_work` is `n x n`, and `eigenvalues` has length `n` with scalar type
///      `uni20::complex<make_real_t<matrix_scalar>>`. Requested vectors have shape
///      `n x n` and the same scalar type as the matrix. Work and output storage must
///      not overlap; no output is resized.
/// \note LAPACK convergence failure uses Uni20's error policy; modified workspaces
///       are not rolled back and no partial-result API is provided.
/// \note `compute_vectors`: If false, vector elements are unchanged and their dimensions
///       are ignored; a tensor of a supported type is still required.
template <class BackendSelector, uni20::MutableRankedTensorView<2> MatrixTensor, class EigenScalar,
          uni20::MutableRankedTensorView<2> SchurVectorTensor>
void schur(BackendSelector&& selector, MatrixTensor&& matrix_work, std::span<EigenScalar> eigenvalues,
           SchurVectorTensor&& schur_vectors, bool compute_vectors)
{
  auto matrix_descriptor = uni20::mdspec_of(matrix_work);
  auto vector_descriptor = uni20::mdspec_of(schur_vectors);
  dispatch_kernel(std::forward<BackendSelector>(selector), schur_op{.compute_vectors = compute_vectors},
                  matrix_descriptor, eigenvalues, vector_descriptor);
}

/// \brief Apply the destructive `schur` contract using tensor storage policy.
template <uni20::MutableRankedTensorView<2> MatrixTensor, class EigenScalar,
          uni20::MutableRankedTensorView<2> SchurVectorTensor>
void schur(MatrixTensor&& matrix_work, std::span<EigenScalar> eigenvalues, SchurVectorTensor&& schur_vectors,
           bool compute_vectors)
{
  auto operation = schur_op{.compute_vectors = compute_vectors};
  auto selector = select_backend(operation, matrix_work, schur_vectors);
  schur(selector, std::forward<MatrixTensor>(matrix_work), eigenvalues, std::forward<SchurVectorTensor>(schur_vectors),
        compute_vectors);
}

/// \brief Replace a real upper-Hessenberg matrix by its real Schur form.
/// \details Shapes, output types, block ordering, and failure behavior follow `schur`.
///          With `compute_vectors == true`, `schur_vectors` is initialized by the
///          backend and returns `Q` for the supplied Hessenberg matrix `H`, satisfying
///          `H = Q * T * transpose(Q)`. It does not accumulate a preceding dense-to-
///          Hessenberg reduction's vectors. With false, vector elements are unchanged.
/// \pre Input is real, square, and already upper Hessenberg. Entries below its first
///      subdiagonal must be zero; this routine does not perform a Hessenberg reduction.
///      Work and output storage must not overlap. Outputs are not resized.
template <class BackendSelector, uni20::MutableRankedTensorView<2> MatrixTensor, class EigenScalar,
          uni20::MutableRankedTensorView<2> SchurVectorTensor>
void hessenberg_schur(BackendSelector&& selector, MatrixTensor&& matrix_work, std::span<EigenScalar> eigenvalues,
                      SchurVectorTensor&& schur_vectors, bool compute_vectors)
{
  auto matrix_descriptor = uni20::mdspec_of(matrix_work);
  auto vector_descriptor = uni20::mdspec_of(schur_vectors);
  dispatch_kernel(std::forward<BackendSelector>(selector), hessenberg_schur_op{.compute_vectors = compute_vectors},
                  matrix_descriptor, eigenvalues, vector_descriptor);
}

/// \brief Apply the destructive `hessenberg_schur` contract using tensor storage policy.
template <uni20::MutableRankedTensorView<2> MatrixTensor, class EigenScalar,
          uni20::MutableRankedTensorView<2> SchurVectorTensor>
void hessenberg_schur(MatrixTensor&& matrix_work, std::span<EigenScalar> eigenvalues, SchurVectorTensor&& schur_vectors,
                      bool compute_vectors)
{
  auto operation = hessenberg_schur_op{.compute_vectors = compute_vectors};
  auto selector = select_backend(operation, matrix_work, schur_vectors);
  hessenberg_schur(selector, std::forward<MatrixTensor>(matrix_work), eigenvalues,
                   std::forward<SchurVectorTensor>(schur_vectors), compute_vectors);
}

/// \brief Reorder a Schur form in place, optionally updating its Schur vectors.
/// \details Uses LAPACK `trexc` semantics. `from` and `to` are zero-based matrix row
///          indices, not ordinal block numbers. For real forms, an index in the second
///          row of a 2x2 block selects that whole block; the final start row may differ
///          from `to` because blocks have different sizes. Complex forms move one
///          diagonal entry. The adjusted destination is not returned.
///          Updating `Q` preserves `A = Q * T * adjoint(Q)`. Previously saved eigenvalue
///          arrays are not updated by this operation.
/// \pre `schur_form` is `n x n` in Schur canonical form (as produced by `schur`),
///      with both indices less than `n`. Requested vector storage has shape `n x n`,
///      the same scalar type, and does not overlap the form. With vector updates enabled,
///      it must contain the existing Schur vectors on entry. No output is resized.
/// \note A failed real block swap uses Uni20's error policy and may leave the form
///       and vectors partially reordered. There is no rollback. An empty form has
///       no valid reordering indices.
/// \note `update_vectors`: If false, vector elements are unchanged and their dimensions
///       are ignored; a tensor of a supported type is still required.
template <class BackendSelector, uni20::MutableRankedTensorView<2> SchurFormTensor,
          uni20::MutableRankedTensorView<2> SchurVectorTensor>
void reorder_schur(BackendSelector&& selector, SchurFormTensor&& schur_form, SchurVectorTensor&& schur_vectors,
                   std::size_t from, std::size_t to, bool update_vectors)
{
  auto form_descriptor = uni20::mdspec_of(schur_form);
  auto vector_descriptor = uni20::mdspec_of(schur_vectors);
  dispatch_kernel(std::forward<BackendSelector>(selector),
                  schur_reorder_op{.from = from, .to = to, .update_vectors = update_vectors}, form_descriptor,
                  vector_descriptor);
}

/// \brief Apply the in-place `reorder_schur` contract using tensor storage policy.
template <uni20::MutableRankedTensorView<2> SchurFormTensor, uni20::MutableRankedTensorView<2> SchurVectorTensor>
void reorder_schur(SchurFormTensor&& schur_form, SchurVectorTensor&& schur_vectors, std::size_t from, std::size_t to,
                   bool update_vectors)
{
  auto operation = schur_reorder_op{.from = from, .to = to, .update_vectors = update_vectors};
  auto selector = select_backend(operation, schur_form, schur_vectors);
  reorder_schur(selector, std::forward<SchurFormTensor>(schur_form), std::forward<SchurVectorTensor>(schur_vectors),
                from, to, update_vectors);
}

} // namespace uni20::linalg
