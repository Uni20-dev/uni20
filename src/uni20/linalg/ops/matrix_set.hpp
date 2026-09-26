#pragma once

/**
 * \file matrix_set.hpp
 * \ingroup linalg
 * \brief Structured dense matrix initialization operation.
 */

#include <uni20/linalg/backends/cpu/matrix_set.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/concepts.hpp>

#include <utility>

namespace uni20::linalg
{

/// \brief Set diagonal and off-diagonal entries in a selected matrix region.
/// \details Selected entries with `row == column` receive `diagonal`; other selected
///          entries receive `off_diagonal`. Upper and Lower both include the diagonal.
///          Entries outside the region are unchanged. Rectangular matrices are allowed;
///          no resize occurs, and an empty matrix has no entries to write.
/// \note `region`: All entries by default, or the inclusive upper/lower triangle.
template <class BackendSelector, uni20::MutableRankedTensorView<2> MatrixTensor, class Scalar>
void set_matrix(BackendSelector&& selector, MatrixTensor&& matrix, Scalar diagonal, Scalar off_diagonal,
                MatrixRegion region = MatrixRegion::All)
{
  auto matrix_descriptor = uni20::mdspec_of(matrix);
  dispatch_kernel(std::forward<BackendSelector>(selector), matrix_set_op{.region = region}, matrix_descriptor, diagonal,
                  off_diagonal);
}

/// \brief Apply the `set_matrix` region contract using the matrix storage's backend selector.
template <uni20::MutableRankedTensorView<2> MatrixTensor, class Scalar>
void set_matrix(MatrixTensor&& matrix, Scalar diagonal, Scalar off_diagonal, MatrixRegion region = MatrixRegion::All)
{
  auto selector = select_backend(matrix_set_op{.region = region}, matrix);
  set_matrix(selector, std::forward<MatrixTensor>(matrix), diagonal, off_diagonal, region);
}

} // namespace uni20::linalg
