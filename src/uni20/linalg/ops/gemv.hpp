#pragma once

/**
 * \file gemv.hpp
 * \ingroup linalg
 * \brief Fixed-output Tensor GEMV front end.
 */

#include <uni20/linalg/backends/cpu/gemv.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/concepts.hpp>

#if UNI20_BACKEND_BLAS
#include <uni20/linalg/backends/blas/gemv.hpp>
#endif

#include <utility>

namespace uni20::linalg
{

/// \brief Update a fixed-size vector as `output = alpha * matrix * input + beta * output`.
/// \details For a matrix of shape `m x n`, `input` has length `n` and `output` must
///          already have length `m`; it is never resized. Operands and coefficients
///          use the same scalar type. Transpose or conjugate the matrix view explicitly
///          when required. With `beta == 0`, old output elements are not read; with
///          `alpha == 0`, matrix and input elements are not read. For finite coefficients,
///          a zero inner dimension leaves only beta scaling. An output of length zero
///          has no elements to update; compatible dimensions are still required.
/// \pre Output storage must not overlap the matrix or input vector.
template <class BackendSelector, uni20::MutableRankedTensorView<1> OutputTensor, class Scalar,
          uni20::RankedTensorView<2> MatrixTensor, uni20::RankedTensorView<1> InputTensor>
void gemv(BackendSelector&& selector, OutputTensor&& output, Scalar alpha, MatrixTensor const& matrix,
          InputTensor const& input, Scalar beta)
{
  auto output_span = uni20::mdspec_of(output);
  auto matrix_span = uni20::mdspec_of(matrix);
  auto input_span = uni20::mdspec_of(input);
  dispatch_kernel(std::forward<BackendSelector>(selector), gemv_op{}, output_span, alpha, matrix_span, input_span,
                  beta);
}

/// \brief Apply the fixed-storage `gemv` contract using the operands' default backend selector.
template <uni20::MutableRankedTensorView<1> OutputTensor, class Scalar, uni20::RankedTensorView<2> MatrixTensor,
          uni20::RankedTensorView<1> InputTensor>
void gemv(OutputTensor&& output, Scalar alpha, MatrixTensor const& matrix, InputTensor const& input, Scalar beta)
{
  auto selector = select_backend(gemv_op{}, output, matrix, input);
  gemv(selector, std::forward<OutputTensor>(output), alpha, matrix, input, beta);
}

} // namespace uni20::linalg
