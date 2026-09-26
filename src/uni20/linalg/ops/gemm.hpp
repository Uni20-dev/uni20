#pragma once

/**
 * \file gemm.hpp
 * \ingroup linalg
 * \brief Fixed-output Tensor GEMM front end.
 */

#include <uni20/linalg/backends/cpu/gemm.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/concepts.hpp>

#if UNI20_BACKEND_BLAS
#include <uni20/linalg/backends/blas/gemm.hpp>
#endif

#if UNI20_BACKEND_CUBLAS
#include <uni20/linalg/backends/cublas/gemm.hpp>
#endif

#include <utility>

namespace uni20::linalg
{
/// \brief Update a fixed-size matrix as `output = alpha * lhs * rhs + beta * output`.
/// \details For `lhs` of shape `m x k` and `rhs` of shape `k x n`, `output` must already
///          have shape `m x n`; it is never resized. Operands and coefficients use the
///          same scalar type. Multiplication observes the supplied views: transpose
///          or conjugate a view explicitly when required.
///          With `beta == 0`, old output elements are not read. With `alpha == 0` or
///          `k == 0`, only the beta scaling remains. An empty output has no elements
///          to update; compatible dimensions are still required.
/// \pre Output storage must not overlap either input. Input views may overlap each other.
/// \note Use `assign_product` for an overwrite operation that may resize its output.
template <class BackendSelector, uni20::MutableRankedTensorView<2> OutputTensor, class Scalar,
          uni20::RankedTensorView<2> LhsTensor, uni20::RankedTensorView<2> RhsTensor>
void gemm(BackendSelector&& selector, OutputTensor&& output, Scalar alpha, LhsTensor const& lhs, RhsTensor const& rhs,
          Scalar beta)
{
  auto output_span = uni20::mdspec_of(output);
  auto lhs_span = uni20::mdspec_of(lhs);
  auto rhs_span = uni20::mdspec_of(rhs);
  dispatch_kernel(std::forward<BackendSelector>(selector), gemm_op{}, output_span, alpha, lhs_span, rhs_span, beta);
}

/// \brief Apply the fixed-storage `gemm` contract using the operands' default backend selector.
template <uni20::MutableRankedTensorView<2> OutputTensor, class Scalar, uni20::RankedTensorView<2> LhsTensor,
          uni20::RankedTensorView<2> RhsTensor>
void gemm(OutputTensor&& output, Scalar alpha, LhsTensor const& lhs, RhsTensor const& rhs, Scalar beta)
{
  auto selector = select_backend(gemm_op{}, output, lhs, rhs);
  gemm(selector, std::forward<OutputTensor>(output), alpha, lhs, rhs, beta);
}

} // namespace uni20::linalg
