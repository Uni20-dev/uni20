#pragma once

/**
 * \file gemm.hpp
 * \ingroup linalg
 * \brief Fixed-output Tensor GEMM front end.
 */

#include <uni20/linalg/backends/cpu/gemm.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_scalar.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/concepts.hpp>
#include <uni20/tensor/output.hpp>
#include <uni20/tensor/precision.hpp>
#if UNI20_ENABLE_MPLAPACK_MPFR
#include <uni20/linalg/backends/mplapack/mpfr.hpp>
#endif

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
///          have shape `m x n`; it is never resized. Operands use the same scalar
///          type; coefficients are converted to the output scalar type. Runtime-precision
///          coefficients also accept integers or exact literals. Multiplication observes
///          the supplied views: transpose or conjugate a view explicitly when required.
///          With `beta == 0`, old output elements are not read. With `alpha == 0` or
///          `k == 0`, only the beta scaling remains. An empty output has no elements
///          to update; compatible dimensions are still required.
/// \pre Output storage must not overlap either input. Input views may overlap each other.
/// \note Use `assign_product` for an overwrite operation that may resize its output.
template <class BackendSelector, uni20::MutableRankedTensorView<2> OutputTensor, class Alpha, class Beta,
          uni20::RankedTensorView<2> LhsTensor, uni20::RankedTensorView<2> RhsTensor>
  requires(OperationScalar<Alpha, tensor_element_t<OutputTensor>> &&
           OperationScalar<Beta, tensor_element_t<OutputTensor>>)
void gemm(BackendSelector&& selector, OutputTensor&& output, Alpha alpha, LhsTensor const& lhs, RhsTensor const& rhs,
          Beta beta)
{
  using scalar_type = tensor_element_t<OutputTensor>;
  auto output_span = uni20::mdspec_of(output);
  auto lhs_span = uni20::mdspec_of(lhs);
  auto rhs_span = uni20::mdspec_of(rhs);
#if UNI20_ENABLE_MPFR
  if constexpr (has_runtime_precision_v<tensor_element_t<OutputTensor>>)
  {
    auto p = common_default_precision(lhs, rhs);
    if (beta != 0) p = common_precision(p, output.default_precision());
    dispatch_kernel(std::forward<BackendSelector>(selector), gemm_op{}, output_span, scalar_type(alpha), lhs_span,
                    rhs_span, scalar_type(beta), p);
    uni20::detail::record_output_precision(output, p);
  }
  else
#endif
    dispatch_kernel(std::forward<BackendSelector>(selector), gemm_op{}, output_span, scalar_type(alpha), lhs_span,
                    rhs_span, scalar_type(beta));
}

/// \brief Apply the fixed-storage `gemm` contract using the operands' default backend selector.
template <uni20::MutableRankedTensorView<2> OutputTensor, class Alpha, class Beta, uni20::RankedTensorView<2> LhsTensor,
          uni20::RankedTensorView<2> RhsTensor>
  requires(OperationScalar<Alpha, tensor_element_t<OutputTensor>> &&
           OperationScalar<Beta, tensor_element_t<OutputTensor>>)
void gemm(OutputTensor&& output, Alpha alpha, LhsTensor const& lhs, RhsTensor const& rhs, Beta beta)
{
  auto selector = select_backend(gemm_op{}, output, lhs, rhs);
  gemm(selector, std::forward<OutputTensor>(output), alpha, lhs, rhs, beta);
}

#if UNI20_ENABLE_MPFR
/// \brief Apply fixed-shape GEMM at explicit precision, converting all participating values at the backend boundary.
template <KernelBackendSelector BackendSelector, MutableRankedTensorView<2> Output, class Alpha, class Beta,
          RankedTensorView<2> A, RankedTensorView<2> B>
  requires(has_runtime_precision_v<tensor_element_t<Output>> && OperationScalar<Alpha, tensor_element_t<Output>> &&
           OperationScalar<Beta, tensor_element_t<Output>>)
void gemm(BackendSelector&& selector, Output&& output, Alpha alpha, A const& a, B const& b, Beta beta, Precision p)
{
  auto out = mdspec_of(output);
  auto ad = mdspec_of(a);
  auto bd = mdspec_of(b);
  using scalar_type = tensor_element_t<Output>;
  dispatch_kernel(std::forward<BackendSelector>(selector), gemm_op{}, out, scalar_type(alpha), ad, bd,
                  scalar_type(beta), p);
  uni20::detail::record_output_precision(output, p);
}

/// \brief Apply explicit-precision GEMM through the storage-selected backend.
template <MutableRankedTensorView<2> Output, class Alpha, class Beta, RankedTensorView<2> A, RankedTensorView<2> B>
  requires(has_runtime_precision_v<tensor_element_t<Output>> && OperationScalar<Alpha, tensor_element_t<Output>> &&
           OperationScalar<Beta, tensor_element_t<Output>>)
void gemm(Output&& output, Alpha alpha, A const& a, B const& b, Beta beta, Precision p)
{
  auto selector = select_backend(gemm_op{}, output, a, b);
  gemm(selector, std::forward<Output>(output), alpha, a, b, beta, p);
}
#endif

} // namespace uni20::linalg
