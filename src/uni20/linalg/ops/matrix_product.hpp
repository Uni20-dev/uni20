#pragma once

/**
 * \file matrix_product.hpp
 * \ingroup linalg
 * \brief Tensor-level matrix product update and overwrite operations.
 */

#include <uni20/common/trace.hpp>
#include <uni20/linalg/matrix_product_shape.hpp>
#include <uni20/linalg/ops/gemm.hpp>
#include <uni20/mdspan/mdspan.hpp>
#include <uni20/tensor/output.hpp>

#include <concepts>
#include <memory>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail
{
template <class OutputTensor, class InputTensor>
[[nodiscard]] constexpr bool is_obvious_tensor_alias(OutputTensor& output, InputTensor const& input) noexcept
{
  if constexpr (std::same_as<std::remove_cvref_t<OutputTensor>, std::remove_cvref_t<InputTensor>>)
  {
    return static_cast<void const*>(std::addressof(output)) == static_cast<void const*>(std::addressof(input));
  }
  return false;
}

template <uni20::MutableRankedTensorView<2> OutputTensor, uni20::RankedTensorView<2> LhsTensor,
          uni20::RankedTensorView<2> RhsTensor>
void validate_matrix_product_aliasing(OutputTensor& output, LhsTensor const& lhs, RhsTensor const& rhs)
{
  ERROR_IF(is_obvious_tensor_alias(output, lhs) || is_obvious_tensor_alias(output, rhs),
           "matrix product output must not alias an input tensor");
}

template <class OutputTensor, class LhsTensor, class RhsTensor>
concept CompatibleMatrixProductTensors =
    std::same_as<uni20::tensor_element_t<OutputTensor>, uni20::tensor_element_t<LhsTensor>> &&
    std::same_as<uni20::tensor_element_t<OutputTensor>, uni20::tensor_element_t<RhsTensor>>;
} // namespace detail

/// \brief Accumulate a matrix product into a fixed-shape Tensor output.
/// \details Computes `output += alpha * lhs * rhs`. The output shape is
///          validated and never resized because its old values participate in
///          the result.
/// \pre Output storage does not overlap either input, beyond the cheap
///      same-object aliases rejected by this front end.
template <class BackendSelector, uni20::MutableRankedTensorView<2> OutputTensor, uni20::RankedTensorView<2> LhsTensor,
          uni20::RankedTensorView<2> RhsTensor>
  requires detail::CompatibleMatrixProductTensors<OutputTensor, LhsTensor, RhsTensor>
void add_product(BackendSelector&& selector, OutputTensor&& output, LhsTensor const& lhs, RhsTensor const& rhs,
                 uni20::tensor_element_t<OutputTensor> alpha)
{
  detail::validate_matrix_product_aliasing(output, lhs, rhs);
  auto const shape = detail::matrix_product_shape(lhs, rhs);
  uni20::require_output(output, shape);
  gemm(std::forward<BackendSelector>(selector), std::forward<OutputTensor>(output), alpha, lhs, rhs,
       uni20::tensor_element_t<OutputTensor>{1});
}

/// \brief Accumulate a matrix product using the operands' default backend selector.
template <uni20::MutableRankedTensorView<2> OutputTensor, uni20::RankedTensorView<2> LhsTensor,
          uni20::RankedTensorView<2> RhsTensor>
  requires detail::CompatibleMatrixProductTensors<OutputTensor, LhsTensor, RhsTensor>
void add_product(OutputTensor&& output, LhsTensor const& lhs, RhsTensor const& rhs,
                 uni20::tensor_element_t<OutputTensor> alpha)
{
  detail::validate_matrix_product_aliasing(output, lhs, rhs);
  auto const shape = detail::matrix_product_shape(lhs, rhs);
  uni20::require_output(output, shape);
  auto selector = select_backend(gemm_op{}, output, lhs, rhs);
  gemm(selector, std::forward<OutputTensor>(output), alpha, lhs, rhs, uni20::tensor_element_t<OutputTensor>{1});
}

/// \brief Overwrite a resizable or already-compatible Tensor with a matrix product.
/// \details Computes `output = alpha * lhs * rhs`. Resizable outputs are
///          prepared before their writable mdspan is resolved; fixed outputs
///          must already have the required shape.
/// \pre Output storage does not overlap either input, beyond the cheap
///      same-object aliases rejected by this front end.
template <class BackendSelector, uni20::MutableRankedTensorView<2> OutputTensor, uni20::RankedTensorView<2> LhsTensor,
          uni20::RankedTensorView<2> RhsTensor>
  requires detail::CompatibleMatrixProductTensors<OutputTensor, LhsTensor, RhsTensor>
void assign_product(BackendSelector&& selector, OutputTensor&& output, LhsTensor const& lhs, RhsTensor const& rhs,
                    uni20::tensor_element_t<OutputTensor> alpha)
{
  detail::validate_matrix_product_aliasing(output, lhs, rhs);
  auto lhs_descriptor = uni20::mdspec_of(lhs);
  auto rhs_descriptor = uni20::mdspec_of(rhs);
#if UNI20_ENABLE_MPFR
  if constexpr (has_runtime_precision_v<tensor_element_t<OutputTensor>>)
  {
    auto p = common_default_precision(lhs, rhs);
    auto shape = detail::matrix_product_shape(lhs, rhs);
    prepare_output(output, shape, p);
    gemm(std::forward<BackendSelector>(selector), output, alpha, lhs, rhs, tensor_element_t<OutputTensor>{}, p);
  }
  else
#endif
    dispatch_kernel(std::forward<BackendSelector>(selector), assign_product_op{}, output, alpha, lhs_descriptor,
                    rhs_descriptor);
}

/// \brief Overwrite a Tensor with a matrix product using its default backend selector.
template <uni20::MutableRankedTensorView<2> OutputTensor, uni20::RankedTensorView<2> LhsTensor,
          uni20::RankedTensorView<2> RhsTensor>
  requires detail::CompatibleMatrixProductTensors<OutputTensor, LhsTensor, RhsTensor>
void assign_product(OutputTensor&& output, LhsTensor const& lhs, RhsTensor const& rhs,
                    uni20::tensor_element_t<OutputTensor> alpha)
{
  auto selector = select_backend(assign_product_op{}, output, lhs, rhs);
  assign_product(selector, std::forward<OutputTensor>(output), lhs, rhs, alpha);
}

/// \brief Overwrite with unit coefficient, deriving runtime precision from matching input defaults.
template <KernelBackendSelector BackendSelector, MutableRankedTensorView<2> Output, RankedTensorView<2> A,
          RankedTensorView<2> B>
  requires detail::CompatibleMatrixProductTensors<Output, A, B>
void assign_product(BackendSelector&& selector, Output&& output, A const& a, B const& b)
{
  assign_product(std::forward<BackendSelector>(selector), std::forward<Output>(output), a, b,
                 tensor_element_t<Output>{1});
}

/// \brief Overwrite with unit coefficient using storage-selected backends.
template <MutableRankedTensorView<2> Output, RankedTensorView<2> A, RankedTensorView<2> B>
  requires detail::CompatibleMatrixProductTensors<Output, A, B>
void assign_product(Output&& output, A const& a, B const& b)
{
  assign_product(select_backend(assign_product_op{}, output, a, b), std::forward<Output>(output), a, b);
}

/// \brief Accumulate with unit coefficient, deriving runtime precision from matching input defaults.
template <KernelBackendSelector BackendSelector, MutableRankedTensorView<2> Output, RankedTensorView<2> A,
          RankedTensorView<2> B>
  requires detail::CompatibleMatrixProductTensors<Output, A, B>
void add_product(BackendSelector&& selector, Output&& output, A const& a, B const& b)
{
  add_product(std::forward<BackendSelector>(selector), std::forward<Output>(output), a, b, tensor_element_t<Output>{1});
}

/// \brief Accumulate with unit coefficient using storage-selected backends.
template <MutableRankedTensorView<2> Output, RankedTensorView<2> A, RankedTensorView<2> B>
  requires detail::CompatibleMatrixProductTensors<Output, A, B>
void add_product(Output&& output, A const& a, B const& b)
{
  add_product(select_backend(gemm_op{}, output, a, b), std::forward<Output>(output), a, b);
}

#if UNI20_ENABLE_MPFR
/// \brief Overwrite at explicit operation precision with a unit coefficient.
template <KernelBackendSelector BackendSelector, MutableRankedTensorView<2> Output, RankedTensorView<2> A,
          RankedTensorView<2> B>
  requires detail::CompatibleMatrixProductTensors<Output, A, B> && has_runtime_precision_v<tensor_element_t<Output>>
void assign_product(BackendSelector&& selector, Output&& output, A const& a, B const& b, Precision p)
{
  detail::validate_matrix_product_aliasing(output, a, b);
  prepare_output(output, detail::matrix_product_shape(a, b), p);
  using S = tensor_element_t<Output>;
  gemm(std::forward<BackendSelector>(selector), output, S{1}, a, b, S{}, p);
}

/// \brief Overwrite at explicit precision using storage-selected backends.
template <MutableRankedTensorView<2> Output, RankedTensorView<2> A, RankedTensorView<2> B>
  requires detail::CompatibleMatrixProductTensors<Output, A, B> && has_runtime_precision_v<tensor_element_t<Output>>
void assign_product(Output&& output, A const& a, B const& b, Precision p)
{
  assign_product(select_backend(assign_product_op{}, output, a, b), std::forward<Output>(output), a, b, p);
}
#endif

#if UNI20_ENABLE_MPFR
/// \brief Overwrite a runtime-precision tensor using an exact coefficient.
template <KernelBackendSelector Selector, MutableRankedTensorView<2> Output, RankedTensorView<2> A,
          RankedTensorView<2> B, ExactRationalSource Alpha>
  requires has_runtime_precision_v<tensor_element_t<Output>> && detail::CompatibleMatrixProductTensors<Output, A, B>
void assign_product(Selector&& selector, Output&& output, A const& a, B const& b, Alpha const& alpha)
{
  assign_product(std::forward<Selector>(selector), std::forward<Output>(output), a, b, tensor_element_t<Output>(alpha));
}

/// \brief Overwrite with an exact coefficient through the storage-selected backend.
template <MutableRankedTensorView<2> Output, RankedTensorView<2> A, RankedTensorView<2> B, ExactRationalSource Alpha>
  requires has_runtime_precision_v<tensor_element_t<Output>> && detail::CompatibleMatrixProductTensors<Output, A, B>
void assign_product(Output&& output, A const& a, B const& b, Alpha const& alpha)
{
  assign_product(select_backend(assign_product_op{}, output, a, b), std::forward<Output>(output), a, b,
                 tensor_element_t<Output>(alpha));
}

/// \brief Accumulate a runtime-precision matrix product with an exact coefficient.
template <KernelBackendSelector Selector, MutableRankedTensorView<2> Output, RankedTensorView<2> A,
          RankedTensorView<2> B, ExactRationalSource Alpha>
  requires has_runtime_precision_v<tensor_element_t<Output>> && detail::CompatibleMatrixProductTensors<Output, A, B>
void add_product(Selector&& selector, Output&& output, A const& a, B const& b, Alpha const& alpha)
{
  add_product(std::forward<Selector>(selector), std::forward<Output>(output), a, b, tensor_element_t<Output>(alpha));
}

/// \brief Accumulate with an exact coefficient through the storage-selected backend.
template <MutableRankedTensorView<2> Output, RankedTensorView<2> A, RankedTensorView<2> B, ExactRationalSource Alpha>
  requires has_runtime_precision_v<tensor_element_t<Output>> && detail::CompatibleMatrixProductTensors<Output, A, B>
void add_product(Output&& output, A const& a, B const& b, Alpha const& alpha)
{
  add_product(select_backend(gemm_op{}, output, a, b), std::forward<Output>(output), a, b,
              tensor_element_t<Output>(alpha));
}
#endif

} // namespace uni20::linalg
