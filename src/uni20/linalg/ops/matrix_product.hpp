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

/// \brief Overwrite a resizable or already-compatible output with alpha * a * b.
/// \details An omitted precision infers participating tensor defaults. An explicit
///          precision must match fixed result types; runtime values convert in the kernel.
/// \pre Output storage does not overlap either input.
template <KernelBackendSelector Selector, MutableRankedTensorView<2> Output, RankedTensorView<2> A,
          RankedTensorView<2> B, class Alpha>
  requires detail::CompatibleMatrixProductTensors<Output, A, B> && OperationScalar<Alpha, tensor_element_t<Output>>
void assign_product(Selector&& selector, Output&& output, A const& a, B const& b, Alpha const& alpha,
                    std::optional<Precision> requested = {})
{
  using S = tensor_element_t<Output>;
  detail::validate_matrix_product_aliasing(output, a, b);
  auto p = requested ? *requested : common_default_precision(a, b);
  require_precision<S>(p);
  if constexpr (has_runtime_precision_v<S>)
  {
    prepare_output(output, detail::matrix_product_shape(a, b), p);
    gemm(std::forward<Selector>(selector), output, S(alpha), a, b, S{}, p);
  }
  else
  {
    auto ad = mdspec_of(a);
    auto bd = mdspec_of(b);
    dispatch_kernel(std::forward<Selector>(selector), assign_product_op{}, output, S(alpha), ad, bd);
  }
}

/// \brief Apply assign_product using the storage-selected backend.
template <MutableRankedTensorView<2> Output, RankedTensorView<2> A, RankedTensorView<2> B, class Alpha>
  requires detail::CompatibleMatrixProductTensors<Output, A, B> && OperationScalar<Alpha, tensor_element_t<Output>>
void assign_product(Output&& output, A const& a, B const& b, Alpha const& alpha, std::optional<Precision> p = {})
{
  assign_product(select_backend(assign_product_op{}, output, a, b), std::forward<Output>(output), a, b, alpha, p);
}

/// \brief Apply assign_product with an exact unit coefficient and optional precision.
template <KernelBackendSelector Selector, MutableRankedTensorView<2> Output, RankedTensorView<2> A,
          RankedTensorView<2> B>
  requires detail::CompatibleMatrixProductTensors<Output, A, B>
void assign_product(Selector&& selector, Output&& output, A const& a, B const& b, std::optional<Precision> p = {})
{
  assign_product(std::forward<Selector>(selector), std::forward<Output>(output), a, b, tensor_element_t<Output>{1}, p);
}

/// \brief Apply unit-coefficient assign_product using storage-selected backends.
template <MutableRankedTensorView<2> Output, RankedTensorView<2> A, RankedTensorView<2> B>
  requires detail::CompatibleMatrixProductTensors<Output, A, B>
void assign_product(Output&& output, A const& a, B const& b, std::optional<Precision> p = {})
{
  assign_product(select_backend(assign_product_op{}, output, a, b), std::forward<Output>(output), a, b,
                 tensor_element_t<Output>{1}, p);
}

/// \brief Accumulate alpha * a * b into an existing fixed-shape output.
/// \details An omitted precision infers participating tensor defaults. An explicit
///          precision must match fixed result types; runtime values convert in the kernel.
/// \pre Output storage does not overlap either input.
template <KernelBackendSelector Selector, MutableRankedTensorView<2> Output, RankedTensorView<2> A,
          RankedTensorView<2> B, class Alpha>
  requires detail::CompatibleMatrixProductTensors<Output, A, B> && OperationScalar<Alpha, tensor_element_t<Output>>
void add_product(Selector&& selector, Output&& output, A const& a, B const& b, Alpha const& alpha,
                 std::optional<Precision> requested = {})
{
  using S = tensor_element_t<Output>;
  detail::validate_matrix_product_aliasing(output, a, b);
  require_output(output, detail::matrix_product_shape(a, b));
  gemm(std::forward<Selector>(selector), std::forward<Output>(output), S(alpha), a, b, S{1}, requested);
}

/// \brief Apply add_product using the storage-selected backend.
template <MutableRankedTensorView<2> Output, RankedTensorView<2> A, RankedTensorView<2> B, class Alpha>
  requires detail::CompatibleMatrixProductTensors<Output, A, B> && OperationScalar<Alpha, tensor_element_t<Output>>
void add_product(Output&& output, A const& a, B const& b, Alpha const& alpha, std::optional<Precision> p = {})
{
  add_product(select_backend(gemm_op{}, output, a, b), std::forward<Output>(output), a, b, alpha, p);
}

/// \brief Apply add_product with an exact unit coefficient and optional precision.
template <KernelBackendSelector Selector, MutableRankedTensorView<2> Output, RankedTensorView<2> A,
          RankedTensorView<2> B>
  requires detail::CompatibleMatrixProductTensors<Output, A, B>
void add_product(Selector&& selector, Output&& output, A const& a, B const& b, std::optional<Precision> p = {})
{
  add_product(std::forward<Selector>(selector), std::forward<Output>(output), a, b, tensor_element_t<Output>{1}, p);
}

/// \brief Apply unit-coefficient add_product using storage-selected backends.
template <MutableRankedTensorView<2> Output, RankedTensorView<2> A, RankedTensorView<2> B>
  requires detail::CompatibleMatrixProductTensors<Output, A, B>
void add_product(Output&& output, A const& a, B const& b, std::optional<Precision> p = {})
{
  add_product(select_backend(gemm_op{}, output, a, b), std::forward<Output>(output), a, b, tensor_element_t<Output>{1},
              p);
}

} // namespace uni20::linalg
