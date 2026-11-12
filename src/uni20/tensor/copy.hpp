#pragma once

/**
 * \file copy.hpp
 * \ingroup tensor
 * \brief Backend-dispatched tensor copies and inferred owning materialization.
 */

#include <uni20/mdspan/concepts.hpp>
#include <uni20/tensor/copy_into.hpp>
#include <uni20/tensor/output.hpp>
#include <uni20/tensor/tensor.hpp>

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace uni20
{
/// \brief Materialize a bare mdspan-like view as an inferred owning host tensor.
/// \details The caller supplies a backend selector because a bare mdspan does
///          not carry storage-domain policy. By default the output keeps the
///          source's canonical layout type. Inputs without a canonical physical
///          layout use the default column-major `Tensor` layout. An explicit
///          `RequestedLayout` overrides either choice. The result has the
///          input's compile-time rank with runtime extents on every axis.
///          The input must be host-readable; CUDA views are transferred only
///          through `to_host` or `copy(destination, source)`.
template <class RequestedLayout = void, class BackendSelector, MdspanLike InputMdspan>
  requires detail::AccessDomainsCompatible<
      stdex::default_accessor<std::remove_cv_t<typename std::remove_cvref_t<InputMdspan>::element_type> const>,
      typename std::remove_cvref_t<InputMdspan>::accessor_type>
[[nodiscard]] auto make_tensor(BackendSelector&& selector, InputMdspan&& input)
{
  using input_type = std::remove_cvref_t<InputMdspan>;
  using layout_type = detail::materialized_layout_t<RequestedLayout, input_type>;
  using result_type =
      Tensor<std::remove_cv_t<typename input_type::element_type>, input_type::rank(), HostStorage, layout_type>;

  result_type result(uninitialized, convert_tensor_extents<typename result_type::extents_type>(input.extents()));
  auto output_span = result.mdspan();
  copy(std::forward<BackendSelector>(selector), output_span, std::forward<InputMdspan>(input));
  return result;
}

namespace detail
{
template <class RequestedLayout, TensorView InputTensor>
using materialized_tensor_t =
    Tensor<tensor_element_t<InputTensor>, tensor_mdspec_t<InputTensor>::rank(),
           materialized_storage_t<typename tensor_mdspec_t<InputTensor>::accessor_type>,
           materialized_layout_t<RequestedLayout, tensor_mdspec_t<InputTensor>>,
           tensor_accessor_factory_t<materialized_storage_t<typename tensor_mdspec_t<InputTensor>::accessor_type>>>;
} // namespace detail

/// \brief Materialize a tensor view as an inferred owning tensor in the view's storage domain.
/// \details This is the explicit eager boundary for lazy views such as
///          `conj(input)`. Host-readable and generated inputs produce host
///          tensors; CUDA inputs produce CUDA tensors on the source device.
///          Materialization never transfers between domains: use `to_host` or
///          `to_device` for that. A canonical resolved layout is preserved
///          unless `RequestedLayout` is supplied. Generated and other
///          noncanonical inputs use the default column-major `Tensor` layout.
template <class RequestedLayout = void, TensorView InputTensor> [[nodiscard]] auto make_tensor(InputTensor const& input)
{
  using result_type = detail::materialized_tensor_t<RequestedLayout, InputTensor>;
  if constexpr (std::same_as<std::remove_cvref_t<InputTensor>, result_type> &&
                !std::is_copy_constructible_v<result_type>)
  {
    // Owning CUDA tensors are deliberately not copy-constructible; materialize on the same device.
    result_type result(uninitialized, detail::source_allocation_context(input),
                       convert_tensor_extents<typename result_type::extents_type>(input.extents()));
    copy(result, input);
    return result;
  }
  else
  {
    return result_type(input);
  }
}

namespace detail
{
/// \brief Tensor whose values are readable on the host without a storage-domain transfer.
template <class InputTensor>
concept HostReadableTensor =
    TensorView<std::remove_cvref_t<InputTensor>> &&
    AccessDomainsCompatible<stdex::default_accessor<tensor_element_t<std::remove_cvref_t<InputTensor>> const>,
                            typename tensor_mdspec_t<std::remove_cvref_t<InputTensor>>::accessor_type>;

/// \brief Materialize host work storage for an operation implemented only by host providers.
/// \details CUDA operands are rejected at compile time rather than copied to
///          the host. Callers who want a host computation transfer the
///          operand explicitly with `to_host`.
template <class RequestedLayout, TensorView InputTensor>
[[nodiscard]] auto make_host_work_tensor(InputTensor const& input)
{
  static_assert(HostReadableTensor<InputTensor>,
                "this linear-algebra operation has no implementation for CUDA operands; transfer the operand "
                "explicitly with uni20::to_host() to compute on the host");
  return make_tensor<RequestedLayout>(input);
}
} // namespace detail

#if UNI20_BACKEND_CUDA
/// \brief Materialize a CUDA tensor as an owning pageable host tensor.
/// \details The transfer waits for prior device work and returns only after
///          the host allocation is readable. The canonical source layout is
///          preserved.
template <TensorView InputTensor>
  requires(std::same_as<tensor_storage_policy_t<InputTensor>, CudaStorage> &&
           detail::CanonicalReshapeLayout<typename tensor_mdspec_t<InputTensor>::layout_type>)
[[nodiscard]] auto to_host(InputTensor const& input)
{
  using input_mdspan = tensor_mdspec_t<InputTensor>;
  using layout_type = typename input_mdspan::layout_type;
  using result_type = Tensor<tensor_element_t<InputTensor>, input_mdspan::rank(), HostStorage, layout_type>;

  result_type result(uninitialized, convert_tensor_extents<typename result_type::extents_type>(input.extents()));
  copy(result, input);
  return result;
}

/// \brief Materialize a tensor in an explicit CUDA device resource domain.
/// \details Pageable host inputs use blocking `cudaMemcpy`. CUDA inputs enqueue
///          a device or peer copy whose completion is retained by the result's
///          CUDA buffer ledger.
template <TensorView InputTensor>
  requires detail::CanonicalReshapeLayout<typename tensor_mdspec_t<InputTensor>::layout_type>
[[nodiscard]] auto to_device(InputTensor const& input, cuda::DeviceResources& resources)
{
  using input_mdspan = tensor_mdspec_t<InputTensor>;
  using layout_type = typename input_mdspan::layout_type;
  using result_type = Tensor<tensor_element_t<InputTensor>, input_mdspan::rank(), CudaStorage, layout_type>;

  result_type result(uninitialized, resources,
                     convert_tensor_extents<typename result_type::extents_type>(input.extents()));
  copy(result, input);
  return result;
}

/// \brief Materialize a tensor on an enrolled CUDA device ordinal.
template <TensorView InputTensor>
  requires detail::CanonicalReshapeLayout<typename tensor_mdspec_t<InputTensor>::layout_type>
[[nodiscard]] auto to_device(InputTensor const& input, int device)
{
  return to_device(input, cuda::device_resources(device));
}

/// \brief Materialize a tensor on an enrolled CUDA device.
template <TensorView InputTensor>
  requires detail::CanonicalReshapeLayout<typename tensor_mdspec_t<InputTensor>::layout_type>
[[nodiscard]] auto to_device(InputTensor const& input, cuda::Device device)
{
  return to_device(input, device.ordinal());
}
#endif

/// \brief Materialize an owning reshape of a non-owning or generated tensor view.
/// \details Immediately accessible, canonically laid-out strided sources are
///          reshaped before copying, preserving their logical contiguous
///          sequence. Deferred, layout-neutral, and noncanonical sources are
///          first materialized in the requested or inferred canonical layout,
///          then reshaped by transferring that allocation. Compatible owning
///          tensors use the move-aware overload in `reshape.hpp`.
template <class RequestedLayout = void, TensorView InputTensor, std::integral... Extents>
  requires(!OwningTensor<InputTensor> &&
           (std::is_void_v<RequestedLayout> || detail::CanonicalReshapeLayout<RequestedLayout>) &&
           (!std::is_void_v<RequestedLayout> || !StridedTensorView<InputTensor> ||
            detail::CanonicalReshapeLayout<typename tensor_mdspec_t<InputTensor>::layout_type>))
[[nodiscard]] auto reshape(InputTensor const& input, Extents... requested_extents)
{
  using input_layout = typename tensor_mdspec_t<InputTensor>::layout_type;
  if constexpr (ImmediateTensorView<InputTensor> && StridedTensorView<InputTensor> &&
                detail::CanonicalReshapeLayout<input_layout>)
  {
    auto view = reshape_view(input, requested_extents...);
    return make_tensor<RequestedLayout>(view);
  }
  else
  {
    auto materialized = make_tensor<RequestedLayout>(input);
    return reshape(std::move(materialized), requested_extents...);
  }
}

} // namespace uni20
