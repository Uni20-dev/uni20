#pragma once

/**
 * \file copy_staging.hpp
 * \ingroup linalg
 * \brief CUDA copy backends that stage through a buffer and dispatch each step.
 */

#include <uni20/backend/cuda/buffer.hpp>
#include <uni20/core/types.hpp>
#include <uni20/linalg/backends/cpu/copy.hpp>
#include <uni20/linalg/backends/cuda/copy.hpp>
#include <uni20/linalg/backends/cuda/staging_backends.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/storage/cuda_storage.hpp>
#include <uni20/storage/host_storage.hpp>

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail::cuda_staging
{

using cuda_reference::active_strides_are_positive;
using cuda_reference::cuda_buffer_view;
using cuda_reference::mapping_is_contiguous;

template <class Mdspan> using scalar_t = std::remove_cv_t<typename std::remove_cvref_t<Mdspan>::element_type>;

template <class Mdspan> inline constexpr std::size_t rank_v = std::remove_cvref_t<Mdspan>::rank();

template <std::size_t Rank> using transfer_mapping_t = stdex::layout_stride::mapping<stdex::dextents<index_type, Rank>>;

/// \brief CUDA staging view in a transfer order.
template <class Element, std::size_t Rank>
using device_stage_t = uni20::mdspec<Element, stdex::dextents<index_type, Rank>, stdex::layout_stride,
                                     uni20::cuda::CudaPointerAccessor<Element>, uni20::cuda::CudaBufferView<Element>>;

/// \brief Host staging view in a transfer order.
template <class Element, std::size_t Rank>
using host_stage_t = stdex::mdspan<Element, stdex::dextents<index_type, Rank>, stdex::layout_stride>;

/// \brief Host staging storage whose elements are never constructed or destroyed.
/// \details Staged elements are trivially copyable, and each is written by a
///          copy before it is read, so the staging buffer places no
///          constructibility requirement on the element type.
template <class Scalar> class HostStagingBuffer {
  public:
    explicit HostStagingBuffer(std::size_t count) : data_(std::allocator<Scalar>{}.allocate(count)), count_(count) {}
    HostStagingBuffer(HostStagingBuffer const&) = delete;
    HostStagingBuffer& operator=(HostStagingBuffer const&) = delete;
    ~HostStagingBuffer() { std::allocator<Scalar>{}.deallocate(data_, count_); }

    [[nodiscard]] Scalar* data() const noexcept { return data_; }

  private:
    Scalar* data_;
    std::size_t count_;
};

/// \brief One-dimensional CUDA view of a storage range.
template <class Element>
using device_range_t = uni20::mdspec<Element, stdex::dextents<index_type, 1>, stdex::layout_right,
                                     uni20::cuda::CudaPointerAccessor<Element>, uni20::cuda::CudaBufferView<Element>>;

/// \brief Outcome of the checks a staging backend makes before any work.
struct StagingCheck
{
    KernelAttempt attempt = KernelAttempt::success;
    bool has_work = false;
};

/// \brief Check extents and strides shared by both staging backends.
template <class OutputMdspan, class InputMdspan>
[[nodiscard]] StagingCheck check_staged_operands(OutputMdspan const& output, InputMdspan const& input)
{
  bool empty = false;
  for (std::size_t axis = 0; axis < rank_v<OutputMdspan>; ++axis)
  {
    if (output.extent(axis) != input.extent(axis)) return {KernelAttempt::unsupported_shape, false};
    if (output.extent(axis) == 0) empty = true;
  }
  if (empty) return {KernelAttempt::success, false};
  if (!active_strides_are_positive(output) || !active_strides_are_positive(input))
    return {KernelAttempt::unsupported_layout, false};
  return {KernelAttempt::success, true};
}

/// \brief Whether `span` addresses one contiguous range in the element order of `mapping`.
template <class Mdspan, class Mapping>
[[nodiscard]] bool matches_transfer_layout(Mdspan const& span, Mapping const& mapping)
{
  if (!mapping_is_contiguous(span.mapping())) return false;
  for (std::size_t axis = 0; axis < rank_v<Mdspan>; ++axis)
  {
    if (span.extent(axis) > 1 && std::cmp_not_equal(span.stride(axis), mapping.stride(axis))) return false;
  }
  return true;
}

/// \brief Choose the contiguous element order a host-device transfer moves.
/// \details The host operand's order is preferred, so that the device performs
///          any layout change; a non-contiguous host operand takes the device
///          operand's order, and column-major order serves when neither side is
///          contiguous.
template <class HostMdspan, class DeviceMdspan>
[[nodiscard]] auto make_transfer_mapping(HostMdspan const& host, DeviceMdspan const& device)
{
  constexpr std::size_t rank = rank_v<HostMdspan>;
  std::array<index_type, rank> extents{};
  std::array<index_type, rank> strides{};
  for (std::size_t axis = 0; axis < rank; ++axis)
    extents[axis] = static_cast<index_type>(host.extent(axis));

  auto copy_strides = [&](auto const& span) {
    for (std::size_t axis = 0; axis < rank; ++axis)
      strides[axis] = static_cast<index_type>(span.stride(axis));
  };
  if (mapping_is_contiguous(host.mapping()))
    copy_strides(host);
  else if (mapping_is_contiguous(device.mapping()))
    copy_strides(device);
  else
  {
    index_type stride = 1;
    for (std::size_t axis = 0; axis < rank; ++axis)
    {
      strides[axis] = stride;
      stride *= extents[axis];
    }
  }
  return transfer_mapping_t<rank>(stdex::dextents<index_type, rank>(extents), strides);
}

template <class Mdspan>
inline constexpr bool is_device_to_device_copy_operand =
    cuda_reference::is_raw_cuda_mdspec<Mdspan> || cuda_reference::is_supported_cuda_mdspan<Mdspan>;

/// \brief Type-level eligibility of a peer-staged copy for a nested selector.
template <class DeviceCopySelector, class OutputMdspan, class InputMdspan>
consteval bool peer_staging_types_compatible()
{
  // A rank-zero copy is one element, which the reference backend always moves.
  if constexpr (!(cuda_reference::SupportedCopyMdspans<OutputMdspan, InputMdspan> &&
                  cuda_reference::is_raw_cuda_mdspec<OutputMdspan> &&
                  cuda_reference::is_supported_cuda_mdspan<InputMdspan>) ||
                rank_v<OutputMdspan> == 0)
  {
    return false;
  }
  else
  {
    using scalar_type = scalar_t<InputMdspan>;
    return requires(DeviceCopySelector const& selector, device_range_t<scalar_type>& stage,
                    device_range_t<scalar_type const>& source, OutputMdspan& output,
                    std::remove_cvref_t<InputMdspan>& staged_input) {
      try_dispatch_kernel(selector, copy_op{}, stage, source);
      try_dispatch_kernel(selector, copy_op{}, output, staged_input);
    };
  }
}

/// \brief Check a peer-staged copy and size its staging buffer.
template <class OutputMdspan, class InputMdspan>
[[nodiscard]] StagingCheck check_peer_staging(OutputMdspan& output, InputMdspan& input, std::size_t& span_size)
{
  auto check = check_staged_operands(output, input);
  if (!kernel_attempt_succeeded(check.attempt) || !check.has_work) return check;
  if (cuda_buffer_view(output).buffer().device() == cuda_buffer_view(input).buffer().device())
    return {KernelAttempt::unsupported_instance, false};

  auto const required_span = input.mapping().required_span_size();
  if (std::cmp_less_equal(required_span, 0) || std::cmp_greater(required_span, std::numeric_limits<index_type>::max()))
    return {KernelAttempt::unsupported_shape, false};
  span_size = static_cast<std::size_t>(required_span);
  return check;
}

/// \brief Views over a peer staging buffer: the gathered range and the source layout on it.
template <class InputMdspan> struct PeerStageViews
{
    using scalar_type = scalar_t<InputMdspan>;
    using input_type = std::remove_cvref_t<InputMdspan>;

    device_range_t<scalar_type> stage_range;
    device_range_t<scalar_type const> source_range;
    input_type staged_input;
};

/// \brief Build the gather step's views and the staged source for the relayout step.
/// \details The staged range starts at the source's element offset, so the
///          source mapping and accessor address it unchanged.
template <class InputMdspan>
[[nodiscard]] PeerStageViews<InputMdspan> make_peer_stage_views(uni20::cuda::CudaBuffer<scalar_t<InputMdspan>>& stage,
                                                                InputMdspan& input, std::size_t span_size)
{
  using views_type = PeerStageViews<InputMdspan>;
  using scalar_type = typename views_type::scalar_type;
  using input_type = typename views_type::input_type;
  using range_extents = stdex::dextents<index_type, 1>;
  using range_mapping = stdex::layout_right::mapping<range_extents>;

  uni20::cuda::CudaBufferView<scalar_type> const stage_view(stage);
  range_mapping const range{range_extents{static_cast<index_type>(span_size)}};
  return views_type{
      device_range_t<scalar_type>{stage_view, range, {}},
      device_range_t<scalar_type const>{
          uni20::cuda::CudaBufferView<scalar_type const>(cuda_buffer_view(input)), range, {}},
      input_type{typename input_type::data_descriptor_type(stage_view), input.mapping(), input.accessor()},
  };
}

/// \brief Peer-copy the source range to the destination device, then copy from it there.
template <class DeviceCopySelector, class OutputMdspan, class InputMdspan>
KernelAttempt copy_through_peer_stage(DeviceCopySelector const& selector, OutputMdspan& output, InputMdspan& input)
{
  std::size_t span_size = 0;
  auto const check = check_peer_staging(output, input, span_size);
  if (!kernel_attempt_succeeded(check.attempt) || !check.has_work) return check.attempt;

  uni20::cuda::CudaBuffer<scalar_t<InputMdspan>> stage(cuda_buffer_view(output).buffer().resources(), span_size);
  auto views = make_peer_stage_views(stage, input, span_size);
  if (!try_dispatch_kernel(selector, copy_op{}, views.stage_range, views.source_range))
    return KernelAttempt::unsupported_instance;
  if (!try_dispatch_kernel(selector, copy_op{}, output, views.staged_input)) return KernelAttempt::unsupported_instance;
  return KernelAttempt::success;
}

/// \brief Type-level eligibility of a host-staged copy for the nested selectors.
template <class DeviceCopySelector, class HostCopySelector, class TransferSelector, class OutputMdspan,
          class InputMdspan>
consteval bool host_staging_types_compatible()
{
  // A rank-zero copy is one element, which the reference backend always moves.
  if constexpr (!cuda_reference::SupportedCopyMdspans<OutputMdspan, InputMdspan> || rank_v<OutputMdspan> == 0)
  {
    return false;
  }
  else
  {
    using scalar_type = scalar_t<OutputMdspan>;
    constexpr std::size_t rank = rank_v<OutputMdspan>;
    using host_output = host_stage_t<scalar_type, rank>;
    using host_input = host_stage_t<scalar_type const, rank>;
    using device_output = device_stage_t<scalar_type, rank>;
    using device_input = device_stage_t<scalar_type const, rank>;

    if constexpr (cuda_reference::is_raw_cuda_mdspec<OutputMdspan> &&
                  cuda_reference::is_supported_host_mdspan<InputMdspan>)
    {
      return requires(DeviceCopySelector const& device, HostCopySelector const& host, TransferSelector const& transfer,
                      OutputMdspan& output, InputMdspan& input, host_output& host_stage, host_input& host_source,
                      device_output& device_stage, device_input& device_source) {
        try_dispatch_kernel(host, copy_op{}, host_stage, input);
        try_dispatch_kernel(transfer, copy_op{}, device_stage, host_source);
        try_dispatch_kernel(device, copy_op{}, output, device_source);
      };
    }
    else if constexpr (cuda_reference::is_raw_host_mdspan<OutputMdspan> &&
                       cuda_reference::is_supported_cuda_mdspan<InputMdspan>)
    {
      return requires(DeviceCopySelector const& device, HostCopySelector const& host, TransferSelector const& transfer,
                      OutputMdspan& output, InputMdspan& input, host_output& host_stage, host_input& host_source,
                      device_output& device_stage, device_input& device_source) {
        try_dispatch_kernel(device, copy_op{}, device_stage, input);
        try_dispatch_kernel(transfer, copy_op{}, host_stage, device_source);
        try_dispatch_kernel(host, copy_op{}, output, host_source);
      };
    }
    else
    {
      return false;
    }
  }
}

/// \brief Copy a host operand to CUDA storage through one contiguous transfer.
/// \details A host copy packs the source into the transfer order unless it
///          already has it as a raw view, the transfer moves the range, and a
///          device copy changes the layout unless the output already has the
///          transfer order.
template <class Backend, class OutputMdspan, class InputMdspan>
KernelAttempt copy_host_to_device_staged(Backend const& backend, OutputMdspan& output, InputMdspan& input)
{
  using scalar_type = scalar_t<OutputMdspan>;
  constexpr std::size_t rank = rank_v<OutputMdspan>;
  auto const transfer = make_transfer_mapping(input, output);
  auto const count = static_cast<std::size_t>(transfer.required_span_size());

  std::optional<HostStagingBuffer<scalar_type>> host_storage;
  scalar_type const* host_data = nullptr;
  if constexpr (cuda_reference::is_raw_host_mdspan<InputMdspan>)
  {
    if (matches_transfer_layout(input, transfer)) host_data = input.data_handle();
  }
  if (host_data == nullptr)
  {
    host_storage.emplace(count);
    host_stage_t<scalar_type, rank> host_stage(host_storage->data(), transfer);
    if (!try_dispatch_kernel(backend.host_copy_selector, copy_op{}, host_stage, input))
      return KernelAttempt::unsupported_instance;
    host_data = host_storage->data();
  }
  host_stage_t<scalar_type const, rank> host_source(host_data, transfer);

  if (matches_transfer_layout(output, transfer))
  {
    device_stage_t<scalar_type, rank> device_target{cuda_buffer_view(output), transfer, {}};
    if (!try_dispatch_kernel(backend.transfer_selector, copy_op{}, device_target, host_source))
      return KernelAttempt::unsupported_instance;
    return KernelAttempt::success;
  }

  uni20::cuda::CudaBuffer<scalar_type> stage(cuda_buffer_view(output).buffer().resources(), count);
  device_stage_t<scalar_type, rank> device_stage{uni20::cuda::CudaBufferView<scalar_type>(stage), transfer, {}};
  if (!try_dispatch_kernel(backend.transfer_selector, copy_op{}, device_stage, host_source))
    return KernelAttempt::unsupported_instance;
  device_stage_t<scalar_type const, rank> device_source{
      uni20::cuda::CudaBufferView<scalar_type const>(device_stage.data_descriptor()), transfer, {}};
  if (!try_dispatch_kernel(backend.device_copy_selector, copy_op{}, output, device_source))
    return KernelAttempt::unsupported_instance;
  return KernelAttempt::success;
}

/// \brief Copy a CUDA operand to host storage through one contiguous transfer.
/// \details A device copy rearranges the source into the transfer order unless
///          it already has it as a raw view, the transfer moves the range, and
///          a host copy scatters it unless the output already has the transfer
///          order.
template <class Backend, class OutputMdspan, class InputMdspan>
KernelAttempt copy_device_to_host_staged(Backend const& backend, OutputMdspan& output, InputMdspan& input)
{
  using scalar_type = scalar_t<OutputMdspan>;
  constexpr std::size_t rank = rank_v<OutputMdspan>;
  auto const transfer = make_transfer_mapping(output, input);
  auto const count = static_cast<std::size_t>(transfer.required_span_size());

  auto to_host = [&](device_stage_t<scalar_type const, rank>& device_source) -> KernelAttempt {
    if (matches_transfer_layout(output, transfer))
    {
      host_stage_t<scalar_type, rank> host_target(output.data_handle(), transfer);
      if (!try_dispatch_kernel(backend.transfer_selector, copy_op{}, host_target, device_source))
        return KernelAttempt::unsupported_instance;
      return KernelAttempt::success;
    }
    HostStagingBuffer<scalar_type> host_storage(count);
    host_stage_t<scalar_type, rank> host_stage(host_storage.data(), transfer);
    if (!try_dispatch_kernel(backend.transfer_selector, copy_op{}, host_stage, device_source))
      return KernelAttempt::unsupported_instance;
    host_stage_t<scalar_type const, rank> host_source(host_storage.data(), transfer);
    if (!try_dispatch_kernel(backend.host_copy_selector, copy_op{}, output, host_source))
      return KernelAttempt::unsupported_instance;
    return KernelAttempt::success;
  };

  if constexpr (cuda_reference::is_raw_cuda_mdspec<InputMdspan>)
  {
    if (matches_transfer_layout(input, transfer))
    {
      device_stage_t<scalar_type const, rank> device_source{
          uni20::cuda::CudaBufferView<scalar_type const>(cuda_buffer_view(input)), transfer, {}};
      return to_host(device_source);
    }
  }

  uni20::cuda::CudaBuffer<scalar_type> stage(cuda_buffer_view(input).buffer().resources(), count);
  device_stage_t<scalar_type, rank> device_stage{uni20::cuda::CudaBufferView<scalar_type>(stage), transfer, {}};
  if (!try_dispatch_kernel(backend.device_copy_selector, copy_op{}, device_stage, input))
    return KernelAttempt::unsupported_instance;
  device_stage_t<scalar_type const, rank> device_source{
      uni20::cuda::CudaBufferView<scalar_type const>(device_stage.data_descriptor()), transfer, {}};
  return to_host(device_source);
}

} // namespace detail::cuda_staging

/// \brief Report peer-staged copy eligibility for the retained device selector.
template <class DeviceCopySelector, uni20::MutableMdspecLike OutputMdspan, uni20::MdspecLike InputMdspan>
consteval auto kernel_accepts_types(CudaPeerStagingBackend<DeviceCopySelector> const&, copy_op const&, OutputMdspan&,
                                    InputMdspan&)
{
  if constexpr (detail::cuda_staging::peer_staging_types_compatible<DeviceCopySelector, OutputMdspan, InputMdspan>())
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Copy between two devices through a staging buffer on the destination device.
/// \details Declines operands on the same device and copies the step selector
///          declines; the output is written only by the last step.
template <class DeviceCopySelector, uni20::MutableMdspecLike OutputMdspan, uni20::MdspecLike InputMdspan>
  requires(detail::cuda_staging::peer_staging_types_compatible<DeviceCopySelector, OutputMdspan, InputMdspan>())
KernelAttempt try_kernel(CudaPeerStagingBackend<DeviceCopySelector> const& backend, copy_op const&,
                         OutputMdspan& output, InputMdspan& input)
{
  return detail::cuda_staging::copy_through_peer_stage(backend.device_copy_selector, output, input);
}

/// \brief Report host-staged copy eligibility for the retained step selectors.
template <class DeviceCopySelector, class HostCopySelector, class TransferSelector,
          uni20::MutableMdspecLike OutputMdspan, uni20::MdspecLike InputMdspan>
consteval auto
kernel_accepts_types(CudaHostStagingBackend<DeviceCopySelector, HostCopySelector, TransferSelector> const&,
                     copy_op const&, OutputMdspan&, InputMdspan&)
{
  if constexpr (detail::cuda_staging::host_staging_types_compatible<DeviceCopySelector, HostCopySelector,
                                                                    TransferSelector, OutputMdspan, InputMdspan>())
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Copy between host and CUDA storage through one contiguous transfer.
/// \details Declines when a step selector declines; the output is written only
///          by the last step. Coroutine dispatch runs this blocking form on the
///          scheduler thread.
template <class DeviceCopySelector, class HostCopySelector, class TransferSelector,
          uni20::MutableMdspecLike OutputMdspan, uni20::MdspecLike InputMdspan>
  requires(detail::cuda_staging::host_staging_types_compatible<DeviceCopySelector, HostCopySelector, TransferSelector,
                                                               OutputMdspan, InputMdspan>())
KernelAttempt try_kernel(CudaHostStagingBackend<DeviceCopySelector, HostCopySelector, TransferSelector> const& backend,
                         copy_op const&, OutputMdspan& output, InputMdspan& input)
{
  auto const check = detail::cuda_staging::check_staged_operands(output, input);
  if (!kernel_attempt_succeeded(check.attempt) || !check.has_work) return check.attempt;
  if constexpr (detail::cuda_reference::is_cuda_mdspan<OutputMdspan>)
    return detail::cuda_staging::copy_host_to_device_staged(backend, output, input);
  else
    return detail::cuda_staging::copy_device_to_host_staged(backend, output, input);
}

/// \brief Backend selector for copies between host and CUDA storage.
/// \details `CudaReferenceBackend` moves operands that share one contiguous
///          element order; `CudaHostStagingBackend` handles every other
///          layout, changing layouts through the `CudaStorage` and
///          `HostStorage` copy selectors, including any user
///          `backend_selector_override` for `copy_op`. The selector lookups
///          depend on the template parameters, so including this header does
///          not instantiate those overrides; library callers pass their operand
///          types as `Dependent` so that a user specialization declared after
///          the headers is still found.
/// \tparam Operation Copy operation tag.
/// \tparam Dependent Types that make a call inside a template dependent.
template <class Operation = copy_op, class... Dependent> [[nodiscard]] auto cuda_host_transfer_selector()
{
  return backend_list{CudaReferenceBackend{},
                      CudaHostStagingBackend{select_backend_for_storage<CudaStorage>(Operation{}),
                                             select_backend_for_storage<HostStorage>(Operation{}),
                                             CudaReferenceBackend{}}};
}

} // namespace uni20::linalg
