#pragma once

/**
 * \file reductions.hpp
 * \ingroup linalg
 * \brief Deterministic CUDA reference reductions over strided operands.
 */

#include <uni20/backend/cuda/buffer.hpp>
#include <uni20/backend/cuda/cuda_error.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/core/scalar_traits.hpp>
#include <uni20/linalg/backends/cuda/elementwise_plan.hpp>
#include <uni20/linalg/backends/cuda/reduction_kernels.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/mdspan/iteration_plan.hpp>
#include <uni20/storage/cuda_accessor.hpp>
#include <uni20/storage/cuda_storage.hpp>

#include <cuda_runtime_api.h>

#include <concepts>
#include <cstddef>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail::cuda_reference
{

/// \brief Scalars the reference reductions are compiled for.
template <class Scalar>
concept CudaReductionScalar = std::same_as<Scalar, float> || std::same_as<Scalar, double> ||
                              std::same_as<Scalar, uni20::cfloat> || std::same_as<Scalar, uni20::cdouble>;

template <class Output, class Scalar>
concept ReductionHostScalarOutput = (!uni20::MdspecLike<Output>) && std::same_as<std::remove_cvref_t<Output>, Scalar> &&
                                    requires(Output& output, Scalar value) { output = value; };

/// \brief Strided CUDA buffer operand read through the raw pointer accessor.
template <class Mdspec, class Scalar>
concept RawCudaReductionMdspec =
    CudaReductionScalar<Scalar> && uni20::StridedMdspecLike<Mdspec> && uni20::cuda::BufferMdspec<Mdspec> &&
    std::same_as<typename std::remove_cvref_t<Mdspec>::element_type, Scalar const> &&
    std::same_as<typename std::remove_cvref_t<Mdspec>::accessor_type, uni20::cuda::CudaPointerAccessor<Scalar const>>;

template <class Scalar> inline constexpr std::size_t reduction_components = uni20::is_complex_v<Scalar> ? 2 : 1;

/// \brief Check that every element a strided operand reaches lies inside its buffer.
template <class Mdspec> void require_reduction_range(Mdspec const& span)
{
  auto const& descriptor = span.data_descriptor();
  auto const required = span.mapping().required_span_size();
  CHECK(std::cmp_greater_equal(required, 0), required);
  auto const size = static_cast<std::size_t>(required);
  auto const offset = descriptor.element_offset();
  auto const buffer_size = descriptor.buffer().size();
  CHECK(offset <= buffer_size && size <= buffer_size - offset, offset, size, buffer_size);
}

/// \brief Number of block partials one launch over a lowered plan's kernel payload writes.
template <std::size_t OperandCount>
[[nodiscard]] unsigned int
reduction_blocks_per_slice(LoweredStridedElementwisePlan<OperandCount, reduction_maximum_rank> const& plan)
{
  return plan.index_kind == ElementwiseIndexKind::index_32 ? reduction_block_count(plan.plan_32.element_count)
                                                           : reduction_block_count(plan.plan_64.element_count);
}

/// \brief Number of partials a lowered plan writes: one per block for every host-loop slice.
template <std::size_t OperandCount>
[[nodiscard]] std::size_t
reduction_partial_count(LoweredStridedElementwisePlan<OperandCount, reduction_maximum_rank> const& plan)
{
  std::size_t slice_count = 1;
  for (auto const& dimension : plan.host_loop_dimensions)
    slice_count *= static_cast<std::size_t>(dimension.extent);
  return slice_count * reduction_blocks_per_slice(plan);
}

/// \brief Reduce a lowered plan to one partial on `stream` and return it on the host.
/// \details Each host-loop slice writes the same number of block partials into
///          its own workspace range; one combine launch then reduces every
///          partial in index order. The stream is synchronized before return.
template <class Partial, std::size_t OperandCount, class EnqueueSlice, class EnqueueCombine>
[[nodiscard]] Partial
reduce_lowered_plan(LoweredStridedElementwisePlan<OperandCount, reduction_maximum_rank> const& plan,
                    uni20::cuda::DeviceResources& resources, uni20::cuda::Stream const& stream,
                    EnqueueSlice&& enqueue_slice, EnqueueCombine&& enqueue_combine)
{
  unsigned int const blocks = reduction_blocks_per_slice(plan);
  std::size_t const partial_count = reduction_partial_count(plan);
  uni20::cuda::CudaBuffer<Partial> workspace(resources, partial_count + 1, stream);

  auto access = workspace.write_synchronized_with(stream);
  Partial* const partials = access.data();
  std::size_t slice = 0;
  plan.visit([&](auto const& kernel_plan) {
    enqueue_slice(kernel_plan, partials + slice * blocks);
    ++slice;
  });
  enqueue_combine(partials, partial_count, partials + partial_count);

  Partial result{};
  uni20::cuda::check(cudaMemcpyAsync(&result, partials + partial_count, sizeof(Partial), cudaMemcpyDeviceToHost,
                                     stream.native_handle()),
                     "copy CUDA reference reduction result", stream.device());
  uni20::cuda::check(cudaStreamSynchronize(stream.native_handle()), "synchronize CUDA reference reduction",
                     stream.device());
  access.release_after_synchronization();
  return result;
}

/// \brief Reduce one strided CUDA buffer to a host partial.
template <class Partial, class Scalar, class InputMdspec, class EnqueueSlice, class EnqueueCombine>
[[nodiscard]] Partial reduce_unary_plan(LoweredStridedElementwisePlan<1, reduction_maximum_rank> const& plan,
                                        InputMdspec const& input, uni20::cuda::Stream const& stream,
                                        EnqueueSlice enqueue_slice, EnqueueCombine enqueue_combine)
{
  static_assert(RawCudaReductionMdspec<InputMdspec, Scalar>);
  using real_type = uni20::make_real_t<Scalar>;
  require_reduction_range(input);
  auto const& descriptor = input.data_descriptor();
  auto& resources = descriptor.buffer().resources();
  uni20::cuda::ScopedDevice guard(stream.device());
  auto access = descriptor.buffer().read_synchronized_with(stream);
  auto const* data = reinterpret_cast<real_type const*>(access.data() + descriptor.element_offset());
  auto const partial = reduce_lowered_plan<Partial>(
      plan, resources, stream,
      [&](auto const& kernel_plan, Partial* partials) {
        enqueue_slice(data, kernel_plan, partials, stream.native_handle(), stream.device());
      },
      [&](Partial const* partials, std::size_t count, Partial* result) {
        enqueue_combine(partials, count, result, stream.native_handle(), stream.device());
      });
  access.release_after_synchronization();
  return partial;
}

template <class InputMdspec>
bool try_lower_unary_reduction(InputMdspec const& input, LoweredStridedElementwisePlan<1, reduction_maximum_rank>& plan)
{
  return try_lower_strided_elementwise_plan<reduction_maximum_rank>(
      make_multi_iteration_plan(std::tuple{input.mapping()}), plan);
}

/// \brief Compute the Euclidean norm of `input` over a lowered plan of its mapping.
/// \pre `plan` lowers `input.mapping()` and is not empty.
template <class Scalar, class InputMdspec>
[[nodiscard]] uni20::make_real_t<Scalar>
reference_norm_of_plan(LoweredStridedElementwisePlan<1, reduction_maximum_rank> const& plan, InputMdspec const& input,
                       uni20::cuda::Stream const& stream)
{
  using real_type = uni20::make_real_t<Scalar>;
  auto const partial = reduce_unary_plan<NormPartial, Scalar>(
      plan, input, stream,
      [](auto data, auto const& kernel_plan, auto partials, auto stream, int device) {
        enqueue_norm_partials<real_type, reduction_components<Scalar>>(data, kernel_plan, partials, stream, device);
      },
      enqueue_norm_combine);
  return static_cast<real_type>(finish_norm(partial));
}

template <class Scalar, class InputMdspec>
Scalar reference_sum_of_plan(LoweredStridedElementwisePlan<1, reduction_maximum_rank> const& plan,
                             InputMdspec const& input, uni20::cuda::Stream const& stream)
{
  using real_type = uni20::make_real_t<Scalar>;
  constexpr auto components = reduction_components<Scalar>;
  using partial_type = SumPartial<components>;
  auto const partial = reduce_unary_plan<partial_type, Scalar>(
      plan, input, stream,
      [](auto data, auto const& kernel_plan, auto partials, auto stream, int device) {
        enqueue_sum_partials<real_type, components>(data, kernel_plan, partials, stream, device);
      },
      enqueue_sum_combine<partial_type>);
  auto const real = static_cast<real_type>(partial.components[0].value());
  if constexpr (uni20::is_complex_v<Scalar>)
    return Scalar(real, static_cast<real_type>(partial.components[1].value()));
  else
    return real;
}

template <class Scalar, class InputMdspec> KernelAttempt try_reference_sum(Scalar& output, InputMdspec const& input)
{
  LoweredStridedElementwisePlan<1, reduction_maximum_rank> plan;
  if (!try_lower_unary_reduction(input, plan)) return KernelAttempt::unsupported_layout;
  if (plan.element_count() == 0)
  {
    output = Scalar{};
    return KernelAttempt::success;
  }
  auto stream = input.data_descriptor().buffer().resources().streams().acquire();
  output = reference_sum_of_plan<Scalar>(plan, input, stream);
  return KernelAttempt::success;
}

template <class Scalar, class InputMdspec> KernelAttempt try_reference_maximum(Scalar& output, InputMdspec const& input)
{
  LoweredStridedElementwisePlan<1, reduction_maximum_rank> plan;
  if (!try_lower_unary_reduction(input, plan)) return KernelAttempt::unsupported_layout;
  ERROR_IF(plan.element_count() == 0, "maximum requires a nonempty tensor");
  auto stream = input.data_descriptor().buffer().resources().streams().acquire();
  auto const partial = reduce_unary_plan<MaximumPartial, Scalar>(
      plan, input, stream,
      [](auto data, auto const& kernel_plan, auto partials, auto stream, int device) {
        enqueue_maximum_partials(data, kernel_plan, partials, stream, device);
      },
      enqueue_maximum_combine);
  output = static_cast<Scalar>(partial.value);
  return KernelAttempt::success;
}

/// \brief Compute a Euclidean norm over any positive-strided layout.
template <class Scalar, class InputMdspec>
KernelAttempt try_reference_norm(uni20::make_real_t<Scalar>& output, InputMdspec const& input)
{
  LoweredStridedElementwisePlan<1, reduction_maximum_rank> plan;
  if (!try_lower_unary_reduction(input, plan)) return KernelAttempt::unsupported_layout;
  if (plan.element_count() == 0)
    output = {};
  else
  {
    auto stream = input.data_descriptor().buffer().resources().streams().acquire();
    output = reference_norm_of_plan<Scalar>(plan, input, stream);
  }
  return KernelAttempt::success;
}

/// \brief Compute `sum(conj(lhs) * rhs)` over a lowered plan of both mappings.
/// \pre `plan` lowers `(lhs.mapping(), rhs.mapping())` and is not empty, and
///      both operands are on one device.
template <class Scalar, class LhsMdspec, class RhsMdspec>
[[nodiscard]] Scalar
reference_inner_product_of_plan(LoweredStridedElementwisePlan<2, reduction_maximum_rank> const& plan,
                                LhsMdspec const& lhs, RhsMdspec const& rhs, uni20::cuda::Stream const& stream)
{
  static_assert(RawCudaReductionMdspec<LhsMdspec, Scalar>);
  static_assert(RawCudaReductionMdspec<RhsMdspec, Scalar>);
  using real_type = uni20::make_real_t<Scalar>;
  auto const& lhs_descriptor = lhs.data_descriptor();
  auto const& rhs_descriptor = rhs.data_descriptor();
  require_reduction_range(lhs);
  require_reduction_range(rhs);

  auto& resources = lhs_descriptor.buffer().resources();
  uni20::cuda::ScopedDevice guard(stream.device());
  auto lhs_access = lhs_descriptor.buffer().read_synchronized_with(stream);
  auto rhs_access = rhs_descriptor.buffer().read_synchronized_with(stream);
  auto const* lhs_data = reinterpret_cast<real_type const*>(lhs_access.data() + lhs_descriptor.element_offset());
  auto const* rhs_data = reinterpret_cast<real_type const*>(rhs_access.data() + rhs_descriptor.element_offset());
  auto const partial = reduce_lowered_plan<InnerProductPartial>(
      plan, resources, stream,
      [&](auto const& kernel_plan, InnerProductPartial* partials) {
        enqueue_inner_product_partials<real_type, reduction_components<Scalar>>(
            lhs_data, rhs_data, kernel_plan, partials, stream.native_handle(), stream.device());
      },
      [&](InnerProductPartial const* partials, std::size_t count, InnerProductPartial* result) {
        enqueue_sum_combine(partials, count, result, stream.native_handle(), stream.device());
      });
  lhs_access.release_after_synchronization();
  rhs_access.release_after_synchronization();
  if constexpr (uni20::is_complex_v<Scalar>)
    return Scalar(static_cast<real_type>(partial.real), static_cast<real_type>(partial.imag));
  else
    return static_cast<Scalar>(partial.real);
}

template <class LhsMdspec, class RhsMdspec>
KernelAttempt prepare_inner_product(LhsMdspec const& lhs, RhsMdspec const& rhs,
                                    LoweredStridedElementwisePlan<2, reduction_maximum_rank>& plan)
{
  for (std::size_t axis = 0; axis < std::remove_cvref_t<LhsMdspec>::rank(); ++axis)
    CHECK_EQUAL(lhs.extent(axis), rhs.extent(axis));

  if (!try_lower_strided_elementwise_plan<reduction_maximum_rank>(
          make_multi_iteration_plan(std::tuple{lhs.mapping(), rhs.mapping()}), plan))
    return KernelAttempt::unsupported_layout;
  if (plan.element_count() != 0 && lhs.data_descriptor().buffer().device() != rhs.data_descriptor().buffer().device())
    return KernelAttempt::incompatible_devices;
  return KernelAttempt::success;
}

} // namespace detail::cuda_reference

/// \brief Report reference eligibility for a full CUDA inner product returning a host scalar.
template <class Output, uni20::MdspecLike LhsMdspec, uni20::MdspecLike RhsMdspec>
consteval auto kernel_accepts_types(CudaReferenceBackend const&, inner_product_op const&, Output&, LhsMdspec&,
                                    RhsMdspec&)
{
  using lhs_type = std::remove_cvref_t<LhsMdspec>;
  using rhs_type = std::remove_cvref_t<RhsMdspec>;
  using scalar_type = typename lhs_type::value_type;
  if constexpr (lhs_type::rank() == rhs_type::rank() && std::same_as<scalar_type, typename rhs_type::value_type> &&
                detail::cuda_reference::ReductionHostScalarOutput<Output, scalar_type> &&
                detail::cuda_reference::RawCudaReductionMdspec<LhsMdspec, scalar_type> &&
                detail::cuda_reference::RawCudaReductionMdspec<RhsMdspec, scalar_type>)
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Compute a full CUDA inner product with a deterministic reduction.
/// \details Accepts any positive-strided layouts, layouts that differ between
///          the operands, and any element count. Results are bitwise
///          reproducible for the same input, shape and kind of device.
///          Declines operands on different devices.
template <class Output, uni20::MdspecLike LhsMdspec, uni20::MdspecLike RhsMdspec>
  requires detail::cuda_reference::ReductionHostScalarOutput<Output,
                                                             typename std::remove_cvref_t<LhsMdspec>::value_type> &&
           detail::cuda_reference::RawCudaReductionMdspec<LhsMdspec,
                                                          typename std::remove_cvref_t<LhsMdspec>::value_type> &&
           detail::cuda_reference::RawCudaReductionMdspec<RhsMdspec,
                                                          typename std::remove_cvref_t<LhsMdspec>::value_type>
KernelAttempt try_kernel(CudaReferenceBackend, inner_product_op const&, Output& output, LhsMdspec& lhs, RhsMdspec& rhs)
{
  namespace ref = detail::cuda_reference;
  ref::LoweredStridedElementwisePlan<2, ref::reduction_maximum_rank> plan;
  auto const attempt = ref::prepare_inner_product(lhs, rhs, plan);
  if (!kernel_attempt_succeeded(attempt)) return attempt;
  if (plan.element_count() == 0)
    output = {};
  else
  {
    auto stream = lhs.data_descriptor().buffer().resources().streams().acquire();
    output = ref::reference_inner_product_of_plan<Output>(plan, lhs, rhs, stream);
  }
  return KernelAttempt::success;
}

/// \brief Report reference eligibility for a full CUDA norm returning a host scalar.
template <class Output, uni20::MdspecLike InputMdspec>
consteval auto kernel_accepts_types(CudaReferenceBackend const&, norm_op const&, Output&, InputMdspec&)
{
  using scalar_type = typename std::remove_cvref_t<InputMdspec>::value_type;
  if constexpr (detail::cuda_reference::CudaReductionScalar<scalar_type> &&
                detail::cuda_reference::ReductionHostScalarOutput<Output, uni20::make_real_t<scalar_type>> &&
                detail::cuda_reference::RawCudaReductionMdspec<InputMdspec, scalar_type>)
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Compute a full CUDA Euclidean norm with a deterministic, overflow-safe reduction.
/// \details Accepts any positive-strided layout and any element count. Results
///          are bitwise reproducible for the same input, shape and kind of
///          device.
template <class Output, uni20::MdspecLike InputMdspec>
  requires detail::cuda_reference::CudaReductionScalar<typename std::remove_cvref_t<InputMdspec>::value_type> &&
           detail::cuda_reference::ReductionHostScalarOutput<
               Output, uni20::make_real_t<typename std::remove_cvref_t<InputMdspec>::value_type>> &&
           detail::cuda_reference::RawCudaReductionMdspec<InputMdspec,
                                                          typename std::remove_cvref_t<InputMdspec>::value_type>
KernelAttempt try_kernel(CudaReferenceBackend, norm_op const&, Output& output, InputMdspec& input)
{
  using scalar_type = typename std::remove_cvref_t<InputMdspec>::value_type;
  return detail::cuda_reference::try_reference_norm<scalar_type>(output, input);
}

/// \brief Report eligibility for a full CUDA sum returning a host scalar.
template <class Scalar, detail::cuda_reference::RawCudaReductionMdspec<Scalar> Input, std::size_t Rank>
consteval auto kernel_accepts_types(CudaReferenceBackend const&, sum_reduction_op<Rank, Rank> const&, Scalar&, Input&)
{
  return kernel_types_maybe;
}

/// \brief Return a strided CUDA sum as a host scalar.
/// \details Each component uses Neumaier compensation in double precision throughout the fixed reduction tree.
template <class Scalar, detail::cuda_reference::RawCudaReductionMdspec<Scalar> Input, std::size_t Rank>
KernelAttempt try_kernel(CudaReferenceBackend, sum_reduction_op<Rank, Rank> const& operation, Scalar& output,
                         Input& input)
{
  CHECK(reduction_axes_are_valid(operation.axes));
  return detail::cuda_reference::try_reference_sum(output, input);
}

/// \brief Report eligibility for a signed CUDA maximum returning a host scalar.
template <uni20::Real Scalar, detail::cuda_reference::RawCudaReductionMdspec<Scalar> Input>
consteval auto kernel_accepts_types(CudaReferenceBackend const&, maximum_op const&, Scalar&, Input&)
{
  return kernel_types_maybe;
}

/// \brief Return the largest real entry of a strided CUDA buffer.
template <uni20::Real Scalar, detail::cuda_reference::RawCudaReductionMdspec<Scalar> Input>
KernelAttempt try_kernel(CudaReferenceBackend, maximum_op const&, Scalar& output, Input& input)
{
  return detail::cuda_reference::try_reference_maximum(output, input);
}

} // namespace uni20::linalg
