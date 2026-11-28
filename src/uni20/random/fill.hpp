#pragma once

#include <uni20/mdspan/iteration_plan.hpp>
#include <uni20/random/value.hpp>
#include <uni20/tensor/concepts.hpp>

#if UNI20_BACKEND_CUDA
#include <uni20/linalg/backends/cuda/random_fill.hpp>
#include <uni20/linalg/backends/cuda/transform.hpp>
#endif

namespace uni20::random
{
namespace detail
{
template <WordTransform Transform, class Mapping> auto storage_fill_plan(Mapping const& mapping, std::uint64_t origin)
{
  if (!mapping.is_unique()) throw std::invalid_argument("random fill requires a unique output mapping");
  auto plan = make_multi_iteration_plan(std::tuple{mapping});
  if (!plan.representable) throw std::length_error("random fill exceeds the iteration index range");
  if (plan.empty()) return plan;
  auto const span = mapping.required_span_size();
  if (!std::in_range<std::ptrdiff_t>(span)) throw std::length_error("random fill storage span exceeds the index range");
  // Validate the caller's address interval once, before writing or launching.
  constexpr auto last = last_element<Transform>;
  if (origin > last || std::cmp_greater(span - 1, last - origin)) address_out_of_range();
  auto const base = static_cast<std::ptrdiff_t>(mapping.base_offset());
  plan.base_offsets[0] += base;
  plan.reachable_offsets[0].minimum += base;
  plan.reachable_offsets[0].maximum += base;
  return plan;
}
} // namespace detail

/// \brief Fill a dense host view at its physical element offsets, leaving holes untouched.
/// \param origin Allocation offset of the start of the view's storage span, in elements.
///               Supply it whenever the leaf is shared with another view. The default
///               assigns the view its own address space under an exclusive leaf.
template <DefaultAccessorMdspanLike View, WordTransform Transform>
  requires MutableMdspanLike<View> && AffineMdspanLike<View> &&
           std::same_as<typename View::element_type, transformed_value_t<Transform>> &&
           uni20::detail::StorageSpanMapping<typename View::mapping_type>
void fill(View output, Identity leaf, Transform transform, std::uint64_t origin = 0)
{
  auto const plan = detail::storage_fill_plan<Transform>(output.mapping(), origin);
  if (plan.empty()) return;
  auto* storage = output.data_handle() - output.mapping().base_offset();
  constexpr auto g = Transform::result_type::size;
  std::uint64_t cached = detail::value_domain;
  typename Transform::result_type block{};
  auto visit = [&](auto& self, std::size_t depth, std::ptrdiff_t offset) -> void {
    if (depth == plan.rank())
    {
      auto const element = origin + static_cast<std::uint64_t>(offset);
      if (element / g != cached)
      {
        cached = element / g;
        block = value_group(leaf, cached, transform);
      }
      storage[offset] = block.values[element % g];
      return;
    }
    auto const& d = plan.dimensions[depth];
    for (std::ptrdiff_t i = 0; i < d.extent; ++i)
      self(self, depth + 1, offset + i * d.strides[0]);
  };
  visit(visit, 0, plan.base_offsets[0]);
}

#if UNI20_BACKEND_CUDA
/// \brief Enqueue a dense CUDA fill under the same storage-span origin convention as a host fill.
template <MutableMdspecLike View, WordTransform Transform>
  requires linalg::detail::cuda_reference::is_raw_mutable_cuda_mdspec<View> && AffineMdspecLike<View> &&
           uni20::detail::StorageSpanMapping<typename View::mapping_type> &&
           linalg::detail::cuda_reference::CudaRandomTransform<Transform> &&
           std::same_as<typename View::element_type, transformed_value_t<Transform>>
void fill(View output, Identity leaf, Transform transform, std::uint64_t origin = 0)
{
  namespace backend = linalg::detail::cuda_reference;
  auto const source = detail::storage_fill_plan<Transform>(output.mapping(), origin);
  if (source.empty()) return;
  backend::FillPlan<transformed_value_t<Transform>> plan;
  if (!backend::try_lower_strided_elementwise_plan<decltype(plan)::maximum_rank>(source, plan.elementwise_plan))
    throw std::length_error("random fill exceeds CUDA iteration limits");
  auto const descriptor = output.data_descriptor();
  plan.output_buffer = std::addressof(descriptor.buffer());
  uni20::detail::validate_cuda_mapping_range(output.mapping(), output.mapping().required_span_size(),
                                             descriptor.element_offset(), plan.output_buffer->size());
  plan.output_offset = descriptor.element_offset() - output.mapping().base_offset();
  plan.has_work = true;
  plan.attempt = linalg::KernelAttempt::success;
  backend::execute_overwrite_transform(plan, [&](auto* storage, auto const&, cudaStream_t stream, int device) {
    backend::enqueue_random_fill(storage, leaf, transform, origin, plan.elementwise_plan, stream, device);
  });
}
#endif

/// \brief Fill a dense tensor or tensor view using its writable descriptor.
template <MutableTensorView Tensor, WordTransform Transform>
  requires requires(Tensor& t, Identity leaf, Transform transform) { random::fill(mdspec_of(t), leaf, transform); }
void fill(Tensor&& output, Identity leaf, Transform transform, std::uint64_t origin = 0)
{
  random::fill(mdspec_of(output), leaf, transform, origin);
}
} // namespace uni20::random
