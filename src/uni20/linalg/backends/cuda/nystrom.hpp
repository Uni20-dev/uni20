#pragma once

#include <uni20/linalg/backends/cuda/mdspec_traits.hpp>
#include <uni20/linalg/detail/nystrom_scaling.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/cuda_access.hpp>

namespace uni20::linalg
{
template <class M, class V, class R>
concept CudaNystromFactors =
    MutableRankedStridedMdspecLike<M, 2> && RankedStridedMdspecLike<V, 1> &&
    detail::cuda_reference::is_raw_mutable_cuda_mdspec<M> && detail::cuda_reference::is_raw_const_cuda_mdspec<V> &&
    std::same_as<typename M::value_type, R> && std::same_as<typename V::value_type, R> &&
    (std::same_as<R, float> || std::same_as<R, double>);

template <class M, class V, class R>
  requires CudaNystromFactors<M, V, R>
consteval auto kernel_accepts_types(CudaReferenceBackend, nystrom_whiten_op<R>, M&, V&)
{
  return kernel_types_maybe;
}

template <class M, class V, class R>
  requires CudaNystromFactors<M, V, R>
KernelAttempt try_kernel(CudaReferenceBackend, nystrom_whiten_op<R> op, M& columns, V& values)
{
  auto const width = values.extent(0);
  if (!std::in_range<int>(width)) return KernelAttempt::unsupported_shape;
  if (columns.stride(0) != 1 || columns.stride(1) != width || values.stride(0) != 1)
    return KernelAttempt::unsupported_layout;
  auto dst = columns.data_descriptor();
  auto src = values.data_descriptor();
  auto& out = dst.buffer();
  auto const& in = src.buffer();
  if (out.device() != in.device()) return KernelAttempt::incompatible_devices;
  uni20::detail::validate_cuda_mapping_range(columns.mapping(), columns.mapping().required_span_size(),
                                             dst.element_offset(), out.size());
  uni20::detail::validate_cuda_mapping_range(values.mapping(), values.mapping().required_span_size(),
                                             src.element_offset(), in.size());
  auto stream = out.resources().streams().acquire();
  cuda::ScopedDevice guard(stream.device());
  auto write = out.write_synchronized_with(stream);
  auto read = in.read_synchronized_with(stream);
  detail::randomized::enqueue_whitening(write.data() + dst.element_offset(), read.data() + src.element_offset(),
                                        static_cast<int>(width), op.tolerance, stream.native_handle(), stream.device());
  cuda::AccessCompletion completion(stream);
  completion.release(write);
  completion.release(read);
  return KernelAttempt::success;
}
} // namespace uni20::linalg
