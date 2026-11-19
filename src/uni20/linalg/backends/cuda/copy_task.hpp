#pragma once

/**
 * \file copy_task.hpp
 * \ingroup linalg
 * \brief Coroutine kernel-dispatch customization for CUDA Tensor copies.
 */

#include <uni20/async/cuda_task.hpp>
#include <uni20/backend/cuda/task_awaiters.hpp>
#include <uni20/linalg/async/dispatch.hpp>
#include <uni20/linalg/async/kernel_task.hpp>
#include <uni20/linalg/backends/cuda/copy.hpp>
#include <uni20/linalg/backends/cuda/copy_staging.hpp>

#include <cstddef>

namespace uni20::linalg
{
namespace detail::cuda_reference
{

template <class Scalar> async::CudaTask co_copy_submission(CopyPlan<Scalar> plan)
{
  auto& resources = plan.output_buffer->resources();
  co_await uni20::cuda::set_device(resources.device());
  auto stream = co_await uni20::cuda::acquire_stream(resources.streams());
  enqueue_device_copy(plan, stream);
  co_return;
}

/// \brief Lower CUDA mdspans into a deferred non-blocking copy submission.
template <class OutputMdspan, class InputMdspan>
  requires(SupportedCopyMdspans<OutputMdspan, InputMdspan> && is_raw_cuda_mdspec<OutputMdspan> &&
           is_supported_cuda_mdspan<InputMdspan>)
auto try_make_copy_task(OutputMdspan& output, InputMdspan& input) -> KernelTaskAttempt<async::CudaTask>
{
  auto preparation = prepare_copy(output, input);
  if (!kernel_attempt_succeeded(preparation.attempt) || !preparation.has_work)
  {
    return KernelTaskAttempt<async::CudaTask>{preparation.attempt};
  }

  int const device = preparation.output_buffer->device().ordinal();
  auto task = co_copy_submission(std::move(preparation));
  async::cuda_promise(task.handle()).bind_device(device);
  return KernelTaskAttempt<async::CudaTask>{std::move(task)};
}
} // namespace detail::cuda_reference

/// \brief Create a CUDA copy task from normalized mdspecs.
template <uni20::MutableMdspecLike OutputMdspan, uni20::MdspecLike InputMdspan>
  requires(detail::cuda_reference::SupportedCopyMdspans<OutputMdspan, InputMdspan> &&
           detail::cuda_reference::is_raw_cuda_mdspec<OutputMdspan> &&
           detail::cuda_reference::is_supported_cuda_mdspan<InputMdspan>)
auto try_make_kernel_task(CudaReferenceBackend, copy_op const&, OutputMdspan& output,
                          InputMdspan& input) -> KernelTaskAttempt<async::CudaTask>
{
  return detail::cuda_reference::try_make_copy_task(output, input);
}

namespace detail::cuda_staging
{

/// \brief Peer-copy the source range, then copy it into the destination, as nested coroutine dispatches.
/// \details A nested decline raises: the outer dispatch has already committed to
///          this backend.
template <class DeviceCopySelector, class OutputMdspan, class InputMdspan>
async::AsyncTask co_copy_through_peer_stage(DeviceCopySelector selector, OutputMdspan output, InputMdspan input,
                                            std::size_t span_size)
{
  uni20::cuda::CudaBuffer<scalar_t<InputMdspan>> stage(cuda_buffer_view(output).buffer().resources(), span_size);
  auto views = make_peer_stage_views(stage, input, span_size);
  co_await co_dispatch_kernel(selector, copy_op{}, views.stage_range, views.source_range);
  co_await co_dispatch_kernel(selector, copy_op{}, output, views.staged_input);
}

} // namespace detail::cuda_staging

/// \brief Create a peer-staged copy task whose steps are submitted without blocking.
template <class DeviceCopySelector, uni20::MutableMdspecLike OutputMdspan, uni20::MdspecLike InputMdspan>
  requires(detail::cuda_staging::peer_staging_types_compatible<DeviceCopySelector, OutputMdspan, InputMdspan>())
auto try_make_kernel_task(CudaPeerStagingBackend<DeviceCopySelector> const& backend, copy_op const&,
                          OutputMdspan& output, InputMdspan& input) -> KernelTaskAttempt<async::AsyncTask>
{
  std::size_t span_size = 0;
  auto const check = detail::cuda_staging::check_peer_staging(output, input, span_size);
  if (!kernel_attempt_succeeded(check.attempt) || !check.has_work)
    return KernelTaskAttempt<async::AsyncTask>{check.attempt};
  return KernelTaskAttempt<async::AsyncTask>{
      detail::cuda_staging::co_copy_through_peer_stage(backend.device_copy_selector, output, input, span_size)};
}

} // namespace uni20::linalg
