#pragma once

/**
 * \file reduction_task.hpp
 * \ingroup linalg
 * \brief Coroutine dispatch for CUDA reductions returning host scalars.
 */

#include <uni20/backend/cuda/task_awaiters.hpp>
#include <uni20/linalg/async/kernel_task.hpp>
#include <uni20/linalg/backends/cuda/reductions.hpp>

namespace uni20::linalg
{
namespace detail::cuda_reference
{
template <class Scalar, class Input>
async::CudaTask co_sum_submission(Scalar& output, Input const& input,
                                  LoweredStridedElementwisePlan<1, reduction_maximum_rank> plan)
{
  auto& resources = input.data_descriptor().buffer().resources();
  co_await uni20::cuda::set_device(resources.device());
  auto stream = co_await uni20::cuda::acquire_stream(resources.streams());
  output = reference_sum_of_plan<Scalar>(plan, input, stream);
}
} // namespace detail::cuda_reference

/// \brief Submit a full CUDA sum once a stream becomes available on the input device.
template <class Scalar, detail::cuda_reference::RawCudaReductionMdspec<Scalar> Input, std::size_t Rank>
auto try_make_kernel_task(CudaReferenceBackend, sum_reduction_op<Rank, Rank> const& op, Scalar& output,
                          Input& input) -> KernelTaskAttempt<async::CudaTask>
{
  using namespace detail::cuda_reference;
  CHECK(reduction_axes_are_valid(op.axes));
  LoweredStridedElementwisePlan<1, reduction_maximum_rank> plan;
  if (!try_lower_unary_reduction(input, plan))
    return KernelTaskAttempt<async::CudaTask>{KernelAttempt::unsupported_layout};
  if (plan.element_count() == 0)
  {
    output = Scalar{};
    return KernelTaskAttempt<async::CudaTask>{KernelAttempt::success};
  }

  auto task = co_sum_submission(output, input, std::move(plan));
  async::cuda_promise(task.handle()).bind_device(input.data_descriptor().buffer().device().ordinal());
  return KernelTaskAttempt<async::CudaTask>{std::move(task)};
}
} // namespace uni20::linalg
