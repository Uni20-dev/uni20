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
template <class Op, class Output, class Input>
async::CudaTask co_unary_reduction(Output& output, Input const& input,
                                   LoweredStridedElementwisePlan<1, reduction_maximum_rank> plan)
{
  auto& resources = input.data_descriptor().buffer().resources();
  co_await uni20::cuda::set_device(resources.device());
  auto stream = co_await uni20::cuda::acquire_stream(resources.streams());
  using Scalar = typename Input::value_type;
  if constexpr (std::same_as<Op, norm_op>)
    output = reference_norm_of_plan<Scalar>(plan, input, stream);
  else
    output = reference_sum_of_plan<Scalar>(plan, input, stream);
}

template <class Op, class Output, class Input>
auto make_unary_reduction_task(Output& output, Input& input) -> KernelTaskAttempt<async::CudaTask>
{
  LoweredStridedElementwisePlan<1, reduction_maximum_rank> plan;
  if (!try_lower_unary_reduction(input, plan))
    return KernelTaskAttempt<async::CudaTask>{KernelAttempt::unsupported_layout};
  if (plan.element_count() == 0)
  {
    output = Output{};
    return KernelTaskAttempt<async::CudaTask>{KernelAttempt::success};
  }

  auto task = co_unary_reduction<Op>(output, input, std::move(plan));
  async::cuda_promise(task.handle()).bind_device(input.data_descriptor().buffer().device().ordinal());
  return KernelTaskAttempt<async::CudaTask>{std::move(task)};
}

template <class Scalar, class Lhs, class Rhs>
async::CudaTask co_inner_product_submission(Scalar& output, Lhs const& lhs, Rhs const& rhs,
                                            LoweredStridedElementwisePlan<2, reduction_maximum_rank> plan)
{
  auto& resources = lhs.data_descriptor().buffer().resources();
  co_await uni20::cuda::set_device(resources.device());
  auto stream = co_await uni20::cuda::acquire_stream(resources.streams());
  output = reference_inner_product_of_plan<Scalar>(plan, lhs, rhs, stream);
}
} // namespace detail::cuda_reference

/// \brief Submit a full CUDA sum once a stream becomes available on the input device.
template <class Scalar, detail::cuda_reference::RawCudaReductionMdspec<Scalar> Input, std::size_t Rank>
auto try_make_kernel_task(CudaReferenceBackend, sum_reduction_op<Rank, Rank> const& op, Scalar& output, Input& input)
{
  CHECK(reduction_axes_are_valid(op.axes));
  return detail::cuda_reference::make_unary_reduction_task<sum_reduction_op<Rank, Rank>>(output, input);
}

/// \brief Submit a reference norm once a stream becomes available on the input device.
template <class Input>
  requires detail::cuda_reference::RawCudaReductionMdspec<Input, typename Input::value_type>
auto try_make_kernel_task(CudaReferenceBackend, norm_op const&, uni20::make_real_t<typename Input::value_type>& output,
                          Input& input)
{
  return detail::cuda_reference::make_unary_reduction_task<norm_op>(output, input);
}

/// \brief Submit a reference inner product once a stream becomes available on the input device.
template <class Scalar, detail::cuda_reference::RawCudaReductionMdspec<Scalar> Lhs,
          detail::cuda_reference::RawCudaReductionMdspec<Scalar> Rhs>
auto try_make_kernel_task(CudaReferenceBackend, inner_product_op const&, Scalar& output, Lhs& lhs,
                          Rhs& rhs) -> KernelTaskAttempt<async::CudaTask>
{
  namespace ref = detail::cuda_reference;
  ref::LoweredStridedElementwisePlan<2, ref::reduction_maximum_rank> plan;
  auto const attempt = ref::prepare_inner_product(lhs, rhs, plan);
  if (!kernel_attempt_succeeded(attempt)) return KernelTaskAttempt<async::CudaTask>{attempt};
  if (plan.element_count() == 0)
  {
    output = {};
    return KernelTaskAttempt<async::CudaTask>{KernelAttempt::success};
  }
  auto task = ref::co_inner_product_submission(output, lhs, rhs, std::move(plan));
  async::cuda_promise(task.handle()).bind_device(lhs.data_descriptor().buffer().device().ordinal());
  return KernelTaskAttempt<async::CudaTask>{std::move(task)};
}
} // namespace uni20::linalg
