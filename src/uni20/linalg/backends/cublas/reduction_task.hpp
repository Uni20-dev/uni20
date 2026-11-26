#pragma once

/**
 * \file reduction_task.hpp
 * \ingroup linalg
 * \brief Coroutine dispatch for cuBLAS reductions returning host scalars.
 */

#include <uni20/backend/cublas/task_awaiters.hpp>
#include <uni20/backend/cuda/task_awaiters.hpp>
#include <uni20/linalg/async/kernel_task.hpp>
#include <uni20/linalg/backends/cublas/reductions.hpp>

namespace uni20::linalg
{
namespace detail::cublas_backend
{
template <class Input>
async::CudaTask co_norm_submission(uni20::make_real_t<typename Input::value_type>& output, Input const& input,
                                   int count)
{
  auto& resources = input.data_descriptor().buffer().resources();
  co_await uni20::cuda::set_device(resources.device());
  auto execution = co_await uni20::cublas::acquire_execution(uni20::cublas::execution_pool(resources));
  output = execute_norm(execution, input, count);
}

template <class Scalar, class Lhs, class Rhs>
async::CudaTask co_inner_product_submission(Scalar& output, Lhs const& lhs, Rhs const& rhs, int count)
{
  auto& resources = lhs.data_descriptor().buffer().resources();
  co_await uni20::cuda::set_device(resources.device());
  auto execution = co_await uni20::cublas::acquire_execution(uni20::cublas::execution_pool(resources));
  output = execute_inner_product(execution, lhs, rhs, count);
}
} // namespace detail::cublas_backend

/// \brief Await the cuBLAS handle and stream for a nonempty dense norm.
template <class Input>
  requires detail::cublas_backend::RawReadableCudaReductionMdspec<Input, typename Input::value_type>
auto try_make_kernel_task(CublasBackend, norm_op const&, uni20::make_real_t<typename Input::value_type>& output,
                          Input& input) -> KernelTaskAttempt<async::CudaTask>
{
  int count = 0;
  auto attempt = detail::cublas_backend::prepare_norm(input, count);
  if (!kernel_attempt_succeeded(attempt)) return KernelTaskAttempt<async::CudaTask>{attempt};
  if (count == 0)
  {
    output = {};
    return KernelTaskAttempt<async::CudaTask>{KernelAttempt::success};
  }
  auto task = detail::cublas_backend::co_norm_submission(output, input, count);
  async::cuda_promise(task.handle()).bind_device(input.data_descriptor().buffer().device().ordinal());
  return KernelTaskAttempt<async::CudaTask>{std::move(task)};
}

/// \brief Await the cuBLAS handle and stream for a nonempty dense inner product.
template <class Scalar, detail::cublas_backend::RawReadableCudaReductionMdspec<Scalar> Lhs,
          detail::cublas_backend::RawReadableCudaReductionMdspec<Scalar> Rhs>
auto try_make_kernel_task(CublasBackend, inner_product_op const&, Scalar& output, Lhs& lhs,
                          Rhs& rhs) -> KernelTaskAttempt<async::CudaTask>
{
  int count = 0;
  auto attempt = detail::cublas_backend::prepare_inner_product(lhs, rhs, count);
  if (!kernel_attempt_succeeded(attempt)) return KernelTaskAttempt<async::CudaTask>{attempt};
  if (count == 0)
  {
    output = {};
    return KernelTaskAttempt<async::CudaTask>{KernelAttempt::success};
  }
  auto task = detail::cublas_backend::co_inner_product_submission(output, lhs, rhs, count);
  async::cuda_promise(task.handle()).bind_device(lhs.data_descriptor().buffer().device().ordinal());
  return KernelTaskAttempt<async::CudaTask>{std::move(task)};
}
} // namespace uni20::linalg
