#include "reduction_kernels.hpp"

#include <uni20/backend/cuda/cuda_error.hpp>

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>

namespace uni20::linalg::detail::cuda_reference
{
namespace
{

// Plan offsets count scalar elements. Native complex arrays expose their
// interleaved real and imaginary components through a pointer to the real type.
template <std::size_t Components, class Real, class Offset>
__device__ Real const* element_components(Real const* data, Offset offset)
{
  return data + static_cast<std::size_t>(offset) * Components;
}

template <class Partial> __device__ Partial initial_partial()
{
  if constexpr (std::is_same_v<Partial, MaximumPartial>)
    return MaximumPartial{-CUDART_INF};
  else
    return Partial{};
}

// Neumaier addition. A nonfinite sum already carries the IEEE result.
// Computing its rounding error could turn a single infinity into a NaN.
__device__ void accumulate_sum(SumComponent& partial, double value)
{
  double const next = partial.sum + value;
  if (isfinite(next))
    partial.correction +=
        fabs(partial.sum) >= fabs(value) ? (partial.sum - next) + value : (value - next) + partial.sum;
  else
    partial.correction = 0;
  partial.sum = next;
}

template <std::size_t Components>
__device__ SumPartial<Components> combine(SumPartial<Components> lhs, SumPartial<Components> const& rhs)
{
  for (std::size_t i = 0; i < Components; ++i)
  {
    auto& component = lhs.components[i];
    accumulate_sum(component, rhs.components[i].sum);
    if (isfinite(component.sum)) component.correction += rhs.components[i].correction;
  }
  return lhs;
}

/// \brief Reduce one value per thread to `shared[0]` with a fixed pairing.
template <class Partial> __device__ void reduce_block(Partial* shared, Partial value)
{
  shared[threadIdx.x] = value;
  __syncthreads();
  for (unsigned int width = reduction_threads / 2; width > 0; width /= 2)
  {
    if (threadIdx.x < width) shared[threadIdx.x] = combine(shared[threadIdx.x], shared[threadIdx.x + width]);
    __syncthreads();
  }
}

template <class Real, std::size_t Components, class Plan>
__global__ void __launch_bounds__(reduction_threads)
    norm_partials_kernel(Real const* input, Plan plan, NormPartial* partials)
{
  using index_type = typename Plan::logical_index_type;
  __shared__ NormPartial shared[reduction_threads];
  NormPartial local{};
  auto index = static_cast<index_type>(blockIdx.x) * static_cast<index_type>(reduction_threads) +
               static_cast<index_type>(threadIdx.x);
  auto const stride = static_cast<index_type>(gridDim.x) * static_cast<index_type>(reduction_threads);
  for (; index < plan.element_count; index += stride)
  {
    auto const* element = element_components<Components>(input, plan.offsets(index)[0]);
    for (std::size_t component = 0; component < Components; ++component)
      accumulate_norm(local, fabs(static_cast<double>(element[component])));
  }
  reduce_block(shared, local);
  if (threadIdx.x == 0) partials[blockIdx.x] = shared[0];
}

template <class Real, std::size_t Components, class Plan>
__global__ void __launch_bounds__(reduction_threads)
    inner_product_partials_kernel(Real const* lhs, Real const* rhs, Plan plan, InnerProductPartial* partials)
{
  using index_type = typename Plan::logical_index_type;
  __shared__ InnerProductPartial shared[reduction_threads];
  InnerProductPartial local{};
  auto index = static_cast<index_type>(blockIdx.x) * static_cast<index_type>(reduction_threads) +
               static_cast<index_type>(threadIdx.x);
  auto const stride = static_cast<index_type>(gridDim.x) * static_cast<index_type>(reduction_threads);
  for (; index < plan.element_count; index += stride)
  {
    auto const offsets = plan.offsets(index);
    auto const* left = element_components<Components>(lhs, offsets[0]);
    auto const* right = element_components<Components>(rhs, offsets[1]);
    double const left_real = static_cast<double>(left[0]);
    double const right_real = static_cast<double>(right[0]);
    if constexpr (Components == 1)
    {
      local.real += left_real * right_real;
    }
    else
    {
      double const left_imag = static_cast<double>(left[1]);
      double const right_imag = static_cast<double>(right[1]);
      local.real += left_real * right_real + left_imag * right_imag;
      local.imag += left_real * right_imag - left_imag * right_real;
    }
  }
  reduce_block(shared, local);
  if (threadIdx.x == 0) partials[blockIdx.x] = shared[0];
}

template <class Partial, class Real, std::size_t Components, class Plan>
__global__ void __launch_bounds__(reduction_threads)
    unary_partials_kernel(Real const* input, Plan plan, Partial* partials)
{
  using index_type = typename Plan::logical_index_type;
  __shared__ Partial shared[reduction_threads];
  auto local = initial_partial<Partial>();
  auto index = static_cast<index_type>(blockIdx.x) * static_cast<index_type>(reduction_threads) +
               static_cast<index_type>(threadIdx.x);
  auto const stride = static_cast<index_type>(gridDim.x) * static_cast<index_type>(reduction_threads);
  for (; index < plan.element_count; index += stride)
  {
    auto const* element = element_components<Components>(input, plan.offsets(index)[0]);
    if constexpr (std::is_same_v<Partial, SumPartial<Components>>)
    {
      for (std::size_t i = 0; i < Components; ++i)
        accumulate_sum(local.components[i], static_cast<double>(element[i]));
    }
    else
    {
      local = combine(local, MaximumPartial{static_cast<double>(element[0])});
    }
  }
  reduce_block(shared, local);
  if (threadIdx.x == 0) partials[blockIdx.x] = shared[0];
}

template <class Partial>
__global__ void __launch_bounds__(reduction_threads)
    combine_partials_kernel(Partial const* partials, std::size_t count, Partial* result)
{
  __shared__ Partial shared[reduction_threads];
  auto local = initial_partial<Partial>();
  for (std::size_t index = threadIdx.x; index < count; index += reduction_threads)
    local = combine(local, partials[index]);
  reduce_block(shared, local);
  if (threadIdx.x == 0) result[0] = shared[0];
}

template <class Partial>
void launch_combine(Partial const* partials, std::size_t count, Partial* result, cudaStream_t stream, int device,
                    char const* operation_name)
{
  combine_partials_kernel<<<1, reduction_threads, 0, stream>>>(partials, count, result);
  uni20::cuda::check(cudaGetLastError(), operation_name, device);
}

template <class Partial, class Real, std::size_t Components, class Plan>
void launch_unary(Real const* input, Plan const& plan, Partial* partials, cudaStream_t stream, int device,
                  char const* operation_name)
{
  unary_partials_kernel<Partial, Real, Components><<<reduction_block_count(plan.element_count), reduction_threads, 0, stream>>>(
      input, plan, partials);
  uni20::cuda::check(cudaGetLastError(), operation_name, device);
}

} // namespace

template <class Real, std::size_t Components, class Plan>
void enqueue_norm_partials(Real const* input, Plan const& plan, NormPartial* partials, cudaStream_t stream, int device)
{
  norm_partials_kernel<Real, Components><<<reduction_block_count(plan.element_count), reduction_threads, 0, stream>>>(
      input, plan, partials);
  uni20::cuda::check(cudaGetLastError(), "launch CUDA reference norm", device);
}

void enqueue_norm_combine(NormPartial const* partials, std::size_t count, NormPartial* result, cudaStream_t stream,
                          int device)
{
  launch_combine(partials, count, result, stream, device, "launch CUDA reference norm combine");
}

template <class Real, std::size_t Components, class Plan>
void enqueue_inner_product_partials(Real const* lhs, Real const* rhs, Plan const& plan, InnerProductPartial* partials,
                                    cudaStream_t stream, int device)
{
  inner_product_partials_kernel<Real, Components>
      <<<reduction_block_count(plan.element_count), reduction_threads, 0, stream>>>(
          lhs, rhs, plan, partials);
  uni20::cuda::check(cudaGetLastError(), "launch CUDA reference inner product", device);
}

template <class Real, std::size_t Components, class Plan>
void enqueue_sum_partials(Real const* input, Plan const& plan, SumPartial<Components>* partials, cudaStream_t stream,
                          int device)
{
  launch_unary<SumPartial<Components>, Real, Components>(input, plan, partials, stream, device,
                                                         "launch CUDA reference sum");
}

template <class Real, class Plan>
void enqueue_maximum_partials(Real const* input, Plan const& plan, MaximumPartial* partials, cudaStream_t stream,
                              int device)
{
  launch_unary<MaximumPartial, Real, 1>(input, plan, partials, stream, device, "launch CUDA reference maximum");
}

template <class Partial>
void enqueue_sum_combine(Partial const* partials, std::size_t count, Partial* result, cudaStream_t stream, int device)
{
  launch_combine(partials, count, result, stream, device, "launch CUDA reference sum combine");
}

void enqueue_maximum_combine(MaximumPartial const* partials, std::size_t count, MaximumPartial* result,
                             cudaStream_t stream, int device)
{
  launch_combine(partials, count, result, stream, device, "launch CUDA reference maximum combine");
}

template void enqueue_sum_combine(SumPartial<1> const*, std::size_t, SumPartial<1>*, cudaStream_t, int);
template void enqueue_sum_combine(SumPartial<2> const*, std::size_t, SumPartial<2>*, cudaStream_t, int);
template void enqueue_sum_combine(InnerProductPartial const*, std::size_t, InnerProductPartial*, cudaStream_t, int);

#define UNI20_CUDA_REFERENCE_REDUCTIONS(Real, Components)                                                              \
  template void enqueue_sum_partials<Real, Components, ReductionPlan32<1>>(                                            \
      Real const*, ReductionPlan32<1> const&, SumPartial<Components>*, cudaStream_t, int);                             \
  template void enqueue_sum_partials<Real, Components, ReductionPlan64<1>>(                                            \
      Real const*, ReductionPlan64<1> const&, SumPartial<Components>*, cudaStream_t, int);                             \
  template void enqueue_norm_partials<Real, Components, ReductionPlan32<1>>(Real const*, ReductionPlan32<1> const&,    \
                                                                            NormPartial*, cudaStream_t, int);          \
  template void enqueue_norm_partials<Real, Components, ReductionPlan64<1>>(Real const*, ReductionPlan64<1> const&,    \
                                                                            NormPartial*, cudaStream_t, int);          \
  template void enqueue_inner_product_partials<Real, Components, ReductionPlan32<2>>(                                  \
      Real const*, Real const*, ReductionPlan32<2> const&, InnerProductPartial*, cudaStream_t, int);                   \
  template void enqueue_inner_product_partials<Real, Components, ReductionPlan64<2>>(                                  \
      Real const*, Real const*, ReductionPlan64<2> const&, InnerProductPartial*, cudaStream_t, int);

UNI20_CUDA_REFERENCE_REDUCTIONS(float, 1)
UNI20_CUDA_REFERENCE_REDUCTIONS(float, 2)
UNI20_CUDA_REFERENCE_REDUCTIONS(double, 1)
UNI20_CUDA_REFERENCE_REDUCTIONS(double, 2)

template void enqueue_maximum_partials<float, ReductionPlan32<1>>(float const*, ReductionPlan32<1> const&,
                                                                  MaximumPartial*, cudaStream_t, int);
template void enqueue_maximum_partials<float, ReductionPlan64<1>>(float const*, ReductionPlan64<1> const&,
                                                                  MaximumPartial*, cudaStream_t, int);
template void enqueue_maximum_partials<double, ReductionPlan32<1>>(double const*, ReductionPlan32<1> const&,
                                                                   MaximumPartial*, cudaStream_t, int);
template void enqueue_maximum_partials<double, ReductionPlan64<1>>(double const*, ReductionPlan64<1> const&,
                                                                   MaximumPartial*, cudaStream_t, int);

#undef UNI20_CUDA_REFERENCE_REDUCTIONS

} // namespace uni20::linalg::detail::cuda_reference
