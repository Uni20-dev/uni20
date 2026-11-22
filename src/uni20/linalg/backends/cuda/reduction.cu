#include "reduction_kernels.hpp"

#include <uni20/backend/cuda/cuda_error.hpp>

#include <cuda_runtime.h>
#include <math_constants.h>

#include <cstddef>

namespace uni20::linalg::detail::cuda_reference
{
namespace
{

/// \brief One element viewed as its real components, so offsets stay in element units.
template <class Real, std::size_t Components> struct ReductionElement
{
    Real components[Components];
};

template <class Partial> __device__ Partial initial_partial()
{
  if constexpr (std::is_same_v<Partial, MaximumPartial>)
    return MaximumPartial{-CUDART_INF};
  else
    return Partial{};
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
    norm_partials_kernel(ReductionElement<Real, Components> const* input, Plan plan, NormPartial* partials)
{
  using index_type = typename Plan::logical_index_type;
  __shared__ NormPartial shared[reduction_threads];
  NormPartial local{};
  auto index = static_cast<index_type>(blockIdx.x) * static_cast<index_type>(reduction_threads) +
               static_cast<index_type>(threadIdx.x);
  auto const stride = static_cast<index_type>(gridDim.x) * static_cast<index_type>(reduction_threads);
  for (; index < plan.element_count; index += stride)
  {
    auto const& element = input[plan.offsets(index)[0]];
    for (std::size_t component = 0; component < Components; ++component)
      accumulate_norm(local, fabs(static_cast<double>(element.components[component])));
  }
  reduce_block(shared, local);
  if (threadIdx.x == 0) partials[blockIdx.x] = shared[0];
}

template <class Real, std::size_t Components, class Plan>
__global__ void __launch_bounds__(reduction_threads)
    inner_product_partials_kernel(ReductionElement<Real, Components> const* lhs,
                                  ReductionElement<Real, Components> const* rhs, Plan plan, SumPartial* partials)
{
  using index_type = typename Plan::logical_index_type;
  __shared__ SumPartial shared[reduction_threads];
  SumPartial local{};
  auto index = static_cast<index_type>(blockIdx.x) * static_cast<index_type>(reduction_threads) +
               static_cast<index_type>(threadIdx.x);
  auto const stride = static_cast<index_type>(gridDim.x) * static_cast<index_type>(reduction_threads);
  for (; index < plan.element_count; index += stride)
  {
    auto const offsets = plan.offsets(index);
    auto const& left = lhs[offsets[0]];
    auto const& right = rhs[offsets[1]];
    double const left_real = static_cast<double>(left.components[0]);
    double const right_real = static_cast<double>(right.components[0]);
    if constexpr (Components == 1)
    {
      local.real += left_real * right_real;
    }
    else
    {
      double const left_imag = static_cast<double>(left.components[1]);
      double const right_imag = static_cast<double>(right.components[1]);
      local.real += left_real * right_real + left_imag * right_imag;
      local.imag += left_real * right_imag - left_imag * right_real;
    }
  }
  reduce_block(shared, local);
  if (threadIdx.x == 0) partials[blockIdx.x] = shared[0];
}

template <class Partial, class Real, std::size_t Components, class Plan>
__global__ void __launch_bounds__(reduction_threads)
    unary_partials_kernel(ReductionElement<Real, Components> const* input, Plan plan, Partial* partials)
{
  using index_type = typename Plan::logical_index_type;
  __shared__ Partial shared[reduction_threads];
  auto local = initial_partial<Partial>();
  auto index = static_cast<index_type>(blockIdx.x) * static_cast<index_type>(reduction_threads) +
               static_cast<index_type>(threadIdx.x);
  auto const stride = static_cast<index_type>(gridDim.x) * static_cast<index_type>(reduction_threads);
  for (; index < plan.element_count; index += stride)
  {
    auto const& element = input[plan.offsets(index)[0]];
    if constexpr (std::is_same_v<Partial, SumPartial>)
    {
      local.real += static_cast<double>(element.components[0]);
      if constexpr (Components == 2) local.imag += static_cast<double>(element.components[1]);
    }
    else
    {
      local = combine(local, MaximumPartial{static_cast<double>(element.components[0])});
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
  using element_type = ReductionElement<Real, Components>;
  static_assert(sizeof(element_type) == Components * sizeof(Real));
  unary_partials_kernel<<<reduction_block_count(plan.element_count), reduction_threads, 0, stream>>>(
      reinterpret_cast<element_type const*>(input), plan, partials);
  uni20::cuda::check(cudaGetLastError(), operation_name, device);
}

} // namespace

template <class Real, std::size_t Components, class Plan>
void enqueue_norm_partials(Real const* input, Plan const& plan, NormPartial* partials, cudaStream_t stream, int device)
{
  using element_type = ReductionElement<Real, Components>;
  static_assert(sizeof(element_type) == Components * sizeof(Real));
  norm_partials_kernel<Real, Components><<<reduction_block_count(plan.element_count), reduction_threads, 0, stream>>>(
      reinterpret_cast<element_type const*>(input), plan, partials);
  uni20::cuda::check(cudaGetLastError(), "launch CUDA reference norm", device);
}

void enqueue_norm_combine(NormPartial const* partials, std::size_t count, NormPartial* result, cudaStream_t stream,
                          int device)
{
  launch_combine(partials, count, result, stream, device, "launch CUDA reference norm combine");
}

template <class Real, std::size_t Components, class Plan>
void enqueue_inner_product_partials(Real const* lhs, Real const* rhs, Plan const& plan, SumPartial* partials,
                                    cudaStream_t stream, int device)
{
  using element_type = ReductionElement<Real, Components>;
  static_assert(sizeof(element_type) == Components * sizeof(Real));
  inner_product_partials_kernel<Real, Components>
      <<<reduction_block_count(plan.element_count), reduction_threads, 0, stream>>>(
          reinterpret_cast<element_type const*>(lhs), reinterpret_cast<element_type const*>(rhs), plan, partials);
  uni20::cuda::check(cudaGetLastError(), "launch CUDA reference inner product", device);
}

template <class Real, std::size_t Components, class Plan>
void enqueue_sum_partials(Real const* input, Plan const& plan, SumPartial* partials, cudaStream_t stream, int device)
{
  launch_unary<SumPartial, Real, Components>(input, plan, partials, stream, device, "launch CUDA reference sum");
}

template <class Real, class Plan>
void enqueue_maximum_partials(Real const* input, Plan const& plan, MaximumPartial* partials, cudaStream_t stream,
                              int device)
{
  launch_unary<MaximumPartial, Real, 1>(input, plan, partials, stream, device, "launch CUDA reference maximum");
}

void enqueue_sum_combine(SumPartial const* partials, std::size_t count, SumPartial* result, cudaStream_t stream,
                         int device)
{
  launch_combine(partials, count, result, stream, device, "launch CUDA reference sum combine");
}

void enqueue_maximum_combine(MaximumPartial const* partials, std::size_t count, MaximumPartial* result,
                             cudaStream_t stream, int device)
{
  launch_combine(partials, count, result, stream, device, "launch CUDA reference maximum combine");
}

#define UNI20_CUDA_REFERENCE_REDUCTIONS(Real, Components)                                                              \
  template void enqueue_sum_partials<Real, Components, ReductionPlan32<1>>(Real const*, ReductionPlan32<1> const&,     \
                                                                           SumPartial*, cudaStream_t, int);            \
  template void enqueue_sum_partials<Real, Components, ReductionPlan64<1>>(Real const*, ReductionPlan64<1> const&,     \
                                                                           SumPartial*, cudaStream_t, int);            \
  template void enqueue_norm_partials<Real, Components, ReductionPlan32<1>>(Real const*, ReductionPlan32<1> const&,    \
                                                                            NormPartial*, cudaStream_t, int);          \
  template void enqueue_norm_partials<Real, Components, ReductionPlan64<1>>(Real const*, ReductionPlan64<1> const&,    \
                                                                            NormPartial*, cudaStream_t, int);          \
  template void enqueue_inner_product_partials<Real, Components, ReductionPlan32<2>>(                                  \
      Real const*, Real const*, ReductionPlan32<2> const&, SumPartial*, cudaStream_t, int);                            \
  template void enqueue_inner_product_partials<Real, Components, ReductionPlan64<2>>(                                  \
      Real const*, Real const*, ReductionPlan64<2> const&, SumPartial*, cudaStream_t, int);

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
