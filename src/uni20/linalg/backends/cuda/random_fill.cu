#include <uni20/linalg/backends/cuda/elementwise_kernel.cuh>
#include <uni20/linalg/backends/cuda/random_fill.hpp>
#include <uni20/storage/cuda_accessor.hpp>

namespace uni20::linalg::detail::cuda_reference
{
template <CudaRandomTransform Transform> struct RandomAtOffset
{
    random::Identity leaf;
    Transform transform;
    std::uint64_t origin;

    __device__ auto operator()(std::uint64_t offset) const
    {
      return random::value_at(leaf, origin + offset, transform);
    }
};

template <CudaRandomTransform Transform>
void enqueue_random_fill(random::transformed_value_t<Transform>* storage, random::Identity leaf, Transform transform,
                         std::uint64_t origin, LoweredElementwiseFillPlan const& plan, cudaStream_t stream, int device)
{
  using accessor = uni20::cuda::CudaPointerAccessor<random::transformed_value_t<Transform>>;
  plan.visit([&](auto const& slice) {
    launch_elementwise_kernel<false, true>(slice, RandomAtOffset{leaf, transform, origin},
                                           ElementwiseOperand{storage, accessor{}}, stream, device,
                                           "launch CUDA random fill");
  });
}

#define UNI20_RANDOM_FILL(Transform)                                                                                   \
  template void enqueue_random_fill(random::transformed_value_t<Transform>*, random::Identity, Transform,              \
                                    std::uint64_t, LoweredElementwiseFillPlan const&, cudaStream_t, int)
UNI20_RANDOM_FILL(random::Uniform<float>);
UNI20_RANDOM_FILL(random::Uniform<double>);
UNI20_RANDOM_FILL(random::Normal);
UNI20_RANDOM_FILL(random::ComplexNormal);
#undef UNI20_RANDOM_FILL
} // namespace uni20::linalg::detail::cuda_reference
