#include "test_cuda_peer_kernel.hpp"

namespace
{
__global__ void copy_doubles(double* output, double const* input, std::size_t count)
{
  for (std::size_t index = blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x; index < count;
       index += static_cast<std::size_t>(gridDim.x) * blockDim.x)
    output[index] = input[index];
}
} // namespace

cudaError_t uni20_test_launch_copy_doubles(double* output, double const* input, std::size_t count, cudaStream_t stream)
{
  copy_doubles<<<4, 128, 0, stream>>>(output, input, count);
  return cudaGetLastError();
}
