#include "initialization.hpp"

#include "cuda_error.hpp"

#include <algorithm>

namespace uni20::cuda::detail
{
namespace
{

template <unsigned ComponentBytes>
__global__ void initialize_pattern(unsigned char* destination, std::size_t bytes, std::uint64_t pattern)
{
  auto const index = static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  auto const stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
  // Byte stores preserve the signaling bit and do not reinterpret complex
  // objects as a different scalar object type.
  for (std::size_t offset = index; offset < bytes; offset += stride)
    destination[offset] = static_cast<unsigned char>(pattern >> (8 * (offset % ComponentBytes)));
}

} // namespace

void enqueue_initialization_pattern(void* destination, std::size_t bytes, std::uint64_t pattern,
                                    unsigned component_bytes, cudaStream_t stream, int device)
{
  if (bytes == 0) return;
  constexpr unsigned threads = 256;
  auto const blocks = static_cast<unsigned>(std::min<std::size_t>(1 + (bytes - 1) / threads, 65535));
  if (component_bytes == 4)
    initialize_pattern<4><<<blocks, threads, 0, stream>>>(static_cast<unsigned char*>(destination), bytes, pattern);
  else
    initialize_pattern<8><<<blocks, threads, 0, stream>>>(static_cast<unsigned char*>(destination), bytes, pattern);
  check(cudaGetLastError(), "initialize CUDA storage diagnostic pattern", device);
}

} // namespace uni20::cuda::detail
