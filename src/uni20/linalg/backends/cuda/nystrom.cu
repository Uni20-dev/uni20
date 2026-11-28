#include <uni20/backend/cuda/cuda_error.hpp>
#include <uni20/config.hpp>
#include <uni20/linalg/detail/nystrom_scaling.hpp>

namespace uni20::linalg::detail::randomized
{
template <class R> __global__ void whiten_columns(R* vectors, R const* values, int width, R tolerance)
{
  int column = blockIdx.x;
  R scale = whitening_scale(values[column], values[width - 1], tolerance);
  auto* start = vectors + static_cast<std::size_t>(column) * width;
  for (std::size_t row = threadIdx.x; row < static_cast<std::size_t>(width); row += blockDim.x)
    start[row] *= scale;
}
template <class R>
void enqueue_whitening(R* columns, R const* values, int width, R tolerance, cudaStream_t stream, int device)
{
  whiten_columns<<<width, 128, 0, stream>>>(columns, values, width, tolerance);
  cuda::check(cudaGetLastError(), "launch Nystrom whitening", device);
}
template void enqueue_whitening(float*, float const*, int, float, cudaStream_t, int);
template void enqueue_whitening(double*, double const*, int, double, cudaStream_t, int);
} // namespace uni20::linalg::detail::randomized
