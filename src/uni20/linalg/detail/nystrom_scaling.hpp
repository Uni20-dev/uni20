#pragma once

#include <cmath>
#include <uni20/config.hpp>
#include <uni20/core/compiler_attributes.hpp>
#if UNI20_BACKEND_CUDA
#include <cuda_runtime_api.h>
#endif

namespace uni20::linalg::detail::randomized
{
// Relative test avoids underflow of the cutoff for a uniformly tiny PSD matrix.
template <class R> UNI20_HOST_DEVICE R whitening_scale(R eigenvalue, R largest, R tolerance)
{
  return eigenvalue > R{0} && eigenvalue / largest > tolerance ? R{1} / std::sqrt(eigenvalue) : R{0};
}
#if UNI20_BACKEND_CUDA
template <class R>
void enqueue_whitening(R* columns, R const* values, int width, R tolerance, cudaStream_t stream, int device);
#endif
} // namespace uni20::linalg::detail::randomized
