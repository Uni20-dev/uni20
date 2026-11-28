#include "qr_upper.hpp"
#include <cstddef>
#include <uni20/backend/cuda/cuda_error.hpp>

namespace uni20::linalg::detail::cusolver_backend
{
template <class S> __global__ void extract_upper(S* r, int ldr, S const* a, int lda, int rows)
{
  std::size_t const col = blockIdx.x;
  for (std::size_t i = threadIdx.x; i < static_cast<std::size_t>(rows); i += blockDim.x)
    r[col * ldr + i] = i <= col ? a[col * lda + i] : S{};
}

template <class S>
void copy_upper_trapezoid(S* r, int ldr, S const* a, int lda, int rows, int cols, cudaStream_t stream, int device)
{
  extract_upper<<<cols, 128, 0, stream>>>(r, ldr, a, lda, rows);
  cuda::check(cudaGetLastError(), "extract upper trapezoid of QR", device);
}
template void copy_upper_trapezoid(float*, int, float const*, int, int, int, cudaStream_t, int);
template void copy_upper_trapezoid(double*, int, double const*, int, int, int, cudaStream_t, int);
} // namespace uni20::linalg::detail::cusolver_backend
