#pragma once
#include <cuda_runtime_api.h>

namespace uni20::linalg::detail::cusolver_backend
{
template <class S>
void copy_upper_trapezoid(S* r, int ldr, S const* a, int lda, int rows, int cols, cudaStream_t stream, int device);
}
