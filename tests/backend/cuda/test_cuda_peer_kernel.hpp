#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>

/// \brief Launch a kernel on the current device that copies `count` doubles from `input` to `output`.
/// \details `input` may be a peer device's allocation; reading it requires peer access.
cudaError_t uni20_test_launch_copy_doubles(double* output, double const* input, std::size_t count, cudaStream_t stream);
