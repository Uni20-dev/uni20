#pragma once

/**
 * \file initialization.hpp
 * \ingroup backend_cuda
 * \brief Stream-ordered diagnostic initialization of CUDA scalar storage.
 */

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace uni20::cuda::detail
{

/// \brief Repeat an exact four- or eight-byte scalar representation on a stream.
/// \details The pattern is passed by value to a device kernel. No floating-point
///          conversion, host staging buffer, or synchronization is performed.
/// \pre `component_bytes` is four or eight, and `bytes` is a multiple of it.
///      The caller holds write access to the destination on `stream` and has
///      selected its CUDA device.
void enqueue_initialization_pattern(void* destination, std::size_t bytes, std::uint64_t pattern,
                                    unsigned component_bytes, cudaStream_t stream, int device);

} // namespace uni20::cuda::detail
