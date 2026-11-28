#pragma once

#include <uni20/linalg/backends/cuda/elementwise_arithmetic.hpp>
#include <uni20/random/value.hpp>

namespace uni20::linalg::detail::cuda_reference
{
template <class T>
concept CudaRandomTransform = std::same_as<T, random::Uniform<float>> || std::same_as<T, random::Uniform<double>> ||
                              std::same_as<T, random::detail::RoundedNormalSketch> || std::same_as<T, random::Normal> ||
                              std::same_as<T, random::ComplexNormal>;

/// \brief Write the mapped elements of a CUDA storage span using allocation-relative random addresses.
template <CudaRandomTransform Transform>
void enqueue_random_fill(random::transformed_value_t<Transform>* storage, random::Identity leaf, Transform transform,
                         std::uint64_t origin, LoweredElementwiseFillPlan const& plan, cudaStream_t stream, int device);
} // namespace uni20::linalg::detail::cuda_reference
