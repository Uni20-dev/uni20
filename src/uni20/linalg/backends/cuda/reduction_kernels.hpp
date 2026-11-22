#pragma once

/**
 * \file reduction_kernels.hpp
 * \ingroup linalg
 * \brief Deterministic CUDA reference reductions: partial states and precompiled launches.
 */

#include <uni20/core/compiler_attributes.hpp>
#include <uni20/linalg/backends/cuda/elementwise_plan.hpp>

#include <cuda_runtime_api.h>

#include <cmath>
#include <cstddef>

namespace uni20::linalg::detail::cuda_reference
{

/// \brief Maximum compact rank of one reduction launch; outer dimensions become host loops.
inline constexpr std::size_t reduction_maximum_rank = 8;

/// \brief Threads per block for every reduction launch.
inline constexpr unsigned int reduction_threads = 256;

/// \brief Upper bound on blocks per reduction launch.
/// \details The block count depends only on the element count, never on the
///          device's size or occupancy, so the same input and shape give the
///          same bits on every run on the same kind of device.
inline constexpr unsigned int reduction_maximum_blocks = 1024;

template <std::size_t OperandCount>
using ReductionPlan32 = StridedElementwisePlan32<OperandCount, reduction_maximum_rank>;

template <std::size_t OperandCount>
using ReductionPlan64 = StridedElementwisePlan64<OperandCount, reduction_maximum_rank>;

/// \brief Return the block count of one reduction launch over `element_count` elements.
template <class Count> [[nodiscard]] constexpr unsigned int reduction_block_count(Count element_count) noexcept
{
  auto const required = (static_cast<unsigned long long>(element_count) + reduction_threads - 1) / reduction_threads;
  if (required == 0) return 1;
  return required < reduction_maximum_blocks ? static_cast<unsigned int>(required) : reduction_maximum_blocks;
}

/// \brief Partial Euclidean-norm state: sums of squares in three scaled ranges.
/// \details Follows the reference BLAS `dnrm2` (Anderson, "Algorithm 978: Safe
///          scaling in the Level 1 BLAS", 2017). Magnitudes above
///          `norm_big_threshold` are scaled down by `norm_big_scale` before
///          squaring, magnitudes below `norm_small_threshold` are scaled up by
///          `norm_small_scale`, and the rest are squared directly, so no square
///          overflows or underflows. Partials combine by adding each range.
///          Every component is accumulated in `double`, so `float` inputs never
///          leave the middle range.
struct NormPartial
{
    double small;
    double medium;
    double big;
};

/// \brief Partial real or complex sum, accumulated in `double`.
struct SumPartial
{
    double real;
    double imag;
};

/// \brief Partial signed maximum of real elements.
struct MaximumPartial
{
    double value;
};

[[nodiscard]] UNI20_HOST_DEVICE constexpr MaximumPartial combine(MaximumPartial lhs, MaximumPartial rhs) noexcept
{
  if (rhs.value != rhs.value || lhs.value < rhs.value) return rhs;
  return lhs;
}

inline constexpr double norm_small_threshold = 0x1p-511;
inline constexpr double norm_big_threshold = 0x1p486;
inline constexpr double norm_small_scale = 0x1p537;
inline constexpr double norm_big_scale = 0x1p-538;

/// \brief Add one real component's magnitude to a norm partial.
UNI20_HOST_DEVICE constexpr void accumulate_norm(NormPartial& partial, double magnitude) noexcept
{
  if (magnitude > norm_big_threshold)
  {
    double const scaled = magnitude * norm_big_scale;
    partial.big += scaled * scaled;
  }
  else if (magnitude < norm_small_threshold)
  {
    double const scaled = magnitude * norm_small_scale;
    partial.small += scaled * scaled;
  }
  else
  {
    // NaN fails both comparisons and lands here, so it propagates.
    partial.medium += magnitude * magnitude;
  }
}

/// \brief Combine two norm partials.
[[nodiscard]] UNI20_HOST_DEVICE constexpr NormPartial combine(NormPartial const& lhs, NormPartial const& rhs) noexcept
{
  return NormPartial{lhs.small + rhs.small, lhs.medium + rhs.medium, lhs.big + rhs.big};
}

/// \brief Combine two additive partials.
[[nodiscard]] UNI20_HOST_DEVICE constexpr SumPartial combine(SumPartial const& lhs, SumPartial const& rhs) noexcept
{
  return SumPartial{lhs.real + rhs.real, lhs.imag + rhs.imag};
}

/// \brief Return the Euclidean norm described by a fully combined partial.
/// \details Combines the three ranges as the reference BLAS `dnrm2` does.
///          Small values are ignored when any big value is present, because
///          their squares cannot change the result.
[[nodiscard]] inline double finish_norm(NormPartial partial)
{
  double scale = 1.0;
  double sum_of_squares = partial.medium;
  bool const medium_contributes = partial.medium > 0.0 || partial.medium != partial.medium;
  if (partial.big > 0.0)
  {
    if (medium_contributes) partial.big += (partial.medium * norm_big_scale) * norm_big_scale;
    scale = 1.0 / norm_big_scale;
    sum_of_squares = partial.big;
  }
  else if (partial.small > 0.0)
  {
    if (medium_contributes)
    {
      double const medium = std::sqrt(partial.medium);
      double const small = std::sqrt(partial.small) / norm_small_scale;
      double const larger = small > medium ? small : medium;
      double const smaller = small > medium ? medium : small;
      double const ratio = smaller / larger;
      sum_of_squares = larger * larger * (1.0 + ratio * ratio);
    }
    else
    {
      scale = 1.0 / norm_small_scale;
      sum_of_squares = partial.small;
    }
  }
  return scale * std::sqrt(sum_of_squares);
}

/// \brief Write one norm partial per block for one launch over `plan`.
/// \tparam Real Component type: `float` or `double`.
/// \tparam Components One for real elements, two for complex elements.
/// \tparam Plan `ReductionPlan32<1>` or `ReductionPlan64<1>`.
/// \param input Element storage, addressed by the plan's element offsets.
/// \param partials Destination for `reduction_block_count(plan.element_count)` partials.
template <class Real, std::size_t Components, class Plan>
void enqueue_norm_partials(Real const* input, Plan const& plan, NormPartial* partials, cudaStream_t stream, int device);

/// \brief Combine `count` norm partials into `result[0]` in a fixed order.
void enqueue_norm_combine(NormPartial const* partials, std::size_t count, NormPartial* result, cudaStream_t stream,
                          int device);

/// \brief Write one additive partial per block for a full sum.
template <class Real, std::size_t Components, class Plan>
void enqueue_sum_partials(Real const* input, Plan const& plan, SumPartial* partials, cudaStream_t stream, int device);

/// \brief Write one signed-maximum partial per block.
template <class Real, class Plan>
void enqueue_maximum_partials(Real const* input, Plan const& plan, MaximumPartial* partials, cudaStream_t stream,
                              int device);

/// \brief Combine additive partials in a fixed order.
void enqueue_sum_combine(SumPartial const* partials, std::size_t count, SumPartial* result, cudaStream_t stream,
                         int device);

/// \brief Combine signed-maximum partials in a fixed order.
void enqueue_maximum_combine(MaximumPartial const* partials, std::size_t count, MaximumPartial* result,
                             cudaStream_t stream, int device);

/// \brief Write one inner-product partial per block for one launch over `plan`.
/// \details Accumulates `conj(lhs) * rhs` with the left operand at plan
///          operand 0 and the right at operand 1.
/// \tparam Plan `ReductionPlan32<2>` or `ReductionPlan64<2>`.
template <class Real, std::size_t Components, class Plan>
void enqueue_inner_product_partials(Real const* lhs, Real const* rhs, Plan const& plan, SumPartial* partials,
                                    cudaStream_t stream, int device);

} // namespace uni20::linalg::detail::cuda_reference
