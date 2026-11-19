#pragma once

/**
 * \file elementwise_plan.hpp
 * \ingroup linalg
 * \brief CUDA lowering for compact affine elementwise iteration plans.
 */

#include <uni20/core/compiler_attributes.hpp>
#include <uni20/mdspan/iteration_plan.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::linalg::detail::cuda_reference
{

/// \brief CUDA integer representation selected for an affine elementwise plan.
enum class ElementwiseIndexKind
{
  index_32,
  index_64
};

/// \brief Device-ready affine elementwise plan.
/// \details Dimensions are stored innermost first. Offset arithmetic is signed
///          even though the initial CUDA lowering accepts only nonnegative
///          strides and offsets.
/// \tparam OperandCount Number of operands, with the writable output first.
/// \tparam MaximumRank Maximum compact rank represented in the kernel payload.
/// \tparam LogicalIndex Unsigned type used for logical linear indices.
/// \tparam Offset Signed type used for handle-relative element offsets.
template <std::size_t OperandCount, std::size_t MaximumRank, std::unsigned_integral LogicalIndex,
          std::signed_integral Offset>
struct StridedElementwisePlan
{
    using logical_index_type = LogicalIndex;
    using offset_type = Offset;
    using offsets_type = std::array<offset_type, OperandCount>;

    static constexpr std::size_t operand_count = OperandCount;
    static constexpr std::size_t maximum_rank = MaximumRank;
    static_assert(MaximumRank <= std::numeric_limits<std::uint8_t>::max());

    logical_index_type element_count = 0;
    std::uint8_t compact_rank = 0;
    std::array<logical_index_type, MaximumRank> extents{};
    std::array<offsets_type, MaximumRank> strides{};
    offsets_type base_offsets{};

    /// \brief Decode a logical linear index into one offset per operand.
    [[nodiscard]] UNI20_HOST_DEVICE constexpr offsets_type offsets(logical_index_type logical_index) const noexcept
    {
      auto result = base_offsets;
      for (std::size_t dimension = 0; dimension < compact_rank; ++dimension)
      {
        auto const extent = extents[dimension];
        auto const coordinate = logical_index % extent;
        logical_index /= extent;
        for (std::size_t operand = 0; operand < OperandCount; ++operand)
        {
          auto const stride = strides[dimension][operand];
          if (stride != 0) result[operand] += static_cast<offset_type>(coordinate) * stride;
        }
      }
      return result;
    }
};

template <std::size_t OperandCount, std::size_t MaximumRank>
using StridedElementwisePlan32 = StridedElementwisePlan<OperandCount, MaximumRank, std::uint32_t, std::int32_t>;

template <std::size_t OperandCount, std::size_t MaximumRank>
using StridedElementwisePlan64 = StridedElementwisePlan<OperandCount, MaximumRank, std::uint64_t, std::int64_t>;

/// \brief One compact dimension iterated on the host rather than in the kernel.
template <std::size_t OperandCount> struct HostLoopDimension
{
    std::int64_t extent = 0;
    std::array<std::int64_t, OperandCount> strides{};
};

/// \brief Host-owned pair of possible CUDA plan representations.
/// \details Only the representation selected by `index_kind` is initialized.
///          A compact rank above `MaximumRank` keeps its innermost
///          `MaximumRank` dimensions in the kernel payload; the remaining
///          outer dimensions, outermost first, become one launch per outer
///          index.
template <std::size_t OperandCount, std::size_t MaximumRank> struct LoweredStridedElementwisePlan
{
    ElementwiseIndexKind index_kind = ElementwiseIndexKind::index_32;
    StridedElementwisePlan32<OperandCount, MaximumRank> plan_32;
    StridedElementwisePlan64<OperandCount, MaximumRank> plan_64;
    std::vector<HostLoopDimension<OperandCount>> host_loop_dimensions;

    /// \brief Return the logical element count independent of representation.
    [[nodiscard]] std::size_t element_count() const noexcept
    {
      std::size_t count = index_kind == ElementwiseIndexKind::index_32
                              ? std::size_t{plan_32.element_count}
                              : static_cast<std::size_t>(plan_64.element_count);
      for (auto const& dimension : host_loop_dimensions)
        count *= static_cast<std::size_t>(dimension.extent);
      return count;
    }

    /// \brief Invoke a callable with the selected 32- or 64-bit payload once per host-loop index.
    /// \details Each call receives the payload with its base offsets advanced to
    ///          one outer index; without host-loop dimensions it is called once.
    /// \param visitor Callable accepting either concrete plan type.
    template <class Visitor> void visit(Visitor&& visitor) const
    {
      if (index_kind == ElementwiseIndexKind::index_32)
        visit_slices(plan_32, visitor);
      else
        visit_slices(plan_64, visitor);
    }

  private:
    template <class Plan, class Visitor> void visit_slices(Plan const& plan, Visitor& visitor) const
    {
      if (host_loop_dimensions.empty())
      {
        visitor(plan);
        return;
      }

      using offset_type = typename Plan::offset_type;
      std::vector<std::int64_t> index(host_loop_dimensions.size(), 0);
      Plan slice = plan;
      for (;;)
      {
        for (std::size_t operand = 0; operand < OperandCount; ++operand)
        {
          std::int64_t offset = plan.base_offsets[operand];
          for (std::size_t dimension = 0; dimension < index.size(); ++dimension)
            offset += index[dimension] * host_loop_dimensions[dimension].strides[operand];
          slice.base_offsets[operand] = static_cast<offset_type>(offset);
        }
        visitor(slice);

        std::size_t dimension = index.size();
        while (dimension > 0 && ++index[dimension - 1] == host_loop_dimensions[dimension - 1].extent)
        {
          index[dimension - 1] = 0;
          --dimension;
        }
        if (dimension == 0) return;
      }
    }
};

namespace elementwise_plan_detail
{

template <class Target, class Source> [[nodiscard]] constexpr bool in_range(Source value) noexcept
{
  return std::in_range<Target>(value);
}

/// \brief Lower the innermost `source.rank() - host_loop_rank` dimensions into a kernel payload.
/// \details Offsets are checked against every reachable offset, so each
///          host-loop slice also fits the payload's offset type.
template <class LogicalIndex, class Offset, std::size_t OperandCount, std::size_t MaximumRank, std::size_t SourceRank>
[[nodiscard]] bool try_lower(multi_iteration_plan<OperandCount, SourceRank> const& source,
                             StridedElementwisePlan<OperandCount, MaximumRank, LogicalIndex, Offset>& destination,
                             std::size_t host_loop_rank)
{
  constexpr auto maximum_logical_index = std::numeric_limits<std::make_signed_t<LogicalIndex>>::max();
  std::size_t const kernel_rank = source.rank() - host_loop_rank;
  std::size_t kernel_element_count = source.element_count;
  for (std::size_t dimension = 0; dimension < host_loop_rank; ++dimension)
    kernel_element_count /= static_cast<std::size_t>(source.dimensions[dimension].extent);
  if (!source.representable || kernel_rank > MaximumRank ||
      std::cmp_greater(kernel_element_count, maximum_logical_index))
    return false;

  destination.element_count = static_cast<LogicalIndex>(kernel_element_count);
  if (source.empty())
  {
    destination.compact_rank = 0;
    return true;
  }
  destination.compact_rank = static_cast<std::uint8_t>(kernel_rank);

  for (std::size_t operand = 0; operand < OperandCount; ++operand)
  {
    auto const range = source.reachable_offsets[operand];
    if ((!range.empty() && (range.minimum < 0 || !in_range<Offset>(range.maximum))) ||
        source.base_offsets[operand] < 0 || !in_range<Offset>(source.base_offsets[operand]))
      return false;
    destination.base_offsets[operand] = static_cast<Offset>(source.base_offsets[operand]);
  }

  for (std::size_t device_dimension = 0; device_dimension < kernel_rank; ++device_dimension)
  {
    auto const& source_dimension = source.dimensions[source.rank() - device_dimension - 1];
    if (source_dimension.extent <= 0 || !in_range<LogicalIndex>(source_dimension.extent)) return false;
    destination.extents[device_dimension] = static_cast<LogicalIndex>(source_dimension.extent);

    for (std::size_t operand = 0; operand < OperandCount; ++operand)
    {
      auto const stride = source_dimension.strides[operand];
      if (stride < 0 || !in_range<Offset>(stride)) return false;
      destination.strides[device_dimension][operand] = static_cast<Offset>(stride);
    }
  }

  return true;
}

} // namespace elementwise_plan_detail

/// \brief Lower a backend-neutral affine plan to a CUDA 32- or 64-bit payload.
/// \details Positive and zero strides are supported. Negative reachable offsets
///          or strides decline until signed CUDA descriptor rebasing is defined.
///          Dimensions beyond `MaximumRank` become host-loop dimensions.
/// \tparam MaximumRank Maximum compact rank of one kernel launch.
/// \param source Backend-neutral compact iteration plan.
/// \param destination Selected CUDA payload.
/// \return True when the plan can be represented by the CUDA executor.
template <std::size_t MaximumRank, std::size_t OperandCount, std::size_t SourceRank>
[[nodiscard]] bool
try_lower_strided_elementwise_plan(multi_iteration_plan<OperandCount, SourceRank> const& source,
                                   LoweredStridedElementwisePlan<OperandCount, MaximumRank>& destination)
{
  std::size_t const host_loop_rank = source.empty() || source.rank() <= MaximumRank ? 0 : source.rank() - MaximumRank;
  destination.host_loop_dimensions.clear();
  for (std::size_t dimension = 0; dimension < host_loop_rank; ++dimension)
  {
    auto const& source_dimension = source.dimensions[dimension];
    HostLoopDimension<OperandCount> loop{.extent = static_cast<std::int64_t>(source_dimension.extent)};
    for (std::size_t operand = 0; operand < OperandCount; ++operand)
    {
      if (source_dimension.strides[operand] < 0) return false;
      loop.strides[operand] = static_cast<std::int64_t>(source_dimension.strides[operand]);
    }
    destination.host_loop_dimensions.push_back(loop);
  }

  StridedElementwisePlan32<OperandCount, MaximumRank> plan_32;
  if (elementwise_plan_detail::try_lower(source, plan_32, host_loop_rank))
  {
    destination.index_kind = ElementwiseIndexKind::index_32;
    destination.plan_32 = plan_32;
    return true;
  }

  StridedElementwisePlan64<OperandCount, MaximumRank> plan_64;
  if (!elementwise_plan_detail::try_lower(source, plan_64, host_loop_rank)) return false;
  destination.index_kind = ElementwiseIndexKind::index_64;
  destination.plan_64 = plan_64;
  return true;
}

} // namespace uni20::linalg::detail::cuda_reference
