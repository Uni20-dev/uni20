#pragma once

/**
 * \file iteration_plan.hpp
 * \ingroup mdspan_ext
 * \brief Helpers for constructing merged iteration plans over strided tensors.
 */

#include <uni20/common/static_vector.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/mdspan/offset_range.hpp>
#include <uni20/mdspan/strides.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20
{

/// \brief Represents a single dimension's extent and stride for iteration planning.
/// \tparam ExtentT Integral type used to store the extent.
/// \tparam StrideT Integral type used to store the stride.
/// \ingroup internal
template <typename ExtentT = std::size_t, typename StrideT = std::ptrdiff_t> struct extent_stride
{
    ExtentT extent{};
    StrideT stride{};

    constexpr extent_stride() = default;
    constexpr extent_stride(ExtentT extent_, StrideT stride_) : extent(extent_), stride(stride_) {}

    /// \brief Returns true when two adjacent dimensions can be merged.
    /// \details An extent-one dimension is always mergeable because its stride
    ///          is not observed by any logical index.
    /// \param inner Metadata describing the inner dimension.
    /// \ingroup internal
    [[nodiscard]] constexpr bool can_merge_with_inner(extent_stride inner) const noexcept
    {
      if (extent == ExtentT{1} || inner.extent == ExtentT{1}) return true;
      return stride == inner.stride * static_cast<StrideT>(inner.extent);
    }

    /// \brief Merge an inner dimension into this one.
    /// \details The merged descriptor retains the non-singleton dimension's
    ///          stride. Two singleton dimensions use the canonical unit stride.
    /// \param inner The dimension that satisfies can_merge_with_inner.
    /// \ingroup internal
    constexpr void merge_with_inner(extent_stride inner) noexcept
    {
      if (extent == ExtentT{1})
      {
        extent = inner.extent;
        stride = inner.extent == ExtentT{1} ? StrideT{1} : inner.stride;
        return;
      }
      if (inner.extent == ExtentT{1}) return;

      extent *= inner.extent;
      stride = inner.stride;
    }
};

/// \brief Compact affine iteration plan shared by host and device backends.
/// \tparam OperandCount Number of operands, with the writable output first.
/// \tparam MaximumRank Maximum number of compact dimensions.
/// \ingroup internal
template <std::size_t OperandCount, std::size_t MaximumRank> struct multi_iteration_plan
{
    static constexpr std::size_t operand_count = OperandCount;
    static constexpr std::size_t maximum_rank = MaximumRank;

    static_vector<extent_strides<OperandCount>, MaximumRank> dimensions;
    std::array<std::ptrdiff_t, OperandCount> base_offsets{};
    std::array<offset_range, OperandCount> reachable_offsets{};
    std::size_t element_count = 1;
    bool representable = true;

    /// \brief Report the compact rank after size-one elimination and coalescing.
    [[nodiscard]] std::size_t rank() const noexcept { return dimensions.size(); }

    /// \brief Report whether the logical iteration space is empty.
    [[nodiscard]] bool empty() const noexcept { return element_count == 0; }
};

namespace detail
{

template <class Extents> constexpr bool has_zero_extent(Extents const& extents)
{
  for (std::size_t axis = 0; axis != Extents::rank(); ++axis)
    if (extents.extent(axis) == 0) return true;
  return false;
}

[[nodiscard]] constexpr bool checked_iteration_multiply(std::ptrdiff_t lhs, std::ptrdiff_t rhs,
                                                        std::ptrdiff_t& result) noexcept
{
  if (lhs == 0 || rhs == 0)
  {
    result = 0;
    return true;
  }

  constexpr auto minimum = std::numeric_limits<std::ptrdiff_t>::min();
  constexpr auto maximum = std::numeric_limits<std::ptrdiff_t>::max();
  if ((lhs == -1 && rhs == minimum) || (rhs == -1 && lhs == minimum)) return false;

  if (lhs > 0)
  {
    if ((rhs > 0 && lhs > maximum / rhs) || (rhs < 0 && rhs < minimum / lhs)) return false;
  }
  else
  {
    if ((rhs > 0 && lhs < minimum / rhs) || (rhs < 0 && lhs < maximum / rhs)) return false;
  }

  result = lhs * rhs;
  return true;
}

[[nodiscard]] constexpr bool checked_iteration_add(std::ptrdiff_t lhs, std::ptrdiff_t rhs,
                                                   std::ptrdiff_t& result) noexcept
{
  constexpr auto minimum = std::numeric_limits<std::ptrdiff_t>::min();
  constexpr auto maximum = std::numeric_limits<std::ptrdiff_t>::max();
  if ((rhs > 0 && lhs > maximum - rhs) || (rhs < 0 && lhs < minimum - rhs)) return false;
  result = lhs + rhs;
  return true;
}

template <std::size_t OperandCount, std::size_t MaximumRank>
void finish_iteration_plan(multi_iteration_plan<OperandCount, MaximumRank>& plan)
{
  if (plan.dimensions.size() == 1 && plan.dimensions[0].extent == 0)
  {
    plan.element_count = 0;
    for (auto& range : plan.reachable_offsets)
      range = offset_range{.minimum = 0, .maximum = -1};
    return;
  }

  plan.element_count = 1;
  for (auto const& dimension : plan.dimensions)
  {
    auto const extent = static_cast<std::size_t>(dimension.extent);
    if (extent != 0 && plan.element_count > std::numeric_limits<std::size_t>::max() / extent)
    {
      plan.representable = false;
      return;
    }
    plan.element_count *= extent;
  }

  for (std::size_t operand = 0; operand < OperandCount; ++operand)
  {
    auto minimum = plan.base_offsets[operand];
    auto maximum = plan.base_offsets[operand];
    for (auto const& dimension : plan.dimensions)
    {
      std::ptrdiff_t delta = 0;
      if (!checked_iteration_multiply(static_cast<std::ptrdiff_t>(dimension.extent - 1), dimension.strides[operand],
                                      delta))
      {
        plan.representable = false;
        return;
      }

      auto& bound = delta < 0 ? minimum : maximum;
      if (!checked_iteration_add(bound, delta, bound))
      {
        plan.representable = false;
        return;
      }
    }
    plan.reachable_offsets[operand] = offset_range{.minimum = minimum, .maximum = maximum};
  }
}

} // namespace detail

/// \brief Build a merged iteration plan for multiple tensors sharing the same extents.
/// \details The first mapping determines iteration order and negative-stride
///          normalization. Remaining mappings may have different layout and
///          extents types, but must have the same rank and runtime extents.
/// \tparam Mappings Layout mapping types modelling the mdspan mapping interface.
/// \param mappings Tuple of mappings, with the writable output mapping first.
/// \return Compact plan with per-operand offsets, ranges, and element count.
/// \ingroup internal
template <typename... Mappings> auto make_multi_iteration_plan(std::tuple<Mappings...> const& mappings)
{
  static_assert(sizeof...(Mappings) >= 1, "At least one mapping is required");

  using first_mapping = std::tuple_element_t<0, std::tuple<Mappings...>>;
  using size_type = typename first_mapping::size_type;
  using index_type = std::ptrdiff_t;
  static constexpr std::size_t N = sizeof...(Mappings);
  static constexpr size_type Rank = first_mapping::extents_type::rank();

  static_assert(((Mappings::extents_type::rank() == Rank) && ...),
                "All iteration-plan mappings must have the same rank");

  auto const& base_mapping = std::get<0>(mappings);
  auto const& base_extents = base_mapping.extents();
  std::size_t logical_element_count = 1;

  auto check_extents = [&](auto const& mapping) {
    for (size_type axis = 0; axis < Rank; ++axis)
      PRECONDITION_EQUAL(mapping.extents().extent(axis), base_extents.extent(axis),
                         "elementwise iteration requires equal extents");
  };
  std::apply([&](auto const&... mapping) { (check_extents(mapping), ...); }, mappings);

  multi_iteration_plan<N, Rank> plan;
  if (detail::has_zero_extent(base_extents))
  {
    plan.dimensions.emplace_back(size_type{0}, std::array<index_type, N>{});
    detail::finish_iteration_plan(plan);
    return plan;
  }

  // Strides describe differences between offsets. Each operand may start at
  // a different offset, which must fit the plan before normalizing the axes.
  std::size_t operand = 0;
  auto set_origin = [&](auto const& mapping) {
    auto const origin = detail::mapping_origin(mapping);
    // Integral promotion preserves the offset and admits plain char to std::in_range.
    if (!std::in_range<index_type>(+origin)) return false;
    plan.base_offsets[operand++] = static_cast<index_type>(origin);
    return true;
  };
  if (!std::apply([&](auto const&... mapping) { return (set_origin(mapping) && ...); }, mappings))
  {
    plan.representable = false;
    return plan;
  }

  if constexpr (Rank > 0)
  {
    for (size_type i = 0; i < Rank; ++i)
    {
      size_type const extent = base_extents.extent(i);

      if (std::cmp_greater(extent, std::numeric_limits<std::ptrdiff_t>::max()) ||
          logical_element_count > std::numeric_limits<std::ptrdiff_t>::max() / static_cast<std::size_t>(extent))
      {
        plan.representable = false;
        return plan;
      }
      logical_element_count *= static_cast<std::size_t>(extent);

      // Size-1 dimensions span index 0 only (0*stride, count factor 1), so drop
      // them; they coalesce with anything by vanishing.
      if (extent == 1) continue;

      auto strides = std::apply(
          [i](auto const&... mapping) {
            return std::array<index_type, N>{static_cast<index_type>(mapping.stride(i))...};
          },
          mappings);

      if (strides[0] < 0)
      {
        for (std::size_t k = 0; k < N; ++k)
        {
          index_type delta = 0;
          if (!detail::checked_iteration_multiply(strides[k], static_cast<index_type>(extent - 1), delta) ||
              !detail::checked_iteration_add(plan.base_offsets[k], delta, plan.base_offsets[k]))
          {
            plan.representable = false;
            return plan;
          }
          if (strides[k] == std::numeric_limits<index_type>::min())
          {
            plan.representable = false;
            return plan;
          }
          strides[k] = -strides[k];
        }
      }

      plan.dimensions.emplace_back(extent, strides);
    }
  }

  // Empty plan => single element at the base offset (rank-0 scalar).
  if (!plan.dimensions.empty())
  {
    std::sort(plan.dimensions.begin(), plan.dimensions.end(),
              [](auto const& lhs, auto const& rhs) { return std::abs(lhs.strides[0]) > std::abs(rhs.strides[0]); });
    merge_strides_right(plan.dimensions);
  }

  detail::finish_iteration_plan(plan);
  return plan;
}

/// \brief Construct a merged iteration plan and offset for a single mapping.
/// \tparam Mapping Layout mapping type modelling the mdspan mapping interface.
/// \param mapping Mapping used to compute strides and extents.
/// \return Pair consisting of the merged plan and the starting offset adjustment.
/// \ingroup internal
template <typename Mapping> auto make_iteration_plan_with_offset(Mapping const& mapping)
{
  auto const plan = make_multi_iteration_plan(std::tuple{mapping});
  CHECK(plan.representable, "iteration plan exceeds host index range");

  static_vector<extent_stride<>, Mapping::extents_type::rank()> dimensions;
  for (auto const& dimension : plan.dimensions)
    dimensions.emplace_back(static_cast<std::size_t>(dimension.extent), dimension.strides[0]);
  return std::pair{dimensions, plan.base_offsets[0]};
}

} // namespace uni20
