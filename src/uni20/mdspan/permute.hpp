#pragma once

/**
 * \file permute.hpp
 * \ingroup mdspan_ext
 * \brief Zero-copy axis permutations of strided multidimensional metadata.
 */

#include <uni20/mdspan/mdspec.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <utility>

namespace uni20
{
namespace detail
{
// Rank-zero canonical mappings have no stride(axis) member to query.
template <class Span>
concept PermutableMdspec =
    MdspecLike<Span> && Span::is_always_strided() && (Span::rank() == 0 || StridedMdspecLike<Span>);

template <std::size_t Size>
consteval auto is_valid_axis_permutation(std::array<std::size_t, Size> const& permutation) -> bool
{
  std::array<bool, Size> seen{};
  for (std::size_t const axis : permutation)
  {
    if (axis >= Size || seen[axis]) return false;
    seen[axis] = true;
  }
  return true;
}

template <std::size_t Size> consteval auto make_identity_axis_permutation()
{
  std::array<std::size_t, Size> permutation{};
  for (std::size_t axis = 0; axis < Size; ++axis)
    permutation[axis] = axis;
  return permutation;
}

/// \brief Positive strides for an empty mapping that satisfy layout_stride's ordering precondition.
/// \details layout_stride needs an order of the axes in which each stride is at
///          least the previous stride times the previous extent. A zero extent
///          ends that product, so the order consists of groups of nonzero axes
///          separated by zero-extent axes: each zero-extent axis takes the
///          product of the group before it, and the next stride restarts at one.
///          The largest extent goes last in the final group, where it multiplies
///          nothing, and every other nonzero axis joins the group whose product
///          is currently smallest.
/// \pre At least one extent is zero.
/// \throws std::overflow_error If a stride in that order is not representable.
template <class Extents>
constexpr auto
empty_mapping_strides(Extents const& extents) -> std::array<typename Extents::index_type, Extents::rank()>
{
  using index_type = typename Extents::index_type;
  constexpr std::size_t rank = Extents::rank();
  constexpr auto limit = static_cast<unsigned long long>(std::numeric_limits<index_type>::max());

  std::array<std::size_t, rank> zero_axes{};
  std::array<std::size_t, rank> nonzero_axes{};
  std::size_t zero_count = 0;
  std::size_t nonzero_count = 0;
  for (std::size_t axis = 0; axis < rank; ++axis)
  {
    if (extents.extent(axis) == 0)
      zero_axes[zero_count++] = axis;
    else
      nonzero_axes[nonzero_count++] = axis;
  }
  std::sort(nonzero_axes.begin(), nonzero_axes.begin() + nonzero_count, [&](std::size_t lhs, std::size_t rhs) {
    if (extents.extent(lhs) != extents.extent(rhs)) return extents.extent(lhs) > extents.extent(rhs);
    return lhs < rhs;
  });

  // Group g < zero_count precedes zero_axes[g]; group zero_count is final, and
  // its largest extent, nonzero_axes[0], is placed last.
  std::array<std::size_t, rank> group{};
  std::array<unsigned long long, rank + 1> product{};
  product.fill(1);
  for (std::size_t position = 1; position < nonzero_count; ++position)
  {
    std::size_t chosen = 0;
    for (std::size_t candidate = 1; candidate <= zero_count; ++candidate)
      if (product[candidate] < product[chosen]) chosen = candidate;
    auto const axis = nonzero_axes[position];
    group[axis] = chosen;
    auto const extent = static_cast<unsigned long long>(extents.extent(axis));
    product[chosen] = product[chosen] > limit / extent ? limit + 1 : product[chosen] * extent;
  }

  std::array<std::size_t, rank> order{};
  std::size_t length = 0;
  for (std::size_t g = 0; g <= zero_count; ++g)
  {
    for (std::size_t position = 1; position < nonzero_count; ++position)
      if (group[nonzero_axes[position]] == g) order[length++] = nonzero_axes[position];
    if (g < zero_count) order[length++] = zero_axes[g];
  }
  if (nonzero_count > 0) order[length++] = nonzero_axes[0];

  std::array<index_type, rank> strides{};
  index_type stride = 1;
  for (std::size_t position = 0; position < rank; ++position)
  {
    auto const axis = order[position];
    strides[axis] = stride;
    if (position + 1 == rank) break;
    auto const extent = extents.extent(axis);
    if (extent == 0)
      stride = 1;
    else
    {
      if (stride > std::numeric_limits<index_type>::max() / extent)
        throw std::overflow_error("empty permutation strides are not representable");
      stride *= extent;
    }
  }
  return strides;
}

template <auto Permutation, PermutableMdspec Span, std::size_t... I>
constexpr auto permute_mdspec_impl(Span source, std::index_sequence<I...>)
{
  if constexpr (Permutation == make_identity_axis_permutation<Span::rank()>())
  {
    return source;
  }
  else
  {
    using index_type = typename Span::index_type;
    using source_extents_type = typename Span::extents_type;
    using extents_type = stdex::extents<index_type, source_extents_type::static_extent(Permutation[I])...>;
    using mapping_type = stdex::layout_stride::mapping<extents_type>;
    using element_type = typename Span::element_type;
    using accessor_type = typename Span::accessor_type;
    extents_type extents{source.extent(Permutation[I])...};
    if (!source.mapping().is_unique()) throw std::invalid_argument("permute requires a unique strided mapping");
    // An empty source's own strides can be unrepresentable products of extents
    // (row-major after a zero extent), so they are read only when it is not empty.
    bool const empty = ((source.extent(I) == 0) || ...);
    std::array<index_type, sizeof...(I)> strides{};
    if (empty)
    {
      // Canonical mappings may have zero strides after an empty dimension.
      // Build positive strides satisfying layout_stride's ordering precondition;
      // replacing every zero by one would fail that precondition at rank > 2.
      strides = empty_mapping_strides(extents);
    }
    else
    {
      strides = {source.stride(Permutation[I])...};
      for (auto const stride : strides)
        if (stride <= 0) throw std::invalid_argument("permute requires positive strides");
    }
    mapping_type mapping{extents, strides};
    if constexpr (MdspanLike<Span>)
    {
      return stdex::mdspan<element_type, extents_type, stdex::layout_stride, accessor_type>{source.data_handle(),
                                                                                            mapping, source.accessor()};
    }
    else
    {
      return mdspec<element_type, extents_type, stdex::layout_stride, accessor_type,
                    typename Span::data_descriptor_type>{std::move(source.data_descriptor()), mapping,
                                                         source.accessor()};
    }
  }
}

template <auto Permutation, PermutableMdspec Span>
  requires(Permutation.size() == Span::rank() && is_valid_axis_permutation(Permutation))
constexpr auto permute_mdspec(Span source)
{
  return permute_mdspec_impl<Permutation>(std::move(source), std::make_index_sequence<Span::rank()>{});
}
} // namespace detail

/// \brief Reorder strided mdspan or mdspec axes without moving elements or acquiring access.
/// \details Output axis j is source axis Axis[j]. Static extents, accessors,
///          and handles or deferred descriptors are preserved. Identity
///          permutations also preserve the original layout type. Other
///          permutations use layout_stride and require a unique mapping with
///          positive strides, except for zero strides in empty sources.
/// \pre Nonempty source mappings are representable by layout_stride using
///      their observed extents and strides, with zero offset at the origin.
/// \tparam Axis A bijection of the source axes.
/// \param source Borrowed metadata; the underlying storage must outlive the result.
template <std::size_t... Axis, detail::PermutableMdspec Span>
  requires(sizeof...(Axis) == Span::rank() &&
           detail::is_valid_axis_permutation(std::array<std::size_t, sizeof...(Axis)>{Axis...}))
[[nodiscard]] constexpr auto permute(Span source)
{
  return detail::permute_mdspec<std::array<std::size_t, sizeof...(Axis)>{Axis...}>(std::move(source));
}
} // namespace uni20
