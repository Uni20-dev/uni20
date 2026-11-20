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
    std::array<index_type, sizeof...(I)> strides{source.stride(Permutation[I])...};
    if (!source.mapping().is_unique()) throw std::invalid_argument("permute requires a unique strided mapping");
    bool const empty = ((source.extent(I) == 0) || ...);
    if (empty)
    {
      // Canonical mappings may have zero strides after an empty dimension.
      // Build positive strides satisfying layout_stride's ordering precondition;
      // replacing every zero by one would fail that precondition at rank > 2.
      // The ordering puts zero-extent axes first and the others by increasing
      // extent. Each stride after a zero extent restarts at one, so the largest
      // stride is the product of every nonzero extent except the largest.
      std::array<std::size_t, sizeof...(I)> order{I...};
      std::sort(order.begin(), order.end(), [&](std::size_t lhs, std::size_t rhs) {
        auto const lhs_extent = extents.extent(lhs);
        auto const rhs_extent = extents.extent(rhs);
        if ((lhs_extent == 0) != (rhs_extent == 0)) return lhs_extent == 0;
        if (lhs_extent != rhs_extent) return lhs_extent < rhs_extent;
        return lhs < rhs;
      });
      index_type stride = 1;
      for (std::size_t position = 0; position < order.size(); ++position)
      {
        auto const axis = order[position];
        strides[axis] = stride;
        if (position + 1 == order.size()) break;
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
    }
    else
      for (auto const stride : strides)
        if (stride <= 0) throw std::invalid_argument("permute requires positive strides");
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
