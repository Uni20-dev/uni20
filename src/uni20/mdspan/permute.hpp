#pragma once

/**
 * \file permute.hpp
 * \ingroup mdspan_ext
 * \brief Zero-copy axis permutations of strided multidimensional metadata.
 */

#include <uni20/mdspan/mdspec.hpp>

#include <array>
#include <cstddef>
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
    using extents_type = uni20::extents<index_type, source_extents_type::static_extent(Permutation[I])...>;
    using mapping_type = layout_stride::mapping<extents_type>;
    using element_type = typename Span::element_type;
    using accessor_type = typename Span::accessor_type;
    extents_type extents{source.extent(Permutation[I])...};
    if (!source.mapping().is_unique()) throw std::invalid_argument("permute requires a unique strided mapping");
    std::array<index_type, sizeof...(I)> const strides{source.stride(Permutation[I])...};
    mapping_type mapping{extents, strides};
    if constexpr (MdspanLike<Span>)
    {
      return uni20::mdspan<element_type, extents_type, layout_stride, accessor_type>{source.data_handle(), mapping,
                                                                                     source.accessor()};
    }
    else
    {
      return mdspec<element_type, extents_type, layout_stride, accessor_type, typename Span::data_descriptor_type>{
          std::move(source.data_descriptor()), mapping, source.accessor()};
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
///          permutations use layout_stride and preserve the source strides,
///          including those of empty and singleton axes.
/// \pre The source extents and strides satisfy the requirements of layout_stride,
///      with zero offset at the origin.
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
