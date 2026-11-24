#pragma once

#include <uni20/mdspan/mdspan.hpp>

#include <array>
#include <concepts>
#include <cstddef>

namespace uni20
{

/// \brief Borrow a pointer-backed view with one axis reversed.
/// \details Empty views retain their handle and strides. An axis with one
///          element retains its stride because the only index is zero.
/// \tparam Axis Axis to reverse.
/// \param source Native view with a default accessor and signed indices.
template <std::size_t Axis, class Element, class Extents, class Layout>
  requires(Axis < Extents::rank() && std::signed_integral<typename Extents::index_type> &&
           (std::same_as<Layout, layout_left> || std::same_as<Layout, layout_right> ||
            std::same_as<Layout, layout_stride>))
[[nodiscard]] UNI20_HOST_DEVICE constexpr auto reverse(mdspan<Element, Extents, Layout> const& source) noexcept
{
  using index_type = typename Extents::index_type;
  std::array<index_type, Extents::rank()> strides{};
  for (std::size_t i = 0; i != Extents::rank(); ++i)
    strides[i] = source.stride(i);

  auto handle = source.data_handle();
  if (!source.empty() && source.extent(Axis) > 1)
  {
    // The final element is an addressable offset. Its displacement and the
    // negated stride have to fit the destination signed range.
    auto const last = (source.extent(Axis) - 1) * strides[Axis];
    handle = source.accessor().offset(handle, last);
    strides[Axis] = -strides[Axis];
  }
  using mapping_type = layout_signed_stride::mapping<Extents>;
  return mdspan{handle, mapping_type{source.extents(), strides}, source.accessor()};
}

} // namespace uni20
