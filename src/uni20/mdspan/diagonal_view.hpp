#pragma once

/**
 * \file diagonal_view.hpp
 * \ingroup mdspan_ext
 * \brief Strided metadata for paired-axis diagonals.
 */

#include <uni20/core/compiler_attributes.hpp>
#include <uni20/mdspan/mdspec.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <stdexcept>
#include <type_traits>

namespace uni20
{
namespace detail
{
// A diagonal of a custom unique strided mapping can fall outside the native layout's stride-ordering rule.
struct DiagonalLayout
{
    template <class Extents> class mapping {
      public:
        using extents_type = Extents;
        using index_type = typename extents_type::index_type;
        using size_type = typename extents_type::size_type;
        using rank_type = typename extents_type::rank_type;
        using layout_type = DiagonalLayout;

        UNI20_HOST_DEVICE constexpr mapping() noexcept = default;

        UNI20_HOST_DEVICE constexpr mapping(extents_type extents,
                                            std::array<index_type, extents_type::rank()> strides) noexcept
            : extents_(extents), strides_(strides)
        {}

        [[nodiscard]] UNI20_HOST_DEVICE constexpr auto extents() const noexcept -> extents_type const&
        {
          return extents_;
        }

        [[nodiscard]] UNI20_HOST_DEVICE constexpr auto stride(rank_type axis) const noexcept -> index_type
        {
          return strides_[axis];
        }

        [[nodiscard]] UNI20_HOST_DEVICE constexpr auto required_span_size() const noexcept -> index_type
        {
          for (rank_type axis = 0; axis < extents_type::rank(); ++axis)
            if (extents_.extent(axis) == 0) return 0;

          index_type last_offset = 0;
          for (rank_type axis = 0; axis < extents_type::rank(); ++axis)
            last_offset += (extents_.extent(axis) - 1) * strides_[axis];
          return last_offset + 1;
        }

        [[nodiscard]] UNI20_HOST_DEVICE constexpr offset_range reachable_range() const noexcept
        {
          return nonnegative_offset_range(this->required_span_size());
        }

        template <class... Indices>
          requires(sizeof...(Indices) == extents_type::rank() && (std::is_convertible_v<Indices, index_type> && ...))
        [[nodiscard]] UNI20_HOST_DEVICE constexpr auto operator()(Indices... indices) const noexcept -> index_type
        {
          std::array<index_type, extents_type::rank()> coordinate{static_cast<index_type>(indices)...};
          index_type offset = 0;
          for (rank_type axis = 0; axis < extents_type::rank(); ++axis)
            offset += coordinate[axis] * strides_[axis];
          return offset;
        }

        [[nodiscard]] UNI20_HOST_DEVICE static constexpr bool is_always_unique() noexcept { return true; }
        [[nodiscard]] UNI20_HOST_DEVICE static constexpr bool is_always_exhaustive() noexcept { return false; }
        [[nodiscard]] UNI20_HOST_DEVICE static constexpr bool is_always_strided() noexcept { return true; }
        [[nodiscard]] UNI20_HOST_DEVICE static constexpr bool is_unique() noexcept { return true; }
        [[nodiscard]] UNI20_HOST_DEVICE static constexpr bool is_strided() noexcept { return true; }

        [[nodiscard]] UNI20_HOST_DEVICE constexpr bool is_exhaustive() const noexcept
        {
          auto const span_size = this->required_span_size();
          if (span_size == 0) return true;
          index_type count = 1;
          for (rank_type axis = 0; axis < extents_type::rank(); ++axis)
            count *= extents_.extent(axis);
          return span_size == count;
        }

        friend UNI20_HOST_DEVICE constexpr bool operator==(mapping const&, mapping const&) = default;

      private:
        extents_type extents_{};
        std::array<index_type, extents_type::rank()> strides_{};
    };
};

template <std::size_t Rank, std::size_t... Axis> consteval bool valid_diagonal_pairs()
{
  if (sizeof...(Axis) == 0 || sizeof...(Axis) % 2 != 0) return false;
  std::array<bool, Rank> seen{};
  for (auto axis : std::array<std::size_t, sizeof...(Axis)>{Axis...})
  {
    if (axis >= Rank || seen[axis]) return false;
    seen[axis] = true;
  }
  return true;
}

template <std::size_t... Axis, StridedMdspecLike Span> auto diagonal_mdspec(Span const& source)
{
  using index_type = typename Span::index_type;
  constexpr std::size_t result_rank = Span::rank() - sizeof...(Axis) / 2;
  using extents_type = stdex::dextents<index_type, result_rank>;
  using element_type = typename Span::element_type;
  using accessor_type = typename Span::accessor_type;

  if (!source.mapping().is_unique()) throw std::invalid_argument("diagonal_view requires a unique mapping");

  bool empty = false;
  for (std::size_t axis = 0; axis < Span::rank(); ++axis)
    empty |= source.extent(axis) == 0;
  auto stride = [&](std::size_t axis) -> index_type {
    if (empty || source.extent(axis) == 1) return 1;
    auto const result = source.stride(axis);
    if (result <= 0) throw std::invalid_argument("diagonal_view requires positive strides");
    return result;
  };

  std::array<bool, Span::rank()> paired{};
  ((paired[Axis] = true), ...);
  std::array<index_type, result_rank> extents{};
  std::array<index_type, result_rank> strides{};
  std::size_t out = 0;
  for (std::size_t axis = 0; axis < Span::rank(); ++axis)
  {
    if (paired[axis]) continue;
    extents[out] = source.extent(axis);
    strides[out++] = stride(axis);
  }

  constexpr std::array pairs{Axis...};
  for (std::size_t i = 0; i < pairs.size(); i += 2)
  {
    auto const first = pairs[i];
    auto const second = pairs[i + 1];
    auto const extent = std::min(source.extent(first), source.extent(second));
    extents[out] = extent;
    // When both indices can be 1, their combined step is an addressable source offset.
    strides[out++] = !empty && extent > 1 ? stride(first) + stride(second) : index_type{1};
  }

  using source_layout = typename Span::layout_type;
  using layout_type =
      std::conditional_t<std::same_as<source_layout, layout_left> || std::same_as<source_layout, layout_right> ||
                             std::same_as<source_layout, layout_stride>,
                         layout_stride, DiagonalLayout>;
  typename layout_type::template mapping<extents_type> const mapping{extents_type(extents), strides};
  if constexpr (MdspanLike<Span>)
  {
    return uni20::mdspan<element_type, extents_type, layout_type, accessor_type>{source.data_handle(), mapping,
                                                                                 source.accessor()};
  }
  else
  {
    using descriptor_type = typename Span::data_descriptor_type;
    return mdspec<element_type, extents_type, layout_type, accessor_type, descriptor_type>{source.data_descriptor(),
                                                                                           mapping, source.accessor()};
  }
}
} // namespace detail
} // namespace uni20
