#pragma once

#include <uni20/core/compiler_attributes.hpp>
#include <uni20/mdspan/offset_range.hpp>

#include <array>
#include <cassert>
#include <concepts>
#include <span>
#include <type_traits>

namespace uni20
{

namespace detail
{

template <class Extents> [[nodiscard]] UNI20_HOST_DEVICE constexpr bool mdspan_empty(Extents const& extents) noexcept
{
  for (std::size_t axis = 0; axis != Extents::rank(); ++axis)
    if (extents.extent(axis) == 0) return true;
  return false;
}

template <class Size, class Extents>
[[nodiscard]] UNI20_HOST_DEVICE constexpr Size mdspan_size(Extents const& extents) noexcept
{
  if (mdspan_empty(extents)) return Size{0};
  Size size = 1;
  for (std::size_t axis = 0; axis != Extents::rank(); ++axis)
    size *= static_cast<Size>(extents.extent(axis));
  return size;
}

template <class Extents> consteval bool has_static_zero_extent()
{
  for (std::size_t axis = 0; axis != Extents::rank(); ++axis)
    if (Extents::static_extent(axis) == 0) return true;
  return false;
}

template <class Index>
[[nodiscard]] UNI20_HOST_DEVICE constexpr offset_range nonnegative_offset_range(Index span_size) noexcept
{
  if (span_size == 0) return {.minimum = 0, .maximum = -1};
  return {.minimum = 0, .maximum = static_cast<std::ptrdiff_t>(span_size - 1)};
}

template <class Layout, class Extents, bool Left> class contiguous_mapping {
  public:
    using extents_type = Extents;
    using index_type = typename extents_type::index_type;
    using size_type = typename extents_type::size_type;
    using rank_type = typename extents_type::rank_type;
    using layout_type = Layout;

    UNI20_HOST_DEVICE constexpr contiguous_mapping() noexcept {}

    /// \brief Construct a contiguous mapping for the given shape.
    /// \pre The product of the nonzero extents fits in `index_type`.
    UNI20_HOST_DEVICE constexpr explicit contiguous_mapping(extents_type const& extents) noexcept : extents_(extents) {}

    template <class OtherLayout, class OtherExtents, bool OtherLeft>
      requires(Left == OtherLeft || extents_type::rank() <= 1) &&
              std::constructible_from<extents_type, OtherExtents const&>
    UNI20_HOST_DEVICE constexpr explicit(!std::is_convertible_v<OtherExtents const&, extents_type>)
        contiguous_mapping(contiguous_mapping<OtherLayout, OtherExtents, OtherLeft> const& source) noexcept
        : contiguous_mapping(extents_type(source.extents()))
    {}

    [[nodiscard]] UNI20_HOST_DEVICE constexpr extents_type const& extents() const noexcept { return extents_; }

    [[nodiscard]] UNI20_HOST_DEVICE constexpr index_type required_span_size() const noexcept
    {
      return mdspan_size<index_type>(extents_);
    }

    /// \brief Bound the mapping's element offsets.
    /// \pre The highest reachable offset fits in `std::ptrdiff_t`.
    [[nodiscard]] UNI20_HOST_DEVICE constexpr offset_range reachable_range() const noexcept
    {
      return nonnegative_offset_range(this->required_span_size());
    }

    /// \brief Return the element stride of an axis.
    /// \pre `axis < extents_type::rank()`.
    [[nodiscard]] UNI20_HOST_DEVICE constexpr index_type stride(rank_type axis) const noexcept
    {
      assert(axis < extents_type::rank());
      index_type stride = 1;
      auto const begin = Left ? rank_type{0} : axis + 1;
      auto const end = Left ? axis : extents_type::rank();
      for (auto i = begin; i < end; ++i)
        stride *= extents_.extent(i);
      return stride;
    }

    template <class... Indices>
      requires(sizeof...(Indices) == extents_type::rank() && (std::convertible_to<Indices, index_type> && ...))
    [[nodiscard]] UNI20_HOST_DEVICE constexpr index_type operator()(Indices... indices) const noexcept
    {
      std::array<index_type, extents_type::rank()> const coordinate{static_cast<index_type>(indices)...};
      index_type offset = 0;
      for (rank_type i = 0; i != extents_type::rank(); ++i)
      {
        auto const axis = Left ? extents_type::rank() - i - 1 : i;
        offset = static_cast<index_type>(offset * extents_.extent(axis) + coordinate[axis]);
      }
      return offset;
    }

    [[nodiscard]] UNI20_HOST_DEVICE static constexpr bool is_always_unique() noexcept { return true; }
    [[nodiscard]] UNI20_HOST_DEVICE static constexpr bool is_always_strided() noexcept { return true; }
    [[nodiscard]] UNI20_HOST_DEVICE static constexpr bool is_always_exhaustive() noexcept { return true; }
    [[nodiscard]] UNI20_HOST_DEVICE constexpr bool is_unique() const noexcept { return true; }
    [[nodiscard]] UNI20_HOST_DEVICE constexpr bool is_strided() const noexcept { return true; }
    [[nodiscard]] UNI20_HOST_DEVICE constexpr bool is_exhaustive() const noexcept { return true; }

    friend UNI20_HOST_DEVICE constexpr bool operator==(contiguous_mapping const&, contiguous_mapping const&) = default;

  private:
    [[no_unique_address]] extents_type extents_{};
};

} // namespace detail

/// \brief Column-major contiguous layout with the first axis varying fastest.
struct layout_left
{
    template <class Extents> using mapping = detail::contiguous_mapping<layout_left, Extents, true>;
};

/// \brief Row-major contiguous layout with the last axis varying fastest.
struct layout_right
{
    template <class Extents> using mapping = detail::contiguous_mapping<layout_right, Extents, false>;
};

/// \brief Unique affine layout with nonnegative element strides.
struct layout_stride
{
    template <class Extents> class mapping {
      public:
        using extents_type = Extents;
        using index_type = typename extents_type::index_type;
        using size_type = typename extents_type::size_type;
        using rank_type = typename extents_type::rank_type;
        using layout_type = layout_stride;

        UNI20_HOST_DEVICE constexpr mapping() noexcept
        {
          layout_right::mapping<extents_type> const contiguous{extents_};
          for (rank_type axis = 0; axis != extents_type::rank(); ++axis)
            strides_[axis] = contiguous.stride(axis);
        }

        /// \brief Construct a mapping with the given shape and element strides.
        /// \pre Extents, their nonzero product, strides and span size fit in `index_type`.
        /// \pre Strides are nonnegative. For a nonempty shape, each varying axis's stride in increasing
        ///      stride order exceeds the sum of the preceding axes' maximum offset contributions.
        UNI20_HOST_DEVICE constexpr mapping(extents_type const& extents,
                                            std::array<index_type, extents_type::rank()> const& strides) noexcept
            : extents_(extents), strides_(strides)
        {}

        template <class OtherIndex>
          requires std::convertible_to<OtherIndex, index_type>
        UNI20_HOST_DEVICE constexpr mapping(extents_type const& extents,
                                            std::span<OtherIndex, extents_type::rank()> strides) noexcept
            : extents_(extents)
        {
          for (rank_type axis = 0; axis != extents_type::rank(); ++axis)
            strides_[axis] = static_cast<index_type>(strides[axis]);
        }

        template <class OtherIndex>
          requires std::convertible_to<OtherIndex, index_type>
        UNI20_HOST_DEVICE constexpr mapping(extents_type const& extents,
                                            std::array<OtherIndex, extents_type::rank()> const& strides) noexcept
            : mapping(extents, std::span{strides})
        {}

        template <class Layout, class OtherExtents, bool Left>
          requires std::constructible_from<extents_type, OtherExtents const&>
        UNI20_HOST_DEVICE constexpr explicit(!std::is_convertible_v<OtherExtents const&, extents_type>)
            mapping(detail::contiguous_mapping<Layout, OtherExtents, Left> const& source) noexcept
            : extents_(source.extents())
        {
          detail::contiguous_mapping<Layout, extents_type, Left> const contiguous{extents_};
          for (rank_type axis = 0; axis != extents_type::rank(); ++axis)
            strides_[axis] = contiguous.stride(axis);
        }

        template <class OtherExtents>
          requires std::constructible_from<extents_type, OtherExtents const&>
        UNI20_HOST_DEVICE constexpr explicit(!std::is_convertible_v<OtherExtents const&, extents_type>)
            mapping(layout_stride::mapping<OtherExtents> const& source) noexcept
            : mapping(extents_type(source.extents()), source.strides())
        {}

        [[nodiscard]] UNI20_HOST_DEVICE constexpr extents_type const& extents() const noexcept { return extents_; }

        /// \brief Return the element stride of an axis.
        /// \pre `axis < extents_type::rank()`.
        [[nodiscard]] UNI20_HOST_DEVICE constexpr index_type stride(rank_type axis) const noexcept
        {
          assert(axis < extents_type::rank());
          return strides_[axis];
        }

        /// \brief Return the strides in logical axis order.
        [[nodiscard]] UNI20_HOST_DEVICE constexpr auto strides() const noexcept { return strides_; }

        [[nodiscard]] UNI20_HOST_DEVICE constexpr index_type required_span_size() const noexcept
        {
          if (detail::mdspan_empty(extents_)) return 0;
          index_type span_size = 1;
          for (rank_type axis = 0; axis != extents_type::rank(); ++axis)
            span_size += static_cast<index_type>((extents_.extent(axis) - 1) * strides_[axis]);
          return span_size;
        }

        /// \brief Bound the mapping's element offsets.
        /// \pre The highest reachable offset fits in `std::ptrdiff_t`.
        [[nodiscard]] UNI20_HOST_DEVICE constexpr offset_range reachable_range() const noexcept
        {
          return detail::nonnegative_offset_range(this->required_span_size());
        }

        template <class... Indices>
          requires(sizeof...(Indices) == extents_type::rank() && (std::convertible_to<Indices, index_type> && ...))
        [[nodiscard]] UNI20_HOST_DEVICE constexpr index_type operator()(Indices... indices) const noexcept
        {
          std::array<index_type, extents_type::rank()> const coordinate{static_cast<index_type>(indices)...};
          index_type offset = 0;
          for (rank_type axis = 0; axis != extents_type::rank(); ++axis)
            offset += static_cast<index_type>(coordinate[axis] * strides_[axis]);
          return offset;
        }

        [[nodiscard]] UNI20_HOST_DEVICE static constexpr bool is_always_unique() noexcept { return true; }
        [[nodiscard]] UNI20_HOST_DEVICE static constexpr bool is_always_strided() noexcept { return true; }
        [[nodiscard]] UNI20_HOST_DEVICE static constexpr bool is_always_exhaustive() noexcept
        {
          return extents_type::rank() == 0 || detail::has_static_zero_extent<extents_type>();
        }

        [[nodiscard]] UNI20_HOST_DEVICE constexpr bool is_unique() const noexcept { return true; }
        [[nodiscard]] UNI20_HOST_DEVICE constexpr bool is_strided() const noexcept { return true; }
        [[nodiscard]] UNI20_HOST_DEVICE constexpr bool is_exhaustive() const noexcept
        {
          return this->required_span_size() == detail::mdspan_size<index_type>(extents_);
        }

        friend UNI20_HOST_DEVICE constexpr bool operator==(mapping const&, mapping const&) = default;

      private:
        [[no_unique_address]] extents_type extents_{};
        std::array<index_type, extents_type::rank()> strides_{};
    };
};

} // namespace uni20
