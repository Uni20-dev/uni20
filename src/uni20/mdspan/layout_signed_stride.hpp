#pragma once

#include <uni20/mdspan/layouts.hpp>

#include <array>
#include <cassert>
#include <concepts>
#include <type_traits>

namespace uni20
{

/// \brief Unique affine layout whose strides may have either sign.
/// \details The handle addresses index zero on every axis.
struct layout_signed_stride
{
    template <class Extents>
      requires std::signed_integral<typename Extents::index_type>
    class mapping {
      public:
        using extents_type = Extents;
        using index_type = typename extents_type::index_type;
        using size_type = typename extents_type::size_type;
        using rank_type = typename extents_type::rank_type;
        using layout_type = layout_signed_stride;

        UNI20_HOST_DEVICE constexpr mapping() noexcept
            : mapping(extents_type{}, layout_stride::mapping<extents_type>{}.strides())
        {}

        /// \brief Construct a mapping with signed element strides.
        /// \pre The product of the nonzero extents and the reachable offsets fit in `index_type`.
        /// \pre For a nonempty shape, each varying axis in increasing absolute stride order
        ///      has an absolute stride greater than the total contribution of the preceding axes.
        UNI20_HOST_DEVICE constexpr mapping(extents_type const& extents,
                                            std::array<index_type, extents_type::rank()> const& strides) noexcept
            : extents_(extents), strides_(strides)
        {}

        [[nodiscard]] UNI20_HOST_DEVICE constexpr extents_type const& extents() const noexcept { return extents_; }

        /// \brief Return the signed stride of an axis.
        /// \pre `axis < extents_type::rank()`.
        [[nodiscard]] UNI20_HOST_DEVICE constexpr index_type stride(rank_type axis) const noexcept
        {
          assert(axis < extents_type::rank());
          return strides_[axis];
        }

        [[nodiscard]] UNI20_HOST_DEVICE constexpr auto strides() const noexcept { return strides_; }

        /// \brief Bound the element offsets relative to the logical origin.
        /// \pre The reachable offsets fit in `std::ptrdiff_t`.
        [[nodiscard]] UNI20_HOST_DEVICE constexpr offset_range reachable_range() const noexcept
        {
          if (detail::mdspan_empty(extents_)) return {.minimum = 0, .maximum = -1};
          offset_range range{};
          for (rank_type axis = 0; axis != extents_type::rank(); ++axis)
          {
            auto const delta = static_cast<std::ptrdiff_t>((extents_.extent(axis) - 1) * strides_[axis]);
            if (delta < 0)
              range.minimum += delta;
            else
              range.maximum += delta;
          }
          return range;
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
          auto const range = this->reachable_range();
          if (range.empty()) return true;
          using unsigned_offset = std::make_unsigned_t<std::ptrdiff_t>;
          // Subtract in the unsigned type so intervals crossing zero can use the full width.
          auto const width = static_cast<unsigned_offset>(range.maximum) - static_cast<unsigned_offset>(range.minimum);
          return width == static_cast<unsigned_offset>(detail::mdspan_size<size_type>(extents_)) - 1;
        }

        friend UNI20_HOST_DEVICE constexpr bool operator==(mapping const&, mapping const&) = default;

      private:
        [[no_unique_address]] extents_type extents_{};
        std::array<index_type, extents_type::rank()> strides_{};
    };
};

namespace detail
{

template <class Layout, class Extents, class Accessor>
concept LayoutOffsetCompatible =
    !std::same_as<Layout, layout_signed_stride> ||
    (std::signed_integral<typename Extents::index_type> && std::signed_integral<typename Accessor::offset_type>);

} // namespace detail

} // namespace uni20
