#pragma once

#include <uni20/mdspan/layouts.hpp>

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace uni20
{

/// \brief Unique affine layout whose strides may have either sign.
/// \details The handle addresses the lowest element of the storage span.
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

        UNI20_HOST_DEVICE constexpr mapping()
            : mapping(extents_type{}, layout_stride::mapping<extents_type>{}.strides())
        {}

        /// \brief Construct a mapping with signed element strides.
        /// \pre The product of the nonzero extents fits in `index_type`.
        /// \pre For a nonempty shape, each varying axis in increasing absolute stride order
        ///      has an absolute stride greater than the total contribution of the preceding axes.
        /// \throws std::overflow_error On the host, if the storage span does not fit in `index_type`.
        UNI20_HOST_DEVICE constexpr mapping(extents_type const& extents,
                                            std::array<index_type, extents_type::rank()> const& strides)
            : extents_(extents), strides_(strides)
        {
          if (detail::mdspan_empty(extents_)) return;
          span_size_ = 1;
          for (rank_type axis = 0; axis != extents_type::rank(); ++axis)
          {
            auto const steps = extents_.extent(axis) - 1;
            if (steps == 0) continue;
            auto const stride = strides_[axis];
            auto const limit = (std::numeric_limits<index_type>::max() - span_size_) / steps;
            if (stride > limit || stride < -limit)
            {
#if defined(__CUDA_ARCH__)
              __trap();
#else
              throw std::overflow_error("signed-stride storage span exceeds its index type");
#endif
            }
            auto const delta = steps * stride;
            if (delta < 0) origin_ -= delta;
            span_size_ += delta < 0 ? -delta : delta;
          }
        }

        [[nodiscard]] UNI20_HOST_DEVICE constexpr extents_type const& extents() const noexcept { return extents_; }

        /// \brief Return the signed stride of an axis.
        /// \pre `axis < extents_type::rank()`.
        [[nodiscard]] UNI20_HOST_DEVICE constexpr index_type stride(rank_type axis) const noexcept
        {
          assert(axis < extents_type::rank());
          return strides_[axis];
        }

        [[nodiscard]] UNI20_HOST_DEVICE constexpr auto strides() const noexcept { return strides_; }

        /// \brief Return the width of the enclosing storage span in elements, including padding.
        [[nodiscard]] UNI20_HOST_DEVICE constexpr index_type required_span_size() const noexcept { return span_size_; }

        template <class... Indices>
          requires(sizeof...(Indices) == extents_type::rank() && (std::convertible_to<Indices, index_type> && ...))
        [[nodiscard]] UNI20_HOST_DEVICE constexpr index_type operator()(Indices... indices) const noexcept
        {
          std::array<index_type, extents_type::rank()> const coordinate{static_cast<index_type>(indices)...};
          index_type offset = origin_;
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
        index_type origin_ = 0;
        index_type span_size_ = 0;
    };
};

} // namespace uni20
