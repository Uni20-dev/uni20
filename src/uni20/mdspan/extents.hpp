#pragma once

/**
 * \file extents.hpp
 * \ingroup mdspan_ext
 * \brief Configured extents vocabulary shared by host and device views.
 */

#include <uni20/config.hpp>
#include <uni20/core/compiler_attributes.hpp>

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <type_traits>
#include <utility>

#if UNI20_USE_STD_EXTENTS
#include <mdspan>
#endif

namespace uni20
{

using std::dynamic_extent;

#if UNI20_USE_STD_EXTENTS
using std::dextents;
using std::extents;
#else

namespace detail
{
template <class Index>
concept ExtentsIndex = std::integral<Index> && std::same_as<Index, std::remove_cv_t<Index>> &&
                       !std::same_as<Index, bool> &&
                       (std::same_as<Index, std::make_signed_t<Index>> ||
                        std::same_as<Index, std::make_unsigned_t<Index>>);
} // namespace detail

/// \brief Shape with static extents and stored values for dynamic axes.
/// \pre Runtime extents are nonnegative, fit in `Index`, and agree with any
///      corresponding static extents.
template <detail::ExtentsIndex Index, std::size_t... StaticExtents> class extents {
    static_assert(((StaticExtents == dynamic_extent || StaticExtents <= std::numeric_limits<Index>::max()) && ...));
    static constexpr std::size_t dynamic_rank = (std::size_t{0} + ... + (StaticExtents == dynamic_extent));

  public:
    using index_type = Index;
    using size_type = std::make_unsigned_t<index_type>;
    using rank_type = std::size_t;

    [[nodiscard]] UNI20_HOST_DEVICE static constexpr rank_type rank() noexcept { return sizeof...(StaticExtents); }

    [[nodiscard]] UNI20_HOST_DEVICE static constexpr rank_type rank_dynamic() noexcept
    {
      return dynamic_rank;
    }

    [[nodiscard]] UNI20_HOST_DEVICE static constexpr std::size_t static_extent(rank_type axis) noexcept
    {
      assert(axis < rank());
      return std::array<std::size_t, sizeof...(StaticExtents)>{StaticExtents...}[axis];
    }

    [[nodiscard]] UNI20_HOST_DEVICE constexpr index_type extent(rank_type axis) const noexcept
    {
      assert(axis < rank());
      if constexpr (dynamic_rank != 0 && dynamic_rank == rank())
      {
        return dynamic_extents_[axis];
      }
      else
      {
        constexpr std::array<std::size_t, sizeof...(StaticExtents)> declared{StaticExtents...};
        if constexpr (rank_dynamic() > 0)
        {
          constexpr auto positions = [] {
            std::array<rank_type, sizeof...(StaticExtents)> result{};
            rank_type axis = 0;
            rank_type next_dynamic = 0;
            ((result[axis++] = next_dynamic, next_dynamic += StaticExtents == dynamic_extent), ...);
            return result;
          }();
          if (declared[axis] == dynamic_extent) return dynamic_extents_[positions[axis]];
        }
        return static_cast<index_type>(declared[axis]);
      }
    }

    UNI20_HOST_DEVICE constexpr extents() noexcept = default;

    template <class... Indices>
      requires(sizeof...(Indices) > 0 && (sizeof...(Indices) == rank() || sizeof...(Indices) == rank_dynamic()) &&
               (std::is_convertible_v<Indices, index_type> && ...) &&
               (std::is_nothrow_constructible_v<index_type, Indices> && ...))
    UNI20_HOST_DEVICE constexpr explicit extents(Indices... values) noexcept
        : extents(std::array<index_type, sizeof...(Indices)>{static_cast<index_type>(values)...})
    {}

    template <class OtherIndex, std::size_t Count>
      requires((Count == rank() || Count == rank_dynamic()) && std::is_convertible_v<OtherIndex const&, index_type> &&
               std::is_nothrow_constructible_v<index_type, OtherIndex const&>)
    UNI20_HOST_DEVICE constexpr explicit(Count != rank_dynamic())
        extents(std::array<OtherIndex, Count> const& values) noexcept
    {
      this->assign<Count>([&values](rank_type axis) { return static_cast<index_type>(values[axis]); });
    }

    template <class OtherIndex, std::size_t Count>
      requires((Count == rank() || Count == rank_dynamic()) && std::is_convertible_v<OtherIndex const&, index_type> &&
               std::is_nothrow_constructible_v<index_type, OtherIndex const&>)
    UNI20_HOST_DEVICE constexpr explicit(Count != rank_dynamic()) extents(std::span<OtherIndex, Count> values) noexcept
    {
      this->assign<Count>([&values](rank_type axis) { return static_cast<index_type>(values[axis]); });
    }

  private:
    template <class Other> static consteval bool compatible()
    {
      if constexpr (Other::rank() != rank()) return false;
      for (rank_type axis = 0; axis != rank(); ++axis)
      {
        auto const from = Other::static_extent(axis);
        auto const to = static_extent(axis);
        if (from != dynamic_extent && to != dynamic_extent && from != to) return false;
      }
      return true;
    }

    template <class Other> static consteval bool explicit_conversion()
    {
      if constexpr (std::numeric_limits<index_type>::digits < std::numeric_limits<typename Other::index_type>::digits)
        return true;
      for (rank_type axis = 0; axis != rank(); ++axis)
        if (static_extent(axis) != dynamic_extent && Other::static_extent(axis) == dynamic_extent) return true;
      return false;
    }

  public:
    template <class OtherIndex, std::size_t... OtherExtents>
      requires(compatible<extents<OtherIndex, OtherExtents...>>())
    UNI20_HOST_DEVICE constexpr explicit(explicit_conversion<extents<OtherIndex, OtherExtents...>>())
        extents(extents<OtherIndex, OtherExtents...> const& other) noexcept
    {
      this->assign<rank()>([&other](rank_type axis) { return static_cast<index_type>(other.extent(axis)); });
    }

    template <class OtherIndex, std::size_t... OtherExtents>
    [[nodiscard]] UNI20_HOST_DEVICE constexpr bool
    operator==(extents<OtherIndex, OtherExtents...> const& other) const noexcept
    {
      if constexpr (sizeof...(OtherExtents) != rank()) return false;
      for (rank_type axis = 0; axis != rank(); ++axis)
        if (static_cast<std::size_t>(this->extent(axis)) != static_cast<std::size_t>(other.extent(axis))) return false;
      return true;
    }

  private:
    template <std::size_t Count, class ValueAt> UNI20_HOST_DEVICE constexpr void assign(ValueAt value_at) noexcept
    {
      if constexpr (Count == rank_dynamic())
      {
        if constexpr (Count != 0)
          for (rank_type axis = 0; axis != Count; ++axis)
            dynamic_extents_[axis] = value_at(axis);
      }
      else
      {
        [[maybe_unused]] rank_type dynamic_axis = 0;
        for (rank_type axis = 0; axis != Count; ++axis)
        {
          [[maybe_unused]] auto const declared = static_extent(axis);
          if constexpr (dynamic_rank != 0)
            if (declared == dynamic_extent)
            {
              dynamic_extents_[dynamic_axis++] = value_at(axis);
              continue;
            }
          assert(declared == static_cast<std::size_t>(value_at(axis)));
        }
      }
    }

    struct EmptyStorage {};
    [[no_unique_address]] std::conditional_t<dynamic_rank == 0, EmptyStorage, std::array<index_type, dynamic_rank>>
        dynamic_extents_{};
};

template <class... Indices>
extents(Indices...) -> extents<std::size_t, ((void)sizeof(Indices), dynamic_extent)...>;

namespace detail
{
template <class Index, std::size_t... Axis>
auto dynamic_extents_type(std::index_sequence<Axis...>) -> extents<Index, ((void)Axis, dynamic_extent)...>;
} // namespace detail

template <class Index, std::size_t Rank>
using dextents = decltype(detail::dynamic_extents_type<Index>(std::make_index_sequence<Rank>{}));

#endif

} // namespace uni20
