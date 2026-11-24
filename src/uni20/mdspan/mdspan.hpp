#pragma once

/**
 * \file mdspan.hpp
 * \ingroup mdspan_ext
 * \brief Native borrowed multidimensional views and checked foreign import.
 */

#include <uni20/mdspan/extents.hpp>
#include <uni20/mdspan/default_accessor.hpp>
#include <uni20/mdspan/layout_signed_stride.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <utility>

#if __cplusplus > 202002L && __has_include(<mdspan>)
#include <mdspan>
#endif

namespace uni20
{

namespace detail
{

template <class Layout> struct foreign_layout
{
    using type = void;
};

template <class Accessor> inline constexpr bool is_foreign_default_accessor = false;

#ifdef __cpp_lib_mdspan
template <> struct foreign_layout<std::layout_left>
{
    using type = layout_left;
};

template <> struct foreign_layout<std::layout_right>
{
    using type = layout_right;
};

template <> struct foreign_layout<std::layout_stride>
{
    using type = layout_stride;
};

template <class Element>
inline constexpr bool is_foreign_default_accessor<std::default_accessor<Element>> = true;
#endif

template <class Span> using imported_layout_t = typename foreign_layout<typename Span::layout_type>::type;

template <class Span>
concept ForeignDefaultMdspan = requires(Span const& source) {
  typename Span::extents_type;
  typename Span::element_type;
  typename Span::mapping_type;
  requires(!std::same_as<imported_layout_t<Span>, void>);
  requires std::same_as<typename Span::mapping_type,
                        typename Span::layout_type::template mapping<typename Span::extents_type>>;
  requires is_foreign_default_accessor<typename Span::accessor_type>;
  requires std::same_as<typename Span::accessor_type::element_type, typename Span::element_type>;
  { source.mapping() } -> std::same_as<typename Span::mapping_type const&>;
  { source.accessor() } -> std::same_as<typename Span::accessor_type const&>;
  { source.data_handle() } -> std::same_as<typename Span::element_type* const&>;
};

template <class Extents, std::size_t... Axes>
auto rebound_import_extents(std::index_sequence<Axes...>) -> extents<std::ptrdiff_t, Extents::static_extent(Axes)...>;

template <class Span>
using imported_extents_t = decltype(rebound_import_extents<typename Span::extents_type>(
    std::make_index_sequence<Span::extents_type::rank()>{}));

template <class To, class From> consteval bool compatible_static_extents()
{
  if constexpr (To::rank() != From::rank())
    return false;
  else
  {
    for (std::size_t axis = 0; axis != To::rank(); ++axis)
    {
      auto const lhs = To::static_extent(axis);
      auto const rhs = From::static_extent(axis);
      if (lhs != dynamic_extent && rhs != dynamic_extent && lhs != rhs) return false;
    }
    return true;
  }
}

// Integer comparisons in import must hold before either operand is narrowed.
template <class To, class From> [[nodiscard]] UNI20_HOST_DEVICE constexpr bool nonnegative_fits(From value) noexcept
{
  if constexpr (std::is_signed_v<From>)
    if (value < 0) return false;
  if constexpr (std::numeric_limits<From>::digits > std::numeric_limits<To>::digits)
    return value <= static_cast<From>(std::numeric_limits<To>::max());
  return true;
}

template <class ToMapping, ForeignDefaultMdspan Span> constexpr ToMapping import_mapping(Span const& source)
{
  using shape = typename ToMapping::extents_type;
  using index_type = typename ToMapping::index_type;
  auto const& mapping = source.mapping();
  std::array<index_type, shape::rank()> values{};
  index_type nonzero_product = 1;
  bool empty = false;
  for (std::size_t axis = 0; axis != shape::rank(); ++axis)
  {
    auto const extent = mapping.extents().extent(axis);
    if (!nonnegative_fits<index_type>(extent)) throw std::overflow_error("mdspan import extent exceeds index range");
    if (shape::static_extent(axis) != dynamic_extent && static_cast<std::size_t>(extent) != shape::static_extent(axis))
      throw std::invalid_argument("mdspan import static extent does not match");
    auto const value = static_cast<index_type>(extent);
    values[axis] = value;
    if (value == 0)
      empty = true;
    else
    {
      if (nonzero_product > std::numeric_limits<index_type>::max() / value)
        throw std::overflow_error("mdspan import nonzero extent product exceeds index range");
      nonzero_product *= value;
    }
  }
  shape const extents(values);
  if constexpr (std::same_as<typename ToMapping::layout_type, layout_stride>)
  {
    std::array<index_type, shape::rank()> strides{};
    if constexpr (shape::rank() > 0)
      for (std::size_t axis = 0; axis != shape::rank(); ++axis)
      {
        auto const stride = mapping.stride(axis);
        if (!nonnegative_fits<index_type>(stride))
          throw std::overflow_error("mdspan import stride exceeds index range");
        strides[axis] = static_cast<index_type>(stride);
      }
    // The source mapping already establishes uniqueness and the layout's stride ordering.
    if (!empty)
    {
      auto const span_size = mapping.required_span_size();
      if (!nonnegative_fits<index_type>(span_size) || !nonnegative_fits<std::ptrdiff_t>(span_size - 1))
        throw std::overflow_error("mdspan import required span exceeds index or offset range");
    }
    return ToMapping{extents, strides};
  }
  else
  {
    if (!empty && !nonnegative_fits<std::ptrdiff_t>(nonzero_product - 1))
      throw std::overflow_error("mdspan import required span exceeds offset range");
    return ToMapping{extents};
  }
}

} // namespace detail

/// \brief Non-owning multidimensional view with an accessor-defined offset type.
/// \pre The mapping's reachable offsets are representable in the accessor's offset type.
template <class Element, class Extents, class Layout = layout_right, class Accessor = default_accessor<Element>>
  requires std::same_as<Element, typename Accessor::element_type> &&
           detail::LayoutOffsetCompatible<Layout, Extents, Accessor> &&
           requires { typename Accessor::offset_type; }
class mdspan {
  public:
    using element_type = Element;
    using value_type = std::remove_cv_t<element_type>;
    using extents_type = Extents;
    using index_type = typename extents_type::index_type;
    using size_type = typename extents_type::size_type;
    using rank_type = typename extents_type::rank_type;
    using layout_type = Layout;
    using mapping_type = typename layout_type::template mapping<extents_type>;
    using accessor_type = Accessor;
    using offset_type = typename accessor_type::offset_type;
    using reference = typename accessor_type::reference;
    using data_handle_type = typename accessor_type::data_handle_type;

    UNI20_HOST_DEVICE constexpr mdspan()
      requires(extents_type::rank_dynamic() > 0 && std::default_initializable<data_handle_type> &&
               std::default_initializable<mapping_type> && std::default_initializable<accessor_type>)
        : handle_{}, mapping_{}, accessor_{}
    {}

    /// \brief Borrow the storage using the supplied mapping and accessor.
    UNI20_HOST_DEVICE constexpr mdspan(data_handle_type handle, mapping_type const& mapping,
                                       accessor_type const& accessor)
        : handle_(std::move(handle)), mapping_(mapping), accessor_(accessor)
    {}

    UNI20_HOST_DEVICE constexpr mdspan(data_handle_type handle, mapping_type const& mapping)
      requires std::default_initializable<accessor_type>
        : mdspan(std::move(handle), mapping, accessor_type{})
    {}

    UNI20_HOST_DEVICE constexpr mdspan(data_handle_type handle, extents_type const& extents)
      requires(std::constructible_from<mapping_type, extents_type const&> && std::default_initializable<accessor_type>)
        : mdspan(std::move(handle), mapping_type{extents})
    {}

    template <class... Indices>
      requires((sizeof...(Indices) == extents_type::rank() || sizeof...(Indices) == extents_type::rank_dynamic()) &&
               (std::convertible_to<Indices, index_type> && ...) && std::constructible_from<extents_type, Indices...> &&
               std::constructible_from<mapping_type, extents_type const&> && std::default_initializable<accessor_type>)
    UNI20_HOST_DEVICE constexpr explicit mdspan(data_handle_type handle, Indices... extents)
        : mdspan(std::move(handle), extents_type{extents...})
    {}

    template <class Index, std::size_t Count>
      requires(std::constructible_from<extents_type, std::span<Index, Count>> &&
               std::constructible_from<mapping_type, extents_type const&> && std::default_initializable<accessor_type>)
    UNI20_HOST_DEVICE constexpr explicit(Count != extents_type::rank_dynamic())
        mdspan(data_handle_type handle, std::span<Index, Count> extents)
        : mdspan(std::move(handle), extents_type(extents))
    {}

    template <class Index, std::size_t Count>
      requires(std::constructible_from<extents_type, std::span<Index const, Count>> &&
               std::constructible_from<mapping_type, extents_type const&> && std::default_initializable<accessor_type>)
    UNI20_HOST_DEVICE constexpr explicit(Count != extents_type::rank_dynamic())
        mdspan(data_handle_type handle, std::array<Index, Count> const& extents)
        : mdspan(std::move(handle), std::span{extents})
    {}

    template <class OtherElement, class OtherExtents, class OtherLayout, class OtherAccessor>
      requires std::constructible_from<mapping_type, typename mdspan<OtherElement, OtherExtents, OtherLayout,
                                                                     OtherAccessor>::mapping_type> &&
               std::constructible_from<accessor_type, OtherAccessor const&> &&
               std::constructible_from<data_handle_type, typename OtherAccessor::data_handle_type const&>
    UNI20_HOST_DEVICE constexpr explicit(
        !std::is_convertible_v<typename mdspan<OtherElement, OtherExtents, OtherLayout, OtherAccessor>::mapping_type,
                               mapping_type> ||
        !std::is_convertible_v<OtherAccessor const&, accessor_type>)
        mdspan(mdspan<OtherElement, OtherExtents, OtherLayout, OtherAccessor> const& source)
        : mdspan(data_handle_type(source.data_handle()), mapping_type(source.mapping()),
                 accessor_type(source.accessor()))
    {}

    /// \brief Borrow a standard mdspan after checking the destination representation.
    /// \throws std::overflow_error If an extent, stride or span size would be narrowed out of range.
    /// \throws std::invalid_argument If a runtime extent disagrees with a static destination extent.
    template <detail::ForeignDefaultMdspan Span>
      requires std::same_as<layout_type, detail::imported_layout_t<Span>> &&
               std::same_as<accessor_type, default_accessor<Element>> &&
               std::is_convertible_v<typename Span::element_type (*)[], Element (*)[]> &&
               (detail::compatible_static_extents<extents_type, typename Span::extents_type>())
    constexpr explicit mdspan(Span const& source)
        : mdspan(source.data_handle(), detail::import_mapping<mapping_type>(source))
    {}

    [[nodiscard]] UNI20_HOST_DEVICE static constexpr rank_type rank() noexcept { return extents_type::rank(); }
    [[nodiscard]] UNI20_HOST_DEVICE static constexpr rank_type rank_dynamic() noexcept
    {
      return extents_type::rank_dynamic();
    }
    [[nodiscard]] UNI20_HOST_DEVICE static constexpr std::size_t static_extent(rank_type axis) noexcept
    {
      return extents_type::static_extent(axis);
    }

    [[nodiscard]] UNI20_HOST_DEVICE constexpr data_handle_type const& data_handle() const noexcept { return handle_; }
    [[nodiscard]] UNI20_HOST_DEVICE constexpr mapping_type const& mapping() const noexcept { return mapping_; }
    [[nodiscard]] UNI20_HOST_DEVICE constexpr accessor_type const& accessor() const noexcept { return accessor_; }
    [[nodiscard]] UNI20_HOST_DEVICE constexpr extents_type const& extents() const noexcept
    {
      return mapping_.extents();
    }
    [[nodiscard]] UNI20_HOST_DEVICE constexpr index_type extent(rank_type axis) const noexcept
    {
      return this->extents().extent(axis);
    }

    /// \brief Return the number of logical elements.
    [[nodiscard]] UNI20_HOST_DEVICE constexpr size_type size() const noexcept
    {
      return detail::mdspan_size<size_type>(this->extents());
    }

    /// \brief Return whether any extent is zero.
    [[nodiscard]] UNI20_HOST_DEVICE constexpr bool empty() const noexcept
    {
      return detail::mdspan_empty(this->extents());
    }

    [[nodiscard]] UNI20_HOST_DEVICE static constexpr bool is_always_unique() noexcept
    {
      return mapping_type::is_always_unique();
    }

    [[nodiscard]] UNI20_HOST_DEVICE static constexpr bool is_always_strided() noexcept
    {
      return mapping_type::is_always_strided();
    }

    [[nodiscard]] UNI20_HOST_DEVICE static constexpr bool is_always_exhaustive() noexcept
    {
      return mapping_type::is_always_exhaustive();
    }

    /// \brief Report whether the mapping's storage span always starts at the handle.
    [[nodiscard]] UNI20_HOST_DEVICE static constexpr bool is_base_offset_always_zero() noexcept
      requires detail::StorageSpanMapping<mapping_type>
    {
      return mapping_type::is_base_offset_always_zero();
    }

    [[nodiscard]] UNI20_HOST_DEVICE constexpr bool is_unique() const noexcept { return mapping_.is_unique(); }
    [[nodiscard]] UNI20_HOST_DEVICE constexpr bool is_strided() const noexcept { return mapping_.is_strided(); }
    [[nodiscard]] UNI20_HOST_DEVICE constexpr bool is_exhaustive() const noexcept { return mapping_.is_exhaustive(); }

    /// \brief Return the stride when the mapping provides it.
    /// \pre `axis < rank()`.
    [[nodiscard]] UNI20_HOST_DEVICE constexpr index_type stride(rank_type axis) const noexcept
      requires requires(mapping_type const& mapping) { mapping.stride(axis); }
    {
      return mapping_.stride(axis);
    }

    template <class... Indices>
      requires(sizeof...(Indices) == extents_type::rank() && (std::convertible_to<Indices, index_type> && ...))
    [[nodiscard]] UNI20_HOST_DEVICE constexpr reference operator[](Indices... indices) const
    {
      return accessor_.access(handle_, static_cast<offset_type>(mapping_(static_cast<index_type>(indices)...)));
    }

    template <class Index>
      requires std::convertible_to<Index const&, index_type>
    [[nodiscard]] UNI20_HOST_DEVICE constexpr reference
    operator[](std::array<Index, extents_type::rank()> const& indices) const
    {
      return this->access_indices(indices, std::make_index_sequence<extents_type::rank()>{});
    }

    template <class Index>
      requires std::convertible_to<Index const&, index_type>
    [[nodiscard]] UNI20_HOST_DEVICE constexpr reference operator[](std::span<Index, extents_type::rank()> indices) const
    {
      return this->access_indices(indices, std::make_index_sequence<extents_type::rank()>{});
    }

  private:
    template <class Indices, std::size_t... Axes>
    [[nodiscard]] UNI20_HOST_DEVICE constexpr reference access_indices(Indices const& indices,
                                                                       std::index_sequence<Axes...>) const
    {
      return accessor_.access(handle_, static_cast<offset_type>(mapping_(static_cast<index_type>(indices[Axes])...)));
    }

    [[no_unique_address]] data_handle_type handle_;
    [[no_unique_address]] mapping_type mapping_;
    [[no_unique_address]] accessor_type accessor_;
};

template <detail::ForeignDefaultMdspan Span>
mdspan(Span const&)
    -> mdspan<typename Span::element_type, detail::imported_extents_t<Span>, detail::imported_layout_t<Span>>;

template <class Element, class Extents>
  requires requires { Extents::rank(); } && (!requires { typename Extents::layout_type; })
mdspan(Element*, Extents const&) -> mdspan<Element, Extents>;

template <class Element, std::integral... Indices>
  requires(sizeof...(Indices) > 0)
mdspan(Element*, Indices...) -> mdspan<Element, dextents<std::ptrdiff_t, sizeof...(Indices)>>;

template <class Array>
  requires(std::rank_v<Array> == 1 && std::extent_v<Array> > 0)
mdspan(Array&) -> mdspan<std::remove_extent_t<Array>, extents<std::ptrdiff_t, std::extent_v<Array>>>;

template <class Pointer>
  requires std::is_pointer_v<std::remove_reference_t<Pointer>>
mdspan(Pointer&&) -> mdspan<std::remove_pointer_t<std::remove_reference_t<Pointer>>, extents<std::ptrdiff_t>>;

template <class Element, class Index, std::size_t Count>
mdspan(Element*, std::array<Index, Count> const&) -> mdspan<Element, dextents<std::ptrdiff_t, Count>>;

template <class Element, class Index, std::size_t Count>
  requires(Count != std::dynamic_extent)
mdspan(Element*, std::span<Index, Count>) -> mdspan<Element, dextents<std::ptrdiff_t, Count>>;

template <class Element, class Mapping>
  requires requires { typename Mapping::layout_type; }
mdspan(Element*, Mapping const&) -> mdspan<Element, typename Mapping::extents_type, typename Mapping::layout_type>;

template <class Mapping, class Accessor>
mdspan(typename Accessor::data_handle_type, Mapping const&, Accessor const&)
    -> mdspan<typename Accessor::element_type, typename Mapping::extents_type, typename Mapping::layout_type, Accessor>;

} // namespace uni20
