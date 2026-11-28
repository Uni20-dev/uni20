/**
 * \file diagonal_accessor.hpp
 * \ingroup mdspan_ext
 * \brief Accessor support for compressed generalized diagonal tensors.
 */

#pragma once

#include <uni20/core/compiler_attributes.hpp>
#include <uni20/core/types.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/mdspan/generated_layout.hpp>

#include <array>
#include <cassert>
#include <concepts>
#include <cstddef>
#include <limits>
#include <stdexcept>
#include <type_traits>
#include <utility>

namespace uni20
{

/// \brief Handle for one compressed generalized diagonal value sequence.
/// \tparam ElementType Stored element type, possibly const-qualified.
template <class ElementType> struct diagonal_data_handle
{
    ElementType* data = nullptr;
    std::size_t logical_offset = 0;

    UNI20_HOST_DEVICE constexpr diagonal_data_handle() noexcept = default;

    UNI20_HOST_DEVICE constexpr diagonal_data_handle(ElementType* data, std::size_t logical_offset) noexcept
        : data(data), logical_offset(logical_offset)
    {}

    template <class OtherElementType>
      requires std::convertible_to<OtherElementType*, ElementType*>
    UNI20_HOST_DEVICE constexpr diagonal_data_handle(diagonal_data_handle<OtherElementType> const& other) noexcept
        : data(other.data), logical_offset(other.logical_offset)
    {}

    friend UNI20_HOST_DEVICE constexpr bool operator==(diagonal_data_handle const&,
                                                       diagonal_data_handle const&) = default;
};

/// \brief Writable proxy for one structural diagonal or off-diagonal element.
/// \details Reading an off-diagonal proxy returns zero. Assigning an
///          off-diagonal proxy has the precondition that the assigned value is
///          zero, preserving the compressed diagonal representation.
/// \tparam ElementType Stored non-const element type.
template <class ElementType> class diagonal_reference {
  public:
    static_assert(!std::is_const_v<ElementType>);

    UNI20_HOST_DEVICE constexpr diagonal_reference() noexcept = default;
    UNI20_HOST_DEVICE constexpr explicit diagonal_reference(ElementType* element) noexcept : element_(element) {}

    [[nodiscard]] UNI20_HOST_DEVICE constexpr operator ElementType() const noexcept
    {
      return element_ == nullptr ? ElementType{} : *element_;
    }

    UNI20_HOST_DEVICE constexpr auto operator=(ElementType const& value) noexcept -> diagonal_reference&
    {
      if (element_ == nullptr)
      {
        assert(value == ElementType{});
        return *this;
      }
      *element_ = value;
      return *this;
    }

    UNI20_HOST_DEVICE constexpr auto operator=(diagonal_reference const& other) noexcept -> diagonal_reference&
    {
      return *this = static_cast<ElementType>(other);
    }

  private:
    ElementType* element_ = nullptr;
};

/// \brief Mdspan accessor over a compressed rank-N generalized diagonal.
/// \details The associated mapping encodes logical tensor indices as offsets.
///          This accessor decodes those offsets and addresses the compressed
///          value only when every logical index is equal. Rectangular tensors
///          store `min(extents...)` values.
/// \tparam ElementType Presented element type, possibly const-qualified.
/// \tparam Extents Logical tensor extents used to decode mapping offsets.
/// \tparam ComponentIndex Index type of the physical component mapping.
/// \tparam ComponentLayout Layout of the recovered rank-one component view.
template <class ElementType, class Extents, class ComponentIndex = std::ptrdiff_t,
          class ComponentLayout = layout_stride>
class diagonal_accessor {
  public:
    using value_type = std::remove_cv_t<ElementType>;
    using element_type = ElementType;
    using reference = std::conditional_t<std::is_const_v<element_type>, value_type, diagonal_reference<value_type>>;
    using data_handle_type = diagonal_data_handle<element_type>;
    using offset_policy = diagonal_accessor;
    using offset_type = std::size_t;
    using extents_type = Extents;
    using index_type = typename extents_type::index_type;

    /// \brief Decoded component coordinate for one logical mapping offset.
    struct component_index_result
    {
        std::size_t index = 0;
        bool present = true;
    };

    UNI20_HOST_DEVICE constexpr diagonal_accessor()
      requires std::default_initializable<extents_type>
    = default;

    /// \brief Construct an accessor for one logical index space and component stride.
    /// \pre The component stride is nonnegative when `ComponentLayout` is `layout_stride`.
    /// \param extents Full logical tensor extents.
    /// \param component_stride Offset between consecutive stored diagonal components.
    UNI20_HOST_DEVICE explicit constexpr diagonal_accessor(extents_type extents, ComponentIndex component_stride = 1)
        : extents_(std::move(extents)), component_stride_(component_stride)
    {}

    /// \brief Read or write the logical element represented by an encoded offset.
    [[nodiscard]] UNI20_HOST_DEVICE constexpr reference access(data_handle_type handle, offset_type offset) const
    {
      auto const diagonal_index = this->component_index(handle, offset);
      auto const component_offset = static_cast<ComponentIndex>(diagonal_index.index) * component_stride_;
      if constexpr (std::is_const_v<element_type>)
      {
        return diagonal_index.present ? handle.data[component_offset] : value_type{};
      }
      else
      {
        return reference{diagonal_index.present ? handle.data + component_offset : nullptr};
      }
    }

    /// \brief Advance a logical handle without changing its original index space.
    [[nodiscard]] UNI20_HOST_DEVICE constexpr data_handle_type offset(data_handle_type handle,
                                                                      offset_type offset) const noexcept
    {
      handle.logical_offset += offset;
      return handle;
    }

    /// \brief Return the extents used to decode logical offsets.
    [[nodiscard]] UNI20_HOST_DEVICE constexpr auto extents() const noexcept -> extents_type const& { return extents_; }

    /// \brief Return the physical offset between consecutive diagonal components.
    [[nodiscard]] UNI20_HOST_DEVICE constexpr auto component_stride() const noexcept -> ComponentIndex
    {
      return component_stride_;
    }

    /// \brief Decode whether a logical offset addresses a stored component and which one.
    [[nodiscard]] UNI20_HOST_DEVICE constexpr auto
    component_index(data_handle_type handle, offset_type offset = 0) const noexcept -> component_index_result
    {
      return this->diagonal_index(handle.logical_offset + offset);
    }

  private:
    [[nodiscard]] UNI20_HOST_DEVICE constexpr auto
    diagonal_index(std::size_t linear) const noexcept -> component_index_result
    {
      if constexpr (extents_type::rank() == 0)
      {
        return {};
      }
      else
      {
        std::array<index_type, extents_type::rank()> indices{};
        for (std::size_t axis = extents_type::rank(); axis > 0; --axis)
        {
          auto const extent = static_cast<std::size_t>(extents_.extent(axis - 1));
          indices[axis - 1] = static_cast<index_type>(linear % extent);
          linear /= extent;
        }
        auto const first = indices[0];
        for (std::size_t axis = 1; axis < extents_type::rank(); ++axis)
          if (indices[axis] != first) return {.present = false};
        return {.index = static_cast<std::size_t>(first), .present = true};
      }
    }

    [[no_unique_address]] extents_type extents_{};
    ComponentIndex component_stride_ = 1;
};

/// \brief Convert a writable diagonal accessor to its read-only form.
template <class ElementType, class Extents, class ComponentIndex, class ComponentLayout>
  requires(!std::is_const_v<ElementType>)
[[nodiscard]] UNI20_HOST_DEVICE constexpr auto
const_accessor(diagonal_accessor<ElementType, Extents, ComponentIndex, ComponentLayout> const& accessor)
{
  return diagonal_accessor<ElementType const, Extents, ComponentIndex, ComponentLayout>{accessor.extents(),
                                                                                        accessor.component_stride()};
}

/// \brief Return an already read-only diagonal accessor unchanged.
template <class ElementType, class Extents, class ComponentIndex, class ComponentLayout>
  requires std::is_const_v<ElementType>
[[nodiscard]] UNI20_HOST_DEVICE constexpr auto
const_accessor(diagonal_accessor<ElementType, Extents, ComponentIndex, ComponentLayout> const& accessor)
{
  return accessor;
}

/// \brief Diagonal accessors may be evaluated by ordinary host code.
template <class ElementType, class Extents, class ComponentIndex, class ComponentLayout>
inline constexpr bool enable_accessor_in_domain<
    diagonal_accessor<ElementType, Extents, ComponentIndex, ComponentLayout>, host_access_domain> = true;

namespace detail
{

template <class Extents>
[[nodiscard]] UNI20_HOST_DEVICE constexpr auto diagonal_component_extent(Extents const& extents) noexcept ->
    typename Extents::index_type
{
  using index_type = typename Extents::index_type;
  if constexpr (Extents::rank() == 0)
  {
    return index_type{1};
  }
  else
  {
    index_type result = extents.extent(0);
    for (std::size_t axis = 1; axis < Extents::rank(); ++axis)
      if (extents.extent(axis) < result) result = extents.extent(axis);
    return result;
  }
}

// GeneratedLayout encodes a full logical coordinate as one linear offset.
template <class Index, std::size_t... Axes>
constexpr auto diagonal_logical_extents(uni20::extents<Index, Axes...> extents, bool empty)
{
  using logical_index = std::common_type_t<Index, std::ptrdiff_t>;
  if (!empty)
  {
    logical_index size = 1;
    for (std::size_t axis = 0; axis < extents.rank(); ++axis)
    {
      auto const extent = static_cast<logical_index>(extents.extent(axis));
      if (size > std::numeric_limits<logical_index>::max() / extent)
        throw std::overflow_error("diagonal logical size exceeds its index range");
      size *= extent;
    }
  }
  return uni20::extents<logical_index, Axes...>{extents};
}

template <class Mapping, std::size_t... Axis>
[[nodiscard]] constexpr auto diagonal_mapping_at_one(Mapping const& mapping, std::index_sequence<Axis...>)
{
  using index_type = typename Mapping::extents_type::index_type;
  return mapping(((void)Axis, index_type{1})...);
}

template <class Layout, std::integral Index>
constexpr void validate_diagonal_component_span(std::size_t count, Index stride)
{
  if constexpr (std::same_as<Layout, layout_stride> && std::signed_integral<Index>)
    if (stride < 0) throw std::invalid_argument("negative diagonal component stride requires a signed-stride layout");
  if (count == 0) return;

  Index last_offset{};
  if (__builtin_mul_overflow(count - 1, stride, &last_offset) || last_offset == std::numeric_limits<Index>::max() ||
      (std::signed_integral<Index> && last_offset <= -std::numeric_limits<Index>::max()))
    throw std::overflow_error("diagonal component span exceeds its index range");
}

} // namespace detail

/// \brief Expose the physically stored values of a generalized-diagonal mdspan.
/// \details The returned rank-one mdspan preserves the component origin and
///          stride of aligned offset or stepped views. The source remains the
///          full logical rank-N mdspan whose off-diagonal accesses observe zero.
/// \tparam ElementType Presented component type, possibly const-qualified.
/// \tparam Extents Full logical tensor extents.
/// \tparam LayoutPolicy Logical mdspan layout policy.
/// \tparam AccessorExtents Original extents used by the diagonal accessor to decode offsets.
/// \tparam ComponentIndex Index type of the stored component view.
/// \tparam ComponentLayout Layout of the stored component view.
/// \param span Full generalized-diagonal mdspan.
/// \return Rank-one strided mdspan over the stored diagonal components.
/// \throws std::invalid_argument If the logical view does not begin on or
///         advance along the represented diagonal.
/// \throws std::overflow_error If the component stride or span cannot be represented.
template <class ElementType, class Extents, class LayoutPolicy, class AccessorExtents, class ComponentIndex,
          class ComponentLayout>
  requires(Extents::rank() == AccessorExtents::rank())
[[nodiscard]] constexpr auto diagonal_components(
    uni20::mdspan<ElementType, Extents, LayoutPolicy,
                  diagonal_accessor<ElementType, AccessorExtents, ComponentIndex, ComponentLayout>> const& span)
{
  using index_type = ComponentIndex;
  using component_extents_type = uni20::dextents<index_type, 1>;
  using component_mapping_type = ComponentLayout::template mapping<component_extents_type>;
  using component_mdspan_type = uni20::mdspan<ElementType, component_extents_type, ComponentLayout>;
  auto const count = detail::diagonal_component_extent(span.extents());
  if (!std::in_range<index_type>(count)) throw std::overflow_error("diagonal component count exceeds its index range");
  auto const extents = component_extents_type{count};
  index_type component_stride = span.accessor().component_stride();
  auto* data = span.data_handle().data;
  if (extents.extent(0) > 0)
  {
    auto const origin = span.accessor().component_index(span.data_handle(), detail::mapping_origin(span.mapping()));
    if (!origin.present)
      throw std::invalid_argument("diagonal component view does not begin on the represented diagonal");
    index_type origin_offset{};
    if (__builtin_mul_overflow(origin.index, component_stride, &origin_offset))
      throw std::overflow_error("diagonal component origin exceeds its index range");
    if (extents.extent(0) > 1)
    {
      auto const next_offset =
          detail::diagonal_mapping_at_one(span.mapping(), std::make_index_sequence<Extents::rank()>{});
      auto const next = span.accessor().component_index(span.data_handle(), next_offset);
      if (!next.present || next.index < origin.index)
        throw std::invalid_argument("diagonal component view does not preserve the represented diagonal");
      if (__builtin_mul_overflow(next.index - origin.index, component_stride, &component_stride))
        throw std::overflow_error("diagonal component stride exceeds its index range");
    }
    data += origin_offset;
  }
  detail::validate_diagonal_component_span<ComponentLayout>(count, component_stride);
  auto const mapping = component_mapping_type{extents, std::array<index_type, 1>{component_stride}};
  if (count != 0) data -= mapping(0);
  return component_mdspan_type{data, mapping};
}

/// \brief Present rank-one strided components as a full generalized-diagonal mdspan.
/// \details The resulting mdspan has the supplied logical rank-N extents.
///          It reads as zero unless every logical index is equal and delegates
///          diagonal values to the supplied component view. The accessor
///          preserves the component layout and stride direction.
/// \throws std::invalid_argument If the component extent differs from the logical diagonal extent
///         or the component stride is negative for a forward component layout.
/// \throws std::overflow_error If the logical size, component count, stride, or rebased span exceeds its index range.
/// \tparam Components Default-accessor rank-one strided mdspan type.
/// \param logical_extents Full logical tensor extents.
/// \param components Stored component view with `min(logical_extents)` entries.
/// \return Full logical mdspan modelling `DiagonalMdspecLike`.
template <class Extents, class Components>
  requires AffineMdspanLike<Components> && RankedMdspanLike<Components, 1> && DefaultAccessorMdspanLike<Components>
[[nodiscard]] constexpr auto make_diagonal_mdspan(Extents logical_extents, Components const& components)
{
  using component_type = std::remove_cvref_t<Components>;
  using element_type = typename component_type::element_type;
  auto const count = components.extent(0);
  if (!std::cmp_equal(count, detail::diagonal_component_extent(logical_extents)))
  {
    throw std::invalid_argument("diagonal component extent does not match the logical tensor extents");
  }
  using component_index = std::remove_cvref_t<decltype(components.mapping()(0))>;
  using component_layout = std::conditional_t<StridedMdspanLike<Components> || !std::signed_integral<component_index>,
                                              layout_stride, layout_signed_stride>;
  auto const stride = components.mapping().stride(0);
  if (!std::in_range<component_index>(count) || !std::in_range<component_index>(stride))
    throw std::overflow_error("diagonal component count or stride exceeds its index range");
  auto const component_stride = static_cast<component_index>(stride);
  detail::validate_diagonal_component_span<component_layout>(static_cast<std::size_t>(count), component_stride);

  auto* data = components.data_handle();
  if (count != 0) data = components.accessor().offset(data, components.mapping()(0));
  auto extents = detail::diagonal_logical_extents(logical_extents, count == 0);
  using extents_type = decltype(extents);
  using accessor_type = diagonal_accessor<element_type, extents_type, component_index, component_layout>;
  using result_type = uni20::mdspan<element_type, extents_type, GeneratedLayout, accessor_type>;
  return result_type{typename accessor_type::data_handle_type{data, 0}, typename result_type::mapping_type{extents},
                     accessor_type{extents, component_stride}};
}

/// \brief Recover the stored value type from a writable diagonal proxy.
template <class ElementType> struct remove_proxy_reference<diagonal_reference<ElementType>>
{
    using type = ElementType;
};

} // namespace uni20
