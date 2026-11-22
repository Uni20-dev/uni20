#pragma once

/**
 * \file reshape.hpp
 * \ingroup tensor
 * \brief Explicit aliasing, in-place, and owning tensor reshape operations.
 */

#include <uni20/common/trace.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/mdspan/mdspan.hpp>
#include <uni20/mdspan/mdspec.hpp>
#include <uni20/tensor/basic_tensor.hpp>
#include <uni20/tensor/concepts.hpp>
#include <uni20/tensor/mdspec_tensor_view.hpp>
#include <uni20/tensor/shape.hpp>

#include <array>
#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <new>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20
{
namespace detail
{

template <class LayoutPolicy>
concept CanonicalReshapeLayout =
    std::same_as<LayoutPolicy, stdex::layout_left> || std::same_as<LayoutPolicy, stdex::layout_right>;

template <class Span>
concept CanonicallyLaidOutMdspan =
    StridedMdspanLike<Span> && CanonicalReshapeLayout<typename std::remove_cvref_t<Span>::layout_type>;

template <class T>
concept CanonicallyLaidOutOwningTensor =
    OwningTensor<T> && MutableStridedTensorView<T> && CanonicalReshapeLayout<typename tensor_mdspec_t<T>::layout_type>;

template <class T, class NewExtents>
using reshape_rebind_t = typename std::remove_cvref_t<T>::template rebind_extents_type<NewExtents>;

template <class T, class NewExtents>
concept StorageTransferReshapableTensor =
    CanonicallyLaidOutOwningTensor<T> &&
    requires(std::remove_cvref_t<T>&& tensor, typename std::remove_cvref_t<T>::storage_type storage,
             typename std::remove_cvref_t<T>::accessor_factory_type accessor_factory,
             typename reshape_rebind_t<T, NewExtents>::mapping_type mapping) {
      { std::move(tensor).release_storage() } -> std::same_as<typename std::remove_cvref_t<T>::storage_type>;
      {
        std::move(tensor).release_accessor_factory()
      } -> std::same_as<typename std::remove_cvref_t<T>::accessor_factory_type>;
      {
        reshape_rebind_t<T, NewExtents>::adopt_storage(std::move(mapping), std::move(storage),
                                                       std::move(accessor_factory))
      } -> std::same_as<reshape_rebind_t<T, NewExtents>>;
    };

template <class Index> [[nodiscard]] constexpr bool positive_stride_equals(Index stride, std::size_t expected)
{
  if constexpr (std::signed_integral<Index>)
    if (stride < 0) return false;
  return std::in_range<std::size_t>(stride) && static_cast<std::size_t>(stride) == expected;
}

template <StridedMdspecLike Span> [[nodiscard]] bool has_row_major_contiguous_mapping(Span const& span)
{
  std::size_t expected = 1;
  for (std::size_t axis = Span::rank(); axis > 0; --axis)
  {
    auto const extent = static_cast<std::size_t>(span.extent(axis - 1));
    if (extent > 1 && !positive_stride_equals(span.stride(axis - 1), expected)) return false;
    if (extent > 0)
    {
      if (expected > std::numeric_limits<std::size_t>::max() / extent) return false;
      expected *= extent;
    }
  }
  return true;
}

template <StridedMdspecLike Span> [[nodiscard]] bool has_column_major_contiguous_mapping(Span const& span)
{
  std::size_t expected = 1;
  for (std::size_t axis = 0; axis < Span::rank(); ++axis)
  {
    auto const extent = static_cast<std::size_t>(span.extent(axis));
    if (extent > 1 && !positive_stride_equals(span.stride(axis), expected)) return false;
    if (extent > 0)
    {
      if (expected > std::numeric_limits<std::size_t>::max() / extent) return false;
      expected *= extent;
    }
  }
  return true;
}

template <CanonicalReshapeLayout LayoutPolicy, StridedMdspecLike Span>
[[nodiscard]] bool has_canonical_contiguous_mapping(Span const& span)
{
  auto const logical_size = checked_element_count(span.extents());
  // An empty index space has no elements to place or order. Test it before
  // is_exhaustive(), which can be false for empty strided mappings (LWG 4266).
  if (logical_size == 0) return true;
  if constexpr (requires { span.mapping().is_unique(); })
    if (!span.mapping().is_unique()) return false;
  if constexpr (requires { span.mapping().is_exhaustive(); })
    if (!span.mapping().is_exhaustive()) return false;
  if constexpr (requires { span.mapping().required_span_size(); })
  {
    auto const required_size = span.mapping().required_span_size();
    if (!std::in_range<std::size_t>(required_size) || static_cast<std::size_t>(required_size) != logical_size)
      return false;
  }

  // A scalar index space has no observable traversal-order distinction.
  if (logical_size == 1) return true;

  if constexpr (std::same_as<LayoutPolicy, stdex::layout_left>)
    return has_column_major_contiguous_mapping(span);
  else
    return has_row_major_contiguous_mapping(span);
}

template <CanonicalReshapeLayout LayoutPolicy, StridedMdspecLike Span>
void require_canonical_contiguous_mapping(Span const& span)
{
  if constexpr (std::same_as<LayoutPolicy, stdex::layout_left>)
  {
    ERROR_IF(!has_canonical_contiguous_mapping<LayoutPolicy>(span),
             "reshape requires a unique, exhaustive, canonical column-major source mapping");
  }
  else
  {
    ERROR_IF(!has_canonical_contiguous_mapping<LayoutPolicy>(span),
             "reshape requires a unique, exhaustive, canonical row-major source mapping");
  }
}

template <CanonicalReshapeLayout LayoutPolicy, StridedMdspecLike Span, std::integral... Extents>
[[nodiscard]] auto make_reshape_view(Span&& source, Extents... requested_extents)
{
  using source_type = std::remove_cvref_t<Span>;
  require_canonical_contiguous_mapping<LayoutPolicy>(source);
  auto const source_size = checked_element_count(source.extents());
  auto new_extents = make_reshape_extents(source_size, requested_extents...);
  using extents_type = decltype(new_extents);
  using accessor_type = typename source_type::accessor_type;
  using element_type = typename source_type::element_type;
  using mapping_type = typename LayoutPolicy::template mapping<extents_type>;
  if constexpr (MdspanLike<source_type>)
  {
    using result_type = stdex::mdspan<element_type, extents_type, LayoutPolicy, accessor_type>;
    return result_type{source.data_handle(), mapping_type{new_extents}, source.accessor()};
  }
  else
  {
    using descriptor_type = typename source_type::data_descriptor_type;
    using result_type = mdspec<element_type, extents_type, LayoutPolicy, accessor_type, descriptor_type>;
    return result_type{source.data_descriptor(), mapping_type{new_extents}, source.accessor()};
  }
}

} // namespace detail

/// \brief Return a no-copy mdspan reshape preserving a canonical source layout.
/// \details Automatic order selection is available only when the source's
///          static layout is `layout_left` or `layout_right`.
template <detail::CanonicallyLaidOutMdspan Span, std::integral... Extents>
[[nodiscard]] auto reshape_view(Span&& source, Extents... requested_extents)
{
  using layout_type = typename std::remove_cvref_t<Span>::layout_type;
  return detail::make_reshape_view<layout_type>(std::forward<Span>(source), requested_extents...);
}

/// \brief Return a no-copy column-major reshape of a compatible strided mdspan.
/// \details Singleton-axis strides do not select an order. All other strides
///          must describe a unique, exhaustive canonical column-major mapping.
template <StridedMdspanLike Span, std::integral... Extents>
[[nodiscard]] auto reshape_view_left(Span&& source, Extents... requested_extents)
{
  return detail::make_reshape_view<stdex::layout_left>(std::forward<Span>(source), requested_extents...);
}

/// \brief Return a no-copy row-major reshape of a compatible strided mdspan.
/// \details Singleton-axis strides do not select an order. All other strides
///          must describe a unique, exhaustive canonical row-major mapping.
template <StridedMdspanLike Span, std::integral... Extents>
[[nodiscard]] auto reshape_view_right(Span&& source, Extents... requested_extents)
{
  return detail::make_reshape_view<stdex::layout_right>(std::forward<Span>(source), requested_extents...);
}

/// \brief Borrowed reshape with separate read and write descriptors stored by value.
template <MdspecLike MutableMdspec, MdspecLike ConstMdspec, class StoragePolicy, class BackendSelector>
class ReshapedTensor : public MdspecTensorView<MutableMdspec, ConstMdspec, StoragePolicy, BackendSelector> {
  public:
    using MdspecTensorView<MutableMdspec, ConstMdspec, StoragePolicy, BackendSelector>::MdspecTensorView;
    using async_alias_tag = void;
    using layout_type = typename ConstMdspec::layout_type;
    using mapping_type = typename ConstMdspec::mapping_type;
};

/// \brief Tensor-level descriptor that resolves a reshape through its parent tensor.
/// \details The descriptor aliases the same element storage as its source and
///          preserves source mutability. Parent indirection allows an async
///          alias to bind reserved storage before the parent value is constructed.
template <class Tensor, detail::CanonicalReshapeLayout LayoutPolicy, std::integral... RequestedExtents>
  requires TensorView<Tensor> && StridedTensorView<Tensor>
class IndirectReshapedTensorView {
  public:
    using tensor_type = std::remove_reference_t<Tensor>;
    using storage_policy = tensor_storage_policy_t<std::remove_cv_t<tensor_type>>;
    using backend_selector_type = std::remove_cvref_t<decltype(std::declval<tensor_type const&>().backend_selector())>;
    using requested_extents_type = std::tuple<RequestedExtents...>;
    using mdspec_type = decltype(detail::make_reshape_view<LayoutPolicy>(mdspec_of(std::declval<tensor_type&>()),
                                                                         std::declval<RequestedExtents>()...));
    using const_mdspec_type = decltype(detail::make_reshape_view<LayoutPolicy>(
        mdspec_of(std::declval<tensor_type const&>()), std::declval<RequestedExtents>()...));
    using async_alias_tag = void;
    using element_type = typename mdspec_type::element_type;
    using value_type = typename mdspec_type::value_type;
    using extents_type = typename mdspec_type::extents_type;
    using index_type = typename mdspec_type::index_type;
    using layout_type = typename mdspec_type::layout_type;
    using mapping_type = typename mdspec_type::mapping_type;

    /// \brief Bind and immediately validate a reshape of an existing tensor.
    IndirectReshapedTensorView(tensor_type& tensor, RequestedExtents... requested_extents)
        : IndirectReshapedTensorView(std::addressof(tensor), requested_extents...)
    {
      static_cast<void>(this->mdspec());
    }

    /// \brief Bind a reshape to externally retained tensor storage.
    /// \warning The pointer may identify reserved but unconstructed storage;
    ///          callers must not resolve the view until the parent epoch is readable.
    explicit IndirectReshapedTensorView(tensor_type* tensor, RequestedExtents... requested_extents)
        : tensor_(tensor), requested_extents_(requested_extents...)
    {
      CHECK(tensor_ != nullptr);
    }

#if UNI20_HAS_MPREAL
    /// \brief Read the parent's current default precision without converting elements.
    /// \pre The parent is constructed and readable, including its async epoch when applicable.
    [[nodiscard]] Precision default_precision() const
      requires has_runtime_precision_v<tensor_element_t<Tensor>>
    {
      return this->base().default_precision();
    }

    /// \brief Read the parent's optional default under the same access rules as its values.
    [[nodiscard]] std::optional<Precision> default_precision_if_set() const
      requires has_runtime_precision_v<tensor_element_t<Tensor>>
    {
      return this->base().default_precision_if_set();
    }
#endif

    /// \brief Return the source storage's backend selector.
    [[nodiscard]] constexpr decltype(auto) backend_selector() const { return this->base().backend_selector(); }

    /// \brief Return the reshaped immediate or deferred metadata with source mutability.
    [[nodiscard]] auto mdspec() & -> mdspec_type
    {
      return std::apply(
          [this](auto... requested_extents) {
            return detail::make_reshape_view<LayoutPolicy>(mdspec_of(this->base()), requested_extents...);
          },
          requested_extents_);
    }

    /// \brief Return the reshaped immediate or deferred metadata for read-only access.
    [[nodiscard]] auto mdspec() const& -> const_mdspec_type
    {
      return std::apply(
          [this](auto... requested_extents) {
            return detail::make_reshape_view<LayoutPolicy>(mdspec_of(this->base()), requested_extents...);
          },
          requested_extents_);
    }

    /// \brief Resolve the reshaped mdspan when the parent is immediately accessible.
    [[nodiscard]] auto mdspan() & -> mdspec_type
      requires MdspanLike<mdspec_type>
    {
      return this->mdspec();
    }

    /// \brief Resolve the read-only reshaped mdspan when the parent is immediately accessible.
    [[nodiscard]] auto mdspan() const& -> const_mdspec_type
      requires MdspanLike<const_mdspec_type>
    {
      return this->mdspec();
    }

    /// \brief Return the reshaped extents.
    [[nodiscard]] auto extents() const -> extents_type { return this->mdspec().extents(); }

    /// \brief Return one reshaped extent.
    [[nodiscard]] auto extent(std::size_t axis) const -> index_type { return this->mdspec().extent(axis); }

    /// \brief Evaluate or assign one reshaped element according to source mutability.
    template <class... Index>
      requires MdspanLike<mdspec_type> && (sizeof...(Index) == extents_type::rank())
    decltype(auto) operator[](Index... indices)
    {
      return this->mdspan()[indices...];
    }

    /// \brief Evaluate one reshaped element through the read-only accessor.
    template <class... Index>
      requires MdspanLike<const_mdspec_type> && (sizeof...(Index) == extents_type::rank())
    decltype(auto) operator[](Index... indices) const
    {
      return this->mdspan()[indices...];
    }

    /// \brief Return the tensor referenced by this descriptor.
    [[nodiscard]] tensor_type& base() noexcept { return *std::launder(tensor_); }

    /// \brief Return the tensor referenced by this descriptor.
    [[nodiscard]] tensor_type const& base() const noexcept { return *std::launder(tensor_); }

  private:
    tensor_type* tensor_;
    requested_extents_type requested_extents_;
};

namespace detail
{

template <CanonicalReshapeLayout LayoutPolicy, StridedTensorView Tensor, std::integral... Extents>
[[nodiscard]] auto make_tensor_reshape_view(Tensor& tensor, Extents... requested_extents)
{
  auto writable = make_reshape_view<LayoutPolicy>(mdspec_of(tensor), requested_extents...);
  auto readable = make_reshape_view<LayoutPolicy>(mdspec_of(std::as_const(tensor)), requested_extents...);
  using selector_type = std::remove_cvref_t<decltype(tensor.backend_selector())>;
  return ReshapedTensor<decltype(writable), decltype(readable), tensor_storage_policy_t<Tensor>, selector_type>{
      std::move(writable), std::move(readable), descriptor_precision_source(tensor), tensor.backend_selector()};
}

} // namespace detail

/// \brief Return a tensor-level no-copy reshape preserving a canonical source layout.
/// \details The view stores the reshaped metadata and borrows the underlying storage.
///          Any live precision source must also outlive the view.
template <StridedTensorView Tensor, std::integral... Extents>
  requires(std::is_lvalue_reference_v<Tensor &&> &&
           detail::CanonicalReshapeLayout<typename tensor_mdspec_t<Tensor>::layout_type>)
[[nodiscard]] auto reshape_view(Tensor&& tensor, Extents... requested_extents)
{
  using layout_type = typename tensor_mdspec_t<Tensor>::layout_type;
  return detail::make_tensor_reshape_view<layout_type>(tensor, requested_extents...);
}

/// \brief Return a tensor-level no-copy column-major reshape of a strided source.
template <StridedTensorView Tensor, std::integral... Extents>
  requires std::is_lvalue_reference_v<Tensor&&>
[[nodiscard]] auto reshape_view_left(Tensor&& tensor, Extents... requested_extents)
{
  return detail::make_tensor_reshape_view<stdex::layout_left>(tensor, requested_extents...);
}

/// \brief Return a tensor-level no-copy row-major reshape of a strided source.
template <StridedTensorView Tensor, std::integral... Extents>
  requires std::is_lvalue_reference_v<Tensor&&>
[[nodiscard]] auto reshape_view_right(Tensor&& tensor, Extents... requested_extents)
{
  return detail::make_tensor_reshape_view<stdex::layout_right>(tensor, requested_extents...);
}

/// \brief Change a canonical owning tensor's shape without reallocating storage.
/// \details The compile-time rank is unchanged. Existing copied mdspan
///          descriptors retain their old mappings.
template <detail::CanonicallyLaidOutOwningTensor TensorType, std::integral... Extents>
  requires(sizeof...(Extents) == tensor_extents_t<TensorType>::rank() &&
           requires(TensorType& tensor, typename TensorType::mapping_type mapping) {
             tensor.replace_mapping(std::move(mapping));
           })
void reshape_inplace(TensorType& tensor, Extents... requested_extents)
{
  using tensor_extents = tensor_extents_t<TensorType>;
  using layout_type = typename tensor_mdspec_t<TensorType>::layout_type;
  auto const source_size = detail::checked_element_count(tensor.extents());
  auto requested = detail::make_reshape_extents(source_size, requested_extents...);
  auto new_extents = detail::convert_reshape_extents<tensor_extents>(requested);
  using mapping_type = typename layout_type::template mapping<tensor_extents>;
  tensor.replace_mapping(mapping_type{new_extents});
}

/// \brief Return an owning reshape of a canonical dense tensor.
/// \details Passing an lvalue copies its allocation; passing an rvalue transfers
///          it. The source's compile-time canonical layout is preserved.
template <class TensorType, std::integral... Extents>
  requires detail::StorageTransferReshapableTensor<TensorType, stdex::dextents<index_type, sizeof...(Extents)>>
[[nodiscard]] auto reshape(TensorType tensor, Extents... requested_extents)
{
  auto const source_size = detail::checked_element_count(tensor.extents());
  auto new_extents = detail::make_reshape_extents(source_size, requested_extents...);

  using source_type = std::remove_cvref_t<TensorType>;
  using result_type = typename source_type::template rebind_extents_type<decltype(new_extents)>;
  using layout_type = typename tensor_mdspec_t<TensorType>::layout_type;
  using mapping_type = typename layout_type::template mapping<decltype(new_extents)>;
  auto accessor_factory = std::move(tensor).release_accessor_factory();
  auto storage = std::move(tensor).release_storage();
  return result_type::adopt_storage(mapping_type{new_extents}, std::move(storage), std::move(accessor_factory));
}

} // namespace uni20
