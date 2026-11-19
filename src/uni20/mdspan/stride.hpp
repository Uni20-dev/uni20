#pragma once

/**
 * \file stride.hpp
 * \ingroup mdspan_ext
 * \brief Stride observer that exists at every rank for strided mappings, mdspans and mdspecs.
 */

#include <uni20/common/trace.hpp>
#include <uni20/mdspan/mdspan.hpp>

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <utility>

namespace uni20
{
namespace detail
{

template <class T>
concept HasLayoutMappingMember = requires(T const& value) { value.mapping(); };

/// \brief Return the layout mapping of an mdspan or mdspec, or a mapping itself.
template <class T> [[nodiscard]] constexpr decltype(auto) stride_mapping(T const& value) noexcept
{
  if constexpr (HasLayoutMappingMember<T>)
    return value.mapping();
  else
    return (value);
}

template <class T> using stride_mapping_t = std::remove_cvref_t<decltype(stride_mapping(std::declval<T const&>()))>;

/// \brief A layout mapping, or an mdspan or mdspec of one, whose strides can be observed.
template <class T>
concept StrideObservableMapping = requires {
  typename stride_mapping_t<T>::extents_type;
  typename stride_mapping_t<T>::index_type;
} && (stride_mapping_t<T>::extents_type::rank() == 0 || requires(stride_mapping_t<T> const& mapping, std::size_t axis) {
                                    {
                                      mapping.stride(axis)
                                    } -> std::convertible_to<typename stride_mapping_t<T>::index_type>;
                                  });

/// \brief Any other object with a callable `stride(axis)` member.
template <class T>
concept HasStrideMember = requires(T const& value, std::size_t axis) {
  { value.stride(axis) } -> std::integral;
};

} // namespace detail

/// \brief Whether `uni20::stride` is available for a type.
/// \details Available for a layout mapping, or an mdspan or mdspec of one, that
///          exposes `stride(rank_type)`, and always at rank zero. A rank-zero
///          mapping has no axes; the standard lets its `stride` be ill-formed
///          ([mdspan.layout.reqmts], note 5), and `layout_left` and
///          `layout_right` constrain it away, while `layout_stride` keeps it.
///          Any other object with a callable `stride(axis)` member is accepted
///          as is.
/// \tparam T Layout mapping, mdspan, mdspec, or object with a stride member.
/// \ingroup mdspan_ext
template <class T>
concept StrideObservable = detail::StrideObservableMapping<T> || detail::HasStrideMember<T>;

/// \brief Return the stride of one axis, for a mapping of any rank.
/// \details The observer exists at every rank, so code that loops over the axes
///          of a strided operand compiles at rank zero, where the loop body never
///          runs. Its precondition `axis < rank()` cannot hold at rank zero, so a
///          call there is a precondition failure.
/// \param value Layout mapping, mdspan, mdspec, or object with a stride member.
/// \param axis Axis whose stride is requested.
/// \pre `axis < rank()`.
/// \ingroup mdspan_ext
template <StrideObservable T> [[nodiscard]] constexpr auto stride(T const& value, std::size_t axis)
{
  if constexpr (detail::StrideObservableMapping<T>)
  {
    using mapping_type = detail::stride_mapping_t<T>;
    using index_type = typename mapping_type::index_type;
    constexpr std::size_t rank = mapping_type::extents_type::rank();
    if constexpr (rank == 0)
    {
      PRECONDITION(axis < rank, "a rank-zero mapping has no axes", axis);
      return index_type{0};
    }
    else
    {
      return static_cast<index_type>(detail::stride_mapping(value).stride(axis));
    }
  }
  else
  {
    return value.stride(axis);
  }
}

} // namespace uni20
