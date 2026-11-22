#pragma once

/**
 * \file diagonal.hpp
 * \ingroup tensor
 * \brief Borrowed views that identify pairs of dense tensor axes.
 */

#include <uni20/mdspan/diagonal_view.hpp>
#include <uni20/tensor/mdspec_tensor_view.hpp>

#include <cstddef>
#include <type_traits>
#include <utility>

namespace uni20
{
/// \brief Borrow a view on which each pair of axes shares one index.
/// \details Unpaired axes retain their order, followed by one axis per pair.
///          Each paired axis has the smaller of its two source extents.
///          The view preserves the source accessor, storage domain, and backend.
///          The tensor lvalue must outlive the view.
/// \pre The source mapping is unique and has its origin at offset zero. Its
///      strides are positive on all axes with more than one entry.
/// \tparam Axis A nonempty sequence of disjoint axis pairs.
template <std::size_t... Axis, StridedTensorView Tensor>
  requires(std::is_lvalue_reference_v<Tensor &&> &&
           detail::valid_diagonal_pairs<tensor_mdspec_t<Tensor>::rank(), Axis...>())
[[nodiscard]] auto diagonal_view(Tensor&& source)
{
  auto writable = detail::diagonal_mdspec<Axis...>(mdspec_of(source));
  auto readable = detail::diagonal_mdspec<Axis...>(mdspec_of(std::as_const(source)));
  using selector_type = std::remove_cvref_t<decltype(source.backend_selector())>;
  return MdspecTensorView<decltype(writable), decltype(readable), tensor_storage_policy_t<Tensor>, selector_type>{
      std::move(writable), std::move(readable), detail::descriptor_precision_source(source), source.backend_selector()};
}

} // namespace uni20
