#pragma once

#include <uni20/mdspan/repeat_axis.hpp>
#include <uni20/tensor/transform.hpp>

#include <stdexcept>

namespace uni20::linalg
{
// Share the multiply kernel's accessor, type and layout support. A backend can specialize
// scale_columns_op directly while receiving the original matrix and vector descriptors.
template <class Backend, MutableRankedMdspecLike<2> M, RankedMdspecLike<1> V>
  requires std::is_const_v<typename V::element_type>
consteval auto kernel_accepts_types(Backend const&, scale_columns_op, M& matrix, V& factors)
{
  using Rows = decltype(uni20::detail::repeat_first_axis(factors, matrix.extents()));
  return KernelAcceptance<detail::backend_type_acceptance<Backend, transform_inplace_op<multiply>, M, Rows>()>{};
}

template <class Backend, MutableRankedMdspecLike<2> M, RankedMdspecLike<1> V>
  requires std::is_const_v<typename V::element_type>
KernelAttempt try_kernel(Backend const& backend, scale_columns_op, M& matrix, V& factors)
{
  auto rows = uni20::detail::repeat_first_axis(factors, matrix.extents());
  return try_kernel(backend, transform_inplace_op{multiply{}}, matrix, rows);
}

/// \brief Multiply each matrix column j in place by factors[j] using an explicit backend selector.
/// \pre The matrix has a unique mapping, and the factors do not overlap the matrix.
/// \throws std::invalid_argument If the factor count differs from the column count.
template <KernelBackendSelector Selector, MutableRankedTensorView<2> M, RankedTensorView<1> V>
void scale_columns(Selector&& selector, M&& matrix, V const& factors)
{
  if (matrix.extent(1) != factors.extent(0)) throw std::invalid_argument("scale_columns needs one factor per column");
  if (matrix.extent(0) == 0 || matrix.extent(1) == 0) return;
  auto out = mdspec_of(matrix);
  auto in = mdspec_of(factors);
  dispatch_kernel(std::forward<Selector>(selector), scale_columns_op{}, out, in);
}

/// \brief Multiply a matrix in place on the right by diag(factors).
template <MutableRankedTensorView<2> M, RankedTensorView<1> V> void scale_columns(M&& matrix, V const& factors)
{
  scale_columns(select_backend(scale_columns_op{}, matrix, factors), matrix, factors);
}
} // namespace uni20::linalg
