#pragma once

/**
 * \file qr.hpp
 * \ingroup linalg
 * \brief LAPACK backend for reduced dense real or complex QR factorization.
 */

#include "common.hpp"
#if UNI20_HAS_MPLAPACK_MPREAL || UNI20_HAS_MPLAPACK_FLOAT80
#include "decompositions_packed.hpp"
#endif

#include <uni20/backend/blas/blas_int.hpp>
#include <uni20/backend/lapack/lapack.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/blas/mdspan_matrix.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/concepts.hpp>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace uni20::linalg
{
namespace lapack_detail
{

template <class QMdspan, class RMdspan, class MatrixMdspan> consteval auto qr_acceptance()
{
  using q_scalar = std::remove_cv_t<typename QMdspan::element_type>;
  using r_scalar = std::remove_cv_t<typename RMdspan::element_type>;
  using matrix_scalar = std::remove_cv_t<typename MatrixMdspan::element_type>;
  if constexpr (uni20::LapackScalar<matrix_scalar> && std::same_as<q_scalar, matrix_scalar> &&
                std::same_as<r_scalar, matrix_scalar> && uni20::DefaultAccessorMdspanLike<MatrixMdspan>)
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

// Form Q in the first min(m,n) columns; full QR consumes the packed R before orgqr overwrites it.
template <MutableRankedStridedMdspanLike<2> Matrix, class SaveR> KernelAttempt form_q(Matrix& matrix_work, SaveR save_r)
{
  using scalar_type = typename Matrix::value_type;
  auto const m = static_cast<std::size_t>(matrix_work.extent(0));
  auto const n = static_cast<std::size_t>(matrix_work.extent(1));
  auto const rank_size = std::min(m, n);
  if (rank_size == 0) return KernelAttempt::success;
  blas_int const rows = uni20::blas::try_blas_int(m), cols = uni20::blas::try_blas_int(n);
  if (!uni20::blas::is_valid_blas_int(rows) || !uni20::blas::is_valid_blas_int(cols))
    return KernelAttempt::unsupported_shape;
  blas_int const rank = std::min(rows, cols);
  auto matrix = blas::try_lapack_writable_matrix(matrix_work);
  if (!matrix) return KernelAttempt::unsupported_layout;

  std::vector<scalar_type> tau(rank_size);
  scalar_type work_query{};
  uni20::lapack::geqrf(rows, cols, matrix->data, matrix->leading_dimension, tau.data(), &work_query, -1);
  blas_int const geqrf_lwork = workspace_size(work_query);
  std::vector<scalar_type> work(static_cast<std::size_t>(geqrf_lwork));
  uni20::lapack::geqrf(rows, cols, matrix->data, matrix->leading_dimension, tau.data(), work.data(), geqrf_lwork);

  save_r();

  work_query = scalar_type{};
  uni20::lapack::orgqr(rows, rank, rank, matrix->data, matrix->leading_dimension, tau.data(), &work_query, -1);
  blas_int const orgqr_lwork = workspace_size(work_query);
  work.resize(static_cast<std::size_t>(orgqr_lwork));
  uni20::lapack::orgqr(rows, rank, rank, matrix->data, matrix->leading_dimension, tau.data(), work.data(), orgqr_lwork);

  return KernelAttempt::success;
}

template <uni20::MutableRankedMdspanLike<2> QMdspan, uni20::MutableRankedMdspanLike<2> RMdspan,
          uni20::MutableRankedStridedMdspanLike<2> MatrixMdspan>
KernelAttempt try_qr(QMdspan& q, RMdspan& r, MatrixMdspan& matrix_work)
{
  using scalar_type = std::remove_cv_t<typename MatrixMdspan::element_type>;
  using q_index_type = typename QMdspan::index_type;
  using r_index_type = typename RMdspan::index_type;
  using matrix_index_type = typename MatrixMdspan::index_type;

  std::size_t const rows_size = static_cast<std::size_t>(matrix_work.extent(0));
  std::size_t const cols_size = static_cast<std::size_t>(matrix_work.extent(1));
  std::size_t const rank_size = std::min(rows_size, cols_size);
  CHECK_EQUAL(q.extent(0), matrix_work.extent(0));
  CHECK_EQUAL(static_cast<std::size_t>(q.extent(1)), rank_size);
  CHECK_EQUAL(static_cast<std::size_t>(r.extent(0)), rank_size);
  CHECK_EQUAL(r.extent(1), matrix_work.extent(1));

  auto const attempt = form_q(matrix_work, [&] {
    for (std::size_t row = 0; row < rank_size; ++row)
    {
      for (std::size_t column = 0; column < cols_size; ++column)
      {
        auto const r_row = static_cast<r_index_type>(row);
        auto const r_column = static_cast<r_index_type>(column);
        if (row <= column)
        {
          r[r_row, r_column] = matrix_work[static_cast<matrix_index_type>(row), static_cast<matrix_index_type>(column)];
        }
        else
        {
          r[r_row, r_column] = scalar_type{};
        }
      }
    }
  });
  if (attempt != KernelAttempt::success) return attempt;
  for (std::size_t row = 0; row < rows_size; ++row)
  {
    for (std::size_t column = 0; column < rank_size; ++column)
    {
      q[static_cast<q_index_type>(row), static_cast<q_index_type>(column)] =
          matrix_work[static_cast<matrix_index_type>(row), static_cast<matrix_index_type>(column)];
    }
  }
  return KernelAttempt::success;
}

} // namespace lapack_detail

/// \brief Report eligibility for a host-accessible reduced real or complex QR factorization.
template <uni20::MutableRankedMdspecLike<2> QMdspec, uni20::MutableRankedMdspecLike<2> RMdspec,
          uni20::MutableRankedStridedMdspecLike<2> MatrixMdspec>
  requires LapackScalar<typename MatrixMdspec::value_type> && uni20::HostWritableMdspec<QMdspec> &&
           uni20::HostWritableMdspec<RMdspec> && uni20::HostWritableMdspec<MatrixMdspec>
consteval auto kernel_accepts_types(LapackBackend const&, qr_op const&, QMdspec&, RMdspec&, MatrixMdspec&)
{
  using q_span = uni20::host_write_mdspan_t<QMdspec>;
  using r_span = uni20::host_write_mdspan_t<RMdspec>;
  using matrix_span = uni20::host_write_mdspan_t<MatrixMdspec>;
  constexpr auto acceptance = lapack_detail::qr_acceptance<q_span, r_span, matrix_span>();
  if constexpr (acceptance == KernelTypeAcceptance::no)
    return kernel_types_no;
  else
    return kernel_types_maybe;
}

/// \brief Resolve host access and compute reduced QR through LAPACK `geqrf` and `orgqr`/`ungqr`.
template <uni20::MutableRankedMdspecLike<2> QMdspec, uni20::MutableRankedMdspecLike<2> RMdspec,
          uni20::MutableRankedStridedMdspecLike<2> MatrixMdspec>
  requires LapackScalar<typename MatrixMdspec::value_type> && uni20::HostWritableMdspec<QMdspec> &&
           uni20::HostWritableMdspec<RMdspec> && uni20::HostWritableMdspec<MatrixMdspec>
KernelAttempt try_kernel(LapackBackend, qr_op const&, QMdspec& q, RMdspec& r, MatrixMdspec& matrix_work)
{
  return lapack_detail::with_host_write_mdspans(
      [](auto& q_span, auto& r_span, auto& matrix_span) { return lapack_detail::try_qr(q_span, r_span, matrix_span); },
      q, r, matrix_work);
}

namespace lapack_detail
{
template <MutableRankedStridedMdspanLike<2> Span>
  requires LapackScalar<typename Span::value_type> && DefaultAccessorMdspanLike<Span>
KernelAttempt form_thin_q(Span& span)
{
  return form_q(span, [] {});
}

#if UNI20_HAS_MPLAPACK_MPREAL || UNI20_HAS_MPLAPACK_FLOAT80
// The Q-only frontend materializes arbitrary input views into this column-major workspace.
template <MutableRankedMdspanLike<2> Span, class... P>
  requires packed_detail::BasisScalar<typename Span::value_type> && DefaultAccessorMdspanLike<Span> &&
           std::same_as<typename Span::layout_type, ColumnMajor> &&
           packed_detail::KernelPrecision<typename Span::value_type, P...>
KernelAttempt form_thin_q(Span& span, P... precision)
{
  auto const p = packed_detail::working_precision<typename Span::value_type>(precision...);
  std::span work(span.data_handle(), static_cast<std::size_t>(span.size()));
  mplapack::orthonormal_basis(span.extent(0), span.extent(1), work, p);
  return KernelAttempt::success;
}
#endif
} // namespace lapack_detail

template <HostWritableMdspec A, class... P>
consteval auto kernel_accepts_types(LapackBackend, orthonormal_basis_op, A&, P... precision)
{
  if constexpr (requires(host_write_mdspan_t<A>& span) { lapack_detail::form_thin_q(span, precision...); })
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

template <class A, class... P> KernelAttempt try_kernel(LapackBackend, orthonormal_basis_op, A& matrix, P... precision)
{
  if (matrix.extent(0) < matrix.extent(1)) return KernelAttempt::unsupported_shape;
  return lapack_detail::with_host_write_mdspans(
      [&](auto& span) { return lapack_detail::form_thin_q(span, precision...); }, matrix);
}

} // namespace uni20::linalg
