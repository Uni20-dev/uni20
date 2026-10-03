#pragma once

#include "linear_solve.hpp"
#include <uni20/linalg/backends/lu_common.hpp>

namespace uni20::linalg
{
/// \brief Accept directly addressable native LAPACK LU workspaces.
template <MutableRankedStridedMdspecLike<2> A>
  requires HostWritableMdspec<A> && LapackScalar<typename A::value_type> &&
           DefaultAccessorMdspanLike<host_write_mdspan_t<A>>
consteval auto kernel_accepts_types(LapackBackend, lu_factor_op const&, A&, std::span<std::size_t>, SolveInfo&,
                                    SolveOptions<make_real_t<typename A::value_type>> const&)
{
  return kernel_types_maybe;
}

/// \brief Compute packed factors through GETRF, preserving all outputs on decline.
template <MutableRankedStridedMdspecLike<2> A>
  requires HostWritableMdspec<A> && LapackScalar<typename A::value_type> &&
           DefaultAccessorMdspanLike<host_write_mdspan_t<A>>
KernelAttempt try_kernel(LapackBackend, lu_factor_op const&, A& a, std::span<std::size_t> pivots, SolveInfo& info,
                         SolveOptions<make_real_t<typename A::value_type>> const& options)
{
  detail::require_lu_shape(a, pivots.size());
  detail::require_solve_options(options);
  auto n = uni20::blas::try_blas_int(a.extent(0));
  if (!uni20::blas::is_valid_blas_int(n)) return KernelAttempt::unsupported_shape;
  if (n == 0)
  {
    info = {};
    return KernelAttempt::success;
  }
  auto access = acquire_host_write_access_sync(a);
  auto s = access.mdspan();
  auto matrix = blas::try_lapack_writable_matrix(s);
  if (!matrix) return KernelAttempt::unsupported_layout;
  std::vector<blas_int> provider_pivots(pivots.size());
  info = {};
  if (!detail::solve_matrix_is_finite(s))
  {
    info.status = SolveStatus::nonfinite_input;
    return KernelAttempt::success;
  }
  auto scale = detail::lu_input_scale(s, make_real_t<typename A::value_type>{});
  auto code = uni20::lapack::unchecked::getrf(n, n, matrix->data, matrix->leading_dimension, provider_pivots.data());
  uni20::lapack::detail::check_invalid_argument("getrf", code);
  for (std::size_t k = 0; k < pivots.size(); ++k)
    pivots[k] = std::size_t(provider_pivots[k] - 1);
  if (!detail::solve_matrix_is_finite(s))
    info.status = SolveStatus::nonfinite_result;
  else if (code > 0)
    info = {.status = SolveStatus::singular, .pivot = std::size_t(code - 1)};
  else
    detail::check_lu_diagonal(s, scale, options.relative_pivot_tolerance, info);
  return KernelAttempt::success;
}

/// \brief Accept directly addressable read-only factors and a writable RHS.
template <RankedStridedMdspecLike<2> A, MutableRankedStridedMdspecLike<2> B>
  requires HostReadableMdspec<A> && HostWritableMdspec<B> && LapackScalar<typename A::value_type> &&
           std::same_as<typename A::value_type, typename B::value_type> &&
           DefaultAccessorMdspanLike<host_read_mdspan_t<A>> && DefaultAccessorMdspanLike<host_write_mdspan_t<B>>
consteval auto kernel_accepts_types(LapackBackend, lu_solve_op const&, A&, std::span<std::size_t const>, B&, SolveInfo&)
{
  return kernel_types_maybe;
}

/// \brief Apply GETRS without modifying the factors.
/// \pre Factors and pivots describe a successful LU and do not overlap the RHS.
template <RankedStridedMdspecLike<2> A, MutableRankedStridedMdspecLike<2> B>
  requires HostReadableMdspec<A> && HostWritableMdspec<B> && LapackScalar<typename A::value_type> &&
           std::same_as<typename A::value_type, typename B::value_type> &&
           DefaultAccessorMdspanLike<host_read_mdspan_t<A>> && DefaultAccessorMdspanLike<host_write_mdspan_t<B>>
KernelAttempt try_kernel(LapackBackend, lu_solve_op const&, A& a, std::span<std::size_t const> pivots, B& b,
                         SolveInfo& info)
{
  detail::require_lu_solve_shape(a, pivots, b);
  auto n = uni20::blas::try_blas_int(a.extent(0));
  auto nrhs = uni20::blas::try_blas_int(b.extent(1));
  if (!uni20::blas::is_valid_blas_int(n) || !uni20::blas::is_valid_blas_int(nrhs))
    return KernelAttempt::unsupported_shape;
  if (n == 0 || nrhs == 0)
  {
    info = {};
    return KernelAttempt::success;
  }
  auto aa = acquire_host_read_access_sync(a);
  auto ba = acquire_host_write_access_sync(b);
  auto as = aa.mdspan();
  auto bs = ba.mdspan();
  auto matrix = blas::try_blas_readable_matrix(as);
  auto rhs = blas::try_lapack_writable_matrix(bs);
  if (!matrix || matrix->transform != blas::MatrixTransform::normal || !rhs) return KernelAttempt::unsupported_layout;
  std::vector<blas_int> provider_pivots(pivots.size());
  for (std::size_t k = 0; k < pivots.size(); ++k)
    provider_pivots[k] = blas_int(pivots[k] + 1);
  info = {};
  if (!detail::solve_matrix_is_finite(bs))
  {
    info.status = SolveStatus::nonfinite_input;
    return KernelAttempt::success;
  }
  // GETRS's factor argument is input-only. The legacy provider declaration
  // lacks const; this cast is confined to that boundary, under a read lease.
  using S = typename A::value_type;
  uni20::lapack::getrs('N', n, nrhs, const_cast<S*>(matrix->data), matrix->leading_dimension, provider_pivots.data(),
                       rhs->data, rhs->leading_dimension);
  if (!detail::solve_matrix_is_finite(bs)) info.status = SolveStatus::nonfinite_result;
  return KernelAttempt::success;
}
} // namespace uni20::linalg
