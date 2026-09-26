#pragma once

/**
 * \file linear_solve.hpp
 * \ingroup linalg
 * \brief LAPACK backend for dense general linear systems.
 */

#include "common.hpp"

#include <uni20/backend/lapack/lapack.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/backends/linear_solve_common.hpp>
#include <uni20/linalg/blas/mdspan_matrix.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/mdspan/concepts.hpp>
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/concepts.hpp>

#include <concepts>
#include <cstddef>
#include <type_traits>
#include <vector>

namespace uni20::linalg
{
namespace lapack_detail
{

template <class CoefficientMdspan, class RhsMdspan> consteval auto linear_solve_acceptance()
{
  using coefficient_scalar = std::remove_cv_t<typename CoefficientMdspan::element_type>;
  using rhs_scalar = std::remove_cv_t<typename RhsMdspan::element_type>;
  if constexpr (uni20::LapackScalar<coefficient_scalar> && std::same_as<coefficient_scalar, rhs_scalar> &&
                uni20::DefaultAccessorMdspanLike<CoefficientMdspan> && uni20::DefaultAccessorMdspanLike<RhsMdspan>)
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

template <uni20::MutableRankedStridedMdspanLike<2> CoefficientMdspan,
          uni20::MutableRankedStridedMdspanLike<2> RhsMdspan>
KernelAttempt try_linear_solve(CoefficientMdspan& coefficients, RhsMdspan& right_hand_sides, SolveInfo& info,
                               SolveOptions<uni20::make_real_t<typename CoefficientMdspan::value_type>> const& options)
{
  CHECK_EQUAL(coefficients.extent(0), coefficients.extent(1));
  CHECK_EQUAL(coefficients.extent(0), right_hand_sides.extent(0));
  std::size_t const order_size = static_cast<std::size_t>(coefficients.extent(0));
  std::size_t const rhs_count_size = static_cast<std::size_t>(right_hand_sides.extent(1));
  detail::require_solve_options(options);
  if (order_size == 0 || rhs_count_size == 0)
  {
    info = {};
    return KernelAttempt::success;
  }

  blas_int const order = uni20::blas::try_blas_int(order_size);
  blas_int const rhs_count = uni20::blas::try_blas_int(rhs_count_size);
  if (!uni20::blas::is_valid_blas_int(order) || !uni20::blas::is_valid_blas_int(rhs_count))
    return KernelAttempt::unsupported_shape;

  auto coefficient_matrix = uni20::linalg::blas::try_lapack_writable_matrix(coefficients);
  auto rhs_matrix = uni20::linalg::blas::try_lapack_writable_matrix(right_hand_sides);
  if (!coefficient_matrix || !rhs_matrix) return KernelAttempt::unsupported_layout;

  CHECK_EQUAL(coefficient_matrix->rows, order);
  CHECK_EQUAL(coefficient_matrix->cols, order);
  CHECK_EQUAL(rhs_matrix->rows, order);
  CHECK_EQUAL(rhs_matrix->cols, rhs_count);

  std::vector<blas_int> pivots(order_size);
  using scalar_type = typename CoefficientMdspan::value_type;
  using real_type = uni20::make_real_t<scalar_type>;
  real_type scale{};
  if (!detail::prepare_linear_solve(coefficients, right_hand_sides, scale, info)) return KernelAttempt::success;

  auto const provider_info = uni20::lapack::unchecked::getrf(order, order, coefficient_matrix->data,
                                                             coefficient_matrix->leading_dimension, pivots.data());
  uni20::lapack::detail::check_invalid_argument("getrf", provider_info);
  if (!detail::solve_matrix_is_finite(coefficients))
  {
    info.status = SolveStatus::nonfinite_result;
    return KernelAttempt::success;
  }
  if (provider_info > 0)
  {
    info = {.status = SolveStatus::singular, .pivot = static_cast<std::size_t>(provider_info - 1)};
    return KernelAttempt::success;
  }
  using std::abs;
  for (typename CoefficientMdspan::index_type k = 0; k < coefficients.extent(0); ++k)
    if (!detail::check_solve_pivot(real_type(abs(coefficients[k, k])), scale, options, std::size_t(k), info))
      return KernelAttempt::success;

  uni20::lapack::getrs('N', order, rhs_count, coefficient_matrix->data, coefficient_matrix->leading_dimension,
                       pivots.data(), rhs_matrix->data, rhs_matrix->leading_dimension);
  if (!detail::solve_matrix_is_finite(right_hand_sides)) info.status = SolveStatus::nonfinite_result;
  return KernelAttempt::success;
}

} // namespace lapack_detail

/// \brief Report eligibility for a host-accessible LAPACK general solve.
template <uni20::MutableRankedStridedMdspecLike<2> CoefficientMdspec,
          uni20::MutableRankedStridedMdspecLike<2> RhsMdspec>
  requires uni20::HostWritableMdspec<CoefficientMdspec> && uni20::HostWritableMdspec<RhsMdspec>
consteval auto kernel_accepts_types(LapackBackend const&, linear_solve_op const&, CoefficientMdspec&, RhsMdspec&,
                                    SolveInfo&,
                                    SolveOptions<uni20::make_real_t<typename CoefficientMdspec::value_type>> const&)
{
  using coefficient_span = uni20::host_write_mdspan_t<CoefficientMdspec>;
  using rhs_span = uni20::host_write_mdspan_t<RhsMdspec>;
  constexpr auto acceptance = lapack_detail::linear_solve_acceptance<coefficient_span, rhs_span>();
  if constexpr (acceptance == KernelTypeAcceptance::no)
    return kernel_types_no;
  else
    return kernel_types_maybe;
}

/// \brief Resolve host access and solve a general system through LAPACK LU factorization.
template <uni20::MutableRankedStridedMdspecLike<2> CoefficientMdspec,
          uni20::MutableRankedStridedMdspecLike<2> RhsMdspec>
  requires uni20::HostWritableMdspec<CoefficientMdspec> && uni20::HostWritableMdspec<RhsMdspec>
KernelAttempt try_kernel(LapackBackend, linear_solve_op const&, CoefficientMdspec& coefficients,
                         RhsMdspec& right_hand_sides, SolveInfo& info,
                         SolveOptions<uni20::make_real_t<typename CoefficientMdspec::value_type>> const& options)
{
  return lapack_detail::with_host_write_mdspans(
      [&](auto& coefficient_span, auto& rhs_span) {
        return lapack_detail::try_linear_solve(coefficient_span, rhs_span, info, options);
      },
      coefficients, right_hand_sides);
}

} // namespace uni20::linalg
