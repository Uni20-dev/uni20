#pragma once

#include "binary80.hpp"
#include <uni20/linalg/backends/lu_common.hpp>

namespace uni20::linalg
{
/// \brief Accept logical host workspaces for native binary80 LU.
template <MutableRankedMdspecLike<2> A>
  requires HostWritableMdspec<A> && mplapack_binary80_detail::Binary80Scalar<typename A::value_type>
consteval auto kernel_accepts_types(MplapackBinary80Backend, lu_factor_op const&, A&, std::span<std::size_t>,
                                    SolveInfo&, SolveOptions<float80> const&)
{
  return kernel_types_yes;
}

/// \brief Convert values at the backend boundary and compute portable owning factors.
template <MutableRankedMdspecLike<2> A>
  requires HostWritableMdspec<A> && mplapack_binary80_detail::Binary80Scalar<typename A::value_type>
KernelAttempt try_kernel(MplapackBinary80Backend, lu_factor_op const&, A& a, std::span<std::size_t> pivots,
                         SolveInfo& info, SolveOptions<float80> const& options)
{
  detail::require_lu_shape(a, pivots.size());
  detail::require_solve_options(options);
  auto access = acquire_host_write_access_sync(a);
  auto s = access.mdspan();
  info = {};
  if (!detail::solve_matrix_is_finite(s))
  {
    info.status = SolveStatus::nonfinite_input;
    return KernelAttempt::success;
  }
  auto work = mplapack_binary80_detail::pack(s);
  auto scale = detail::lu_input_scale(work.mdspan(), float80{0});
  auto tolerance = options.relative_pivot_tolerance;
  std::vector<std::size_t> provider_pivots;
  auto singular = mplapack::getrf(pivots.size(), mplapack_binary80_detail::values(work), provider_pivots);
  mplapack_binary80_detail::unpack(work, s);
  std::copy(provider_pivots.begin(), provider_pivots.end(), pivots.begin());
  if (!detail::solve_matrix_is_finite(work.mdspan()))
    info.status = SolveStatus::nonfinite_result;
  else if (singular)
    info = {.status = SolveStatus::singular, .pivot = singular - 1};
  else
    detail::check_lu_diagonal(work.mdspan(), scale, tolerance, info);
  return KernelAttempt::success;
}

/// \brief Accept read-only binary80 factors and logical host RHS elements.
template <RankedMdspecLike<2> A, MutableRankedMdspecLike<2> B>
  requires HostReadableMdspec<A> && HostWritableMdspec<B> &&
           mplapack_binary80_detail::Binary80Scalar<typename A::value_type> &&
           std::same_as<typename A::value_type, typename B::value_type>
consteval auto kernel_accepts_types(MplapackBinary80Backend, lu_solve_op const&, A&, std::span<std::size_t const>, B&,
                                    SolveInfo&)
{
  return kernel_types_yes;
}

/// \brief Solve using binary80 arithmetic; RHS failures leave the factors reusable.
/// \pre Factors and pivots describe a successful LU and do not overlap the RHS.
template <RankedMdspecLike<2> A, MutableRankedMdspecLike<2> B>
  requires HostReadableMdspec<A> && HostWritableMdspec<B> &&
           mplapack_binary80_detail::Binary80Scalar<typename A::value_type> &&
           std::same_as<typename A::value_type, typename B::value_type>
KernelAttempt try_kernel(MplapackBinary80Backend, lu_solve_op const&, A& a, std::span<std::size_t const> pivots, B& b,
                         SolveInfo& info)
{
  detail::require_lu_solve_shape(a, pivots, b);
  info = {};
  if (a.extent(0) == 0 || b.extent(1) == 0) return KernelAttempt::success;
  auto aa = acquire_host_read_access_sync(a);
  auto ba = acquire_host_write_access_sync(b);
  auto as = aa.mdspan();
  auto bs = ba.mdspan();
  if (!detail::solve_matrix_is_finite(bs))
  {
    info.status = SolveStatus::nonfinite_input;
    return KernelAttempt::success;
  }
  auto av = mplapack_binary80_detail::pack(as);
  auto bv = mplapack_binary80_detail::pack(bs);
  mplapack::getrs(pivots.size(), b.extent(1), mplapack_binary80_detail::values(std::as_const(av)), pivots,
                  mplapack_binary80_detail::values(bv));
  mplapack_binary80_detail::unpack(bv, bs);
  if (!detail::solve_matrix_is_finite(bv.mdspan())) info.status = SolveStatus::nonfinite_result;
  return KernelAttempt::success;
}
} // namespace uni20::linalg
