#pragma once

#include <uni20/backend/mplapack/mpfr.hpp>
#include <uni20/linalg/backends/linear_solve_common.hpp>
#include <uni20/linalg/backends/mpfr_common.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/access.hpp>

namespace uni20::linalg
{
/// \brief Accept mutable host operands for an arbitrary-precision LU solve.
template <MutableRankedMdspecLike<2> A, MutableRankedMdspecLike<2> B>
  requires HostWritableMdspec<A> && HostWritableMdspec<B> && mpfr_detail::RuntimeScalar<typename A::value_type> &&
           std::same_as<typename A::value_type, typename B::value_type>
consteval auto kernel_accepts_types(LapackBackend, linear_solve_op const&, A&, B&, SolveInfo&,
                                    SolveOptions<mpreal> const&, Precision)
{
  return kernel_types_yes;
}

/// \brief Convert workspaces at the boundary and report LU solve diagnostics without native-float intermediates.
template <MutableRankedMdspecLike<2> A, MutableRankedMdspecLike<2> B>
  requires HostWritableMdspec<A> && HostWritableMdspec<B> && mpfr_detail::RuntimeScalar<typename A::value_type> &&
           std::same_as<typename A::value_type, typename B::value_type>
KernelAttempt try_kernel(LapackBackend, linear_solve_op const&, A& a, B& b, SolveInfo& info,
                         SolveOptions<mpreal> const& options, Precision p)
{
  (void)p.bit_count(); // Numerical provider kernels require finite working precision.
  ERROR_IF(a.extent(0) != a.extent(1) || a.extent(0) != b.extent(0), "solve operand shapes do not agree");
  detail::require_solve_options(options);
  info = {};
  if (a.extent(0) == 0 || b.extent(1) == 0) return KernelAttempt::success;
  auto aa = acquire_host_write_access_sync(a);
  auto ba = acquire_host_write_access_sync(b);
  auto as = aa.mdspan();
  auto bs = ba.mdspan();
  if (!detail::solve_matrix_is_finite(as) || !detail::solve_matrix_is_finite(bs))
  {
    info.status = SolveStatus::nonfinite_input;
    return KernelAttempt::success;
  }
  auto av = mpfr_detail::pack(as, p), bv = mpfr_detail::pack(bs, p);
  mpreal scale(0, p);
  if (!detail::prepare_linear_solve(av.mdspan(), bv.mdspan(), scale, info))
  {
    // The original inputs were finite; failure arose during boundary conversion.
    info.status = SolveStatus::nonfinite_result;
    return KernelAttempt::success;
  }
  auto tolerance = options.relative_pivot_tolerance.at(p);
  std::vector<std::size_t> pivots;
  auto singular = mplapack::getrf(a.extent(0), mpfr_detail::values(av), pivots, p);
  mpfr_detail::unpack(av, as);
  if (!detail::solve_matrix_is_finite(av.mdspan()))
    info.status = SolveStatus::nonfinite_result;
  else if (singular)
    info = {.status = SolveStatus::singular, .pivot = singular - 1};
  else
  {
    for (std::size_t i = 0; i < std::size_t(a.extent(0)); ++i)
    {
      auto magnitude = uni20::abs(av[i, i]);
      if (!uni20::isfinite(magnitude))
      {
        info.status = SolveStatus::nonfinite_result;
        break;
      }
      if (tolerance > 0 && magnitude / scale <= tolerance)
      {
        info = {.status = SolveStatus::small_pivot, .pivot = i};
        break;
      }
    }
    if (info.succeeded())
    {
      mplapack::getrs(a.extent(0), b.extent(1), mpfr_detail::values(av), pivots, mpfr_detail::values(bv), p);
      mpfr_detail::unpack(bv, bs);
      if (!detail::solve_matrix_is_finite(bv.mdspan())) info.status = SolveStatus::nonfinite_result;
    }
  }
  return KernelAttempt::success;
}
} // namespace uni20::linalg
