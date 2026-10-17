#pragma once

#include <uni20/linalg/backends/binary80_common.hpp>
#include <uni20/linalg/backends/linear_solve_common.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/access.hpp>

namespace uni20::linalg
{
/// \brief Accept mutable host operands for a binary80 LU solve.
template <MutableRankedMdspecLike<2> A, MutableRankedMdspecLike<2> B>
  requires HostWritableMdspec<A> && HostWritableMdspec<B> &&
           mplapack_binary80_detail::Binary80LuScalar<typename A::value_type> &&
           std::same_as<typename A::value_type, typename B::value_type>
consteval auto kernel_accepts_types(LapackBackend, linear_solve_op const&, A&, B&, SolveInfo&,
                                    SolveOptions<float80> const&)
{
  return kernel_types_yes;
}

/// \brief Convert workspaces at the boundary and report LU solve diagnostics without double-precision intermediates.
template <MutableRankedMdspecLike<2> A, MutableRankedMdspecLike<2> B>
  requires HostWritableMdspec<A> && HostWritableMdspec<B> &&
           mplapack_binary80_detail::Binary80LuScalar<typename A::value_type> &&
           std::same_as<typename A::value_type, typename B::value_type>
KernelAttempt try_kernel(LapackBackend, linear_solve_op const&, A& a, B& b, SolveInfo& info,
                         SolveOptions<float80> const& options)
{
  ERROR_IF(a.extent(0) != a.extent(1) || a.extent(0) != b.extent(0), "solve operand shapes do not agree");
  detail::require_solve_options(options);
  info = {};
  if (a.extent(0) == 0 || b.extent(1) == 0) return KernelAttempt::success;
  auto aa = acquire_host_write_access_sync(a);
  auto ba = acquire_host_write_access_sync(b);
  auto as = aa.mdspan();
  auto bs = ba.mdspan();
  float80 scale{};
  if (!detail::prepare_linear_solve(as, bs, scale, info)) return KernelAttempt::success;
  auto av = mplapack_binary80_detail::pack(as), bv = mplapack_binary80_detail::pack(bs);
  auto tolerance = options.relative_pivot_tolerance;
  std::vector<std::size_t> pivots;
  auto singular = mplapack::getrf(a.extent(0), mplapack_binary80_detail::values(av), pivots);
  mplapack_binary80_detail::unpack(av, as);
  if (!detail::solve_matrix_is_finite(av.mdspan()))
    info.status = SolveStatus::nonfinite_result;
  else if (singular)
    info = {.status = SolveStatus::singular, .pivot = singular - 1};
  else
  {
    for (std::size_t i = 0; i < std::size_t(a.extent(0)); ++i)
    {
      auto magnitude = math::abs(av[i, i]);
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
      mplapack::getrs(a.extent(0), b.extent(1), mplapack_binary80_detail::values(av), pivots,
                      mplapack_binary80_detail::values(bv));
      mplapack_binary80_detail::unpack(bv, bs);
      if (!detail::solve_matrix_is_finite(bv.mdspan())) info.status = SolveStatus::nonfinite_result;
    }
  }
  return KernelAttempt::success;
}
} // namespace uni20::linalg
