#pragma once

#include "linear_solve.hpp"
#include <uni20/linalg/backends/lu_common.hpp>

namespace uni20::linalg
{
/// \brief Accept native real/complex packed LU on mutable logical host elements.
template <MutableRankedMdspecLike<2> A>
  requires HostWritableMdspec<A> && RealOrComplex<typename A::value_type> &&
           (!has_runtime_precision_v<typename A::value_type>)
consteval auto kernel_accepts_types(CpuReferenceBackend, lu_factor_op const&, A&, std::span<std::size_t>, SolveInfo&,
                                    SolveOptions<make_real_t<typename A::value_type>> const&)
{
  return kernel_types_yes;
}

/// \brief Factor with partial row pivoting, retaining L multipliers below U.
template <MutableRankedMdspecLike<2> A>
  requires HostWritableMdspec<A> && RealOrComplex<typename A::value_type> &&
           (!has_runtime_precision_v<typename A::value_type>)
KernelAttempt try_kernel(CpuReferenceBackend, lu_factor_op const&, A& a, std::span<std::size_t> pivots, SolveInfo& info,
                         SolveOptions<make_real_t<typename A::value_type>> const& options)
{
  detail::require_lu_shape(a, pivots.size());
  detail::require_solve_options(options);
  auto access = acquire_host_write_access_sync(a);
  auto s = access.mdspan();
  using S = typename A::value_type;
  using R = make_real_t<S>;
  info = {};
  if (!detail::solve_matrix_is_finite(s))
  {
    info.status = SolveStatus::nonfinite_input;
    return KernelAttempt::success;
  }
  auto scale = detail::lu_input_scale(s, R{});
  auto n = pivots.size();
  for (std::size_t k = 0; k < n; ++k)
  {
    auto magnitude = detail::lu_magnitude(S(s[k, k]));
    std::size_t pivot = k;
    for (std::size_t i = k + 1; i < n; ++i)
    {
      auto candidate = detail::lu_magnitude(S(s[i, k]));
      if (detail::lu_magnitude_less(magnitude, candidate))
      {
        magnitude = std::move(candidate);
        pivot = i;
      }
    }
    if (magnitude.scale == 0)
      info = {.status = SolveStatus::singular, .pivot = k};
    else if (detail::lu_small_pivot(magnitude, scale, options.relative_pivot_tolerance))
      info = {.status = SolveStatus::small_pivot, .pivot = k};
    if (!info.succeeded()) return KernelAttempt::success;
    pivots[k] = pivot;
    // Previously computed columns of L must participate in the row swap.
    detail::cpu_reference::swap_rows(s, k, pivot, n);
    for (std::size_t i = k + 1; i < n; ++i)
    {
      S multiplier = S(s[i, k]) / S(s[k, k]);
      if (!uni20::isfinite(multiplier))
      {
        info.status = SolveStatus::nonfinite_result;
        return KernelAttempt::success;
      }
      s[i, k] = multiplier;
      for (std::size_t j = k + 1; j < n; ++j)
      {
        S value = S(s[i, j]) - multiplier * S(s[k, j]);
        if (!uni20::isfinite(value))
        {
          info.status = SolveStatus::nonfinite_result;
          return KernelAttempt::success;
        }
        s[i, j] = value;
      }
    }
  }
  return KernelAttempt::success;
}

/// \brief Accept native host solves from read-only factors.
template <RankedMdspecLike<2> A, MutableRankedMdspecLike<2> B>
  requires HostReadableMdspec<A> && HostWritableMdspec<B> && RealOrComplex<typename A::value_type> &&
           std::same_as<typename A::value_type, typename B::value_type> &&
           (!has_runtime_precision_v<typename A::value_type>)
consteval auto kernel_accepts_types(CpuReferenceBackend, lu_solve_op const&, A&, std::span<std::size_t const>, B&,
                                    SolveInfo&)
{
  return kernel_types_yes;
}

/// \brief Apply stored row swaps, forward substitution and back substitution.
/// \pre Factors and pivots describe a successful LU and do not overlap the RHS.
template <RankedMdspecLike<2> A, MutableRankedMdspecLike<2> B>
  requires HostReadableMdspec<A> && HostWritableMdspec<B> && RealOrComplex<typename A::value_type> &&
           std::same_as<typename A::value_type, typename B::value_type> &&
           (!has_runtime_precision_v<typename A::value_type>)
KernelAttempt try_kernel(CpuReferenceBackend, lu_solve_op const&, A& a, std::span<std::size_t const> pivots, B& b,
                         SolveInfo& info)
{
  detail::require_lu_solve_shape(a, pivots, b);
  info = {};
  if (a.extent(0) == 0 || b.extent(1) == 0) return KernelAttempt::success;
  auto aa = acquire_host_read_access_sync(a);
  auto ba = acquire_host_write_access_sync(b);
  auto as = aa.mdspan();
  auto bs = ba.mdspan();
  using S = typename A::value_type;
  if (!detail::solve_matrix_is_finite(bs))
  {
    info.status = SolveStatus::nonfinite_input;
    return KernelAttempt::success;
  }
  auto n = pivots.size();
  auto count = std::size_t(bs.extent(1));
  for (std::size_t k = 0; k < n; ++k)
    detail::cpu_reference::swap_rows(bs, k, pivots[k], count);
  for (std::size_t col = 0; col < count; ++col)
  {
    for (std::size_t i = 0; i < n; ++i)
    {
      S value = bs[i, col];
      for (std::size_t k = 0; k < i; ++k)
        value -= S(as[i, k]) * S(bs[k, col]);
      if (!uni20::isfinite(value))
      {
        info.status = SolveStatus::nonfinite_result;
        return KernelAttempt::success;
      }
      bs[i, col] = value;
    }
    for (std::size_t i = n; i-- > 0;)
    {
      S value = bs[i, col];
      for (std::size_t k = i + 1; k < n; ++k)
        value -= S(as[i, k]) * S(bs[k, col]);
      value /= S(as[i, i]);
      if (!uni20::isfinite(value))
      {
        info.status = SolveStatus::nonfinite_result;
        return KernelAttempt::success;
      }
      bs[i, col] = value;
    }
  }
  return KernelAttempt::success;
}
} // namespace uni20::linalg
