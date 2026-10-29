#pragma once

#include "lu_common.hpp"
#include <uni20/linalg/backends/packed_matrix.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/access.hpp>

namespace uni20::linalg
{
namespace detail::cpu_reference
{
template <class S, class... P>
concept LuPrecisionArgs = (has_runtime_precision_v<S> && sizeof...(P) == 1 && (std::same_as<P, Precision> && ...)) ||
                          (!has_runtime_precision_v<S> && sizeof...(P) == 0);
} // namespace detail::cpu_reference

/// \brief Accept real/complex LU on mutable logical host elements.
template <MutableRankedMdspecLike<2> A, class... P>
  requires HostWritableMdspec<A> && RealOrComplex<typename A::value_type> &&
           detail::cpu_reference::LuPrecisionArgs<typename A::value_type, P...>
consteval auto kernel_accepts_types(CpuReferenceBackend, lu_factor_op const&, A&, std::span<std::size_t>,
                                    LuNorms<make_real_t<typename A::value_type>>*, SolveInfo&,
                                    SolveOptions<make_real_t<typename A::value_type>> const&, P...)
{
  return kernel_types_yes;
}

/// \brief Factor with partial row pivoting, converting runtime values to finite working precision.
template <MutableRankedMdspecLike<2> A, class... P>
  requires HostWritableMdspec<A> && RealOrComplex<typename A::value_type> &&
           detail::cpu_reference::LuPrecisionArgs<typename A::value_type, P...>
KernelAttempt try_kernel(CpuReferenceBackend, lu_factor_op const&, A& a, std::span<std::size_t> pivots,
                         LuNorms<make_real_t<typename A::value_type>>* norms, SolveInfo& info,
                         SolveOptions<make_real_t<typename A::value_type>> const& options, P... precision)
{
  detail::require_lu_shape(a, pivots.size());
  detail::require_solve_options(options);
  using S = typename A::value_type;
  using R = make_real_t<S>;
  auto access = acquire_host_write_access_sync(a);
  auto s = access.mdspan();
  if constexpr (has_runtime_precision_v<S>)
  {
    Precision p(precision...);
    (void)p.bit_count();
    info = {};
    if (!detail::solve_matrix_is_finite(s))
    {
      info.status = SolveStatus::nonfinite_input;
      return KernelAttempt::success;
    }
    auto work = packed_detail::pack(s, p);
    auto ws = work.mdspan();
    if (!detail::solve_matrix_is_finite(ws))
    {
      info.status = SolveStatus::nonfinite_result;
      return KernelAttempt::success;
    }
    SolveOptions<R> converted{.relative_pivot_tolerance = scalar_at_precision<R>(options.relative_pivot_tolerance, p)};
    detail::cpu_reference::lu_factor(ws, pivots, norms, info, converted, scalar_at_precision<R>(0, p));
    packed_detail::unpack(work, s);
    return KernelAttempt::success;
  }
  else
    return detail::cpu_reference::lu_factor(s, pivots, norms, info, options, R{});
}

/// \brief Accept host solves from read-only factors at their working precision.
template <RankedMdspecLike<2> A, MutableRankedMdspecLike<2> B, class... P>
  requires HostReadableMdspec<A> && HostWritableMdspec<B> && RealOrComplex<typename A::value_type> &&
           std::same_as<typename A::value_type, typename B::value_type> &&
           detail::cpu_reference::LuPrecisionArgs<typename A::value_type, P...>
consteval auto kernel_accepts_types(CpuReferenceBackend, lu_solve_op const&, A&, std::span<std::size_t const>, B&,
                                    SolveInfo&, P...)
{
  return kernel_types_yes;
}

/// \brief Apply row swaps and triangular substitution, converting runtime RHS values to factor precision.
/// \pre Factors and pivots describe a successful LU and do not overlap the RHS.
template <RankedMdspecLike<2> A, MutableRankedMdspecLike<2> B, class... P>
  requires HostReadableMdspec<A> && HostWritableMdspec<B> && RealOrComplex<typename A::value_type> &&
           std::same_as<typename A::value_type, typename B::value_type> &&
           detail::cpu_reference::LuPrecisionArgs<typename A::value_type, P...>
KernelAttempt try_kernel(CpuReferenceBackend, lu_solve_op const&, A& a, std::span<std::size_t const> pivots, B& b,
                         SolveInfo& info, P... precision)
{
  detail::require_lu_solve_shape(a, pivots, b);
  using S = typename A::value_type;
  if constexpr (has_runtime_precision_v<S>) (void)Precision(precision...).bit_count();
  info = {};
  if (a.extent(0) == 0 || b.extent(1) == 0) return KernelAttempt::success;
  auto aa = acquire_host_read_access_sync(a);
  auto ba = acquire_host_write_access_sync(b);
  auto as = aa.mdspan();
  auto bs = ba.mdspan();
  if constexpr (has_runtime_precision_v<S>)
  {
    Precision p(precision...);
    if (!detail::solve_matrix_is_finite(bs))
    {
      info.status = SolveStatus::nonfinite_input;
      return KernelAttempt::success;
    }
    // The factor contract already establishes values at p; only RHS values
    // need conversion. Reading through the accessor preserves view semantics.
    auto work = packed_detail::pack(bs, p);
    auto ws = work.mdspan();
    if (!detail::solve_matrix_is_finite(ws))
    {
      info.status = SolveStatus::nonfinite_result;
      return KernelAttempt::success;
    }
    detail::cpu_reference::lu_solve(as, pivots, ws, info);
    packed_detail::unpack(work, bs);
    return KernelAttempt::success;
  }
  else
    return detail::cpu_reference::lu_solve(as, pivots, bs, info);
}
} // namespace uni20::linalg
