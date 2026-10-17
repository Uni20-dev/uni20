#pragma once

#include <uni20/backend/mplapack/mpfr.hpp>
#include <uni20/linalg/backends/mpfr_common.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/access.hpp>

namespace uni20::linalg
{
/// \brief Accept host-readable runtime-precision operands, including transforming accessors.
template <MutableRankedMdspecLike<2> C, mpfr_detail::RuntimeScalar S, RankedMdspecLike<2> A, RankedMdspecLike<2> B>
  requires HostWritableMdspec<C> && HostReadableMdspec<A> && HostReadableMdspec<B> &&
           std::same_as<typename C::value_type, S> && std::same_as<typename A::value_type, S> &&
           std::same_as<typename B::value_type, S>
consteval auto kernel_accepts_types(BlasBackend, gemm_op const&, C&, S const&, A&, B&, S const&, Precision)
{
  return kernel_types_maybe;
}

/// \brief Pack logical host values and execute GEMM at one explicit precision.
template <MutableRankedMdspecLike<2> C, mpfr_detail::RuntimeScalar S, RankedMdspecLike<2> A, RankedMdspecLike<2> B>
  requires HostWritableMdspec<C> && HostReadableMdspec<A> && HostReadableMdspec<B> &&
           std::same_as<typename C::value_type, S> && std::same_as<typename A::value_type, S> &&
           std::same_as<typename B::value_type, S>
KernelAttempt try_kernel(BlasBackend, gemm_op const&, C& c, S const& alpha, A& a, B& b, S const& beta, Precision p)
{
  if (p.is_exact()) return KernelAttempt::unsupported_instance;
  ERROR_IF(a.extent(1) != b.extent(0) || c.extent(0) != a.extent(0) || c.extent(1) != b.extent(1),
           "GEMM operand shapes do not agree");
  if (c.extent(0) == 0 || c.extent(1) == 0) return KernelAttempt::success;
  auto ca = acquire_host_write_access_sync(c);
  auto aa = acquire_host_read_access_sync(a);
  auto ba = acquire_host_read_access_sync(b);
  auto cs = ca.mdspan();
  auto as = aa.mdspan();
  auto bs = ba.mdspan();
  auto av = mpfr_detail::pack(as, p, alpha != 0 && a.extent(1) != 0);
  auto bv = mpfr_detail::pack(bs, p, alpha != 0 && a.extent(1) != 0);
  auto cv = mpfr_detail::pack(cs, p, beta != 0);
  mplapack::gemm(a.extent(0), b.extent(1), a.extent(1), alpha, mpfr_detail::values(av), mpfr_detail::values(bv), beta,
                 mpfr_detail::values(cv), p);
  mpfr_detail::unpack(cv, cs);
  return KernelAttempt::success;
}

} // namespace uni20::linalg
