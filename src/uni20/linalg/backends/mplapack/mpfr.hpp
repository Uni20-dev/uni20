#pragma once

#include <uni20/backend/mplapack/mpfr.hpp>
#include <uni20/linalg/backends/linear_solve_common.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/tensor.hpp>

namespace uni20::linalg
{
namespace mplapack_detail
{
template <class T>
concept RuntimeScalar = std::same_as<T, mpreal> || std::same_as<T, mpcomplex>;

template <class Span> auto pack(Span const& span, Precision p, bool read = true)
{
  using S = typename Span::value_type;
  DenseMatrix<S> result(uninitialized, span.extent(0), span.extent(1), p);
  if (read)
    for (std::size_t j = 0; j < std::size_t(span.extent(1)); ++j)
      for (std::size_t i = 0; i < std::size_t(span.extent(0)); ++i)
        result[i, j] = S(span[i, j]).at(p);
  return result;
}
template <class Tensor> auto values(Tensor& tensor)
{
  return std::span(tensor.storage().data(), tensor.storage().size());
}
template <class Tensor, class Span> void unpack(Tensor const& source, Span& target)
{
  for (std::size_t j = 0; j < std::size_t(target.extent(1)); ++j)
    for (std::size_t i = 0; i < std::size_t(target.extent(0)); ++i)
      target[i, j] = source[i, j];
}
} // namespace mplapack_detail

/// \brief Accept host-readable runtime-precision operands, including transforming accessors.
template <MutableRankedMdspecLike<2> C, mplapack_detail::RuntimeScalar S, RankedMdspecLike<2> A, RankedMdspecLike<2> B>
  requires HostWritableMdspec<C> && HostReadableMdspec<A> && HostReadableMdspec<B> &&
           std::same_as<typename C::value_type, S> && std::same_as<typename A::value_type, S> &&
           std::same_as<typename B::value_type, S>
consteval auto kernel_accepts_types(MplapackMpfrBackend, gemm_op const&, C&, S const&, A&, B&, S const&, Precision)
{
  return kernel_types_yes;
}

/// \brief Pack logical host values and execute GEMM at one explicit precision.
template <MutableRankedMdspecLike<2> C, mplapack_detail::RuntimeScalar S, RankedMdspecLike<2> A, RankedMdspecLike<2> B>
  requires HostWritableMdspec<C> && HostReadableMdspec<A> && HostReadableMdspec<B> &&
           std::same_as<typename C::value_type, S> && std::same_as<typename A::value_type, S> &&
           std::same_as<typename B::value_type, S>
KernelAttempt try_kernel(MplapackMpfrBackend, gemm_op const&, C& c, S const& alpha, A& a, B& b, S const& beta,
                         Precision p)
{
  ERROR_IF(a.extent(1) != b.extent(0) || c.extent(0) != a.extent(0) || c.extent(1) != b.extent(1),
           "GEMM operand shapes do not agree");
  if (c.extent(0) == 0 || c.extent(1) == 0) return KernelAttempt::success;
  auto ca = acquire_host_write_access_sync(c);
  auto aa = acquire_host_read_access_sync(a);
  auto ba = acquire_host_read_access_sync(b);
  auto cs = ca.mdspan();
  auto as = aa.mdspan();
  auto bs = ba.mdspan();
  auto av = mplapack_detail::pack(as, p, alpha != 0 && a.extent(1) != 0);
  auto bv = mplapack_detail::pack(bs, p, alpha != 0 && a.extent(1) != 0);
  auto cv = mplapack_detail::pack(cs, p, beta != 0);
  mplapack::gemm(a.extent(0), b.extent(1), a.extent(1), alpha, mplapack_detail::values(av), mplapack_detail::values(bv),
                 beta, mplapack_detail::values(cv), p);
  mplapack_detail::unpack(cv, cs);
  return KernelAttempt::success;
}

/// \brief Accept mutable host operands for an arbitrary-precision LU solve.
template <MutableRankedMdspecLike<2> A, MutableRankedMdspecLike<2> B>
  requires HostWritableMdspec<A> && HostWritableMdspec<B> && mplapack_detail::RuntimeScalar<typename A::value_type> &&
           std::same_as<typename A::value_type, typename B::value_type>
consteval auto kernel_accepts_types(MplapackMpfrBackend, linear_solve_op const&, A&, B&, SolveInfo&,
                                    SolveOptions<mpreal> const&, Precision)
{
  return kernel_types_yes;
}

/// \brief Convert workspaces at the boundary and report LU solve diagnostics without native-float intermediates.
template <MutableRankedMdspecLike<2> A, MutableRankedMdspecLike<2> B>
  requires HostWritableMdspec<A> && HostWritableMdspec<B> && mplapack_detail::RuntimeScalar<typename A::value_type> &&
           std::same_as<typename A::value_type, typename B::value_type>
KernelAttempt try_kernel(MplapackMpfrBackend, linear_solve_op const&, A& a, B& b, SolveInfo& info,
                         SolveOptions<mpreal> const& options, Precision p)
{
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
  auto av = mplapack_detail::pack(as, p), bv = mplapack_detail::pack(bs, p);
  mpreal scale(0, p);
  if (!detail::prepare_linear_solve(av.mdspan(), bv.mdspan(), scale, info))
  {
    // The original inputs were finite; failure arose during boundary conversion.
    info.status = SolveStatus::nonfinite_result;
    return KernelAttempt::success;
  }
  auto tolerance = options.relative_pivot_tolerance ? options.relative_pivot_tolerance->at(p) : mpreal(0, p);
  std::vector<std::size_t> pivots;
  auto singular = mplapack::getrf(a.extent(0), mplapack_detail::values(av), pivots, p);
  mplapack_detail::unpack(av, as);
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
      mplapack::getrs(a.extent(0), b.extent(1), mplapack_detail::values(av), pivots, mplapack_detail::values(bv), p);
      mplapack_detail::unpack(bv, bs);
      if (!detail::solve_matrix_is_finite(bv.mdspan())) info.status = SolveStatus::nonfinite_result;
    }
  }
  return KernelAttempt::success;
}
} // namespace uni20::linalg
