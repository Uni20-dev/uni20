#pragma once

#include <uni20/backend/mplapack/binary80.hpp>
#include <uni20/linalg/backends/linear_solve_common.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/matrix_product_shape.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/output.hpp>
#include <uni20/tensor/tensor.hpp>

namespace uni20::linalg
{
namespace mplapack_binary80_detail
{
template <class T>
concept Binary80Scalar = std::same_as<T, float80> || std::same_as<T, complex160>;

template <class Span> auto pack(Span const& span, bool read = true)
{
  using S = typename Span::value_type;
  DenseMatrix<S> result(uninitialized, span.extent(0), span.extent(1));
  if (read)
    for (std::size_t j = 0; j < std::size_t(span.extent(1)); ++j)
      for (std::size_t i = 0; i < std::size_t(span.extent(0)); ++i)
        result[i, j] = S(span[i, j]);
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
} // namespace mplapack_binary80_detail

/// \brief Accept host-readable native binary80 operands, including transforming accessors.
template <MutableRankedMdspecLike<2> C, mplapack_binary80_detail::Binary80Scalar S, RankedMdspecLike<2> A,
          RankedMdspecLike<2> B>
  requires HostWritableMdspec<C> && HostReadableMdspec<A> && HostReadableMdspec<B> &&
           std::same_as<typename C::value_type, S> && std::same_as<typename A::value_type, S> &&
           std::same_as<typename B::value_type, S>
consteval auto kernel_accepts_types(MplapackBinary80Backend, gemm_op const&, C&, S const&, A&, B&, S const&)
{
  return kernel_types_maybe;
}

/// \brief Pack logical host values and execute GEMM through value-preserving provider conversion.
template <MutableRankedMdspecLike<2> C, mplapack_binary80_detail::Binary80Scalar S, RankedMdspecLike<2> A,
          RankedMdspecLike<2> B>
  requires HostWritableMdspec<C> && HostReadableMdspec<A> && HostReadableMdspec<B> &&
           std::same_as<typename C::value_type, S> && std::same_as<typename A::value_type, S> &&
           std::same_as<typename B::value_type, S>
KernelAttempt try_kernel(MplapackBinary80Backend, gemm_op const&, C& c, S const& alpha, A& a, B& b, S const& beta)
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
  auto av = mplapack_binary80_detail::pack(as, alpha != S{0} && a.extent(1) != 0);
  auto bv = mplapack_binary80_detail::pack(bs, alpha != S{0} && a.extent(1) != 0);
  auto cv = mplapack_binary80_detail::pack(cs, beta != S{0});
  mplapack::gemm(a.extent(0), b.extent(1), a.extent(1), alpha, mplapack_binary80_detail::values(av),
                 mplapack_binary80_detail::values(bv), beta, mplapack_binary80_detail::values(cv));
  mplapack_binary80_detail::unpack(cv, cs);
  return KernelAttempt::success;
}

/// \brief Accept replaceable binary80 host outputs for matrix-product assignment.
template <MutableRankedTensorView<2> C, mplapack_binary80_detail::Binary80Scalar S, RankedMdspecLike<2> A,
          RankedMdspecLike<2> B>
  requires HostWritableMdspec<mutable_tensor_mdspec_t<C>> && HostReadableMdspec<A> && HostReadableMdspec<B> &&
           std::same_as<tensor_element_t<C>, S> && std::same_as<typename A::value_type, S> &&
           std::same_as<typename B::value_type, S>
consteval auto kernel_accepts_types(MplapackBinary80Backend, assign_product_op const&, C&, S const&, A&, B&)
{
  return kernel_types_yes;
}

/// \brief Prepare the host output and lower assignment to binary80 GEMM.
template <MutableRankedTensorView<2> C, mplapack_binary80_detail::Binary80Scalar S, RankedMdspecLike<2> A,
          RankedMdspecLike<2> B>
  requires HostWritableMdspec<mutable_tensor_mdspec_t<C>> && HostReadableMdspec<A> && HostReadableMdspec<B> &&
           std::same_as<tensor_element_t<C>, S> && std::same_as<typename A::value_type, S> &&
           std::same_as<typename B::value_type, S>
KernelAttempt try_kernel(MplapackBinary80Backend backend, assign_product_op const&, C& c, S const& alpha, A& a, B& b)
{
  auto shape = detail::matrix_product_shape(a, b);
  prepare_output(c, shape);
  auto out = mdspec_of(c);
  return try_kernel(backend, gemm_op{}, out, alpha, a, b, S{});
}

/// \brief Accept deferred host binary80 matrix-product outputs.
template <MutableRankedTensorView<2> C, mplapack_binary80_detail::Binary80Scalar S, RankedMdspecLike<2> A,
          RankedMdspecLike<2> B>
  requires HostWritableMdspec<mutable_tensor_mdspec_t<C>> && HostReadableMdspec<A> && HostReadableMdspec<B> &&
           std::same_as<tensor_element_t<C>, S> && std::same_as<typename A::value_type, S> &&
           std::same_as<typename B::value_type, S>
consteval auto kernel_accepts_types(MplapackBinary80Backend, assign_product_op const&, async::shared_storage<C>&,
                                    S const&, A&, B&)
{
  return kernel_types_yes;
}

/// \brief Construct or resize a deferred output before binary80 GEMM.
template <MutableRankedTensorView<2> C, mplapack_binary80_detail::Binary80Scalar S, RankedMdspecLike<2> A,
          RankedMdspecLike<2> B>
  requires HostWritableMdspec<mutable_tensor_mdspec_t<C>> && HostReadableMdspec<A> && HostReadableMdspec<B> &&
           std::same_as<tensor_element_t<C>, S> && std::same_as<typename A::value_type, S> &&
           std::same_as<typename B::value_type, S>
KernelAttempt try_kernel(MplapackBinary80Backend backend, assign_product_op const&, async::shared_storage<C>& storage,
                         S const& alpha, A& a, B& b)
{
  auto& output = prepare_output(storage, detail::matrix_product_shape(a, b));
  auto out = mdspec_of(output);
  return try_kernel(backend, gemm_op{}, out, alpha, a, b, S{});
}

/// \brief Accept mutable host operands for a binary80 LU solve.
template <MutableRankedMdspecLike<2> A, MutableRankedMdspecLike<2> B>
  requires HostWritableMdspec<A> && HostWritableMdspec<B> &&
           mplapack_binary80_detail::Binary80Scalar<typename A::value_type> &&
           std::same_as<typename A::value_type, typename B::value_type>
consteval auto kernel_accepts_types(MplapackBinary80Backend, linear_solve_op const&, A&, B&, SolveInfo&,
                                    SolveOptions<float80> const&)
{
  return kernel_types_yes;
}

/// \brief Convert workspaces at the boundary and report LU solve diagnostics without double-precision intermediates.
template <MutableRankedMdspecLike<2> A, MutableRankedMdspecLike<2> B>
  requires HostWritableMdspec<A> && HostWritableMdspec<B> &&
           mplapack_binary80_detail::Binary80Scalar<typename A::value_type> &&
           std::same_as<typename A::value_type, typename B::value_type>
KernelAttempt try_kernel(MplapackBinary80Backend, linear_solve_op const&, A& a, B& b, SolveInfo& info,
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
      using std::abs;
      auto magnitude = abs(av[i, i]);
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
