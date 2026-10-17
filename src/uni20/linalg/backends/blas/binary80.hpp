#pragma once

#include <uni20/linalg/backends/binary80_common.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/matrix_product_shape.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/output.hpp>

namespace uni20::linalg
{

/// \brief Accept host-readable native binary80 operands, including transforming accessors.
template <MutableRankedMdspecLike<2> C, mplapack_binary80_detail::Binary80Scalar S, RankedMdspecLike<2> A,
          RankedMdspecLike<2> B>
  requires HostWritableMdspec<C> && HostReadableMdspec<A> && HostReadableMdspec<B> &&
           std::same_as<typename C::value_type, S> && std::same_as<typename A::value_type, S> &&
           std::same_as<typename B::value_type, S>
consteval auto kernel_accepts_types(BlasBackend, gemm_op const&, C&, S const&, A&, B&, S const&)
{
  return kernel_types_maybe;
}

/// \brief Pack logical host values and execute GEMM through value-preserving provider conversion.
template <MutableRankedMdspecLike<2> C, mplapack_binary80_detail::Binary80Scalar S, RankedMdspecLike<2> A,
          RankedMdspecLike<2> B>
  requires HostWritableMdspec<C> && HostReadableMdspec<A> && HostReadableMdspec<B> &&
           std::same_as<typename C::value_type, S> && std::same_as<typename A::value_type, S> &&
           std::same_as<typename B::value_type, S>
KernelAttempt try_kernel(BlasBackend, gemm_op const&, C& c, S const& alpha, A& a, B& b, S const& beta)
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
consteval auto kernel_accepts_types(BlasBackend, assign_product_op const&, C&, S const&, A&, B&)
{
  return kernel_types_yes;
}

/// \brief Prepare the host output and lower assignment to binary80 GEMM.
template <MutableRankedTensorView<2> C, mplapack_binary80_detail::Binary80Scalar S, RankedMdspecLike<2> A,
          RankedMdspecLike<2> B>
  requires HostWritableMdspec<mutable_tensor_mdspec_t<C>> && HostReadableMdspec<A> && HostReadableMdspec<B> &&
           std::same_as<tensor_element_t<C>, S> && std::same_as<typename A::value_type, S> &&
           std::same_as<typename B::value_type, S>
KernelAttempt try_kernel(BlasBackend backend, assign_product_op const&, C& c, S const& alpha, A& a, B& b)
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
consteval auto kernel_accepts_types(BlasBackend, assign_product_op const&, async::shared_storage<C>&, S const&, A&, B&)
{
  return kernel_types_yes;
}

/// \brief Construct or resize a deferred output before binary80 GEMM.
template <MutableRankedTensorView<2> C, mplapack_binary80_detail::Binary80Scalar S, RankedMdspecLike<2> A,
          RankedMdspecLike<2> B>
  requires HostWritableMdspec<mutable_tensor_mdspec_t<C>> && HostReadableMdspec<A> && HostReadableMdspec<B> &&
           std::same_as<tensor_element_t<C>, S> && std::same_as<typename A::value_type, S> &&
           std::same_as<typename B::value_type, S>
KernelAttempt try_kernel(BlasBackend backend, assign_product_op const&, async::shared_storage<C>& storage,
                         S const& alpha, A& a, B& b)
{
  auto& output = prepare_output(storage, detail::matrix_product_shape(a, b));
  auto out = mdspec_of(output);
  return try_kernel(backend, gemm_op{}, out, alpha, a, b, S{});
}

} // namespace uni20::linalg
