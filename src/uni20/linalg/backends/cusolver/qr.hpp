#pragma once

#include "qr_upper.hpp"
#include <algorithm>
#include <array>
#include <uni20/linalg/backends/cusolver/workspace.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>

namespace uni20::linalg::detail::cusolver_backend
{
// Full QR saves the upper trapezoid before orgqr overwrites it. Q-only needs neither output copy.
template <CusolverRealScalar R, class SaveR, class SaveQ>
void form_q(MatrixDescriptor<R> const& a, cusolver::ExecutionLease& execution, R* matrix, SaveR save_r, SaveQ save_q)
{
  auto& resources = a.data.buffer().resources();
  int const device = resources.device().ordinal();
  cuda::ScopedDevice guard(device);
  auto const& stream = execution.stream();
  auto handle = execution.handle().native_handle();
  auto const m = a.rows, n = a.columns, k = std::min(m, n), ld = a.leading_dimension;
  bool const pools = resources.device().capabilities().memory_pools_supported;
  StreamAllocation<R> tau(stream, k, pools);
  constexpr auto geqrf_size = real_routine<R, cusolverDnSgeqrf_bufferSize, cusolverDnDgeqrf_bufferSize>;
  constexpr auto orgqr_size = real_routine<R, cusolverDnSorgqr_bufferSize, cusolverDnDorgqr_bufferSize>;
  constexpr auto geqrf = real_routine<R, cusolverDnSgeqrf, cusolverDnDgeqrf>;
  constexpr auto orgqr = real_routine<R, cusolverDnSorgqr, cusolverDnDorgqr>;
  int first{}, second{};
  cusolver::check(geqrf_size(handle, m, n, matrix, ld, &first), "geqrf_bufferSize", device);
  cusolver::check(orgqr_size(handle, m, k, k, matrix, ld, tau.data(), &second), "orgqr_bufferSize", device);
  StreamAllocation<R> work(stream, std::max(first, second), pools);
  StreamAllocation<int> info(stream, 2, pools);
  cusolver::check(geqrf(handle, m, n, matrix, ld, tau.data(), work.data(), first, info.data()), "geqrf", device);
  save_r(matrix);
  cusolver::check(orgqr(handle, m, k, k, matrix, ld, tau.data(), work.data(), second, info.data() + 1), "orgqr",
                  device);
  save_q(matrix);
  int status[2]{};
  cuda::check(cudaMemcpyAsync(status, info.data(), sizeof(status), cudaMemcpyDeviceToHost, stream.native_handle()),
              "copy QR status", device);
  stream.synchronize();
  ERROR_IF(status[0] || status[1], "cuSOLVER QR failed", status[0], status[1]);
}
} // namespace uni20::linalg::detail::cusolver_backend

namespace uni20::linalg
{
template <class A>
concept CusolverQrMatrix =
    MutableRankedStridedMdspecLike<A, 2> && detail::cusolver_backend::CusolverRealScalar<typename A::element_type> &&
    detail::cusolver_backend::RawWritableCudaMdspec<A, typename A::element_type, 2>;

template <CusolverQrMatrix A> consteval auto kernel_accepts_types(CusolverBackend, orthonormal_basis_op, A&)
{
  return kernel_types_maybe;
}

template <CusolverQrMatrix A> KernelAttempt try_kernel(CusolverBackend, orthonormal_basis_op, A& work)
{
  using namespace detail::cusolver_backend;
  if (work.extent(0) < work.extent(1)) return KernelAttempt::unsupported_shape;
  if (work.extent(1) == 0) return KernelAttempt::success;
  auto a = try_matrix_descriptor(work);
  if (!a) return KernelAttempt::unsupported_layout;
  require_descriptor_range(a->data, required_elements(*a));
  auto execution = cusolver::execution_pool(a->data.buffer().resources()).acquire();
  auto access = a->data.buffer().write_synchronized_with(execution.stream());
  form_q(*a, execution, offset_pointer(access.data(), a->data.element_offset()), [](auto*) {}, [](auto*) {});
  access.release_after_synchronization();
  return KernelAttempt::success;
}

template <class Q, class R, class A>
concept CusolverQrOperands =
    CusolverQrMatrix<A> && detail::cusolver_backend::RawWritableCudaMdspec<Q, typename A::element_type, 2> &&
    detail::cusolver_backend::RawWritableCudaMdspec<R, typename A::element_type, 2>;

template <class Q, class R, class A>
  requires CusolverQrOperands<Q, R, A>
consteval auto kernel_accepts_types(CusolverBackend, qr_op, Q&, R&, A&)
{
  return kernel_types_maybe;
}

template <class Q, class R, class A>
  requires CusolverQrOperands<Q, R, A>
KernelAttempt try_kernel(CusolverBackend, qr_op, Q& q, R& r, A& work)
{
  using namespace detail::cusolver_backend;
  using S = typename A::value_type;
  auto const k = std::min(work.extent(0), work.extent(1));
  CHECK_EQUAL(q.extent(0), work.extent(0));
  CHECK_EQUAL(q.extent(1), k);
  CHECK_EQUAL(r.extent(0), k);
  CHECK_EQUAL(r.extent(1), work.extent(1));
  if (k == 0) return KernelAttempt::success;
  auto a = try_matrix_descriptor(work);
  auto qd = try_matrix_descriptor(q);
  auto rd = try_matrix_descriptor(r);
  if (!a || !qd || !rd) return KernelAttempt::unsupported_layout;
  auto& ab = a->data.buffer();
  auto& qb = qd->data.buffer();
  auto& rb = rd->data.buffer();
  if (qb.device() != ab.device() || rb.device() != ab.device()) return KernelAttempt::incompatible_devices;
  for (auto const* desc : {&*a, &*qd, &*rd})
    require_descriptor_range(desc->data, required_elements(*desc));

  auto const device = ab.device().ordinal();
  cuda::ScopedDevice guard(device);
  auto execution = cusolver::execution_pool(ab.resources()).acquire();
  auto const& stream = execution.stream();
  // Disjoint views may share a buffer. Acquire each completion ledger just once.
  std::array<std::optional<cuda::WriteAccess<S>>, 3> writes;
  auto acquire = [&](auto const& desc) -> S* {
    auto& buffer = desc.data.buffer();
    for (auto& write : writes)
    {
      if (!write) write.emplace(buffer.write_synchronized_with(stream));
      if (&write->storage() == &buffer) return offset_pointer(write->data(), desc.data.element_offset());
    }
    throw std::logic_error("QR has at most three buffers");
  };
  S* matrix = acquire(*a);
  S* qp = acquire(*qd);
  S* rp = acquire(*rd);
  form_q(
      *a, execution, matrix,
      [&](S const* packed) {
        copy_upper_trapezoid(rp, rd->leading_dimension, packed, a->leading_dimension, rd->rows, rd->columns,
                             stream.native_handle(), device);
      },
      [&](S const* basis) {
        cuda::check(cudaMemcpy2DAsync(qp, sizeof(S) * qd->leading_dimension, basis, sizeof(S) * a->leading_dimension,
                                      sizeof(S) * qd->rows, qd->columns, cudaMemcpyDeviceToDevice,
                                      stream.native_handle()),
                    "copy reduced QR basis", device);
      });
  for (auto& write : writes)
    if (write) write->release_after_synchronization();
  return KernelAttempt::success;
}
} // namespace uni20::linalg
