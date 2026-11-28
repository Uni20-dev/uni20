#pragma once

#include <algorithm>
#include <uni20/linalg/backends/cusolver/workspace.hpp>

namespace uni20::linalg::detail::cusolver_backend
{
// Replace a tall column-major work matrix by the thin Q, without allocating R.
template <CusolverRealScalar R> void range_qr(MatrixDescriptor<R> const& a)
{
  auto& resources = a.data.buffer().resources();
  int const device = resources.device().ordinal();
  cuda::ScopedDevice guard(device);
  auto execution = cusolver::execution_pool(resources).acquire();
  auto const& stream = execution.stream();
  auto handle = execution.handle().native_handle();
  auto access = a.data.buffer().write_synchronized_with(stream);
  auto matrix = offset_pointer(access.data(), a.data.element_offset());
  auto const m = a.rows, n = a.columns, ld = a.leading_dimension;
  bool const pools = resources.device().capabilities().memory_pools_supported;
  StreamAllocation<R> tau(stream, n, pools);
  constexpr auto geqrf_size = real_routine<R, cusolverDnSgeqrf_bufferSize, cusolverDnDgeqrf_bufferSize>;
  constexpr auto orgqr_size = real_routine<R, cusolverDnSorgqr_bufferSize, cusolverDnDorgqr_bufferSize>;
  constexpr auto geqrf = real_routine<R, cusolverDnSgeqrf, cusolverDnDgeqrf>;
  constexpr auto orgqr = real_routine<R, cusolverDnSorgqr, cusolverDnDorgqr>;
  int first{}, second{};
  cusolver::check(geqrf_size(handle, m, n, matrix, ld, &first), "geqrf_bufferSize", device);
  cusolver::check(orgqr_size(handle, m, n, n, matrix, ld, tau.data(), &second), "orgqr_bufferSize", device);
  StreamAllocation<R> work(stream, std::max(first, second), pools);
  StreamAllocation<int> info(stream, 2, pools);
  cusolver::check(geqrf(handle, m, n, matrix, ld, tau.data(), work.data(), first, info.data()), "geqrf", device);
  cusolver::check(orgqr(handle, m, n, n, matrix, ld, tau.data(), work.data(), second, info.data() + 1), "orgqr",
                  device);
  int status[2]{};
  cuda::check(cudaMemcpyAsync(status, info.data(), sizeof(status), cudaMemcpyDeviceToHost, stream.native_handle()),
              "copy QR status", device);
  stream.synchronize();
  access.release_after_synchronization();
  ERROR_IF(status[0] || status[1], "cuSOLVER range QR failed", status[0], status[1]);
}
} // namespace uni20::linalg::detail::cusolver_backend
