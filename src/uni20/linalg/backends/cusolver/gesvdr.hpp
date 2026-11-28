#pragma once

#include <uni20/linalg/backends/cusolver/workspace.hpp>

#include <algorithm>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

namespace uni20::linalg
{
/// \brief Requested Xgesvdr rank, sketch settings and singular vectors.
struct CudaGesvdrOptions
{
    index_type rank;
    index_type oversampling = 8;
    index_type power_iterations = 2;
    bool left_vectors = true;
    bool right_vectors = true;
};

namespace detail::cusolver_backend
{
// The provider documents square backing arrays for requested vectors even in its reduced mode.
template <class R> std::size_t gesvdr_vector_workspace(std::size_t n, bool wanted)
{
  if (!wanted) return 0;
  if (n > (std::numeric_limits<std::size_t>::max() / sizeof(R)) / n)
    throw std::length_error("Xgesvdr vector storage exceeds the addressable size");
  return n * n;
}

template <CusolverRealScalar R>
void execute_gesvdr(std::int64_t m, std::int64_t n, CudaGesvdrOptions opts, cuda::CudaBuffer<R>& work,
                    cuda::CudaBuffer<R>& values, cuda::CudaBuffer<R>& left, cuda::CudaBuffer<R>& right)
{
  auto& resources = work.resources();
  auto const device = resources.device().ordinal();
  cuda::ScopedDevice guard(device);
  auto lease = cusolver::execution_pool(resources).acquire();
  auto const& stream = lease.stream();
  auto handle = lease.handle().native_handle();
  cusolverDnParams_t raw = nullptr;
  cusolver::check(cusolverDnCreateParams(&raw), "create Xgesvdr parameters", device);
  std::unique_ptr<std::remove_pointer_t<cusolverDnParams_t>, decltype(&cusolverDnDestroyParams)> params(
      raw, cusolverDnDestroyParams);

  bool const pools = resources.device().capabilities().memory_pools_supported;
  StreamAllocation<R> s(stream, std::min(m, n), pools);
  StreamAllocation<R> u(stream, gesvdr_vector_workspace<R>(m, opts.left_vectors), pools);
  StreamAllocation<R> v(stream, gesvdr_vector_workspace<R>(n, opts.right_vectors), pools);
  auto a = work.write_synchronized_with(stream);
  constexpr auto dtype = std::same_as<R, float> ? CUDA_R_32F : CUDA_R_64F;
  auto invoke = [&](auto fn, auto... workspace) {
    return fn(handle, params.get(), opts.left_vectors ? 'S' : 'N', opts.right_vectors ? 'S' : 'N', m, n, opts.rank,
              opts.oversampling, opts.power_iterations, dtype, a.data(), m, dtype, s.data(), dtype, u.data(), m, dtype,
              v.data(), n, dtype, workspace...);
  };
  std::size_t device_bytes{}, host_bytes{};
  cusolver::check(invoke(cusolverDnXgesvdr_bufferSize, &device_bytes, &host_bytes), "Xgesvdr_bufferSize", device);
  StreamAllocation<std::byte> scratch(stream, device_bytes, pools);
  StreamAllocation<int> info(stream, 1, pools);
  std::vector<std::byte> host(host_bytes);
  cusolver::check(invoke(cusolverDnXgesvdr, scratch.data(), device_bytes, host.data(), host_bytes, info.data()),
                  "Xgesvdr", device);
  int status{};
  cuda::check(cudaMemcpyAsync(&status, info.data(), sizeof(status), cudaMemcpyDeviceToHost, stream.native_handle()),
              "read Xgesvdr status", device);
  auto compact = [&](cuda::CudaBuffer<R>& output, R const* source) {
    auto access = output.write_synchronized_with(stream);
    cuda::check(cudaMemcpyAsync(access.data(), source, output.size() * sizeof(R), cudaMemcpyDeviceToDevice,
                                stream.native_handle()),
                "copy compact Xgesvdr factors", device);
    return access;
  };
  auto sw = compact(values, s.data());
  std::optional<cuda::WriteAccess<R>> uw, vw;
  if (opts.left_vectors) uw.emplace(compact(left, u.data()));
  if (opts.right_vectors) vw.emplace(compact(right, v.data()));
  stream.synchronize();
  a.release_after_synchronization();
  sw.release_after_synchronization();
  if (uw) uw->release_after_synchronization();
  if (vw) vw->release_after_synchronization();
  ERROR_IF(status != 0, "cuSOLVER Xgesvdr failed", status);
}
} // namespace detail::cusolver_backend
} // namespace uni20::linalg
