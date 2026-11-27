#pragma once

/**
 * \file jacobi.hpp
 * \ingroup linalg
 * \brief Blocking cuSOLVER Jacobi solvers for contiguous real CUDA work and factors.
 */

#include <uni20/linalg/backends/cusolver/workspace.hpp>
#include <uni20/linalg/jacobi_options.hpp>
#include <uni20/linalg/operation_tags.hpp>

#include <array>
#include <memory>
#include <tuple>

namespace uni20::linalg::detail::cusolver_backend
{

template <class Scalar, auto Single, auto Double>
inline constexpr auto real_jacobi_routine = [] {
  if constexpr (std::same_as<Scalar, float>)
    return Single;
  else
    return Double;
}();

struct SyevjParameters
{
    using info_type = syevjInfo_t;
    static constexpr auto create = cusolverDnCreateSyevjInfo;
    static constexpr auto destroy = cusolverDnDestroySyevjInfo;
    static constexpr auto set_tolerance = cusolverDnXsyevjSetTolerance;
    static constexpr auto set_max_sweeps = cusolverDnXsyevjSetMaxSweeps;
    static constexpr auto get_residual = cusolverDnXsyevjGetResidual;
    static constexpr auto get_sweeps = cusolverDnXsyevjGetSweeps;
};

struct GesvdjParameters
{
    using info_type = gesvdjInfo_t;
    static constexpr auto create = cusolverDnCreateGesvdjInfo;
    static constexpr auto destroy = cusolverDnDestroyGesvdjInfo;
    static constexpr auto set_tolerance = cusolverDnXgesvdjSetTolerance;
    static constexpr auto set_max_sweeps = cusolverDnXgesvdjSetMaxSweeps;
    static constexpr auto get_residual = cusolverDnXgesvdjGetResidual;
    static constexpr auto get_sweeps = cusolverDnXgesvdjGetSweeps;
};

/// \brief Run a Jacobi provider call with private, same-device work and factor buffers.
template <class Parameters, CusolverRealScalar Scalar, class Query, class Solve, class... Outputs>
[[nodiscard]] JacobiConvergence run_jacobi(CudaJacobiOptions options, Query query, Solve solve,
                                           cuda::CudaBuffer<Scalar>& work, Outputs&... outputs)
{
  auto& resources = work.resources();
  auto const device = resources.device().ordinal();
  auto execution = cusolver::execution_pool(resources).acquire();
  auto const& stream = execution.stream();
  cuda::ScopedDevice device_guard(device);
  auto handle = execution.handle().native_handle();

  using Info = typename Parameters::info_type;
  Info raw = nullptr;
  cusolver::check(Parameters::create(&raw), "create cuSOLVER Jacobi parameters", device);
  std::unique_ptr<std::remove_pointer_t<Info>, decltype(Parameters::destroy)> params(raw, Parameters::destroy);
  if (options.tolerance)
    cusolver::check(Parameters::set_tolerance(raw, *options.tolerance), "set cuSOLVER Jacobi tolerance", device);
  if (options.maximum_sweeps)
    cusolver::check(Parameters::set_max_sweeps(raw, *options.maximum_sweeps), "set cuSOLVER Jacobi sweep limit",
                    device);

  auto access = std::array{work.write_synchronized_with(stream), outputs.write_synchronized_with(stream)...};
  int workspace_size = 0;
  auto const query_status =
      std::apply([&](auto&... buffers) { return query(handle, raw, &workspace_size, buffers.data()...); }, access);
  cusolver::check(query_status, "query cuSOLVER Jacobi workspace", device);
  CHECK(workspace_size >= 0, workspace_size);

  bool const memory_pools = resources.device().capabilities().memory_pools_supported;
  StreamAllocation<Scalar> workspace(stream, static_cast<std::size_t>(workspace_size), memory_pools);
  StreamAllocation<int> device_info(stream, 1, memory_pools);
  int info = 0;
  try
  {
    auto const status = std::apply(
        [&](auto&... buffers) {
          return solve(handle, raw, workspace.data(), workspace_size, device_info.data(), buffers.data()...);
        },
        access);
    cusolver::check(status, "run cuSOLVER Jacobi decomposition", device);
    cuda::check(
        cudaMemcpyAsync(&info, device_info.data(), sizeof(info), cudaMemcpyDeviceToHost, stream.native_handle()),
        "copy cuSOLVER Jacobi devInfo", device);
    stream.synchronize();
  }
  catch (...)
  {
    // Retain the host parameters and devInfo destination until submitted work has finished.
    (void)cudaStreamSynchronize(stream.native_handle());
    throw;
  }
  for (auto& buffer : access)
    buffer.release_after_synchronization();

  ERROR_IF(info < 0, "cuSOLVER Jacobi decomposition reported an invalid parameter", -info);
  JacobiConvergence result{.converged = info == 0};
  cusolver::check(Parameters::get_residual(handle, raw, &result.residual), "query cuSOLVER Jacobi residual", device);
  cusolver::check(Parameters::get_sweeps(handle, raw, &result.sweeps), "query cuSOLVER Jacobi sweeps", device);
  return result;
}

template <CusolverRealScalar Scalar>
[[nodiscard]] JacobiConvergence execute_syevj(int n, MatrixTriangle triangle, CudaJacobiOptions options,
                                              cuda::CudaBuffer<Scalar>& matrix, cuda::CudaBuffer<Scalar>& values)
{
  constexpr auto query = real_jacobi_routine<Scalar, cusolverDnSsyevj_bufferSize, cusolverDnDsyevj_bufferSize>;
  constexpr auto solve = real_jacobi_routine<Scalar, cusolverDnSsyevj, cusolverDnDsyevj>;
  auto const fill = triangle == MatrixTriangle::Upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  return run_jacobi<SyevjParameters>(
      options,
      [=](cusolverDnHandle_t handle, syevjInfo_t params, int* size, Scalar* a, Scalar* w) {
        return query(handle, CUSOLVER_EIG_MODE_VECTOR, fill, n, a, n, w, size, params);
      },
      [=](cusolverDnHandle_t handle, syevjInfo_t params, Scalar* work, int size, int* info, Scalar* a, Scalar* w) {
        return solve(handle, CUSOLVER_EIG_MODE_VECTOR, fill, n, a, n, w, work, size, info, params);
      },
      matrix, values);
}

template <CusolverRealScalar Scalar>
[[nodiscard]] JacobiConvergence execute_gesvdj(int rows, int cols, CudaJacobiOptions options,
                                               cuda::CudaBuffer<Scalar>& matrix, cuda::CudaBuffer<Scalar>& values,
                                               cuda::CudaBuffer<Scalar>& left, cuda::CudaBuffer<Scalar>& right)
{
  constexpr auto query = real_jacobi_routine<Scalar, cusolverDnSgesvdj_bufferSize, cusolverDnDgesvdj_bufferSize>;
  constexpr auto solve = real_jacobi_routine<Scalar, cusolverDnSgesvdj, cusolverDnDgesvdj>;
  constexpr int economy = 1;
  return run_jacobi<GesvdjParameters>(
      options,
      [=](cusolverDnHandle_t handle, gesvdjInfo_t params, int* size, Scalar* a, Scalar* s, Scalar* u, Scalar* v) {
        return query(handle, CUSOLVER_EIG_MODE_VECTOR, economy, rows, cols, a, rows, s, u, rows, v, cols, size, params);
      },
      [=](cusolverDnHandle_t handle, gesvdjInfo_t params, Scalar* work, int size, int* info, Scalar* a, Scalar* s,
          Scalar* u, Scalar* v) {
        return solve(handle, CUSOLVER_EIG_MODE_VECTOR, economy, rows, cols, a, rows, s, u, rows, v, cols, work, size,
                     info, params);
      },
      matrix, values, left, right);
}

} // namespace uni20::linalg::detail::cusolver_backend
