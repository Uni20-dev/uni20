#pragma once

/**
 * \file self_adjoint_eigh.hpp
 * \ingroup linalg
 * \brief Blocking cuSOLVER symmetric eigensolver over CUDA-buffer mdspec operands.
 */

#include <uni20/linalg/backends/cusolver/workspace.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>

#include <memory>

namespace uni20::linalg
{
namespace detail::cusolver_backend
{

template <class Values, class Matrix>
concept CusolverEighMdspecs = CusolverRealScalar<typename std::remove_cvref_t<Matrix>::element_type> &&
                              RawWritableCudaMdspec<Values, typename std::remove_cvref_t<Matrix>::element_type, 1> &&
                              RawWritableCudaMdspec<Matrix, typename std::remove_cvref_t<Matrix>::element_type, 2>;

template <CusolverRealScalar Scalar>
void execute_syevd(MatrixDescriptor<Scalar> const& matrix, cuda::CudaBufferView<Scalar> values,
                   self_adjoint_eigh_op const& op)
{
  auto& resources = matrix.data.buffer().resources();
  auto const device = resources.device().ordinal();
  auto execution = cusolver::execution_pool(resources).acquire();
  cuda::ScopedDevice device_guard(device);
  auto handle = execution.handle().native_handle();
  auto const job = op.compute_vectors ? CUSOLVER_EIG_MODE_VECTOR : CUSOLVER_EIG_MODE_NOVECTOR;
  auto const triangle = op.triangle == MatrixTriangle::Upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  auto const n = matrix.rows;
  auto const ld = matrix.leading_dimension;

  auto matrix_access = matrix.data.buffer().write_synchronized_with(execution.stream());
  auto values_access = values.buffer().write_synchronized_with(execution.stream());
  auto* a = offset_pointer(matrix_access.data(), matrix.data.element_offset());
  auto* w = offset_pointer(values_access.data(), values.element_offset());
  int workspace_size = 0;
  if constexpr (std::same_as<Scalar, float>)
    cusolver::check(cusolverDnSsyevd_bufferSize(handle, job, triangle, n, a, ld, w, &workspace_size),
                    "cusolverDnSsyevd_bufferSize", device);
  else
    cusolver::check(cusolverDnDsyevd_bufferSize(handle, job, triangle, n, a, ld, w, &workspace_size),
                    "cusolverDnDsyevd_bufferSize", device);
  CHECK(workspace_size >= 0, workspace_size);

  bool const memory_pools_supported = resources.device().capabilities().memory_pools_supported;
  StreamAllocation<Scalar> workspace(execution.stream(), static_cast<std::size_t>(workspace_size),
                                     memory_pools_supported);
  StreamAllocation<int> device_info(execution.stream(), 1, memory_pools_supported);
  if constexpr (std::same_as<Scalar, float>)
    cusolver::check(
        cusolverDnSsyevd(handle, job, triangle, n, a, ld, w, workspace.data(), workspace_size, device_info.data()),
        "cusolverDnSsyevd", device);
  else
    cusolver::check(
        cusolverDnDsyevd(handle, job, triangle, n, a, ld, w, workspace.data(), workspace_size, device_info.data()),
        "cusolverDnDsyevd", device);

  int info = 0;
  cuda::check(cudaMemcpyAsync(&info, device_info.data(), sizeof(info), cudaMemcpyDeviceToHost,
                              execution.stream().native_handle()),
              "cudaMemcpyAsync cuSOLVER devInfo", device);
  execution.stream().synchronize();
  values_access.release_after_synchronization();
  matrix_access.release_after_synchronization();
  ERROR_IF(info < 0, "cuSOLVER syevd reported an invalid parameter", -info);
  ERROR_IF(info > 0, "cuSOLVER syevd failed to converge", info);
}

} // namespace detail::cusolver_backend

/// \brief Report cuSOLVER eligibility for CUDA-buffer symmetric eigensystems.
template <uni20::MutableRankedStridedMdspecLike<1> Values, uni20::MutableRankedStridedMdspecLike<2> Matrix>
consteval auto kernel_accepts_types(CusolverBackend const&, self_adjoint_eigh_op const&, Values&, Matrix&)
{
  if constexpr (detail::cusolver_backend::CusolverEighMdspecs<Values, Matrix>)
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Compute a symmetric eigensystem in real, column-major CUDA workspace.
template <uni20::MutableRankedStridedMdspecLike<1> Values, uni20::MutableRankedStridedMdspecLike<2> Matrix>
KernelAttempt try_kernel(CusolverBackend, self_adjoint_eigh_op const& operation, Values& values, Matrix& matrix)
{
  using namespace detail::cusolver_backend;
  CHECK_EQUAL(matrix.extent(0), matrix.extent(1));
  CHECK_EQUAL(values.extent(0), matrix.extent(0));
  if (matrix.extent(0) == 0) return KernelAttempt::success;

  auto descriptor = try_matrix_descriptor(matrix);
  if (!descriptor || (values.extent(0) > 1 && values.stride(0) != 1)) return KernelAttempt::unsupported_layout;

  auto values_view = values.data_descriptor();
  auto& matrix_buffer = descriptor->data.buffer();
  if (values_view.buffer().device().ordinal() != matrix_buffer.device().ordinal())
    return KernelAttempt::incompatible_devices;
  CHECK(std::addressof(matrix_buffer) != std::addressof(values_view.buffer()),
        "cuSOLVER syevd workspace and eigenvalues must use distinct buffers");
  require_descriptor_range(values_view, static_cast<std::size_t>(descriptor->rows));
  require_descriptor_range(descriptor->data, required_elements(*descriptor));
  execute_syevd(*descriptor, values_view, operation);
  return KernelAttempt::success;
}

} // namespace uni20::linalg
