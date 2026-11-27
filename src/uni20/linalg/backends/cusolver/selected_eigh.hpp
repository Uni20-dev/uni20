#pragma once

#include <uni20/linalg/backends/cusolver/self_adjoint_eigh.hpp>

namespace uni20::linalg
{
namespace detail::cusolver_backend
{

template <CusolverRealScalar Scalar>
void execute_syevdx(MatrixDescriptor<Scalar> const& matrix, MatrixDescriptor<Scalar> const& vectors,
                    cuda::CudaBufferView<Scalar> values, selected_self_adjoint_eigh_op const& op)
{
  auto& resources = matrix.data.buffer().resources();
  auto const device = resources.device().ordinal();
  auto execution = cusolver::execution_pool(resources).acquire();
  cuda::ScopedDevice device_guard(device);
  auto handle = execution.handle().native_handle();
  auto const job = CUSOLVER_EIG_MODE_VECTOR;
  auto const range = CUSOLVER_EIG_RANGE_I;
  auto const triangle = op.triangle == MatrixTriangle::Upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
  auto const n = matrix.rows;
  auto const ld = matrix.leading_dimension;
  auto const il = static_cast<int>(op.indices.first + 1);
  auto const iu = static_cast<int>(op.indices.last);
  auto const pool = resources.device().capabilities().memory_pools_supported;
  StreamAllocation<Scalar> scratch_values(execution.stream(), static_cast<std::size_t>(n), pool);
  StreamAllocation<int> device_info(execution.stream(), 1, pool);

  auto matrix_access = matrix.data.buffer().write_synchronized_with(execution.stream());
  auto vector_access = vectors.data.buffer().write_synchronized_with(execution.stream());
  auto value_access = values.buffer().write_synchronized_with(execution.stream());
  auto* a = offset_pointer(matrix_access.data(), matrix.data.element_offset());
  auto* z = offset_pointer(vector_access.data(), vectors.data.element_offset());
  auto* w = offset_pointer(value_access.data(), values.element_offset());

  int selected = 0;
  int workspace_size = 0;
  auto* device_values = scratch_values.data();
  if constexpr (std::same_as<Scalar, float>)
    cusolver::check(cusolverDnSsyevdx_bufferSize(handle, job, range, triangle, n, a, ld, 0, 0, il, iu, &selected,
                                                 device_values, &workspace_size),
                    "cusolverDnSsyevdx_bufferSize", device);
  else
    cusolver::check(cusolverDnDsyevdx_bufferSize(handle, job, range, triangle, n, a, ld, 0, 0, il, iu, &selected,
                                                 device_values, &workspace_size),
                    "cusolverDnDsyevdx_bufferSize", device);
  CHECK(workspace_size >= 0, workspace_size);
  StreamAllocation<Scalar> workspace(execution.stream(), static_cast<std::size_t>(workspace_size), pool);
  if constexpr (std::same_as<Scalar, float>)
    cusolver::check(cusolverDnSsyevdx(handle, job, range, triangle, n, a, ld, 0, 0, il, iu, &selected, device_values,
                                      workspace.data(), workspace_size, device_info.data()),
                    "cusolverDnSsyevdx", device);
  else
    cusolver::check(cusolverDnDsyevdx(handle, job, range, triangle, n, a, ld, 0, 0, il, iu, &selected, device_values,
                                      workspace.data(), workspace_size, device_info.data()),
                    "cusolverDnDsyevdx", device);

  auto const stream = execution.stream().native_handle();
  // cuSOLVER writes the selected eigenvectors to the first columns of A.
  auto const count = static_cast<std::size_t>(vectors.columns);
  cuda::check(cudaMemcpyAsync(w, device_values, count * sizeof(Scalar), cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpyAsync cuSOLVER eigenvalues", device);
  cuda::check(cudaMemcpy2DAsync(z, static_cast<std::size_t>(vectors.leading_dimension) * sizeof(Scalar), a,
                                static_cast<std::size_t>(ld) * sizeof(Scalar),
                                static_cast<std::size_t>(n) * sizeof(Scalar), count, cudaMemcpyDeviceToDevice, stream),
              "cudaMemcpy2DAsync cuSOLVER eigenvectors", device);
  int info = 0;
  cuda::check(cudaMemcpyAsync(&info, device_info.data(), sizeof(info), cudaMemcpyDeviceToHost, stream),
              "cudaMemcpyAsync cuSOLVER devInfo", device);
  execution.stream().synchronize();
  value_access.release_after_synchronization();
  vector_access.release_after_synchronization();
  matrix_access.release_after_synchronization();
  ERROR_IF(info < 0, "cuSOLVER syevdx reported an invalid parameter", -info);
  ERROR_IF(info > 0, "cuSOLVER syevdx failed to converge", info);
  ERROR_IF(!std::cmp_equal(selected, count), "cuSOLVER syevdx returned an unexpected eigenvalue count", selected,
           count);
}

} // namespace detail::cusolver_backend

/// \brief Identify device descriptors supported by the selected cuSOLVER eigensolver.
template <uni20::MutableRankedStridedMdspecLike<1> Values, uni20::MutableRankedStridedMdspecLike<2> Vectors,
          uni20::MutableRankedStridedMdspecLike<2> Matrix>
consteval auto kernel_accepts_types(CusolverBackend const&, selected_self_adjoint_eigh_op const&, Values&, Vectors&,
                                    Matrix&)
{
  using namespace detail::cusolver_backend;
  if constexpr (CusolverEighMdspecs<Values, Matrix> && RawWritableCudaMdspec<Vectors, typename Matrix::element_type, 2>)
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Compute compact selected eigenpairs on the operand device.
template <class Values, class Vectors, class Matrix>
KernelAttempt try_kernel(CusolverBackend, selected_self_adjoint_eigh_op const& op, Values& values, Vectors& vectors,
                         Matrix& matrix)
{
  using namespace detail::cusolver_backend;
  auto const n = static_cast<std::size_t>(matrix.extent(0));
  auto const [first, last] = op.indices;
  CHECK(0 <= first && first <= last && std::cmp_less_equal(last, n));
  auto const count = static_cast<std::size_t>(last - first);
  CHECK(std::cmp_equal(matrix.extent(1), n));
  CHECK(std::cmp_equal(values.extent(0), count));
  CHECK(std::cmp_equal(vectors.extent(0), n));
  CHECK(std::cmp_equal(vectors.extent(1), count));
  if (count == 0) return KernelAttempt::success;

  auto a = try_matrix_descriptor(matrix);
  auto z = try_matrix_descriptor(vectors);
  if (!a || !z || (count > 1 && values.stride(0) != 1)) return KernelAttempt::unsupported_layout;
  auto w = values.data_descriptor();
  auto& matrix_buffer = a->data.buffer();
  auto& vector_buffer = z->data.buffer();
  auto& value_buffer = w.buffer();
  auto const device = matrix_buffer.device().ordinal();
  if (vector_buffer.device().ordinal() != device || value_buffer.device().ordinal() != device)
    return KernelAttempt::incompatible_devices;
  CHECK(std::addressof(matrix_buffer) != std::addressof(vector_buffer) &&
            std::addressof(matrix_buffer) != std::addressof(value_buffer) &&
            std::addressof(vector_buffer) != std::addressof(value_buffer),
        "cuSOLVER syevdx needs distinct buffers");
  require_descriptor_range(a->data, required_elements(*a));
  require_descriptor_range(z->data, required_elements(*z));
  require_descriptor_range(w, count);
  execute_syevdx(*a, *z, w, op);
  return KernelAttempt::success;
}

} // namespace uni20::linalg
