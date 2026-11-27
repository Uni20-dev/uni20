#pragma once

/**
 * \file svd.hpp
 * \ingroup linalg
 * \brief Blocking cuSOLVER exact SVD and singular values over CUDA-buffer mdspec operands.
 */

#include <uni20/core/scalar_traits.hpp>
#include <uni20/linalg/backends/cusolver/workspace.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>

#include <cuComplex.h>

#include <algorithm>
#include <concepts>
#include <cstddef>
#include <limits>
#include <memory>
#include <optional>
#include <type_traits>
#include <utility>

namespace uni20::linalg
{
namespace detail::cusolver_backend
{

template <class Values, class Left, class Right, class Matrix>
concept CusolverSvdMdspecs = CusolverRealScalar<typename std::remove_cvref_t<Matrix>::element_type> &&
                             RawWritableCudaMdspec<Values, typename std::remove_cvref_t<Matrix>::element_type, 1> &&
                             RawWritableCudaMdspec<Left, typename std::remove_cvref_t<Matrix>::element_type, 2> &&
                             RawWritableCudaMdspec<Right, typename std::remove_cvref_t<Matrix>::element_type, 2> &&
                             RawWritableCudaMdspec<Matrix, typename std::remove_cvref_t<Matrix>::element_type, 2>;

template <CusolverRealScalar Scalar> struct SvdPlan
{
    cuda::CudaBufferView<Scalar> singular_values;
    MatrixDescriptor<Scalar> left;
    MatrixDescriptor<Scalar> right_adjoint;
    MatrixDescriptor<Scalar> matrix;
    int rank = 0;
    SvdVectorExtent left_extent = SvdVectorExtent::Reduced;
    SvdVectorExtent right_extent = SvdVectorExtent::Reduced;
    int device = -1;
};

template <CusolverRealScalar Scalar> int query_workspace(cusolver::ExecutionLease& execution, int rows, int columns)
{
  int size = 0;
  if constexpr (std::same_as<Scalar, float>)
    cusolver::check(cusolverDnSgesvd_bufferSize(execution.handle().native_handle(), rows, columns, &size),
                    "cusolverDnSgesvd_bufferSize", execution.handle().device());
  else
    cusolver::check(cusolverDnDgesvd_bufferSize(execution.handle().native_handle(), rows, columns, &size),
                    "cusolverDnDgesvd_bufferSize", execution.handle().device());
  CHECK(size >= 0, size);
  return size;
}

template <CusolverRealScalar Scalar>
void invoke_gesvd(cusolver::ExecutionLease& execution, SvdPlan<Scalar> const& plan, Scalar* matrix,
                  Scalar* singular_values, Scalar* left, Scalar* right_adjoint, Scalar* workspace, int workspace_size,
                  int* info)
{
  signed char const jobu = static_cast<signed char>(plan.left_extent == SvdVectorExtent::Full ? 'A' : 'S');
  signed char const jobvt = static_cast<signed char>(plan.right_extent == SvdVectorExtent::Full ? 'A' : 'S');
  auto handle = execution.handle().native_handle();
  if constexpr (std::same_as<Scalar, float>)
  {
    cusolver::check(cusolverDnSgesvd(handle, jobu, jobvt, plan.matrix.rows, plan.matrix.columns, matrix,
                                     plan.matrix.leading_dimension, singular_values, left, plan.left.leading_dimension,
                                     right_adjoint, plan.right_adjoint.leading_dimension, workspace, workspace_size,
                                     nullptr, info),
                    "cusolverDnSgesvd", plan.device);
  }
  else
  {
    cusolver::check(cusolverDnDgesvd(handle, jobu, jobvt, plan.matrix.rows, plan.matrix.columns, matrix,
                                     plan.matrix.leading_dimension, singular_values, left, plan.left.leading_dimension,
                                     right_adjoint, plan.right_adjoint.leading_dimension, workspace, workspace_size,
                                     nullptr, info),
                    "cusolverDnDgesvd", plan.device);
  }
}

template <CusolverRealScalar Scalar> KernelAttempt execute_svd(SvdPlan<Scalar> const& plan)
{
  auto& resources = plan.matrix.data.buffer().resources();
  auto execution = cusolver::execution_pool(resources).acquire();
  // cuSOLVER launches into the calling thread's current device, which need not be the operand device.
  cuda::ScopedDevice device_guard(plan.device);
  int const workspace_size = query_workspace<Scalar>(execution, plan.matrix.rows, plan.matrix.columns);
  bool const memory_pools_supported = resources.device().capabilities().memory_pools_supported;
  StreamAllocation<Scalar> workspace(execution.stream(), static_cast<std::size_t>(workspace_size),
                                     memory_pools_supported);
  StreamAllocation<int> device_info(execution.stream(), 1, memory_pools_supported);

  auto values_access = plan.singular_values.buffer().write_synchronized_with(execution.stream());
  auto left_access = plan.left.data.buffer().write_synchronized_with(execution.stream());
  auto right_access = plan.right_adjoint.data.buffer().write_synchronized_with(execution.stream());
  auto matrix_access = plan.matrix.data.buffer().write_synchronized_with(execution.stream());

  auto* values = offset_pointer(values_access.data(), plan.singular_values.element_offset());
  auto* left = offset_pointer(left_access.data(), plan.left.data.element_offset());
  auto* right = offset_pointer(right_access.data(), plan.right_adjoint.data.element_offset());
  auto* matrix = offset_pointer(matrix_access.data(), plan.matrix.data.element_offset());
  invoke_gesvd(execution, plan, matrix, values, left, right, workspace.data(), workspace_size, device_info.data());

  int info = 0;
  cuda::check(cudaMemcpyAsync(&info, device_info.data(), sizeof(info), cudaMemcpyDeviceToHost,
                              execution.stream().native_handle()),
              "cudaMemcpyAsync cuSOLVER devInfo", plan.device);
  execution.stream().synchronize();
  matrix_access.release_after_synchronization();
  right_access.release_after_synchronization();
  left_access.release_after_synchronization();
  values_access.release_after_synchronization();
  ERROR_IF(info < 0, "cuSOLVER gesvd reported an invalid parameter", -info);
  ERROR_IF(info > 0, "cuSOLVER gesvd failed to converge", info);
  return KernelAttempt::success;
}

template <class Values, class Left, class Right, class Matrix>
  requires CusolverSvdMdspecs<Values, Left, Right, Matrix>
KernelAttempt try_svd(svd_op const& operation, Values& values, Left& left, Right& right_adjoint, Matrix& matrix)
{
  using scalar_type = typename std::remove_cvref_t<Matrix>::element_type;
  if (operation.overwrite != SvdOverwrite::None) return KernelAttempt::unsupported_instance;
  if (matrix.extent(0) == 0 || matrix.extent(1) == 0) return KernelAttempt::unsupported_shape;
  if (matrix.extent(0) < matrix.extent(1)) return KernelAttempt::unsupported_shape;

  auto matrix_descriptor = try_matrix_descriptor(matrix);
  auto left_descriptor = try_matrix_descriptor(left);
  auto right_descriptor = try_matrix_descriptor(right_adjoint);
  if (!matrix_descriptor || !left_descriptor || !right_descriptor) return KernelAttempt::unsupported_layout;
  if (values.extent(0) > 1 && values.stride(0) != 1) return KernelAttempt::unsupported_layout;

  int const rows = matrix_descriptor->rows;
  int const columns = matrix_descriptor->columns;
  int const rank = std::min(rows, columns);
  int const left_columns = operation.left == SvdVectorExtent::Full ? rows : rank;
  int const right_rows = operation.right == SvdVectorExtent::Full ? columns : rank;
  CHECK_EQUAL(values.extent(0), rank);
  CHECK_EQUAL(left_descriptor->rows, rows);
  CHECK_EQUAL(left_descriptor->columns, left_columns);
  CHECK_EQUAL(right_descriptor->rows, right_rows);
  CHECK_EQUAL(right_descriptor->columns, columns);

  auto values_view = values.data_descriptor();
  int const device = matrix_descriptor->data.buffer().device().ordinal();
  if (values_view.buffer().device().ordinal() != device ||
      left_descriptor->data.buffer().device().ordinal() != device ||
      right_descriptor->data.buffer().device().ordinal() != device)
    return KernelAttempt::incompatible_devices;

  auto const* matrix_buffer = std::addressof(matrix_descriptor->data.buffer());
  CHECK(matrix_buffer != std::addressof(values_view.buffer()) &&
            matrix_buffer != std::addressof(left_descriptor->data.buffer()) &&
            matrix_buffer != std::addressof(right_descriptor->data.buffer()),
        "cuSOLVER SVD matrix workspace must not alias an output buffer");
  CHECK(std::addressof(values_view.buffer()) != std::addressof(left_descriptor->data.buffer()) &&
            std::addressof(values_view.buffer()) != std::addressof(right_descriptor->data.buffer()) &&
            std::addressof(left_descriptor->data.buffer()) != std::addressof(right_descriptor->data.buffer()),
        "cuSOLVER SVD output buffers must be distinct");

  require_descriptor_range(values_view, static_cast<std::size_t>(rank));
  require_descriptor_range(matrix_descriptor->data, required_elements(*matrix_descriptor));
  require_descriptor_range(left_descriptor->data, required_elements(*left_descriptor));
  require_descriptor_range(right_descriptor->data, required_elements(*right_descriptor));

  return execute_svd(SvdPlan<scalar_type>{.singular_values = values_view,
                                          .left = *left_descriptor,
                                          .right_adjoint = *right_descriptor,
                                          .matrix = *matrix_descriptor,
                                          .rank = rank,
                                          .left_extent = operation.left,
                                          .right_extent = operation.right,
                                          .device = device});
}

template <class Scalar>
concept CusolverSingularValueScalar = CusolverRealScalar<Scalar> || std::same_as<Scalar, uni20::complex<float>> ||
                                      std::same_as<Scalar, uni20::complex<double>>;

template <class Values, class Matrix>
concept CusolverSingularValueMdspecs =
    CusolverSingularValueScalar<typename std::remove_cvref_t<Matrix>::element_type> &&
    RawWritableCudaMdspec<Values, uni20::make_real_t<typename std::remove_cvref_t<Matrix>::element_type>, 1> &&
    RawWritableCudaMdspec<Matrix, typename std::remove_cvref_t<Matrix>::element_type, 2>;

/// \brief Provider-shaped column-major destructive matrix for values-only SVD.
/// \details A unit stride on logical axis 1 describes the transpose, which has
///          the same singular values. cuSOLVER `gesvd` requires the provider
///          shape to satisfy `rows >= columns`; the caller checks that.
template <class Mdspec> [[nodiscard]] auto try_provider_matrix_descriptor(Mdspec& span)
{
  using scalar_type = typename std::remove_cvref_t<Mdspec>::element_type;
  auto stage = blas::try_mdspan_matrix_stage(span);
  if (!stage) return std::optional<MatrixDescriptor<scalar_type>>{};
  auto matrix = blas::blas_writable_matrix(*stage);
  if (!std::in_range<int>(matrix.rows) || !std::in_range<int>(matrix.cols) ||
      !std::in_range<int>(matrix.leading_dimension))
    return std::optional<MatrixDescriptor<scalar_type>>{};
  return std::optional{MatrixDescriptor<scalar_type>{.data = matrix.data,
                                                     .rows = static_cast<int>(matrix.rows),
                                                     .columns = static_cast<int>(matrix.cols),
                                                     .leading_dimension = static_cast<int>(matrix.leading_dimension)}};
}

template <CusolverSingularValueScalar Scalar> struct SingularValuesPlan
{
    cuda::CudaBufferView<uni20::make_real_t<Scalar>> singular_values;
    MatrixDescriptor<Scalar> matrix;
    int rank = 0;
    int device = -1;
};

template <CusolverSingularValueScalar Scalar>
int query_singular_values_workspace(cusolver::ExecutionLease& execution, int rows, int columns)
{
  auto handle = execution.handle().native_handle();
  auto const device = execution.handle().device();
  int size = 0;
  if constexpr (std::same_as<Scalar, float>)
    cusolver::check(cusolverDnSgesvd_bufferSize(handle, rows, columns, &size), "cusolverDnSgesvd_bufferSize", device);
  else if constexpr (std::same_as<Scalar, double>)
    cusolver::check(cusolverDnDgesvd_bufferSize(handle, rows, columns, &size), "cusolverDnDgesvd_bufferSize", device);
  else if constexpr (std::same_as<Scalar, uni20::complex<float>>)
    cusolver::check(cusolverDnCgesvd_bufferSize(handle, rows, columns, &size), "cusolverDnCgesvd_bufferSize", device);
  else
    cusolver::check(cusolverDnZgesvd_bufferSize(handle, rows, columns, &size), "cusolverDnZgesvd_bufferSize", device);
  CHECK(size >= 0, size);
  return size;
}

/// \brief Invoke `gesvd` with `jobu = jobvt = 'N'`.
/// \details LAPACK and cuSOLVER do not reference U or VT for job `'N'`, so
///          both are null. Their leading dimensions are still valid for the
///          provider shape. `rwork` receives unconverged superdiagonal
///          elements on failure and is allocated by the caller.
template <CusolverSingularValueScalar Scalar>
void invoke_singular_values_gesvd(cusolver::ExecutionLease& execution, SingularValuesPlan<Scalar> const& plan,
                                  Scalar* matrix, uni20::make_real_t<Scalar>* singular_values, Scalar* workspace,
                                  int workspace_size, uni20::make_real_t<Scalar>* rwork, int* info)
{
  signed char const job = 'N';
  int const rows = plan.matrix.rows;
  int const columns = plan.matrix.columns;
  int const ld = plan.matrix.leading_dimension;
  int const ldu = std::max(1, rows);
  int const ldvt = std::max(1, columns);
  auto handle = execution.handle().native_handle();
  if constexpr (std::same_as<Scalar, float>)
  {
    cusolver::check(cusolverDnSgesvd(handle, job, job, rows, columns, matrix, ld, singular_values, nullptr, ldu,
                                     nullptr, ldvt, workspace, workspace_size, rwork, info),
                    "cusolverDnSgesvd", plan.device);
  }
  else if constexpr (std::same_as<Scalar, double>)
  {
    cusolver::check(cusolverDnDgesvd(handle, job, job, rows, columns, matrix, ld, singular_values, nullptr, ldu,
                                     nullptr, ldvt, workspace, workspace_size, rwork, info),
                    "cusolverDnDgesvd", plan.device);
  }
  else if constexpr (std::same_as<Scalar, uni20::complex<float>>)
  {
    static_assert(sizeof(Scalar) == sizeof(cuComplex) && alignof(Scalar) <= alignof(cuComplex));
    cusolver::check(cusolverDnCgesvd(handle, job, job, rows, columns, reinterpret_cast<cuComplex*>(matrix), ld,
                                     singular_values, nullptr, ldu, nullptr, ldvt,
                                     reinterpret_cast<cuComplex*>(workspace), workspace_size, rwork, info),
                    "cusolverDnCgesvd", plan.device);
  }
  else
  {
    static_assert(sizeof(Scalar) == sizeof(cuDoubleComplex) && alignof(Scalar) <= alignof(cuDoubleComplex));
    cusolver::check(cusolverDnZgesvd(handle, job, job, rows, columns, reinterpret_cast<cuDoubleComplex*>(matrix), ld,
                                     singular_values, nullptr, ldu, nullptr, ldvt,
                                     reinterpret_cast<cuDoubleComplex*>(workspace), workspace_size, rwork, info),
                    "cusolverDnZgesvd", plan.device);
  }
}

template <CusolverSingularValueScalar Scalar>
KernelAttempt execute_singular_values(SingularValuesPlan<Scalar> const& plan)
{
  using real_type = uni20::make_real_t<Scalar>;
  auto& resources = plan.matrix.data.buffer().resources();
  auto execution = cusolver::execution_pool(resources).acquire();
  // cuSOLVER launches into the calling thread's current device, which need not be the operand device.
  cuda::ScopedDevice device_guard(plan.device);
  int const workspace_size = query_singular_values_workspace<Scalar>(execution, plan.matrix.rows, plan.matrix.columns);
  bool const memory_pools_supported = resources.device().capabilities().memory_pools_supported;
  StreamAllocation<Scalar> workspace(execution.stream(), static_cast<std::size_t>(workspace_size),
                                     memory_pools_supported);
  StreamAllocation<real_type> rwork(execution.stream(), static_cast<std::size_t>(std::max(1, plan.rank - 1)),
                                    memory_pools_supported);
  StreamAllocation<int> device_info(execution.stream(), 1, memory_pools_supported);

  auto values_access = plan.singular_values.buffer().write_synchronized_with(execution.stream());
  auto matrix_access = plan.matrix.data.buffer().write_synchronized_with(execution.stream());

  auto* values = offset_pointer(values_access.data(), plan.singular_values.element_offset());
  auto* matrix = offset_pointer(matrix_access.data(), plan.matrix.data.element_offset());
  invoke_singular_values_gesvd(execution, plan, matrix, values, workspace.data(), workspace_size, rwork.data(),
                               device_info.data());

  int info = 0;
  cuda::check(cudaMemcpyAsync(&info, device_info.data(), sizeof(info), cudaMemcpyDeviceToHost,
                              execution.stream().native_handle()),
              "cudaMemcpyAsync cuSOLVER devInfo", plan.device);
  execution.stream().synchronize();
  matrix_access.release_after_synchronization();
  values_access.release_after_synchronization();
  ERROR_IF(info < 0, "cuSOLVER gesvd reported an invalid parameter", -info);
  ERROR_IF(info > 0, "cuSOLVER gesvd failed to converge", info);
  return KernelAttempt::success;
}

template <class Values, class Matrix>
  requires CusolverSingularValueMdspecs<Values, Matrix>
KernelAttempt try_singular_values(Values& values, Matrix& matrix)
{
  using scalar_type = typename std::remove_cvref_t<Matrix>::element_type;
  auto const rank = std::min(matrix.extent(0), matrix.extent(1));
  CHECK_EQUAL(values.extent(0), rank);
  if (rank == 0) return KernelAttempt::success;

  auto matrix_descriptor = try_provider_matrix_descriptor(matrix);
  if (!matrix_descriptor) return KernelAttempt::unsupported_layout;
  // A wide column-major or tall row-major workspace would need a transpose copy.
  if (matrix_descriptor->rows < matrix_descriptor->columns) return KernelAttempt::unsupported_layout;
  if (values.extent(0) > 1 && values.stride(0) != 1) return KernelAttempt::unsupported_layout;

  auto values_view = values.data_descriptor();
  int const device = matrix_descriptor->data.buffer().device().ordinal();
  if (values_view.buffer().device().ordinal() != device) return KernelAttempt::incompatible_devices;
  // Complex matrices have real value buffers of another type, so compare object addresses.
  CHECK(static_cast<void const*>(std::addressof(matrix_descriptor->data.buffer())) !=
            static_cast<void const*>(std::addressof(values_view.buffer())),
        "cuSOLVER singular-value matrix workspace must not alias the value output buffer");

  require_descriptor_range(values_view, static_cast<std::size_t>(rank));
  require_descriptor_range(matrix_descriptor->data, required_elements(*matrix_descriptor));

  return execute_singular_values(SingularValuesPlan<scalar_type>{.singular_values = values_view,
                                                                 .matrix = *matrix_descriptor,
                                                                 .rank = matrix_descriptor->columns,
                                                                 .device = device});
}

} // namespace detail::cusolver_backend

/// \brief Report cuSOLVER eligibility for a CUDA-buffer exact SVD.
template <uni20::MutableRankedStridedMdspecLike<1> Values, uni20::MutableRankedStridedMdspecLike<2> Left,
          uni20::MutableRankedStridedMdspecLike<2> Right, uni20::MutableRankedStridedMdspecLike<2> Matrix>
consteval auto kernel_accepts_types(CusolverBackend const&, svd_op const&, Values&, Left&, Right&, Matrix&)
{
  if constexpr (detail::cusolver_backend::CusolverSvdMdspecs<Values, Left, Right, Matrix>)
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Compute one blocking exact SVD through cuSOLVER `gesvd`.
template <uni20::MutableRankedStridedMdspecLike<1> Values, uni20::MutableRankedStridedMdspecLike<2> Left,
          uni20::MutableRankedStridedMdspecLike<2> Right, uni20::MutableRankedStridedMdspecLike<2> Matrix>
  requires detail::cusolver_backend::CusolverSvdMdspecs<Values, Left, Right, Matrix>
KernelAttempt try_kernel(CusolverBackend, svd_op const& operation, Values& values, Left& left, Right& right,
                         Matrix& matrix)
{
  return detail::cusolver_backend::try_svd(operation, values, left, right, matrix);
}

/// \brief Report cuSOLVER eligibility for CUDA-buffer exact singular values.
template <uni20::MutableRankedStridedMdspecLike<1> Values, uni20::MutableRankedStridedMdspecLike<2> Matrix>
consteval auto kernel_accepts_types(CusolverBackend const&, singular_values_op const&, Values&, Matrix&)
{
  if constexpr (detail::cusolver_backend::CusolverSingularValueMdspecs<Values, Matrix>)
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Compute blocking exact singular values through cuSOLVER `gesvd` with `jobu = jobvt = 'N'`.
/// \details `matrix` is destructive workspace. Real and complex `float` and
///          `double` are supported. The provider shape must be tall: a tall
///          column-major or a wide row-major workspace. Other layouts decline
///          before resource admission.
template <uni20::MutableRankedStridedMdspecLike<1> Values, uni20::MutableRankedStridedMdspecLike<2> Matrix>
  requires detail::cusolver_backend::CusolverSingularValueMdspecs<Values, Matrix>
KernelAttempt try_kernel(CusolverBackend, singular_values_op const&, Values& values, Matrix& matrix)
{
  return detail::cusolver_backend::try_singular_values(values, matrix);
}

} // namespace uni20::linalg
