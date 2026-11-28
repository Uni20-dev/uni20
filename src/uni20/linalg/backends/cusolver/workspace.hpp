#pragma once

/**
 * \file workspace.hpp
 * \ingroup linalg
 * \brief Workspace and CUDA operand lowering shared by the cuSOLVER decompositions.
 */

#include <uni20/backend/cuda/buffer.hpp>
#include <uni20/backend/cuda/cuda_error.hpp>
#include <uni20/backend/cusolver/cusolver_error.hpp>
#include <uni20/backend/cusolver/execution.hpp>
#include <uni20/linalg/blas/mdspan_matrix.hpp>
#include <uni20/storage/cuda_accessor.hpp>
#include <uni20/storage/cuda_storage.hpp>

#include <concepts>
#include <cstddef>
#include <limits>
#include <optional>
#include <type_traits>
#include <utility>

namespace uni20::linalg::detail::cusolver_backend
{

template <class Scalar>
concept CusolverRealScalar = std::same_as<Scalar, float> || std::same_as<Scalar, double>;

template <class Scalar, auto Single, auto Double>
inline constexpr auto real_routine = [] {
  if constexpr (std::same_as<Scalar, float>)
    return Single;
  else
    return Double;
}();

template <class Mdspec, class Scalar, std::size_t Rank>
concept RawWritableCudaMdspec =
    uni20::MutableRankedStridedMdspecLike<Mdspec, Rank> && uni20::cuda::BufferMdspec<Mdspec> &&
    std::same_as<typename std::remove_cvref_t<Mdspec>::element_type, Scalar> &&
    std::same_as<typename std::remove_cvref_t<Mdspec>::accessor_type, uni20::cuda::CudaPointerAccessor<Scalar>>;

template <class Scalar> struct MatrixDescriptor
{
    cuda::CudaBufferView<Scalar> data;
    int rows = 0;
    int columns = 0;
    int leading_dimension = 0;
};

template <class Mdspec> [[nodiscard]] auto try_matrix_descriptor(Mdspec& span)
{
  using scalar_type = typename std::remove_cvref_t<Mdspec>::element_type;
  auto stage = blas::try_mdspan_matrix_stage(span);
  if (!stage || stage->unit_stride_axis != 0 || stage->needs_conjugation)
    return std::optional<MatrixDescriptor<scalar_type>>{};
  auto matrix = blas::blas_writable_matrix(*stage);
  if (!std::in_range<int>(matrix.rows) || !std::in_range<int>(matrix.cols) ||
      !std::in_range<int>(matrix.leading_dimension))
    return std::optional<MatrixDescriptor<scalar_type>>{};
  return std::optional{MatrixDescriptor<scalar_type>{.data = matrix.data,
                                                     .rows = static_cast<int>(matrix.rows),
                                                     .columns = static_cast<int>(matrix.cols),
                                                     .leading_dimension = static_cast<int>(matrix.leading_dimension)}};
}

template <class Scalar> [[nodiscard]] std::size_t required_elements(MatrixDescriptor<Scalar> const& matrix)
{
  if (matrix.rows == 0 || matrix.columns == 0) return 0;
  auto const rows = static_cast<std::size_t>(matrix.rows);
  auto const columns = static_cast<std::size_t>(matrix.columns);
  auto const leading_dimension = static_cast<std::size_t>(matrix.leading_dimension);
  CHECK(columns - 1 <= (std::numeric_limits<std::size_t>::max() - rows) / leading_dimension);
  return (columns - 1) * leading_dimension + rows;
}

template <class View> void require_descriptor_range(View const& view, std::size_t required)
{
  auto const offset = view.element_offset();
  auto const size = view.buffer().size();
  CHECK(offset <= size && required <= size - offset, offset, required, size);
}

template <class Pointer> [[nodiscard]] Pointer offset_pointer(Pointer pointer, std::size_t offset)
{
  if (offset == 0) return pointer;
  CHECK(pointer != nullptr, offset);
  return pointer + offset;
}

template <class T> class StreamAllocation {
  public:
    StreamAllocation(cuda::Stream const& stream, std::size_t size, bool memory_pools_supported)
        : stream_(&stream), stream_ordered_(memory_pools_supported)
    {
      if (size == 0) return;
      CHECK(size <= std::numeric_limits<std::size_t>::max() / sizeof(T), size, sizeof(T));
      cuda::ScopedDevice guard(stream.device());
      if (stream_ordered_)
        cuda::check(cudaMallocAsync(reinterpret_cast<void**>(&data_), size * sizeof(T), stream.native_handle()),
                    "cudaMallocAsync cuSOLVER workspace", stream.device());
      else
        cuda::check(cudaMalloc(reinterpret_cast<void**>(&data_), size * sizeof(T)), "cudaMalloc cuSOLVER workspace",
                    stream.device());
    }

    StreamAllocation(StreamAllocation const&) = delete;
    StreamAllocation& operator=(StreamAllocation const&) = delete;
    ~StreamAllocation() noexcept
    {
      if (data_ == nullptr) return;
      try
      {
        cuda::ScopedDevice guard(stream_->device());
        auto const status = stream_ordered_ ? cudaFreeAsync(data_, stream_->native_handle()) : cudaFree(data_);
        if (status != cudaSuccess)
          PANIC("cuSOLVER workspace cleanup failed", stream_->device(), cudaGetErrorName(status),
                cudaGetErrorString(status));
      }
      catch (...)
      {
        PANIC("CUDA device selection failed while freeing cuSOLVER workspace", stream_->device());
      }
    }

    [[nodiscard]] T* data() const noexcept { return data_; }

  private:
    cuda::Stream const* stream_ = nullptr;
    T* data_ = nullptr;
    bool stream_ordered_ = false;
};

} // namespace uni20::linalg::detail::cusolver_backend
