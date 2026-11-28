#pragma once

#include <uni20/config.hpp>
#if UNI20_BACKEND_CUSOLVER
#include <uni20/linalg/backends/cusolver/gesvdr.hpp>
#include <uni20/linalg/ops/svd.hpp>
#include <uni20/tensor/allocate_like.hpp>

namespace uni20::linalg
{
/// \brief Compute leading real singular triplets with the cuSOLVER Xgesvdr provider sketch.
/// \details The input is preserved. U, descending values and the right adjoint remain on its CUDA device.
///          A disabled vector side has zero retained columns (U) or rows (right adjoint).
///          The provider controls the sketch and accepts no Uni20 replay identity.
template <RankedTensorView<2> Matrix>
  requires cuda::BufferMdspec<tensor_mdspec_t<Matrix>> &&
           detail::cusolver_backend::CusolverRealScalar<tensor_element_t<Matrix>>
[[nodiscard]] auto svd_gesvdr(Matrix const& matrix, CudaGesvdrOptions options)
{
  using S = tensor_element_t<Matrix>;
  using RightAdjoint = CudaMatrix<S, RowMajor>;
  auto const m = static_cast<index_type>(matrix.extent(0)), n = static_cast<index_type>(matrix.extent(1));
  auto const k = options.rank, limit = std::min(m, n);
  if (k < 0 || options.oversampling < 0 || options.power_iterations < 0 ||
      (k > 0 && (k >= limit || options.oversampling >= limit - k)))
    throw std::invalid_argument(
        "Xgesvdr requires nonnegative settings and rank + oversampling < min(m,n) for nonzero rank");
  auto u = make_tensor_like(uninitialized, matrix, m, options.left_vectors ? k : 0);
  auto s = make_tensor_like(uninitialized, matrix, k);
  auto v = make_tensor_like(uninitialized, matrix, n, options.right_vectors ? k : 0);
  if (k != 0)
  {
    auto work = make_tensor_like(uninitialized, matrix, m, n);
    copy(work, matrix);
    detail::cusolver_backend::execute_gesvdr(m, n, options, work.storage(), s.storage(), u.storage(), v.storage());
  }
  // Contiguous real columns of V are already contiguous rows of its adjoint.
  auto vh = RightAdjoint::adopt_storage(
      typename RightAdjoint::mapping_type{typename RightAdjoint::extents_type{v.extent(1), n}},
      std::move(v).release_storage());
  return SvdResult<decltype(u), decltype(s), RightAdjoint>{std::move(u), std::move(s), std::move(vh)};
}
} // namespace uni20::linalg
#endif
