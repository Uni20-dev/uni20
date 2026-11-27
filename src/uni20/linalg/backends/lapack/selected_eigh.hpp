#pragma once

#include <uni20/linalg/backends/lapack/self_adjoint_eigh.hpp>

namespace uni20::linalg
{
namespace lapack_detail
{

template <class Values, class Vectors, class Matrix>
KernelAttempt try_selected_eigh(selected_self_adjoint_eigh_op const& op, Values& values, Vectors& vectors,
                                Matrix& matrix)
{
  using scalar = typename Matrix::value_type;
  using real = uni20::make_real_t<scalar>;
  auto const n = static_cast<std::size_t>(matrix.extent(0));
  auto const order = uni20::blas::try_blas_int(n);
  if (!uni20::blas::is_valid_blas_int(order)) return KernelAttempt::unsupported_shape;
  auto const [first, last] = op.indices;
  CHECK(0 <= first && first <= last && std::cmp_less_equal(last, n));
  auto const count = static_cast<std::size_t>(last - first);
  CHECK(std::cmp_equal(matrix.extent(1), n));
  CHECK(std::cmp_equal(values.extent(0), count));
  CHECK(std::cmp_equal(vectors.extent(0), n));
  CHECK(std::cmp_equal(vectors.extent(1), count));
  if (count == 0) return KernelAttempt::success;

  auto a = blas::try_lapack_writable_matrix(matrix);
  auto z = blas::try_lapack_writable_matrix(vectors);
  auto w = blas::try_lapack_writable_vector(values);
  if (!a || !z || !w) return KernelAttempt::unsupported_layout;
  // The MPLAPACK wrapper copies its full 2*n support array back to the caller.
  auto const support_size = try_size_product(2, n);
  if (!support_size || !uni20::blas::is_valid_blas_int(uni20::blas::try_blas_int(*support_size)))
    return KernelAttempt::unsupported_shape;
  std::vector<blas_int> support(*support_size);
  // LAPACK requires room for n values even when only count are requested.
  std::vector<real> scratch_values(n);
  auto const uplo = lapack_triangle(op.triangle);
  auto const il = static_cast<blas_int>(first + 1);
  auto const iu = static_cast<blas_int>(last);
  blas_int selected = 0;
  auto run = [&](scalar* work, blas_int lwork, real* rwork, blas_int lrwork, blas_int* iwork, blas_int liwork) {
    if constexpr (uni20::LapackReal<scalar>)
      uni20::lapack::syevr('V', 'I', uplo, order, a->data, a->leading_dimension, real{}, real{}, il, iu, real{},
                           selected, scratch_values.data(), z->data, z->leading_dimension, support.data(), work, lwork,
                           iwork, liwork);
    else
      uni20::lapack::heevr('V', 'I', uplo, order, a->data, a->leading_dimension, real{}, real{}, il, iu, real{},
                           selected, scratch_values.data(), z->data, z->leading_dimension, support.data(), work, lwork,
                           rwork, lrwork, iwork, liwork);
  };

  scalar work_query{};
  real rwork_query{};
  blas_int iwork_query{};
  run(&work_query, -1, &rwork_query, -1, &iwork_query, -1);
  auto const lwork = workspace_size(work_query);
  auto const lrwork = uni20::LapackReal<scalar> ? blas_int{0} : workspace_size(rwork_query);
  CHECK(iwork_query >= 1 && std::in_range<std::size_t>(iwork_query), iwork_query);
  std::vector<scalar> work(static_cast<std::size_t>(lwork));
  std::vector<real> rwork(static_cast<std::size_t>(lrwork));
  std::vector<blas_int> iwork(static_cast<std::size_t>(iwork_query));
  run(work.data(), lwork, rwork.data(), lrwork, iwork.data(), iwork_query);
  ERROR_IF(!std::cmp_equal(selected, count), "LAPACK eigensolver returned an unexpected eigenvalue count", selected,
           count);
  std::copy_n(scratch_values.data(), count, w->data);
  return KernelAttempt::success;
}

} // namespace lapack_detail

/// \brief Identify host descriptors supported by the selected LAPACK eigensolver.
template <uni20::MutableRankedStridedMdspecLike<1> Values, uni20::MutableRankedStridedMdspecLike<2> Vectors,
          uni20::MutableRankedStridedMdspecLike<2> Matrix>
  requires uni20::HostWritableMdspec<Values> && uni20::HostWritableMdspec<Vectors> && uni20::HostWritableMdspec<Matrix>
consteval auto kernel_accepts_types(LapackBackend const&, selected_self_adjoint_eigh_op const&, Values&, Vectors&,
                                    Matrix&)
{
  using value_span = uni20::host_write_mdspan_t<Values>;
  using vector_span = uni20::host_write_mdspan_t<Vectors>;
  using matrix_span = uni20::host_write_mdspan_t<Matrix>;
  if constexpr (std::same_as<typename vector_span::value_type, typename matrix_span::value_type> &&
                uni20::DefaultAccessorMdspanLike<vector_span> &&
                lapack_detail::self_adjoint_eigh_acceptance<value_span, matrix_span>() != KernelTypeAcceptance::no)
    return kernel_types_maybe;
  else
    return kernel_types_no;
}

/// \brief Compute compact selected eigenpairs into distinct column-major host outputs.
template <class Values, class Vectors, class Matrix>
KernelAttempt try_kernel(LapackBackend, selected_self_adjoint_eigh_op const& op, Values& values, Vectors& vectors,
                         Matrix& matrix)
{
  return lapack_detail::with_host_write_mdspans(
      [&](auto& w, auto& z, auto& a) { return lapack_detail::try_selected_eigh(op, w, z, a); }, values, vectors,
      matrix);
}

} // namespace uni20::linalg
