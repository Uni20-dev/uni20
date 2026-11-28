#pragma once

#include <uni20/linalg/detail/nystrom_whitening.hpp>
#include <uni20/linalg/detail/randomized_range.hpp>
#include <uni20/linalg/ops/self_adjoint_eigh.hpp>
#include <uni20/linalg/ops/svd.hpp>
#include <uni20/tensor/transform.hpp>

namespace uni20::linalg
{
/// \brief Leading Nyström eigenpairs in descending order, and the dimensions of the sketch.
template <class Values, class Vectors> struct NystromResult
{
    Values eigenvalues;
    Vectors eigenvectors;
    RandomizedRankInfo truncation;
};
/// \brief Approximate the leading eigenpairs of a real symmetric positive semidefinite matrix.
/// \pre Both triangles of the square input contain the symmetric matrix.
/// \details Each power step applies the input once and reorthogonalizes the sketch.
///          Null directions of the projected Gram matrix use a numerical pseudoinverse.
template <RankedTensorView<2> Matrix>
  requires detail::randomized::SketchReal<tensor_element_t<Matrix>>
[[nodiscard]] auto randomized_nystrom(Matrix const& a, RandomizedRangeOptions options, random::Identity identity)
{
  namespace impl = detail::randomized;
  index_type const n = a.extent(0), k = options.rank;
  if (n != a.extent(1)) throw std::invalid_argument("randomized Nystrom requires a square PSD matrix");
  auto width = impl::sketch_width(n, n, options);
  auto eigenvalues = make_tensor_like(uninitialized, a, k);
  auto eigenvectors = make_tensor_like(uninitialized, a, n, k);
  if (k != 0)
  {
    auto q = orthonormal_basis(impl::gaussian(a, width, identity));
    for (index_type step = 0; step < options.power_iterations; ++step)
      q = orthonormal_basis(impl::product(a, q));
    auto y = impl::product(a, q);
    auto projected = impl::product(permute<1, 0>(q), y);
    auto spectrum = make_tensor_like(uninitialized, a, width);
    self_adjoint_eigh(spectrum, projected);
    // If W = V D V^T, the Nyström approximation is F F^T with F = Y V (D^+)^{1/2}.
    impl::whiten_columns(projected, spectrum);
    auto factor = impl::product(y, projected);
    auto basis = make_tensor_like(uninitialized, a, n, width);
    auto right = make_tensor_like(uninitialized, a, width, width);
    singular_value_decomposition(spectrum, basis, right, factor);
    impl::copy_prefix(eigenvectors, basis);
    impl::apply_prefix(transform_op{square{}}, eigenvalues, spectrum);
  }
  return NystromResult<decltype(eigenvalues), decltype(eigenvectors)>{
      std::move(eigenvalues), std::move(eigenvectors), {k, width}};
}
} // namespace uni20::linalg
