#pragma once

#include <uni20/linalg/detail/randomized_range.hpp>
#include <uni20/linalg/ops/truncated_svd.hpp>

namespace uni20::linalg
{
/// \brief Approximate the leading real singular triplets using a Gaussian range finder.
/// \details Supply a fresh identity for an independent sketch. The returned U, s and Vh are
///          in the input storage domain; the fourth member records retained rank and sketch width.
template <RankedTensorView<2> Matrix>
  requires detail::randomized::SketchMatrix<Matrix>
[[nodiscard]] auto randomized_truncated_svd(Matrix const& a, RandomizedRangeOptions options, random::Identity identity)
{
  namespace impl = detail::randomized;
  index_type const m = a.extent(0), n = a.extent(1), k = options.rank;
  auto const width = impl::sketch_width(m, n, options);
  auto u = make_tensor_like(uninitialized, a, m, k);
  auto s = make_tensor_like(uninitialized, a, k);
  auto vh = make_tensor_like(uninitialized, a, k, n);
  if (k != 0)
  {
    auto at = permute<1, 0>(a);
    auto q = orthonormal_basis(impl::product(a, impl::gaussian(a, width, identity)));
    for (index_type i = 0; i < options.power_iterations; ++i)
    {
      auto z = orthonormal_basis(impl::product(at, q));
      q = orthonormal_basis(impl::product(a, z));
    }
    // Factor B^T, so the projected factorization always has at least as many rows as columns.
    auto bt = impl::product(at, q);
    auto values = make_tensor_like(uninitialized, a, width);
    auto v = make_tensor_like(uninitialized, a, n, width);
    auto ubt = make_tensor_like(uninitialized, a, width, width);
    singular_value_decomposition(values, v, ubt, bt);
    auto ub = make_tensor_like(uninitialized, a, width, k);
    impl::copy_prefix(ub, permute<1, 0>(ubt));
    using R = tensor_element_t<Matrix>;
    gemm(u, R{1}, q, ub, R{0});
    impl::copy_prefix(s, values);
    impl::copy_prefix(vh, permute<1, 0>(v));
  }
  return TruncatedSvdResult<decltype(u), decltype(s), decltype(vh), RandomizedRankInfo>{
      std::move(u), std::move(s), std::move(vh), {k, width}};
}
} // namespace uni20::linalg
