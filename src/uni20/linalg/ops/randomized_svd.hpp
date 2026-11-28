#pragma once

#include <algorithm>
#include <array>
#include <stdexcept>
#include <uni20/linalg/ops/gemm.hpp>
#include <uni20/linalg/ops/qr.hpp>
#include <uni20/linalg/ops/truncated_svd.hpp>
#include <uni20/random/fill.hpp>
#include <uni20/tensor/allocate_like.hpp>
#include <uni20/tensor/permute.hpp>

namespace uni20::linalg
{
/// \brief Target rank, extra Gaussian columns, and number of stabilized power steps.
struct RandomizedRangeOptions
{
    index_type rank;
    index_type oversampling = 8;
    index_type power_iterations = 2;
};

/// \brief Retained factor extent and actual number of Gaussian columns used.
struct RandomizedRankInfo
{
    index_type retained_rank;
    index_type sketch_width;
};

namespace detail::randomized
{
template <class R>
concept SketchReal = std::same_as<R, float> || std::same_as<R, double>;

inline index_type sketch_width(index_type m, index_type n, RandomizedRangeOptions options)
{
  auto limit = std::min(m, n);
  if (options.rank < 0 || options.rank > limit || options.oversampling < 0 || options.power_iterations < 0)
    throw std::invalid_argument("randomized decomposition requires 0 <= rank <= min(rows, columns) and nonnegative "
                                "oversampling and power iterations");
  return options.rank == 0 ? 0 : options.rank + std::min(options.oversampling, limit - options.rank);
}

template <class A, class B> auto product(A const& a, B const& b)
{
  using R = tensor_element_t<A>;
  auto out = make_tensor_like(uninitialized, a, a.extent(0), b.extent(1));
  linalg::gemm(out, R{1}, a, b, R{0});
  return out;
}

template <class Matrix> auto gaussian(Matrix const& a, index_type width, random::Identity identity)
{
  auto omega = make_tensor_like(uninitialized, a, a.extent(1), width);
  auto leaf = random::child(identity, 0);
  random::fill(omega, leaf, random::detail::SketchNormal<tensor_element_t<Matrix>>{});
  return omega;
}

// Copy the upper-left prefix from an internal factor, retaining the source's actual strides.
template <class Output, class Source> void copy_prefix(Output& out, Source const& source)
{
  auto input = mdspec_of(source);
  auto dest = mdspec_of(out);
  using S = decltype(input);
  using E = typename decltype(dest)::extents_type;
  std::array<index_type, S::rank()> strides{};
  for (unsigned d = 0; d < S::rank(); ++d)
    strides[d] = input.stride(d);
  layout_stride::mapping<E> map(dest.extents(), strides);
  using T = typename S::element_type;
  using A = typename S::accessor_type;
  auto cropped = [&] {
    if constexpr (MdspanLike<S>)
      return mdspan<T, E, layout_stride, A>{input.data_handle(), map, input.accessor()};
    else
      return mdspec<T, E, layout_stride, A, typename S::data_descriptor_type>{input.data_descriptor(), map,
                                                                              input.accessor()};
  }();
  dispatch_kernel(select_backend(copy_op{}, out, source), copy_op{}, dest, cropped);
}
} // namespace detail::randomized

/// \brief Approximate the leading real singular triplets using a Gaussian range finder.
/// \details Supply a fresh identity for an independent sketch. The returned U, s and Vh are
///          in the input storage domain; the fourth member records retained rank and sketch width.
template <RankedTensorView<2> Matrix>
  requires detail::randomized::SketchReal<tensor_element_t<Matrix>>
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
