#pragma once

#include <algorithm>
#include <array>
#include <stdexcept>
#include <uni20/linalg/ops/gemm.hpp>
#include <uni20/linalg/ops/qr.hpp>
#include <uni20/linalg/ops/truncated_svd.hpp>
#include <uni20/random/fill.hpp>
#include <uni20/tensor/permute.hpp>
#if UNI20_BACKEND_CUSOLVER
#include <uni20/linalg/backends/cusolver/range_qr.hpp>
#endif

namespace uni20::linalg
{
/// \brief Target rank, extra Gaussian columns, and number of stabilized power steps.
struct RandomizedRangeOptions
{
    index_type rank;
    index_type oversampling = 8;
    index_type power_iterations = 1;
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

// All internal matrices use column-major storage in the input's execution domain.
template <class Matrix, class... Dims> auto allocate(Matrix const& input, Dims... dims)
{
  using T = tensor_element_t<Matrix>;
#if UNI20_BACKEND_CUDA
  if constexpr (std::same_as<tensor_storage_policy_t<Matrix>, CudaStorage>)
  {
    static_assert((UNI20_BACKEND_CUSOLVER && UNI20_BACKEND_CUBLAS) ||
                      !std::same_as<tensor_storage_policy_t<Matrix>, CudaStorage>,
                  "randomized CUDA decomposition requires cuBLAS and cuSOLVER");
    auto descriptor = mdspec_of(input).data_descriptor();
    auto& resources = descriptor ? descriptor.buffer().resources() : cuda::device_resources();
    return CudaTensor<T, sizeof...(dims)>(uninitialized, resources, dims...);
  }
  else
#endif
    return Tensor<T, sizeof...(dims)>(uninitialized, dims...);
}

template <class A, class B> auto product(A const& a, B const& b)
{
  using R = tensor_element_t<A>;
  auto out = allocate(a, a.extent(0), b.extent(1));
  linalg::gemm(out, R{1}, a, b, R{0});
  return out;
}

template <class Matrix> auto orthonormalize(Matrix work)
{
#if UNI20_BACKEND_CUSOLVER
  if constexpr (std::same_as<tensor_storage_policy_t<Matrix>, CudaStorage>)
  {
    auto span = mdspec_of(work);
    auto descriptor = cusolver_backend::try_matrix_descriptor(span);
    if (!descriptor) throw std::length_error("randomized QR exceeds cuSOLVER matrix dimensions");
    cusolver_backend::range_qr(*descriptor);
    return work;
  }
  else
#endif
    return linalg::qr(std::move(work)).q;
}

template <class Matrix> auto gaussian(Matrix const& a, index_type width, random::Identity identity)
{
  auto omega = allocate(a, a.extent(1), width);
  auto leaf = random::child(identity, 0);
  if constexpr (std::same_as<tensor_element_t<Matrix>, float>)
    random::fill(omega, leaf, random::detail::RoundedNormalSketch{});
  else
    random::fill(omega, leaf, random::Normal{});
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
  layout_stride::mapping<E> map(out.extents(), strides);
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
  auto const m = a.extent(0), n = a.extent(1), k = options.rank;
  auto const width = impl::sketch_width(m, n, options);
  auto u = impl::allocate(a, m, k);
  auto s = impl::allocate(a, k);
  auto vh = impl::allocate(a, k, n);
  if (k != 0)
  {
    auto at = permute<1, 0>(a);
    auto q = impl::orthonormalize(impl::product(a, impl::gaussian(a, width, identity)));
    for (index_type i = 0; i < options.power_iterations; ++i)
    {
      auto z = impl::orthonormalize(impl::product(at, q));
      q = impl::orthonormalize(impl::product(a, z));
    }
    // Factor B^T, which is always tall and is accepted by cuSOLVER gesvd.
    auto bt = impl::product(at, q);
    auto values = impl::allocate(a, width);
    auto v = impl::allocate(a, n, width);
    auto ubt = impl::allocate(a, width, width);
    singular_value_decomposition(values, v, ubt, bt);
    auto ub = impl::allocate(a, width, k);
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
