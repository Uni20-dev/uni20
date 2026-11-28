#pragma once

#include <algorithm>
#include <array>
#include <stdexcept>
#include <uni20/linalg/ops/gemm.hpp>
#include <uni20/linalg/ops/qr.hpp>
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
concept SketchReal =
#if UNI20_HAS_MPLAPACK_FLOAT80 || UNI20_HAS_MPLAPACK_MPREAL
    packed_detail::BasisScalar<R> ||
#endif
    LapackReal<R>;

template <class Matrix>
concept SketchMatrix = SketchReal<tensor_element_t<Matrix>> &&
                       (HostReadableTensor<Matrix> || std::same_as<tensor_element_t<Matrix>, float> ||
                        std::same_as<tensor_element_t<Matrix>, double>);

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
  using R = tensor_element_t<Matrix>;
  if constexpr (has_runtime_precision_v<R>)
    random::fill(omega, leaf, random::NormalFor<R>{.converter = {precision_of(a)}});
  else
    random::fill(omega, leaf, random::detail::SketchNormal<R>{});
  return omega;
}

// Apply a unary operation to the upper-left prefix, preserving the internal factor's actual strides.
template <class Operation, class Output, class Source>
void apply_prefix(Operation op, Output& out, Source const& source)
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
  dispatch_kernel(select_backend(op, out, source), op, dest, cropped);
}

template <class Output, class Source> void copy_prefix(Output& out, Source const& source)
{
  apply_prefix(copy_op{}, out, source);
}
} // namespace detail::randomized

} // namespace uni20::linalg
