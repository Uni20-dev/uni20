#pragma once

/**
 * \file packed_matrix.hpp
 * \ingroup linalg
 * \brief Logical host packing for value-converting BLAS and LAPACK adapters.
 */

#include <concepts>
#include <cstddef>
#include <span>
#include <uni20/core/runtime_precision.hpp>
#include <uni20/tensor/tensor.hpp>

namespace uni20::linalg
{
namespace packed_detail
{
template <class Span> auto pack(Span const& span, Precision p, bool read = true)
{
  using S = typename Span::value_type;
  DenseMatrix<S> result(uninitialized, span.extent(0), span.extent(1), p);
  if (read)
    for (std::size_t j = 0; j < std::size_t(span.extent(1)); ++j)
      for (std::size_t i = 0; i < std::size_t(span.extent(0)); ++i)
        result[i, j] = scalar_at_precision<S>(S(span[i, j]), p);
  return result;
}
template <class Tensor> auto values(Tensor& tensor)
{
  return std::span(tensor.storage().data(), tensor.storage().size());
}
template <class Tensor, class Span> void unpack(Tensor const& source, Span& target)
{
  for (std::size_t j = 0; j < std::size_t(target.extent(1)); ++j)
    for (std::size_t i = 0; i < std::size_t(target.extent(0)); ++i)
      target[i, j] = source[i, j];
}
} // namespace packed_detail

} // namespace uni20::linalg
