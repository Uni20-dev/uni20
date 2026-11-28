#pragma once

#include <uni20/core/numeric_limits.hpp>
#include <uni20/linalg/backends/cpu/nystrom.hpp>
#if UNI20_BACKEND_CUDA
#include <uni20/linalg/backends/cuda/nystrom.hpp>
#endif

namespace uni20::linalg::detail::randomized
{
// The projected eigensolver produces disjoint factors, with eigenvalues in ascending order.
template <class Matrix, class Vector> void whiten_columns(Matrix& columns, Vector const& values)
{
  using R = tensor_element_t<Matrix>;
  auto const width = values.extent(0);
  CHECK_EQUAL(columns.extent(0), width);
  CHECK_EQUAL(columns.extent(1), width);
  if (width == 0) return;
  nystrom_whiten_op<R> op{static_cast<R>(width) * numeric_limits<R>::epsilon()};
  auto out = mdspec_of(columns);
  auto in = mdspec_of(values);
  dispatch_kernel(select_backend(op, columns, values), op, out, in);
}
} // namespace uni20::linalg::detail::randomized
