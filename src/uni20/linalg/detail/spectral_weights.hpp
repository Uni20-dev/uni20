#pragma once

#include <uni20/core/numeric_limits.hpp>
#include <uni20/mdspan/repeat_axis.hpp>
#include <uni20/tensor/allocate_like.hpp>
#include <uni20/tensor/transform.hpp>

namespace uni20::linalg::detail::randomized
{
// The eigensolver returns a nonempty ascending spectrum in canonical storage.
// Broadcast its last element from the original allocation so the device never returns a scalar to the host.
template <RankedTensorView<1> Vector> auto inverse_sqrt_weights(Vector const& spectrum)
{
  using R = tensor_element_t<Vector>;
  auto const width = spectrum.extent(0);
  auto weights = make_tensor_like(uninitialized, spectrum, width);
  auto in = mdspec_of(spectrum);
  auto reference = uni20::detail::repeat_vector_element(in, width - 1);
  auto dest = mdspec_of(weights);
  auto op = transform_op{
      relative_inverse_sqrt<R>{static_cast<R>(width) * numeric_limits<R>::epsilon(precision_of(spectrum))}};
  dispatch_kernel(select_backend(op, weights, spectrum), op, dest, in, reference);
  return weights;
}
} // namespace uni20::linalg::detail::randomized
