#pragma once

#include <optional>

#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/precision.hpp>
#include <uni20/tensor/precision.hpp>

namespace uni20::linalg::detail
{
/// \brief Scalar families admitted by dense decomposition frontends; leaf probes decide provider availability.
template <class S>
concept DecompositionScalar = LapackScalar<S> || has_runtime_precision_v<S>;

/// \brief Resolve and validate finite working precision before consuming or resizing tensors.
template <TensorView A> Precision decomposition_precision(A const& a, std::optional<Precision> requested)
{
  auto p = requested ? *requested : precision_of(a);
  require_precision<tensor_element_t<A>>(p);
  (void)p.bit_count();
  return p;
}
} // namespace uni20::linalg::detail
