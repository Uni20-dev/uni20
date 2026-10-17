#pragma once

#include "basic_tensor.hpp"

namespace uni20
{
/// \brief Resolve a tensor's default without reading elements; fixed types use native precision.
template <TensorView Input> [[nodiscard]] constexpr Precision precision_of(Input const& input)
{
  using S = tensor_element_t<Input>;
  if constexpr (has_runtime_precision_v<S>)
  {
    if constexpr (requires { input.default_precision(); })
      return input.default_precision();
    else
      throw std::logic_error("tensor: no default working precision was supplied");
  }
  else
    return native_precision<S>();
}

/// \brief Materialize an owning tensor with every element converted to p.
/// \details The source is unchanged; the result's construction default is p.
///          Unset source elements are not numerical values and conversion throws.
template <TensorView Input> [[nodiscard]] auto at_precision(Input const& input, Precision p)
{
  using S = tensor_element_t<Input>;
  require_precision<S>(p);
  auto result = Tensor(input);
  if constexpr (has_runtime_precision_v<S>)
  {
    result.default_precision(p);
    for (auto& value : result.storage())
      value = value.at(p);
  }
  return result;
}

/// \brief Select a binary operation's default without inspecting element precisions.
/// \details Exact defaults are neutral. Missing or differing finite defaults
///          require explicit operation precision. Two exact defaults remain exact;
///          approximation-only kernels require a finite override.
template <TensorView A, TensorView B> [[nodiscard]] constexpr Precision common_default_precision(A const& a, B const& b)
{
  return common_precision(precision_of(a), precision_of(b));
}
} // namespace uni20
