#pragma once

#include "basic_tensor.hpp"

#if UNI20_ENABLE_MPFR
namespace uni20
{
/// \brief Materialize an owning tensor with every element converted to p.
/// \details The source is unchanged; the result's construction default is p.
///          Unset source elements are not numerical values and conversion throws.
template <TensorView Input>
  requires has_runtime_precision_v<tensor_element_t<Input>>
[[nodiscard]] auto at_precision(Input const& input, Precision p)
{
  auto result = Tensor(input);
  result.default_precision(p);
  for (auto& value : result.storage())
    value = value.at(p);
  return result;
}

/// \brief Select a binary operation's default without inspecting element precisions.
/// \details Missing or differing tensor defaults require an explicit operation precision.
template <TensorView A, TensorView B>
  requires(has_runtime_precision_v<tensor_element_t<A>> && has_runtime_precision_v<tensor_element_t<B>>)
[[nodiscard]] Precision common_default_precision(A const& a, B const& b)
{
  auto p = a.default_precision();
  if (p != b.default_precision())
    throw std::invalid_argument("tensor operation: differing defaults require explicit working precision");
  return p;
}
} // namespace uni20
#endif
