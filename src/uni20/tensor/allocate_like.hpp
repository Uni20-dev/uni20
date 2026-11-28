#pragma once

#include <uni20/tensor/precision.hpp>
#include <uni20/tensor/tensor.hpp>

namespace uni20
{
/// \brief Allocate uninitialized column-major work storage of a new shape in the source's memory domain.
/// \details A buffer-backed source retains its allocation context. A source without a buffer uses
///          its domain's default placement. Runtime-precision outputs record the supplied precision.
template <TensorView Input, std::integral... Dims>
[[nodiscard]] auto make_tensor_like(uninitialized_t, Input const& input, Precision p, Dims... dims)
{
  using S = tensor_element_t<Input>;
  using Policy = detail::materialized_storage_t<typename tensor_mdspec_t<Input>::accessor_type>;
  using Result = Tensor<S, sizeof...(Dims), Policy>;
  typename Result::extents_type shape{dims...};
  require_precision<S>(p);
  if constexpr (has_runtime_precision_v<S>)
    return Result(uninitialized, shape, p);
  else
  {
    if constexpr (detail::SourceAllocationContext<Policy, Input>)
    {
      auto const descriptor = mdspec_of(input).data_descriptor();
      if (descriptor) return Result(uninitialized, descriptor.buffer().resources(), shape);
    }
    return Result(uninitialized, shape);
  }
}

/// \brief Allocate work storage at the source's default precision.
template <TensorView Input, std::integral... Dims>
[[nodiscard]] auto make_tensor_like(uninitialized_t tag, Input const& input, Dims... dims)
{
  return make_tensor_like(tag, input, precision_of(input), dims...);
}
} // namespace uni20
