#pragma once

#include <uni20/core/runtime_precision.hpp>
#include <uni20/symmetry/block_tensor_concepts.hpp>

namespace uni20
{
/// \brief Read a BlockTensor's logical operation default without inspecting its numerical blocks.
template <BlockTensorView Input> [[nodiscard]] Precision precision_of(Input const& input)
{
  if constexpr (has_runtime_precision_v<block_tensor_value_t<Input>>)
  {
    if constexpr (requires { input.default_precision(); })
      return input.default_precision();
    else
      throw std::logic_error("BlockTensor: no default working precision was supplied");
  }
  else
    return native_precision<block_tensor_value_t<Input>>();
}

namespace detail
{
// Scalar promotion determines a fixed result's arithmetic type. Runtime result
// precision instead comes from the logical input defaults or an explicit request.
template <class S, BlockTensorView... Inputs>
[[nodiscard]] Precision block_operation_precision(std::optional<Precision> requested, Inputs const&... inputs)
{
  if (requested)
  {
    require_precision<S>(*requested);
    return *requested;
  }
  if constexpr (has_runtime_precision_v<S>)
  {
    auto p = Precision::exact();
    ((p = common_precision(p, precision_of(inputs))), ...);
    return p;
  }
  else
    return native_precision<S>();
}

template <MutableBlockTensorView Output> void record_block_output_precision(Output& output, Precision p)
{
  if constexpr (requires { output.default_precision(p); }) output.default_precision(p);
}
} // namespace detail
} // namespace uni20
