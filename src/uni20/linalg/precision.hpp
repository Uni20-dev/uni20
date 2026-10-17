#pragma once

#include <uni20/core/runtime_precision.hpp>
#include <uni20/linalg/dispatch.hpp>

namespace uni20::linalg::detail
{
/// \brief Validate the result precision and lower it to a scalar-specific kernel signature.
/// \details Fixed scalar types encode precision in their type; runtime scalar
///          kernels receive the concrete operation precision explicitly.
template <class S, class Selector, class Operation, class... Args>
void dispatch_at_precision(Selector&& selector, Operation const& operation, Precision p, Args&&... args)
{
  require_precision<S>(p);
  if constexpr (has_runtime_precision_v<S>)
    dispatch_kernel(std::forward<Selector>(selector), operation, std::forward<Args>(args)..., p);
  else
    dispatch_kernel(std::forward<Selector>(selector), operation, std::forward<Args>(args)...);
}
} // namespace uni20::linalg::detail
