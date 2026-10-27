#pragma once

#include <type_traits>

namespace uni20::detail
{
template <auto Function> struct function_identity;

// Select provider operations by template identity, without evaluating pointer
// equality. GCC with UBSan rejects constant-expression comparisons of distinct
// externally declared functions, including MPFR and MPC entry points.
template <auto Left, auto Right>
inline constexpr bool same_function_v = std::is_same_v<function_identity<Left>, function_identity<Right>>;
} // namespace uni20::detail
