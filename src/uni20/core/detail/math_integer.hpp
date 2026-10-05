#pragma once

#include <concepts>
#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <uni20/core/numeric_limits.hpp>

namespace uni20::detail
{
template <class T>
concept MathInteger = std::integral<T> && !std::same_as<T, bool>;

// Accept ordinary integer literals without an implicit narrowing conversion at
// the public API boundary. Exponents and orders share a signed 64-bit domain.
template <MathInteger I> std::int64_t math_integer(I value)
{
  if constexpr (sizeof(I) >= sizeof(std::int64_t) && !std::is_signed_v<I>)
    if (value > static_cast<I>(numeric_limits<std::int64_t>::max()))
      throw std::out_of_range("math exponent/order does not fit int64_t");
  if constexpr (sizeof(I) > sizeof(std::int64_t) && std::is_signed_v<I>)
    if (value < static_cast<I>(numeric_limits<std::int64_t>::min()) ||
        value > static_cast<I>(numeric_limits<std::int64_t>::max()))
      throw std::out_of_range("math exponent/order does not fit int64_t");
  return static_cast<std::int64_t>(value);
}
} // namespace uni20::detail
