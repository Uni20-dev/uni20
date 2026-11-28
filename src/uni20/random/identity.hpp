#pragma once

#include <uni20/random/philox.hpp>

#include <stdexcept>

namespace uni20::random
{
/// \brief Provisional single-process Philox identity. Labels and positions each use 63 bits.
struct Identity
{
    std::uint64_t key;
    std::uint64_t prefix;
    UNI20_HOST_DEVICE constexpr bool operator==(Identity const&) const = default;
};

namespace detail
{
inline constexpr std::uint64_t value_domain = std::uint64_t{1} << 63;

[[noreturn]] UNI20_HOST_DEVICE inline void address_out_of_range()
{
#ifdef __CUDA_ARCH__
  __trap();
  __builtin_unreachable();
#else
  throw std::out_of_range("Philox label or position exceeds its 63-bit domain");
#endif
}

UNI20_HOST_DEVICE constexpr Philox4x32Block evaluate(Identity node, std::uint64_t local)
{
  return philox4x32_10({{static_cast<std::uint32_t>(local), static_cast<std::uint32_t>(local >> 32),
                         static_cast<std::uint32_t>(node.prefix), static_cast<std::uint32_t>(node.prefix >> 32)}},
                       {{static_cast<std::uint32_t>(node.key), static_cast<std::uint32_t>(node.key >> 32)}});
}
} // namespace detail

/// \brief Seed the provisional identity mapping for single-process dense draws.
[[nodiscard]] UNI20_HOST_DEVICE constexpr Identity root(std::uint64_t seed) { return {seed, 1}; }

/// \brief Derive a child from a caller-assigned label unique among its siblings.
[[nodiscard]] UNI20_HOST_DEVICE constexpr Identity child(Identity node, std::uint64_t label)
{
  if (label >= detail::value_domain) detail::address_out_of_range();
  auto const w = detail::evaluate(node, label);
  return {(std::uint64_t{w.words[1]} << 32) | w.words[0], (std::uint64_t{w.words[3]} << 32) | w.words[2]};
}

/// \brief Return the four words at a leaf's zero-based evaluation position.
[[nodiscard]] UNI20_HOST_DEVICE constexpr Philox4x32Block value_block(Identity leaf, std::uint64_t position)
{
  if (position >= detail::value_domain) detail::address_out_of_range();
  return detail::evaluate(leaf, detail::value_domain | position);
}
} // namespace uni20::random
