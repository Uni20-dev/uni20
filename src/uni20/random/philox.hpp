#pragma once

#include <uni20/core/compiler_attributes.hpp>

#include <cstdint>

namespace uni20::random
{

/// \brief Four 32-bit words in Philox4x32 counter and output lane order.
struct Philox4x32Block
{
    std::uint32_t words[4] = {};
};

/// \brief Two 32-bit words in Philox4x32 key order.
struct Philox4x32Key
{
    std::uint32_t words[2] = {};
};

/// \brief Permute a 128-bit counter under a 64-bit key with ten Philox4x32 rounds.
/// \details Counter, key and output use Random123 word order. Each call evaluates
///          exactly one block and all additions of key words are modulo 2^32.
[[nodiscard]] UNI20_HOST_DEVICE constexpr Philox4x32Block philox4x32_10(Philox4x32Block counter, Philox4x32Key key)
{
  for (int round = 0; round < 10; ++round)
  {
    std::uint64_t const product0 = std::uint64_t{0xd2511f53} * counter.words[0];
    std::uint64_t const product2 = std::uint64_t{0xcd9e8d57} * counter.words[2];
    counter = {{static_cast<std::uint32_t>(product2 >> 32) ^ counter.words[1] ^ key.words[0],
                static_cast<std::uint32_t>(product2),
                static_cast<std::uint32_t>(product0 >> 32) ^ counter.words[3] ^ key.words[1],
                static_cast<std::uint32_t>(product0)}};
    key.words[0] += 0x9e3779b9u;
    key.words[1] += 0xbb67ae85u;
  }
  return counter;
}

} // namespace uni20::random
