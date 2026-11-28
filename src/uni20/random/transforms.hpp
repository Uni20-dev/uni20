#pragma once

#include <uni20/core/types.hpp>
#include <uni20/random/philox.hpp>

#include <cmath>
#include <concepts>
#include <numbers>

namespace uni20::random
{

/// \brief Values produced by one evaluation, in increasing position order.
template <class T, unsigned N> struct ValueBlock
{
    static constexpr unsigned size = N;
    T values[N];
};

namespace detail
{
UNI20_HOST_DEVICE constexpr std::uint64_t uniform_bits53(Philox4x32Block block, unsigned i)
{
  auto const word = (std::uint64_t{block.words[2 * i + 1]} << 32) | block.words[2 * i];
  return word >> 11;
}

UNI20_HOST_DEVICE inline ValueBlock<double, 2> standard_normal_pair(Philox4x32Block block)
{
  double const u1 = static_cast<double>(uniform_bits53(block, 0) + 1) * 0x1p-53;
  double const u2 = static_cast<double>(uniform_bits53(block, 1)) * 0x1p-53;
  double const radius = ::sqrt(-2.0 * ::log(u1));

  // Reduce in exact binary quarter turns before multiplying by pi. This retains
  // the small component when the sampled angle is close to a coordinate axis.
  double const turns = 4 * u2;
  int const quadrant = static_cast<int>(turns + 0.5);
  double const a = (turns - quadrant) * (std::numbers::pi / 2);
  double const c = radius * ::cos(a);
  double const s = radius * ::sin(a);
  switch (quadrant & 3)
  {
    case 1:
      return {{-s, c}};
    case 2:
      return {{-c, -s}};
    case 3:
      return {{s, -c}};
    default:
      return {{c, s}};
  }
}
} // namespace detail

/// \brief Transform a block into four float or two double uniform values.
/// \pre The endpoints are finite and ordered, and their difference is representable.
template <class T>
  requires(std::same_as<T, float> || std::same_as<T, double>)
struct Uniform
{
    T low = 0;
    T high = 1;

    UNI20_HOST_DEVICE constexpr auto operator()(Philox4x32Block block) const
    {
      ValueBlock<T, std::same_as<T, float> ? 4 : 2> result{};
      for (unsigned i = 0; i < result.size; ++i)
      {
        T u;
        if constexpr (std::same_as<T, float>)
          u = static_cast<float>(block.words[i] >> 8) * 0x1p-24f;
        else
          u = static_cast<double>(detail::uniform_bits53(block, i)) * 0x1p-53;
        result.values[i] = low + (high - low) * u;
      }
      return result;
    }
};

/// \brief Transform a block into two real normals with mean mu and standard deviation sigma.
/// \pre sigma is finite and nonnegative.
struct Normal
{
    double mu = 0;
    double sigma = 1;

    UNI20_HOST_DEVICE auto operator()(Philox4x32Block block) const
    {
      auto pair = detail::standard_normal_pair(block);
      for (auto& x : pair.values)
        x = mu + sigma * x;
      return pair;
    }
};

/// \brief Transform a block into one circular complex normal with E|z - mu|² = sigma².
/// \pre sigma is finite and nonnegative.
struct ComplexNormal
{
    complex<double> mu = {};
    double sigma = 1;

    UNI20_HOST_DEVICE ValueBlock<complex<double>, 1> operator()(Philox4x32Block block) const
    {
      auto const pair = detail::standard_normal_pair(block);
      double const scale = sigma * (std::numbers::sqrt2 / 2);
      return {{complex<double>{mu.real() + scale * pair.values[0], mu.imag() + scale * pair.values[1]}}};
    }
};

} // namespace uni20::random
