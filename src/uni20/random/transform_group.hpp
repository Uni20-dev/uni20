#pragma once

#include <uni20/random/philox.hpp>

#include <array>
#include <cstddef>
#include <utility>

namespace uni20::random
{
/// \brief Consecutive Philox evaluations consumed together by one fixed-width transform.
template <unsigned B> struct BlockGroup
{
    static_assert(B > 0);
    static constexpr unsigned size = B;
    Philox4x32Block blocks[B];
};

/// \brief Values produced by one group, in increasing position order.
template <class T, unsigned N> struct ValueBlock
{
    static_assert(N > 0);
    using value_type = T;
    static constexpr unsigned size = N;
    T values[N];
};

/// \brief Scalar customization for converting a fixed count of consecutive 32-bit words to a unit uniform.
/// \details closed_open produces [0,1) and open_closed produces (0,1]. Words are in Philox output order.
template <class Real> struct UnitUniform;

template <> struct UnitUniform<float>
{
    static constexpr unsigned word_count = 1;
    using words_type = std::array<std::uint32_t, word_count>;

    UNI20_HOST_DEVICE static constexpr float closed_open(words_type const& w)
    {
      return static_cast<float>(w[0] >> 8) * 0x1p-24f;
    }
    UNI20_HOST_DEVICE static constexpr float open_closed(words_type const& w) { return closed_open(w) + 0x1p-24f; }
};

template <> struct UnitUniform<double>
{
    static constexpr unsigned word_count = 2;
    using words_type = std::array<std::uint32_t, word_count>;

    UNI20_HOST_DEVICE static constexpr double closed_open(words_type const& w)
    {
      auto const bits = (std::uint64_t{w[1]} << 32) | w[0];
      return static_cast<double>(bits >> 11) * 0x1p-53;
    }
    UNI20_HOST_DEVICE static constexpr double open_closed(words_type const& w) { return closed_open(w) + 0x1p-53; }
};

/// \brief Endpoint convention for one distribution input.
enum class UnitInterval
{
  closed_open,
  open_closed
};

/// \brief Compose a scalar word conversion with one distribution application per sequence of unit intervals.
/// \details Sample is the ValueBlock returned by distribution.from_uniforms(u...). Each group contains
///          enough whole Philox blocks for an application, and packs as many complete applications as fit.
template <class Real, class Sample, UnitInterval... Inputs> struct TransformGroup
{
    using converter_type = UnitUniform<Real>;
    static constexpr unsigned uniform_count = sizeof...(Inputs);
    static_assert(uniform_count > 0 && converter_type::word_count > 0);
    static constexpr unsigned words_per_sample = uniform_count * converter_type::word_count;
    using input_type = BlockGroup<(words_per_sample + 3) / 4>;
    static constexpr unsigned samples_per_group = input_type::size * 4 / words_per_sample;
    using result_type = ValueBlock<typename Sample::value_type, samples_per_group * Sample::size>;

    /// \brief Apply a distribution to each set of uniforms, optionally with a stateful scalar converter.
    template <class Distribution>
    UNI20_HOST_DEVICE static constexpr result_type apply(input_type const& group, Distribution const& distribution,
                                                         converter_type const& converter = {})
    {
      result_type result{};
      for (unsigned i = 0; i < samples_per_group; ++i)
      {
        auto values =
            sample(group, i * words_per_sample, distribution, converter, std::make_index_sequence<uniform_count>{});
        for (unsigned j = 0; j < Sample::size; ++j)
          result.values[i * Sample::size + j] = values.values[j];
      }
      return result;
    }

  private:
    template <UnitInterval Interval>
    UNI20_HOST_DEVICE static constexpr Real unit(input_type const& group, unsigned offset,
                                                 converter_type const& converter)
    {
      std::array<std::uint32_t, converter_type::word_count> words{};
      for (unsigned i = 0; i < words.size(); ++i)
        words[i] = group.blocks[(offset + i) / 4].words[(offset + i) % 4];
      if constexpr (Interval == UnitInterval::closed_open)
        return converter.closed_open(words);
      else
        return converter.open_closed(words);
    }

    template <class Distribution, std::size_t... I>
    UNI20_HOST_DEVICE static constexpr Sample sample(input_type const& group, unsigned offset,
                                                     Distribution const& distribution, converter_type const& converter,
                                                     std::index_sequence<I...>)
    {
      return distribution.from_uniforms(unit<Inputs>(group, offset + I * converter_type::word_count, converter)...);
    }
};

} // namespace uni20::random
