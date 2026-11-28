#pragma once

#include <uni20/core/types.hpp>
#include <uni20/random/identity.hpp>
#if UNI20_HAS_MPREAL
#include <uni20/core/mpreal.hpp>
#include <uni20/random/stream.hpp>
#include <memory>
#endif

#include <array>

namespace uni20::random
{
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
      auto const bits = detail::word_pair(w[0], w[1]);
      return static_cast<double>(bits >> 11) * 0x1p-53;
    }
    UNI20_HOST_DEVICE static constexpr double open_closed(words_type const& w) { return closed_open(w) + 0x1p-53; }
};

#if UNI20_HAS_FLOAT80
template <> struct UnitUniform<float80>
{
    static constexpr unsigned word_count = 2;
    using words_type = std::array<std::uint32_t, word_count>;

    static constexpr float80 closed_open(words_type const& w)
    {
      return float80(detail::word_pair(w[0], w[1])) * 0x1p-64L;
    }
    static constexpr float80 open_closed(words_type const& w) { return closed_open(w) + 0x1p-64L; }
};
#endif

#if UNI20_HAS_FLOAT128
template <> struct UnitUniform<float128>
{
    static constexpr unsigned word_count = 4;
    using words_type = std::array<std::uint32_t, word_count>;

    static constexpr float128 closed_open(words_type const& w)
    {
      auto const high = detail::word_pair(w[2], w[3]);
      auto const low = detail::word_pair(w[0], w[1]);
      return float128(high) * 0x1p-64 + float128(low >> 15) * 0x1p-113;
    }
    static constexpr float128 open_closed(words_type const& w) { return closed_open(w) + 0x1p-113; }
};
#endif

#if UNI20_HAS_MPREAL
/// \brief Use four outer words as a nested Philox identity for one uniform at finite working precision.
template <> struct UnitUniform<mpreal>
{
    static constexpr unsigned word_count = 4;
    using words_type = std::array<std::uint32_t, word_count>;

    UnitUniform(Precision p) : precision_(p) { (void)uni20::detail::mpfr_precision(p); }
    mpreal closed_open(words_type const& w) const { return this->draw(w, false); }
    mpreal open_closed(words_type const& w) const { return this->draw(w, true); }

  private:
    Precision precision_;

    mpreal draw(words_type const& w, bool upper_endpoint) const
    {
      PhiloxStream stream{detail::words_to_identity(w.data())};
      auto const bits = precision_.bit_count();
      auto const count = static_cast<std::size_t>(1 + (bits - 1) / 32);
      std::array<std::uint32_t, 64> local_words;
      std::unique_ptr<std::uint32_t[]> large_words;
      auto* words = local_words.data();
      if (count > local_words.size())
      {
        large_words = std::make_unique_for_overwrite<std::uint32_t[]>(count);
        words = large_words.get();
      }
      for (std::size_t i = 0; i < count; ++i)
        words[i] = stream();

      uni20::detail::mp_integer mantissa;
      // Import whole lanes in one pass, then discard the unused low bits of the final lane.
      mpz_import(mantissa.value, count, 1, sizeof(words[0]), 0, 0, words);
      mpz_fdiv_q_2exp(mantissa.value, mantissa.value, (32 - bits % 32) % 32);
      if (upper_endpoint) mpz_add_ui(mantissa.value, mantissa.value, 1);
      return uni20::detail::mpreal_access::finite_result(
          precision_, [&](mpfr_ptr out) { mpfr_set_z_2exp(out, mantissa.value, -bits, MPFR_RNDN); });
    }
};
#endif
} // namespace uni20::random
