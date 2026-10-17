#pragma once

#include <cstdint>
#include <limits>
#include <stdexcept>

namespace uni20
{

/// \brief Exact arithmetic or an explicit binary working precision.
/// \details A provider-independent description, available in every build. It
///          never changes numerical defaults. Providers validate their own ranges.
class Precision {
  public:
    using bit_count_type = std::int64_t;
    /// \brief No approximation has been selected; rational arithmetic remains exact.
    static constexpr Precision exact() noexcept { return Precision(0); }
    constexpr bool is_exact() const noexcept { return bits_ == 0; }

    /// \brief Select a positive binary significand size.
    static constexpr Precision bits(bit_count_type count)
    {
      if (count < 1) throw std::invalid_argument("Precision::bits: unsupported significand size");
      return Precision(count);
    }

    /// \brief Request at least this many decimal digits of significand precision.
    /// \details Uses the upper bound 3.32193 for log2(10); may allocate extra bits.
    static constexpr Precision decimal_digits(std::int64_t count)
    {
      if (count <= 0) throw std::invalid_argument("Precision::decimal_digits: expected a positive count");
      auto const digits = static_cast<std::uint64_t>(count);
      auto const major = digits / 100000;
      auto const minor = ((digits % 100000) * 332193 + 99999) / 100000;
      auto const maximum = static_cast<std::uint64_t>(std::numeric_limits<bit_count_type>::max());
      if (major > maximum / 332193 || minor > maximum - major * 332193)
        throw std::invalid_argument("Precision::decimal_digits: unsupported significand size");
      return Precision::bits(static_cast<bit_count_type>(major * 332193 + minor));
    }

    /// \brief Number of binary significand bits.
    /// \throws std::logic_error If this precision denotes exact arithmetic.
    constexpr bit_count_type bit_count() const
    {
      if (this->is_exact())
        throw std::logic_error("exact arithmetic requires an explicit finite working precision here");
      return bits_;
    }
    constexpr bool operator==(Precision const&) const = default;

  private:
    explicit constexpr Precision(bit_count_type bits) : bits_(bits) {}
    bit_count_type bits_;
};

/// \brief Combine precisions, treating exact values as neutral and rejecting unequal finite precisions.
constexpr Precision common_precision(Precision a, Precision b)
{
  if (a.is_exact()) return b;
  if (b.is_exact()) return a;
  if (a != b) throw std::invalid_argument("mixed precisions require explicit conversion");
  return a;
}

} // namespace uni20
