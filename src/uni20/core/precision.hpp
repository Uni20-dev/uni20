#pragma once

#include <uni20/config.hpp>
#if !UNI20_ENABLE_MPFR
#error "Configure Uni20 with UNI20_ENABLE_MPFR=ON to use arbitrary-precision scalars"
#endif

#include <compare>
#include <cstdint>
#include <mpfr.h>
#include <stdexcept>

namespace uni20
{

/// \brief Explicit binary working precision for arbitrary-precision scalars.
/// \details This value never changes an MPFR default or thread-local setting.
class Precision {
  public:
    /// \brief Select a significand size, checked against MPFR's supported range.
    static Precision bits(mpfr_prec_t count)
    {
      if (count < MPFR_PREC_MIN || count > MPFR_PREC_MAX)
        throw std::invalid_argument("Precision::bits: unsupported significand size");
      return Precision(count);
    }

    /// \brief Request at least this many decimal digits of significand precision.
    /// \details Uses the upper bound 3.32193 for log2(10); may allocate extra bits.
    static Precision decimal_digits(std::int64_t count)
    {
      if (count <= 0) throw std::invalid_argument("Precision::decimal_digits: expected a positive count");
      auto const digits = static_cast<std::uint64_t>(count);
      auto const major = digits / 100000;
      auto const minor = ((digits % 100000) * 332193 + 99999) / 100000;
      auto const maximum = static_cast<std::uint64_t>(MPFR_PREC_MAX);
      if (major > maximum / 332193 || minor > maximum - major * 332193)
        throw std::invalid_argument("Precision::decimal_digits: unsupported significand size");
      return Precision::bits(static_cast<mpfr_prec_t>(major * 332193 + minor));
    }

    /// \brief Number of binary significand bits.
    mpfr_prec_t bit_count() const noexcept { return bits_; }
    auto operator<=>(Precision const&) const = default;

  private:
    friend class mpreal;
    explicit Precision(mpfr_prec_t bits) : bits_(bits) {}
    mpfr_prec_t bits_;
};

} // namespace uni20
