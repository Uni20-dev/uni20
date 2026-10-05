#pragma once

#include "mpreal.hpp"

namespace uni20
{
namespace detail
{
// Rational values of sin(pi*x) occur on this grid. Reduce exactly, including
// for numerators too large for any native floating-point format.
inline mpreal rational_sinpi(exact_constant const& x)
{
  auto sixths = x * exact_constant{6};
  auto q = sixths.native_handle();
  if (mpz_cmp_ui(mpq_denref(q), 1) == 0)
  {
    switch (mpz_fdiv_ui(mpq_numref(q), 12))
    {
      case 0:
      case 6:
        return mpreal{};
      case 1:
      case 5:
        return mpreal(exact_constant{1} / exact_constant{2});
      case 3:
        return mpreal{1};
      case 7:
      case 11:
        return mpreal(exact_constant{-1} / exact_constant{2});
      case 9:
        return mpreal{-1};
    }
  }
  throw std::logic_error("sinpi requires finite precision for an irrational result");
}
inline mpreal rational_asinpi(exact_constant const& x)
{
  if (x == 0) return mpreal{};
  if (x == 1 || x == -1) return mpreal(x / exact_constant{2});
  if (x * exact_constant{2} == 1 || x * exact_constant{2} == -1) return mpreal(x / exact_constant{3});
  throw std::logic_error("asinpi requires finite precision for this input");
}

template <auto Operation> mpreal extended_unary(mpreal const& x)
{
  auto p = x.precision();
  if (p.is_exact())
  {
    auto const& q = x.exact_value();
    if constexpr (Operation == mpfr_exp10)
      return mpreal(uni20::pow(exact_constant{10}, q));
    else if constexpr (Operation == mpfr_exp2m1 || Operation == mpfr_exp10m1)
      return mpreal(uni20::pow(exact_constant{Operation == mpfr_exp2m1 ? 2 : 10}, q) - exact_constant{1});
    else if constexpr (Operation == mpfr_log2p1 || Operation == mpfr_log10p1)
      return mpreal_access::exact_logarithm(q + exact_constant{1}, Operation == mpfr_log2p1 ? 2 : 10);
    else if constexpr (Operation == mpfr_sinpi)
      return rational_sinpi(q);
    else if constexpr (Operation == mpfr_cospi)
      return rational_sinpi(q + exact_constant{1} / exact_constant{2});
    else if constexpr (Operation == mpfr_tanpi)
    {
      auto quarters = q * exact_constant{4};
      auto value = quarters.native_handle();
      if (mpz_cmp_ui(mpq_denref(value), 1) == 0)
      {
        switch (mpz_fdiv_ui(mpq_numref(value), 4))
        {
          case 0:
            return mpreal{};
          case 1:
            return mpreal{1};
          case 3:
            return mpreal{-1};
          case 2:
            throw std::domain_error("exact tanpi is undefined at half-integers");
        }
      }
    }
    else if constexpr (Operation == mpfr_asinpi)
      return rational_asinpi(q);
    else if constexpr (Operation == mpfr_acospi)
      return mpreal(exact_constant{1} / exact_constant{2} - rational_asinpi(q).exact_value());
    else if constexpr (Operation == mpfr_atanpi)
    {
      if (q == 0) return mpreal{};
      if (q == 1 || q == -1) return mpreal(q / exact_constant{4});
    }
    else if constexpr (Operation == mpfr_sec || Operation == mpfr_sech)
    {
      if (q == 0) return mpreal{1};
    }
    throw std::logic_error("extended elementary function requires finite working precision");
  }
  return mpreal_access::finite_result(p, [&](mpfr_ptr out) { Operation(out, x.native_handle(), MPFR_RNDN); });
}
template <auto Operation> mpreal extended_unary(mpreal const& x, Precision p)
{
  (void)p.bit_count();
  return extended_unary<Operation>(x.at(p));
}
} // namespace detail

/// \brief Base-ten exponential.
inline mpreal exp10(mpreal const& x) { return detail::extended_unary<mpfr_exp10>(x); }
/// \brief Convert x to finite precision p, then evaluate exp10 at p.
inline mpreal exp10(mpreal const& x, Precision p) { return detail::extended_unary<mpfr_exp10>(x, p); }

/// \brief Two to the power x minus one, using the cancellation-safe provider.
inline mpreal exp2m1(mpreal const& x) { return detail::extended_unary<mpfr_exp2m1>(x); }
/// \brief Convert x to finite precision p, then evaluate exp2m1 at p.
inline mpreal exp2m1(mpreal const& x, Precision p) { return detail::extended_unary<mpfr_exp2m1>(x, p); }

/// \brief Ten to the power x minus one, using the cancellation-safe provider.
inline mpreal exp10m1(mpreal const& x) { return detail::extended_unary<mpfr_exp10m1>(x); }
/// \brief Convert x to finite precision p, then evaluate exp10m1 at p.
inline mpreal exp10m1(mpreal const& x, Precision p) { return detail::extended_unary<mpfr_exp10m1>(x, p); }

/// \brief Base-two logarithm of one plus x, without rounding the sum first.
inline mpreal log2p1(mpreal const& x) { return detail::extended_unary<mpfr_log2p1>(x); }
/// \brief Convert x to finite precision p, then evaluate log2p1 at p.
inline mpreal log2p1(mpreal const& x, Precision p) { return detail::extended_unary<mpfr_log2p1>(x, p); }

/// \brief Base-ten logarithm of one plus x, without rounding the sum first.
inline mpreal log10p1(mpreal const& x) { return detail::extended_unary<mpfr_log10p1>(x); }
/// \brief Convert x to finite precision p, then evaluate log10p1 at p.
inline mpreal log10p1(mpreal const& x, Precision p) { return detail::extended_unary<mpfr_log10p1>(x, p); }

/// \brief Sine of pi times x, with provider argument reduction.
inline mpreal sinpi(mpreal const& x) { return detail::extended_unary<mpfr_sinpi>(x); }
/// \brief Convert x to finite precision p, then evaluate sinpi at p.
inline mpreal sinpi(mpreal const& x, Precision p) { return detail::extended_unary<mpfr_sinpi>(x, p); }

/// \brief Cosine of pi times x, with provider argument reduction.
inline mpreal cospi(mpreal const& x) { return detail::extended_unary<mpfr_cospi>(x); }
/// \brief Convert x to finite precision p, then evaluate cospi at p.
inline mpreal cospi(mpreal const& x, Precision p) { return detail::extended_unary<mpfr_cospi>(x, p); }

/// \brief Tangent of pi times x, with provider argument reduction.
inline mpreal tanpi(mpreal const& x) { return detail::extended_unary<mpfr_tanpi>(x); }
/// \brief Convert x to finite precision p, then evaluate tanpi at p.
inline mpreal tanpi(mpreal const& x, Precision p) { return detail::extended_unary<mpfr_tanpi>(x, p); }

/// \brief Principal arcsine divided by pi.
inline mpreal asinpi(mpreal const& x) { return detail::extended_unary<mpfr_asinpi>(x); }
/// \brief Convert x to finite precision p, then evaluate asinpi at p.
inline mpreal asinpi(mpreal const& x, Precision p) { return detail::extended_unary<mpfr_asinpi>(x, p); }

/// \brief Principal arccosine divided by pi.
inline mpreal acospi(mpreal const& x) { return detail::extended_unary<mpfr_acospi>(x); }
/// \brief Convert x to finite precision p, then evaluate acospi at p.
inline mpreal acospi(mpreal const& x, Precision p) { return detail::extended_unary<mpfr_acospi>(x, p); }

/// \brief Principal arctangent divided by pi.
inline mpreal atanpi(mpreal const& x) { return detail::extended_unary<mpfr_atanpi>(x); }
/// \brief Convert x to finite precision p, then evaluate atanpi at p.
inline mpreal atanpi(mpreal const& x, Precision p) { return detail::extended_unary<mpfr_atanpi>(x, p); }

/// \brief Reciprocal cosine, evaluated directly by MPFR.
inline mpreal sec(mpreal const& x) { return detail::extended_unary<mpfr_sec>(x); }
/// \brief Convert x to finite precision p, then evaluate sec at p.
inline mpreal sec(mpreal const& x, Precision p) { return detail::extended_unary<mpfr_sec>(x, p); }

/// \brief Reciprocal sine, evaluated directly by MPFR.
inline mpreal csc(mpreal const& x) { return detail::extended_unary<mpfr_csc>(x); }
/// \brief Convert x to finite precision p, then evaluate csc at p.
inline mpreal csc(mpreal const& x, Precision p) { return detail::extended_unary<mpfr_csc>(x, p); }

/// \brief Reciprocal tangent, evaluated directly by MPFR.
inline mpreal cot(mpreal const& x) { return detail::extended_unary<mpfr_cot>(x); }
/// \brief Convert x to finite precision p, then evaluate cot at p.
inline mpreal cot(mpreal const& x, Precision p) { return detail::extended_unary<mpfr_cot>(x, p); }

/// \brief Reciprocal hyperbolic cosine, evaluated directly by MPFR.
inline mpreal sech(mpreal const& x) { return detail::extended_unary<mpfr_sech>(x); }
/// \brief Convert x to finite precision p, then evaluate sech at p.
inline mpreal sech(mpreal const& x, Precision p) { return detail::extended_unary<mpfr_sech>(x, p); }

/// \brief Reciprocal hyperbolic sine, evaluated directly by MPFR.
inline mpreal csch(mpreal const& x) { return detail::extended_unary<mpfr_csch>(x); }
/// \brief Convert x to finite precision p, then evaluate csch at p.
inline mpreal csch(mpreal const& x, Precision p) { return detail::extended_unary<mpfr_csch>(x, p); }

/// \brief Reciprocal hyperbolic tangent, evaluated directly by MPFR.
inline mpreal coth(mpreal const& x) { return detail::extended_unary<mpfr_coth>(x); }
/// \brief Convert x to finite precision p, then evaluate coth at p.
inline mpreal coth(mpreal const& x, Precision p) { return detail::extended_unary<mpfr_coth>(x, p); }

/// \brief Quadrant-aware arctangent of y/x, divided by pi.
/// \details Exact axes and diagonals retain rational results; two exact zeros
///          have no direction and throw. Finite inputs follow MPFR signed-zero rules.
inline mpreal atan2pi(mpreal const& y, mpreal const& x)
{
  auto p = common_precision(y.precision(), x.precision());
  if (p.is_exact())
  {
    if (y == 0)
    {
      if (x == 0) throw std::domain_error("exact atan2pi needs a nonzero direction");
      return mpreal{x < 0 ? 1 : 0};
    }
    int sign = y < 0 ? -1 : 1;
    if (x == 0) return mpreal(exact_constant{sign} / exact_constant{2});
    if (abs(x) == abs(y)) return mpreal(exact_constant{sign * (x < 0 ? 3 : 1)} / exact_constant{4});
    throw std::logic_error("atan2pi requires finite precision for this direction");
  }
  auto yv = y.at(p), xv = x.at(p);
  return detail::mpreal_access::finite_result(
      p, [&](mpfr_ptr out) { mpfr_atan2pi(out, yv.native_handle(), xv.native_handle(), MPFR_RNDN); });
}
/// \brief Convert y and x to p, then evaluate atan2pi(y,x) at p.
inline mpreal atan2pi(mpreal const& y, mpreal const& x, Precision p)
{
  (void)p.bit_count();
  return atan2pi(y.at(p), x.at(p));
}

} // namespace uni20
