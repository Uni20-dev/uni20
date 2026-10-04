#pragma once

#include "detail/math_integer.hpp"
#include "math_results.hpp"
#include "mpreal.hpp"
#include <cstdint>

namespace uni20
{
namespace detail
{
enum class integer_rounding
{
  down,
  up,
  toward_zero,
  away_ties,
  even_ties
};

inline exact_constant round_rational(exact_constant const& x, integer_rounding mode)
{
  return mpreal_access::rational_result([&](mpq_ptr result) {
    auto q = x.native_handle();
    auto numerator = mpq_numref(q), denominator = mpq_denref(q);
    auto integer = mpq_numref(result);
    if (mode == integer_rounding::down)
      mpz_fdiv_q(integer, numerator, denominator);
    else if (mode == integer_rounding::up)
      mpz_cdiv_q(integer, numerator, denominator);
    else
    {
      mp_integer remainder;
      mpz_tdiv_qr(integer, remainder.value, numerator, denominator);
      if (mode == integer_rounding::toward_zero) return;
      mpz_abs(remainder.value, remainder.value);
      mpz_mul_2exp(remainder.value, remainder.value, 1);
      int comparison = mpz_cmp(remainder.value, denominator);
      if (comparison > 0 || (comparison == 0 && (mode == integer_rounding::away_ties || mpz_odd_p(integer))))
      {
        if (mpz_sgn(numerator) < 0)
          mpz_sub_ui(integer, integer, 1);
        else
          mpz_add_ui(integer, integer, 1);
      }
    }
  });
}

template <integer_rounding Mode, auto Operation> mpreal round_integer(mpreal const& x, Precision p)
{
  (void)x.precision();
  if (x.is_exact()) return mpreal(round_rational(x.exact_value(), Mode), p);
  return mpreal_access::finite_result(p, [&](mpfr_ptr result) { Operation(result, x.native_handle(), MPFR_RNDN); });
}
} // namespace detail

/// \brief Round toward negative infinity; exact inputs produce exact integers.
inline mpreal floor(mpreal const& x)
{
  return detail::round_integer<detail::integer_rounding::down, mpfr_rint_floor>(x, x.precision());
}
/// \brief Round toward negative infinity, then round the result to finite precision p.
/// \details The input is not rounded first. Approximate signed zero is retained.
inline mpreal floor(mpreal const& x, Precision p)
{
  (void)p.bit_count();
  return detail::round_integer<detail::integer_rounding::down, mpfr_rint_floor>(x, p);
}

/// \brief Round toward positive infinity; exact inputs produce exact integers.
inline mpreal ceil(mpreal const& x)
{
  return detail::round_integer<detail::integer_rounding::up, mpfr_rint_ceil>(x, x.precision());
}
/// \brief Round toward positive infinity, then round the result to finite precision p.
/// \details The input is not rounded first. Approximate signed zero is retained.
inline mpreal ceil(mpreal const& x, Precision p)
{
  (void)p.bit_count();
  return detail::round_integer<detail::integer_rounding::up, mpfr_rint_ceil>(x, p);
}

/// \brief Round toward zero; exact inputs produce exact integers.
inline mpreal trunc(mpreal const& x)
{
  return detail::round_integer<detail::integer_rounding::toward_zero, mpfr_rint_trunc>(x, x.precision());
}
/// \brief Round toward zero, then round the result to finite precision p.
/// \details The input is not rounded first. Approximate signed zero is retained.
inline mpreal trunc(mpreal const& x, Precision p)
{
  (void)p.bit_count();
  return detail::round_integer<detail::integer_rounding::toward_zero, mpfr_rint_trunc>(x, p);
}

/// \brief Round to nearest integer, with ties away from zero; exact inputs produce exact integers.
inline mpreal round(mpreal const& x)
{
  return detail::round_integer<detail::integer_rounding::away_ties, mpfr_rint_round>(x, x.precision());
}
/// \brief Round to nearest integer, with ties away from zero, then round the result to finite precision p.
/// \details The input is not rounded first. Approximate signed zero is retained.
inline mpreal round(mpreal const& x, Precision p)
{
  (void)p.bit_count();
  return detail::round_integer<detail::integer_rounding::away_ties, mpfr_rint_round>(x, p);
}

/// \brief Round to nearest integer, with ties to even; exact inputs produce exact integers.
inline mpreal round_even(mpreal const& x)
{
  return detail::round_integer<detail::integer_rounding::even_ties, mpfr_rint_roundeven>(x, x.precision());
}
/// \brief Round to nearest integer, with ties to even, then round the result to finite precision p.
/// \details The input is not rounded first. Approximate signed zero is retained.
inline mpreal round_even(mpreal const& x, Precision p)
{
  (void)p.bit_count();
  return detail::round_integer<detail::integer_rounding::even_ties, mpfr_rint_roundeven>(x, p);
}

namespace detail
{
inline mpreal fused_product_sum(mpreal const& a, mpreal const& b, mpreal const& c, Precision p)
{
  (void)a.precision();
  (void)b.precision();
  (void)c.precision();
  if (p.is_exact()) return mpreal(a.exact_value() * b.exact_value() + c.exact_value());
  if (!a.is_exact() && !b.is_exact() && !c.is_exact())
    return mpreal_access::finite_result(p, [&](mpfr_ptr result) {
      mpfr_fma(result, a.native_handle(), b.native_handle(), c.native_handle(), MPFR_RNDN);
    });
  if (!isfinite(a) || !isfinite(b) || !isfinite(c))
  {
    // Nonfinite results depend only on the exact operands' signs and zeros.
    // Materializing a huge rational here could manufacture an extra infinity.
    auto marker = [p](mpreal const& x) {
      return x.is_exact() ? mpreal(mpq_sgn(x.exact_value().native_handle()), p) : x;
    };
    auto av = marker(a), bv = marker(b), cv = marker(c);
    return mpreal_access::finite_result(p, [&](mpfr_ptr result) {
      mpfr_fma(result, av.native_handle(), bv.native_handle(), cv.native_handle(), MPFR_RNDN);
    });
  }
  return mpreal_access::finite_result(p, [&](mpfr_ptr result) {
    if ((a == 0 || b == 0) && c == 0)
    {
      bool negative = (signbit(a) != signbit(b)) && signbit(c);
      mpfr_set_zero(result, negative ? -1 : 1);
      return;
    }
    scaled_rational_ratio({mpreal_access::stored_scaled(a) * mpreal_access::stored_scaled(b),
                           mpreal_access::stored_scaled(c),
                           {exact_constant{1}, 0},
                           {}})
        .round_to(result);
  });
}

inline mpreal binary_scale(mpreal const& x, std::int64_t exponent, Precision p)
{
  (void)x.precision();
  if (p.is_exact())
  {
    return mpreal(mpreal_access::rational_result([&](mpq_ptr q) {
      mpq_set(q, x.exact_value().native_handle());
      if (mpq_sgn(q) == 0) return;
      if (exponent >= 0)
        mpq_mul_2exp(q, q, static_cast<mp_bitcnt_t>(exponent));
      else
        mpq_div_2exp(q, q, static_cast<mp_bitcnt_t>(std::uint64_t(0) - std::uint64_t(exponent)));
    }));
  }
  return mpreal_access::finite_result(p, [&](mpfr_ptr result) {
    if (x.is_exact())
    {
      scaled_rational_ratio({scaled_rational{x.exact_value(), exponent}, {}, {exact_constant{1}, 0}, {}})
          .round_to(result);
    }
    else
    {
      static_assert(sizeof(long) >= sizeof(std::int64_t));
      mpfr_mul_2si(result, x.native_handle(), static_cast<long>(exponent), MPFR_RNDN);
    }
  });
}
} // namespace detail

/// \brief Compute a*b+c with one rounding, including mixed exact rational operands.
inline mpreal fma(mpreal const& a, mpreal const& b, mpreal const& c)
{
  return detail::fused_product_sum(a, b, c,
                                   common_precision(common_precision(a.precision(), b.precision()), c.precision()));
}
/// \brief Round a*b+c once at p, without converting operands first.
inline mpreal fma(mpreal const& a, mpreal const& b, mpreal const& c, Precision p)
{
  (void)p.bit_count();
  return detail::fused_product_sum(a, b, c, p);
}

/// \brief Multiply by 2^exponent, preserving exact rational state or input precision.
template <detail::MathInteger I> inline mpreal ldexp(mpreal const& x, I exponent)
{
  return detail::binary_scale(x, detail::math_integer(exponent), x.precision());
}
/// \brief Multiply the original value by 2^exponent and round the result at p.
template <detail::MathInteger I> inline mpreal ldexp(mpreal const& x, I exponent, Precision p)
{
  (void)p.bit_count();
  return detail::binary_scale(x, detail::math_integer(exponent), p);
}
/// \brief Binary scaling, identical to ldexp for Uni20's binary scalar formats.
template <detail::MathInteger I> inline mpreal scalbn(mpreal const& x, I exponent) { return ldexp(x, exponent); }
/// \brief Binary scaling with one rounding to finite result precision p.
template <detail::MathInteger I> inline mpreal scalbn(mpreal const& x, I exponent, Precision p)
{
  return ldexp(x, exponent, p);
}

/// \brief Transfer the sign while preserving the magnitude's precision and exactness.
inline mpreal copysign(mpreal const& magnitude, mpreal const& sign)
{
  auto p = magnitude.precision();
  bool negative = signbit(sign);
  if (p.is_exact())
  {
    auto q = magnitude.exact_value();
    if ((q < 0) != negative) q = -q;
    return mpreal(q);
  }
  return detail::mpreal_access::finite_result(
      p, [&](mpfr_ptr result) { mpfr_setsign(result, magnitude.native_handle(), negative, MPFR_RNDN); });
}
/// \brief Round the magnitude to p and transfer the sign; sign precision is immaterial.
inline mpreal copysign(mpreal const& magnitude, mpreal const& sign, Precision p)
{
  (void)p.bit_count();
  return copysign(magnitude.at(p), sign);
}

namespace detail
{
inline mpreal select_extreme(mpreal const& a, mpreal const& b, Precision p, bool maximum)
{
  (void)a.precision();
  (void)b.precision();
  if (isnan(a)) return b.at(p);
  if (isnan(b)) return a.at(p);
  if (a == 0 && b == 0 && !p.is_exact())
    return mpreal_access::finite_result(p, [&](mpfr_ptr result) {
      bool negative = maximum ? signbit(a) && signbit(b) : signbit(a) || signbit(b);
      mpfr_set_zero(result, negative ? -1 : 1);
    });
  return ((maximum ? a > b : a < b) ? a : b).at(p);
}
} // namespace detail
/// \brief Minimum with NaN suppression and negative-zero preference.
inline mpreal fmin(mpreal const& a, mpreal const& b)
{
  return detail::select_extreme(a, b, common_precision(a.precision(), b.precision()), false);
}
/// \brief Select the minimum before rounding it to result precision p.
inline mpreal fmin(mpreal const& a, mpreal const& b, Precision p)
{
  (void)p.bit_count();
  return detail::select_extreme(a, b, p, false);
}
/// \brief Maximum with NaN suppression and positive-zero preference.
inline mpreal fmax(mpreal const& a, mpreal const& b)
{
  return detail::select_extreme(a, b, common_precision(a.precision(), b.precision()), true);
}
/// \brief Select the maximum before rounding it to result precision p.
inline mpreal fmax(mpreal const& a, mpreal const& b, Precision p)
{
  (void)p.bit_count();
  return detail::select_extreme(a, b, p, true);
}
namespace detail
{
inline mpreal positive_difference(mpreal const& a, mpreal const& b, Precision p)
{
  (void)a.precision();
  (void)b.precision();
  if (isnan(a) || isnan(b)) return mpreal("nan", p);
  if (a <= b) return mpreal(p);
  return fused_product_sum(a, mpreal{1}, -b, p);
}
} // namespace detail
/// \brief Positive difference max(a-b,0), propagating NaNs.
inline mpreal fdim(mpreal const& a, mpreal const& b)
{
  return detail::positive_difference(a, b, common_precision(a.precision(), b.precision()));
}
/// \brief Positive difference with one rounding at p, without input conversion.
inline mpreal fdim(mpreal const& a, mpreal const& b, Precision p)
{
  (void)p.bit_count();
  return detail::positive_difference(a, b, p);
}

/// \brief Next value toward positive infinity on the input's finite grid.
inline mpreal next_up(mpreal const& x)
{
  return detail::mpreal_access::finite_result(x.precision(), [&](mpfr_ptr result) {
    mpfr_set(result, x.native_handle(), MPFR_RNDN);
    mpfr_nextabove(result);
  });
}
/// \brief Select grid p by rounding x, then move toward positive infinity.
inline mpreal next_up(mpreal const& x, Precision p)
{
  (void)p.bit_count();
  return next_up(x.at(p));
}
/// \brief Next value toward negative infinity on the input's finite grid.
inline mpreal next_down(mpreal const& x)
{
  return detail::mpreal_access::finite_result(x.precision(), [&](mpfr_ptr result) {
    mpfr_set(result, x.native_handle(), MPFR_RNDN);
    mpfr_nextbelow(result);
  });
}
/// \brief Select grid p by rounding x, then move toward negative infinity.
inline mpreal next_down(mpreal const& x, Precision p)
{
  (void)p.bit_count();
  return next_down(x.at(p));
}
/// \brief Adjacent value on x's grid toward direction, whose precision may differ.
/// \details Equal zeros take the direction's sign. An exact x needs an explicit grid.
inline mpreal nextafter(mpreal const& x, mpreal const& direction)
{
  auto p = x.precision();
  (void)p.bit_count();
  (void)direction.precision();
  if (isnan(x) || isnan(direction)) return mpreal("nan", p);
  if (x == direction) return copysign(x, direction);
  return x < direction ? next_up(x) : next_down(x);
}
/// \brief Round x onto grid p, then step toward the unrounded direction.
inline mpreal nextafter(mpreal const& x, mpreal const& direction, Precision p)
{
  (void)p.bit_count();
  return nextafter(x.at(p), direction);
}

namespace detail
{
inline std::int64_t rational_binary_exponent(exact_constant const& x)
{
  auto q = x.native_handle();
  if (mpq_sgn(q) == 0) return 0;
  mp_integer numerator, denominator;
  mpz_abs(numerator.value, mpq_numref(q));
  mpz_set(denominator.value, mpq_denref(q));
  auto exponent = static_cast<std::int64_t>(mpz_sizeinbase(numerator.value, 2)) -
                  static_cast<std::int64_t>(mpz_sizeinbase(denominator.value, 2));
  if (exponent >= 0)
    mpz_mul_2exp(denominator.value, denominator.value, exponent);
  else
    mpz_mul_2exp(numerator.value, numerator.value, -exponent);
  return exponent + (mpz_cmp(numerator.value, denominator.value) >= 0 ? 1 : 0);
}
inline frexp_result<mpreal> decompose_binary(mpreal const& x, Precision p)
{
  (void)x.precision();
  if (x.is_exact())
  {
    auto exponent = rational_binary_exponent(x.exact_value());
    auto fraction = binary_scale(x, -exponent, p);
    // Rounding the normalized rational can carry into the next binade.
    if (!p.is_exact() && abs(fraction) == 1)
    {
      fraction = binary_scale(fraction, -1, p);
      ++exponent;
    }
    return {std::move(fraction), exponent};
  }
  if (!isfinite(x)) return {x.at(p), 0};
  mpfr_exp_t exponent = 0;
  auto fraction = mpreal_access::finite_result(
      p, [&](mpfr_ptr result) { mpfr_frexp(&exponent, result, x.native_handle(), MPFR_RNDN); });
  return {std::move(fraction), exponent};
}
inline modf_result<mpreal> decompose_fraction(mpreal const& x, Precision p)
{
  (void)x.precision();
  if (x.is_exact())
  {
    auto integer = round_rational(x.exact_value(), integer_rounding::toward_zero);
    return {mpreal(x.exact_value() - integer, p), mpreal(integer, p)};
  }
  mpreal fraction(uninitialized);
  auto integer = mpreal_access::finite_result(p, [&](mpfr_ptr whole) {
    fraction =
        mpreal_access::finite_result(p, [&](mpfr_ptr part) { mpfr_modf(whole, part, x.native_handle(), MPFR_RNDN); });
  });
  return {std::move(fraction), std::move(integer)};
}

struct scaled_remainder
{
    scaled_rational value;
    int quotient;
};
// Integer modular arithmetic retains non-dyadic rational operands without
// expanding binary exponents. The quotient is needed only modulo eight.
inline scaled_remainder rational_remainder(scaled_rational const& a, scaled_rational const& b, bool nearest)
{
  auto aq = a.coefficient.native_handle(), bq = b.coefficient.native_handle();
  if (mpq_sgn(bq) == 0) throw std::domain_error("exact remainder: zero divisor");
  mp_integer numerator, divisor, remainder, quotient, modulus, factor;
  mpz_mul(numerator.value, mpq_numref(aq), mpq_denref(bq));
  mpz_abs(numerator.value, numerator.value);
  mpz_mul(divisor.value, mpq_numref(bq), mpq_denref(aq));
  mpz_abs(divisor.value, divisor.value);
  auto difference = a.exponent - b.exponent;
  auto exponent = b.exponent;
  if (difference >= 0)
  {
    mpz_mul_ui(modulus.value, divisor.value, 8);
    mpz_set_ui(factor.value, 2);
    mpz_powm_ui(factor.value, factor.value, static_cast<unsigned long>(difference), modulus.value);
    mpz_mul(numerator.value, numerator.value, factor.value);
    mpz_mod(numerator.value, numerator.value, modulus.value);
  }
  else
  {
    exponent = a.exponent;
    mpz_neg(factor.value, divisor.value);
    if (nearest) mpz_mul_2exp(numerator.value, numerator.value, 1);
    int comparison = scaled_sum_sign<2>({numerator.value, factor.value}, {0, -difference});
    if (nearest) mpz_tdiv_q_2exp(numerator.value, numerator.value, 1);
    if (comparison < 0 || (nearest && comparison == 0)) return {a, 0};
    // The shifted divisor is now bounded by twice the existing numerator size.
    mpz_mul_2exp(divisor.value, divisor.value, static_cast<mp_bitcnt_t>(-difference));
  }
  mpz_tdiv_qr(quotient.value, remainder.value, numerator.value, divisor.value);
  auto quotient_bits = static_cast<int>(mpz_fdiv_ui(quotient.value, 8));
  if (nearest)
  {
    mpz_mul_2exp(factor.value, remainder.value, 1);
    int comparison = mpz_cmp(factor.value, divisor.value);
    if (comparison > 0 || (comparison == 0 && (quotient_bits & 1)))
    {
      mpz_sub(remainder.value, remainder.value, divisor.value);
      quotient_bits = (quotient_bits + 1) % 8;
    }
  }
  if (mpq_sgn(aq) < 0) mpz_neg(remainder.value, remainder.value);
  if ((mpq_sgn(aq) < 0) != (mpq_sgn(bq) < 0)) quotient_bits = -quotient_bits;
  auto coefficient = mpreal_access::rational_result([&](mpq_ptr q) {
    mpz_set(mpq_numref(q), remainder.value);
    mpz_mul(mpq_denref(q), mpq_denref(aq), mpq_denref(bq));
  });
  return {{std::move(coefficient), exponent}, quotient_bits};
}
inline remquo_result<mpreal> remainder_value(mpreal const& a, mpreal const& b, Precision p, bool nearest)
{
  (void)a.precision();
  (void)b.precision();
  if (p.is_exact())
  {
    auto result = rational_remainder(mpreal_access::stored_scaled(a), mpreal_access::stored_scaled(b), nearest);
    return {mpreal(result.value.coefficient), result.quotient};
  }
  if (!isfinite(a) || isnan(b) || b == 0) return {mpreal("nan", p), 0};
  if (isinf(b)) return {a.at(p), 0};
  if (!a.is_exact() && !b.is_exact())
  {
    long quotient = 0;
    auto result = mpreal_access::finite_result(p, [&](mpfr_ptr out) {
      if (nearest)
        mpfr_remquo(out, &quotient, a.native_handle(), b.native_handle(), MPFR_RNDN);
      else
        mpfr_fmodquo(out, &quotient, a.native_handle(), b.native_handle(), MPFR_RNDN);
    });
    return {std::move(result), static_cast<int>(quotient % 8)};
  }
  auto result = rational_remainder(mpreal_access::stored_scaled(a), mpreal_access::stored_scaled(b), nearest);
  auto value = mpreal_access::finite_result(p, [&](mpfr_ptr out) {
    if (result.value.coefficient == 0)
      mpfr_set_zero(out, signbit(a) ? -1 : 1);
    else
      scaled_rational_ratio({result.value, {}, {exact_constant{1}, 0}, {}}).round_to(out);
  });
  return {std::move(value), result.quotient};
}
} // namespace detail

/// \brief Decompose into a binary fraction and signed 64-bit exponent.
/// \details Zero/nonfinite inputs return exponent zero. Nonzero finite fractions
///          have magnitude in [1/2,1); exact rational inputs remain exact.
inline frexp_result<mpreal> frexp(mpreal const& x) { return detail::decompose_binary(x, x.precision()); }
/// \brief Decompose and round the fraction at p, normalizing any rounding carry.
inline frexp_result<mpreal> frexp(mpreal const& x, Precision p)
{
  (void)p.bit_count();
  return detail::decompose_binary(x, p);
}
/// \brief Signed fractional and integral parts, using truncation toward zero.
inline modf_result<mpreal> modf(mpreal const& x) { return detail::decompose_fraction(x, x.precision()); }
/// \brief Decompose the original value and round each result component at p.
inline modf_result<mpreal> modf(mpreal const& x, Precision p)
{
  (void)p.bit_count();
  return detail::decompose_fraction(x, p);
}
/// \brief Base-two exponent of a finite nonzero value.
/// \throws std::domain_error For zero, infinity or NaN.
inline std::int64_t ilogb(mpreal const& x)
{
  if (!isfinite(x) || x == 0) throw std::domain_error("ilogb requires a finite nonzero value");
  return x.is_exact() ? detail::rational_binary_exponent(x.exact_value()) - 1 : mpfr_get_exp(x.native_handle()) - 1;
}

/// \brief Truncating-quotient remainder; exact operands retain exact rational results.
inline mpreal fmod(mpreal const& a, mpreal const& b)
{
  return detail::remainder_value(a, b, common_precision(a.precision(), b.precision()), false).remainder;
}
/// \brief Truncating-quotient remainder, rounding only the result at p.
inline mpreal fmod(mpreal const& a, mpreal const& b, Precision p)
{
  (void)p.bit_count();
  return detail::remainder_value(a, b, p, false).remainder;
}

/// \brief Nearest-even-quotient remainder; exact operands retain exact rational results.
inline mpreal remainder(mpreal const& a, mpreal const& b)
{
  return detail::remainder_value(a, b, common_precision(a.precision(), b.precision()), true).remainder;
}
/// \brief Nearest-even-quotient remainder, rounding only the result at p.
inline mpreal remainder(mpreal const& a, mpreal const& b, Precision p)
{
  (void)p.bit_count();
  return detail::remainder_value(a, b, p, true).remainder;
}

/// \brief Nearest-even remainder with the signed low three quotient bits.
inline remquo_result<mpreal> remquo(mpreal const& a, mpreal const& b)
{
  return detail::remainder_value(a, b, common_precision(a.precision(), b.precision()), true);
}
/// \brief Remainder and quotient bits computed before rounding the remainder at p.
inline remquo_result<mpreal> remquo(mpreal const& a, mpreal const& b, Precision p)
{
  (void)p.bit_count();
  return detail::remainder_value(a, b, p, true);
}

namespace detail
{
template <bool Hyperbolic> auto paired_functions(mpreal const& x)
{
  using result_type = std::conditional_t<Hyperbolic, sinhcosh_result<mpreal>, sincos_result<mpreal>>;
  auto p = x.precision();
  if (x.is_exact())
  {
    if constexpr (Hyperbolic)
      return result_type{sinh(x), cosh(x)};
    else
      return result_type{sin(x), cos(x)};
  }
  mpreal cosine(uninitialized);
  auto sine = mpreal_access::finite_result(p, [&](mpfr_ptr s) {
    cosine = mpreal_access::finite_result(p, [&](mpfr_ptr c) {
      if constexpr (Hyperbolic)
        mpfr_sinh_cosh(s, c, x.native_handle(), MPFR_RNDN);
      else
        mpfr_sin_cos(s, c, x.native_handle(), MPFR_RNDN);
    });
  });
  return result_type{std::move(sine), std::move(cosine)};
}
} // namespace detail
/// \brief Evaluate sine and cosine using the paired provider routine.
inline sincos_result<mpreal> sincos(mpreal const& x) { return detail::paired_functions<false>(x); }
/// \brief Paired elementary evaluation at p, matching sin(x,p) and cos(x,p).
inline sincos_result<mpreal> sincos(mpreal const& x, Precision p)
{
  (void)p.bit_count();
  return sincos(x.at(p));
}
/// \brief Evaluate hyperbolic sine and cosine using the paired provider routine.
inline sinhcosh_result<mpreal> sinhcosh(mpreal const& x) { return detail::paired_functions<true>(x); }
/// \brief Paired elementary evaluation at p, matching sinh(x,p) and cosh(x,p).
inline sinhcosh_result<mpreal> sinhcosh(mpreal const& x, Precision p)
{
  (void)p.bit_count();
  return sinhcosh(x.at(p));
}

namespace detail
{
// The callers remove rational-result cases first. The remaining result is
// irrational/non-dyadic and cannot equal a binary rounding boundary, so an
// enclosing interval eventually selects one rounded value.
template <class F> mpreal rational_monotone(exact_constant const& positive, Precision p, bool decreasing, F operation)
{
  auto bits = std::max<mpfr_prec_t>(64, p.bit_count());
  while (true)
  {
    auto work = Precision::bits(bits);
    mpfr_value lower(work), upper(work);
    mpfr_set_q(lower.value, positive.native_handle(), MPFR_RNDD);
    mpfr_set_q(upper.value, positive.native_handle(), MPFR_RNDU);
    if (mpfr_zero_p(lower.value) || mpfr_inf_p(upper.value))
      throw std::overflow_error("rational power/root input exceeds the MPFR exponent range");
    if (decreasing) mpfr_swap(lower.value, upper.value);
    operation(lower.value, lower.value, MPFR_RNDD);
    operation(upper.value, upper.value, MPFR_RNDU);
    auto lo = mpreal_access::finite_result(p, [&](mpfr_ptr out) { mpfr_set(out, lower.value, MPFR_RNDN); });
    auto hi = mpreal_access::finite_result(p, [&](mpfr_ptr out) { mpfr_set(out, upper.value, MPFR_RNDN); });
    if (lo == hi) return lo;
    if (bits > MPFR_PREC_MAX / 2) throw std::overflow_error("rational power/root needs unsupported working precision");
    bits *= 2;
  }
}
inline mpreal integer_power_value(mpreal const& x, std::int64_t exponent, Precision p)
{
  (void)x.precision();
  if (p.is_exact()) return mpreal(uni20::pow(x.exact_value(), exact_constant(exponent)));
  if (!x.is_exact())
  {
    auto n = exact_constant(exponent);
    return mpreal_access::finite_result(
        p, [&](mpfr_ptr out) { mpfr_pow_z(out, x.native_handle(), mpq_numref(n.native_handle()), MPFR_RNDN); });
  }
  if (exponent == 0) return mpreal(1, p);
  if (x == 0) return exponent < 0 ? mpreal("inf", p) : mpreal(0, p);
  auto q = x.exact_value();
  if (exponent < 0) q = exact_constant{1} / q;
  bool negative = q < 0 && exponent % 2 != 0;
  if (q < 0) q = -q;
  auto magnitude = exponent < 0 ? std::uint64_t(0) - std::uint64_t(exponent) : std::uint64_t(exponent);
  auto n = exact_constant(magnitude);
  auto operation = [&](mpfr_ptr out, mpfr_srcptr in, mpfr_rnd_t rounding) {
    mpfr_pow_z(out, in, mpq_numref(n.native_handle()), rounding);
  };
  mpreal result(uninitialized);
  if (mpz_popcount(mpq_denref(q.native_handle())) == 1)
  {
    // A dyadic base is exactly representable. Direct evaluation also resolves
    // exact halfway results that an interval test could not separate.
    auto bits = std::max<mpfr_prec_t>(MPFR_PREC_MIN, mpz_sizeinbase(mpq_numref(q.native_handle()), 2));
    auto input = mpreal(q, Precision::bits(bits));
    if (!isfinite(input) || input == 0) throw std::overflow_error("power input exceeds the MPFR exponent range");
    result = mpreal_access::finite_result(p, [&](mpfr_ptr out) { operation(out, input.native_handle(), MPFR_RNDN); });
  }
  else
    result = rational_monotone(q, p, false, operation);
  return negative ? -result : result;
}
inline mpreal integer_root_value(mpreal const& x, std::int64_t degree, Precision p)
{
  (void)x.precision();
  if (degree == 0)
  {
    if (p.is_exact()) throw std::domain_error("exact root: zeroth root");
    return mpreal("nan", p);
  }
  static_assert(sizeof(long) >= sizeof(std::int64_t));
  if (!x.is_exact())
    return mpreal_access::finite_result(
        p, [&](mpfr_ptr out) { mpfr_rootn_si(out, x.native_handle(), static_cast<long>(degree), MPFR_RNDN); });
  auto magnitude = degree < 0 ? std::uint64_t(0) - std::uint64_t(degree) : std::uint64_t(degree);
  if (auto root = rational_root(x.exact_value(), static_cast<unsigned long>(magnitude)))
  {
    if (degree < 0)
    {
      if (*root == 0 && !p.is_exact()) return mpreal("inf", p);
      *root = exact_constant{1} / *root;
    }
    return mpreal(*root, p);
  }
  if (p.is_exact()) throw std::logic_error("rootn requires finite working precision for an irrational result");
  auto q = x.exact_value();
  bool negative = q < 0;
  if (negative && degree % 2 == 0) return mpreal("nan", p);
  if (negative) q = -q;
  auto result = rational_monotone(q, p, degree < 0, [&](mpfr_ptr out, mpfr_srcptr in, mpfr_rnd_t rounding) {
    mpfr_rootn_si(out, in, static_cast<long>(degree), rounding);
  });
  return negative ? -result : result;
}
} // namespace detail

/// \brief Integer power, preserving exact rational results or the input's finite precision.
template <detail::MathInteger I> inline mpreal pown(mpreal const& x, I exponent)
{
  return detail::integer_power_value(x, detail::math_integer(exponent), x.precision());
}
/// \brief Integer power with one final rounding at p, without input conversion.
template <detail::MathInteger I> inline mpreal pown(mpreal const& x, I exponent, Precision p)
{
  (void)p.bit_count();
  return detail::integer_power_value(x, detail::math_integer(exponent), p);
}
/// \brief Real integer-order root; negative orders give reciprocal roots.
/// \details Exact inputs remain rational when their root is rational, otherwise
///          finite precision is required. Even-order negative roots are not real.
template <detail::MathInteger I> inline mpreal rootn(mpreal const& x, I degree)
{
  return detail::integer_root_value(x, detail::math_integer(degree), x.precision());
}
/// \brief Integer-order root with one final rounding at p, without input conversion.
template <detail::MathInteger I> inline mpreal rootn(mpreal const& x, I degree, Precision p)
{
  (void)p.bit_count();
  return detail::integer_root_value(x, detail::math_integer(degree), p);
}

} // namespace uni20
