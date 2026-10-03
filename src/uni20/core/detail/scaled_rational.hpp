#pragma once

#include <uni20/core/exact_constant.hpp>

#include <algorithm>
#include <array>

namespace uni20::detail
{
// Products and quotient comparisons can have exponents outside MPFR's range.
// Widen the exponent, not the significand: no large power of two is allocated.
using scaled_exponent = __int128_t;

struct scaled_rational
{
    exact_constant coefficient;
    scaled_exponent exponent = 0;
};

inline scaled_rational operator*(scaled_rational const& a, scaled_rational const& b)
{
  return {a.coefficient * b.coefficient, a.exponent + b.exponent};
}
inline scaled_rational operator-(scaled_rational const& a) { return {-a.coefficient, a.exponent}; }

// Exact sign of a short sum of integer * 2^exponent terms. Only align a lower
// term when it can affect the sign. Otherwise a huge exponent gap costs no
// storage. If alignment is necessary, its size is bounded by the coefficient
// sizes already present (plus two bits for the at-most-four remaining terms).
template <std::size_t N>
int scaled_sum_sign(std::array<mpz_srcptr, N> const& coefficients, std::array<scaled_exponent, N> const& exponents)
{
  static_assert(N <= 4);
  std::array<std::size_t, N> order{};
  std::size_t count = 0;
  for (std::size_t i = 0; i < N; ++i)
    if (mpz_sgn(coefficients[i]) != 0) order[count++] = i;
  if (count == 0) return 0;
  std::sort(order.begin(), order.begin() + count, [&](auto a, auto b) { return exponents[a] > exponents[b]; });
  mp_integer accumulator;
  mpz_set(accumulator.value, coefficients[order[0]]);
  auto exponent = exponents[order[0]];
  for (std::size_t j = 1; j < count; ++j)
  {
    auto next = order[j];
    if (mpz_sgn(accumulator.value) == 0)
      mpz_set(accumulator.value, coefficients[next]);
    else
    {
      auto top = exponents[next] + mpz_sizeinbase(coefficients[next], 2);
      for (std::size_t k = j + 1; k < count; ++k)
        top = std::max(top, exponents[order[k]] + mpz_sizeinbase(coefficients[order[k]], 2));
      auto lower_bound = exponent + mpz_sizeinbase(accumulator.value, 2) - 1;
      if (lower_bound >= top + 2) return mpz_sgn(accumulator.value);
      mpz_mul_2exp(accumulator.value, accumulator.value, static_cast<mp_bitcnt_t>(exponent - exponents[next]));
      mpz_add(accumulator.value, accumulator.value, coefficients[next]);
    }
    exponent = exponents[next];
  }
  return mpz_sgn(accumulator.value);
}

// Round (a + b) / (c + d) once. This is the large-exponent path for mixed
// rational/MPFR arithmetic; the ordinary path continues to use GMP rationals.
// Exact comparisons locate the binade and significand, including halfway cases,
// without a Ziv loop that could fail to terminate on an exact rounding boundary.
class scaled_rational_ratio {
  public:
    explicit scaled_rational_ratio(std::array<scaled_rational, 4> const& terms)
    {
      mp_integer denominator, factor;
      mpz_set_ui(denominator.value, 1);
      for (auto const& term : terms)
        mpz_lcm(denominator.value, denominator.value, mpq_denref(term.coefficient.native_handle()));
      for (std::size_t i = 0; i < 4; ++i)
      {
        auto q = terms[i].coefficient.native_handle();
        mpz_divexact(factor.value, denominator.value, mpq_denref(q));
        mpz_mul(coefficients_[i].value, mpq_numref(q), factor.value);
        exponents_[i] = terms[i].exponent;
      }
      int numerator_sign =
          scaled_sum_sign<2>({coefficients_[0].value, coefficients_[1].value}, {exponents_[0], exponents_[1]});
      int denominator_sign =
          scaled_sum_sign<2>({coefficients_[2].value, coefficients_[3].value}, {exponents_[2], exponents_[3]});
      if (denominator_sign == 0) throw std::domain_error("scaled rational: zero denominator");
      sign_ = numerator_sign * denominator_sign;
      for (std::size_t i = 0; i < 4; ++i)
        if ((i < 2 ? numerator_sign : denominator_sign) < 0) mpz_neg(coefficients_[i].value, coefficients_[i].value);
    }

    void round_to(mpfr_ptr out) const
    {
      if (sign_ == 0)
      {
        mpfr_set_zero(out, 1);
        return;
      }
      mp_integer mantissa, midpoint;
      mpz_set_ui(mantissa.value, 1);
      auto const emin = mpfr_get_emin(), emax = mpfr_get_emax();
      auto const precision = mpfr_get_prec(out);
      if (this->compare(mantissa.value, emax) >= 0)
      {
        mpfr_set_ui_2exp(out, 1, emax, MPFR_RNDN); // overflow, including flags
        if (sign_ < 0) mpfr_neg(out, out, MPFR_RNDN);
        return;
      }
      if (this->compare(mantissa.value, scaled_exponent(emin) - 1) < 0)
      {
        // No subnormals in MPFR. The halfway case rounds toward signed zero.
        if (this->compare(mantissa.value, scaled_exponent(emin) - 2) <= 0)
          mpfr_set_zero(out, sign_);
        else
          mpfr_set_si_2exp(out, sign_, emin - 1, MPFR_RNDN);
        // MPFR detects tininess after rounding at an unbounded exponent range.
        mpz_mul_2exp(midpoint.value, mantissa.value, precision + 1);
        mpz_sub_ui(midpoint.value, midpoint.value, 1);
        if (this->compare(midpoint.value, scaled_exponent(emin) - precision - 2) < 0) mpfr_set_underflow();
        mpfr_set_inexflag();
        return;
      }
      // Largest e such that 2^e <= |value|. Binary search needs only logarithmic
      // work in the exponent range, even for values such as 2^1000000000.
      scaled_exponent lo = scaled_exponent(emin) - 1, hi = emax;
      while (hi - lo > 1)
      {
        auto mid = lo + (hi - lo) / 2;
        if (this->compare(mantissa.value, mid) >= 0)
          lo = mid;
        else
          hi = mid;
      }
      auto exponent = lo - precision + 1;
      mpz_mul_2exp(mantissa.value, mantissa.value, precision - 1);
      for (mpfr_prec_t bit = precision - 1; bit-- > 0;)
      {
        mpz_setbit(mantissa.value, bit);
        if (this->compare(mantissa.value, exponent) < 0) mpz_clrbit(mantissa.value, bit);
      }
      bool const inexact = this->compare(mantissa.value, exponent) != 0;
      if (inexact)
      {
        mpz_mul_2exp(midpoint.value, mantissa.value, 1);
        mpz_add_ui(midpoint.value, midpoint.value, 1);
        int comparison = this->compare(midpoint.value, exponent - 1);
        if (comparison > 0 || (comparison == 0 && mpz_odd_p(mantissa.value)))
          mpz_add_ui(mantissa.value, mantissa.value, 1);
      }
      if (sign_ < 0) mpz_neg(mantissa.value, mantissa.value);
      mpfr_set_z_2exp(out, mantissa.value, static_cast<mpfr_exp_t>(exponent), MPFR_RNDN);
      if (inexact) mpfr_set_inexflag();
    }

  private:
    // Compare the positive magnitude with candidate * 2^exponent, exactly.
    int compare(mpz_srcptr candidate, scaled_exponent exponent) const
    {
      mp_integer c, d;
      mpz_mul(c.value, coefficients_[2].value, candidate);
      mpz_mul(d.value, coefficients_[3].value, candidate);
      mpz_neg(c.value, c.value);
      mpz_neg(d.value, d.value);
      return scaled_sum_sign<4>({coefficients_[0].value, coefficients_[1].value, c.value, d.value},
                                {exponents_[0], exponents_[1], exponents_[2] + exponent, exponents_[3] + exponent});
    }
    std::array<mp_integer, 4> coefficients_;
    std::array<scaled_exponent, 4> exponents_{};
    int sign_ = 0;
};
} // namespace uni20::detail
