#pragma once

#include <cstdint>
#include <limits>
#include <uni20/core/mpreal.hpp>

namespace uni20::check::detail
{
struct mpreal_ulp_result
{
    bool equal = false;
    long long distance = std::numeric_limits<long long>::max();
    char const* reason = nullptr;
};

// Zero has rank zero. Each positive binade contains 2^(p-1) values, with
// the smallest positive value (exponent emin) at rank one. Negative ranks
// mirror positive ranks, collapsing the two signed zeros. Storage depends
// on significand precision, not on the magnitude of the floating exponent.
inline void mpfr_ulp_key(mpz_ptr key, mpfr_srcptr value, mpfr_exp_t emin)
{
  if (mpfr_zero_p(value))
  {
    mpz_set_ui(key, 0);
    return;
  }
  uni20::detail::mp_integer significand, minimum;
  (void)mpfr_get_z_2exp(significand.value, value);
  mpz_abs(significand.value, significand.value);
  mpz_set_si(key, mpfr_get_exp(value));
  mpz_set_si(minimum.value, emin);
  mpz_sub(key, key, minimum.value);
  mpz_sub_ui(key, key, 1);
  mpz_mul_2exp(key, key, static_cast<mp_bitcnt_t>(mpfr_get_prec(value) - 1));
  mpz_add(key, key, significand.value);
  mpz_add_ui(key, key, 1);
  if (mpfr_signbit(value)) mpz_neg(key, key);
}

inline mpreal_ulp_result compare_mpfr(mpfr_srcptr a, mpfr_srcptr b, std::int64_t max_ulps)
{
  if (max_ulps < 0) return {.reason = "negative ULP tolerance"};
  if (mpfr_get_prec(a) != mpfr_get_prec(b)) return {.reason = "working precisions differ"};
  if (mpfr_nan_p(a) || mpfr_nan_p(b)) return {.reason = "NaN is not comparable"};
  if (mpfr_inf_p(a) || mpfr_inf_p(b))
  {
    if (mpfr_equal_p(a, b)) return {.equal = true, .distance = 0};
    return {.reason = "infinities differ or only one operand is infinite"};
  }
  auto const emin = mpfr_get_emin(), emax = mpfr_get_emax();
  auto outside_grid = [=](mpfr_srcptr value) {
    return !mpfr_zero_p(value) && (mpfr_get_exp(value) < emin || mpfr_get_exp(value) > emax);
  };
  if (outside_grid(a) || outside_grid(b)) return {.reason = "operand is outside the current MPFR exponent range"};
  if (mpfr_equal_p(a, b)) return {.equal = true, .distance = 0};

  uni20::detail::mp_integer left, right, magnitude, limit;
  mpfr_ulp_key(left.value, a, emin);
  mpfr_ulp_key(right.value, b, emin);
  mpz_sub(magnitude.value, right.value, left.value);
  int const sign = mpz_sgn(magnitude.value);
  mpz_abs(magnitude.value, magnitude.value);
  mpz_import(limit.value, 1, 1, sizeof(max_ulps), 0, 0, &max_ulps);
  bool const equal = mpz_cmp(magnitude.value, limit.value) <= 0;

  // Saturation is only for diagnostics. A distance beyond this range must
  // not pass a comparison whose tolerance happens to equal the sentinel.
  constexpr auto maximum = std::numeric_limits<long long>::max();
  mpz_import(limit.value, 1, 1, sizeof(maximum), 0, 0, &maximum);
  long long distance = maximum;
  if (mpz_cmp(magnitude.value, limit.value) < 0)
  {
    unsigned long long value = 0;
    mpz_export(&value, nullptr, 1, sizeof(value), 0, 0, magnitude.value);
    distance = static_cast<long long>(value);
  }
  return {.equal = equal, .distance = sign < 0 ? -distance : distance};
}

// Rounding an exact operand is local to the assertion. Preserve the caller's
// exception flags as well as its values, precision, and exponent-range settings.
struct mpfr_ulp_flags
{
    mpfr_flags_t saved = mpfr_flags_save();
    ~mpfr_ulp_flags() { mpfr_flags_restore(saved, MPFR_FLAGS_ALL); }
};

inline mpreal_ulp_result compare_mpreal(mpreal const& a, mpreal const& b, std::int64_t max_ulps)
{
  mpfr_ulp_flags flags;
  if (max_ulps < 0) return {.reason = "negative ULP tolerance"};
  if (!a.initialized() || !b.initialized()) return {.reason = "unset operand"};
  if (a.is_exact() && b.is_exact())
  {
    if (a == b) return {.equal = true, .distance = 0};
    return {.reason = "unequal exact values have no ULP grid"};
  }
  if (a.is_exact())
  {
    auto rounded = a.at(b.precision());
    return compare_mpfr(rounded.native_handle(), b.native_handle(), max_ulps);
  }
  if (b.is_exact())
  {
    auto rounded = b.at(a.precision());
    return compare_mpfr(a.native_handle(), rounded.native_handle(), max_ulps);
  }
  return compare_mpfr(a.native_handle(), b.native_handle(), max_ulps);
}
} // namespace uni20::check::detail
