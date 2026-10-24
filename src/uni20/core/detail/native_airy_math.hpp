#pragma once

#include "native_airy_expansion.hpp"
#include "native_airy_initial_values.hpp"
#include "native_airy_phase.hpp"

namespace uni20::detail::native_math::airy_detail
{
template <NativeReal R> Expansion<R> constant(std::uint16_t const* a)
{
  Expansion<R> r;
  constexpr int n = (4 * uni20::numeric_limits<R>::digits() + 16 + 15) / 16;
  for (int k = 0; k < n; ++k)
    r = r + Expansion<R>(std::ldexp(R(a[k]), -16 * (k + 1)));
  return r;
}
template <NativeReal R> Expansion<R> wide_sqrt(R x)
{
  using E = Expansion<R>;
  E s(std::sqrt(x));
  for (int k = 0; k < 3; ++k)
    s = s + (E(x) - s * s) / (E(2) * s);
  return s;
}
template <NativeReal R> Expansion<R> wide_log(Expansion<R> x)
{
  using E = Expansion<R>;
  using L = uni20::numeric_limits<R>;
  int n;
  R m = std::frexp(x.high(), &n);
  if (m < R(.75)) --n;
  x = x.scaled(-n);
  E t = (x - E(1)) / (x + E(1)), square = t * t, term = t, sum = t;
  R eps = std::ldexp(R(1), -4 * L::digits() + 8);
  for (int k = 1; k < 4 * L::digits(); ++k)
  {
    term = term * square;
    E v = term / R(2 * k + 1);
    sum = sum + v;
    if (std::abs(v.high()) < eps * std::abs(sum.high())) break;
  }
  return E(2) * sum + E(R(n)) * constant<R>(log_two_bits);
}
template <NativeReal R> R wide_exp(Expansion<R> x)
{
  using E = Expansion<R>;
  using L = uni20::numeric_limits<R>;
  E ln2 = constant<R>(log_two_bits);
  int n = int(std::nearbyint(x.high() / ln2.high()));
  E r = x - E(R(n)) * ln2, sum(1), term(1);
  R eps = std::ldexp(R(1), -4 * L::digits() + 8);
  for (int k = 1; k < 4 * L::digits(); ++k)
  {
    term = term * r / R(k);
    sum = sum + term;
    if (std::abs(term.high()) < eps) break;
  }
  if (n < L::min_exponent)
  {
    constexpr int d = L::min_exponent - L::digits();
    E units = sum.scaled(n - d);
    R q = std::floor(units.high());
    E rem = units - E(q);
    if (rem.high() < 0)
    {
      q -= 1;
      rem = rem + E(1);
    }
    E half = rem - E(R(.5));
    if (half.high() > 0 || (half.high() == 0 && std::fmod(q, R(2)) != 0)) q += 1;
    return std::ldexp(q, d);
  }
  return std::ldexp(sum.rounded(), n);
}
// Recompute the exponent and asymptotic sum before rounding to a subnormal.
// A rounded native exponential multiplied by the prefactor loses the last bit.
template <NativeReal R> R guarded_positive(R x)
{
  using E = Expansion<R>;
  using L = uni20::numeric_limits<R>;
  E z = E(2) * E(x) * wide_sqrt(x) / R(3), inv = E(1) / z, term(1), sum(1);
  R eps = std::ldexp(R(1), -4 * L::digits() + 8);
  bool done = false;
  for (int k = 1; k < 4 * L::digits(); ++k)
  {
    term = term * E(R((6 * k - 5) * (6 * k - 1))) * inv / R(72 * k);
    sum = sum + (k % 2 ? -term : term);
    if (term.high() < eps)
    {
      done = true;
      break;
    }
  }
  if (!done) return L::quiet_NaN();
  E logpi = E(1) + constant<R>(log_pi_fraction_bits);
  return wide_exp(-z - logpi / R(2) - wide_log(E(x)) / R(4) - constant<R>(log_two_bits) + wide_log(sum));
}
// Ai(0)*F(x^3) + Ai'(0)*x*G(x^3), with recurrence coefficients from
// y''=x*y. Four words retain the cancellation at the precision-specific join.
template <NativeReal R> R central(R x)
{
  using L = uni20::numeric_limits<R>;
  using E = Expansion<R>;
  E z = E(x) * E(x) * E(x), f(1), g(1), tf(1), tg(1);
  R eps = std::ldexp(R(1), -4 * L::digits() + 8);
  for (int k = 1; k < 4 * L::digits() + 64; ++k)
  {
    tf = tf * z / R((3 * k) * (3 * k - 1));
    tg = tg * z / R((3 * k) * (3 * k + 1));
    f = f + tf;
    g = g + tg;
    if (std::abs(tf.high()) < eps * std::abs(f.high()) && std::abs(tg.high()) < eps * std::abs(g.high()))
      return (constant<R>(ai_zero_bits) * f - constant<R>(ai_prime_zero_bits) * E(x) * g).rounded();
  }
  return L::quiet_NaN();
}
// Real-variable remainders are bounded by the first omitted term in each
// series. The chosen joins reach the stopping target before terms grow.
template <NativeReal R> R asymptotic(R x)
{
  using L = uni20::numeric_limits<R>;
  bool positive = x > 0;
  R v = std::abs(x), inverse = (R(1.5) / v) / std::sqrt(v), term = 1, even = 1, odd = 0, previous = 1;
  bool done = false;
  for (int k = 1; k < 4 * L::digits() + 64; ++k)
  {
    term *= R((6 * k - 5) * (6 * k - 1)) / R(72 * k) * inverse;
    if (term > previous) return L::quiet_NaN();
    if (positive)
      even += k % 2 ? -term : term;
    else if (k % 2)
      odd += (k / 2) % 2 ? -term : term;
    else
      even += (k / 2) % 2 ? -term : term;
    if (term < L::epsilon() / 32)
    {
      done = true;
      break;
    }
    previous = term;
  }
  if (!done) return L::quiet_NaN();
  R amplitude = 1 / (std::sqrt(uni20::pi<R>) * std::sqrt(std::sqrt(v)));
  if (positive)
  {
    R zeta = (R(2) / 3) * v * std::sqrt(v);
    R estimate = -zeta + std::log(amplitude * (even / 2));
    if (std::abs(estimate - std::log(L::denorm_min())) < 8) return guarded_positive(v);
    R half = std::exp(-zeta / 2);
    return half * (half * (amplitude * (even / 2)));
  }
  auto [s, c] = phase(v);
  return amplitude * std::fma(s, even, -c * odd);
}
template <NativeReal R> R airy(R x)
{
  using L = uni20::numeric_limits<R>;
  if (std::isnan(x)) return x;
  if (std::isinf(x)) return 0;
  constexpr int join = L::digits() <= 24 ? 8 : L::digits() <= 53 ? 12 : L::digits() <= 64 ? 14 : 20;
  if (std::abs(x) <= join) return central(x);
  if (x > R(2 * (L::digits() - L::min_exponent))) return 0;
  return asymptotic(x);
}
} // namespace uni20::detail::native_math::airy_detail

namespace uni20::detail::native_math
{
/// \brief Native real Airy Ai, with phase reduction across the full exponent range.
/// \details See docs/tensor/native_airy.md for domains, accuracy and algorithm bounds.
template <NativeReal R> R airy_ai(R x) { return airy_detail::airy(x); }
} // namespace uni20::detail::native_math
