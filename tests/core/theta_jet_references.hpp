#pragma once
#include <uni20/core/elliptic_theta_jet.hpp>

#if UNI20_HAS_MPREAL
namespace uni20::test::theta_jet_reference
{
using R = mpreal;
using C = complex<R>;
// Independent Jacobi products, differentiated through logarithmic derivatives.
// No period reduction, Gaussian series or production jet is used in this oracle.
inline std::array<C, 3> product_reference(unsigned kind, C const& u, R const& t)
{
  auto p = precision_of(t);
  R zero(0, p), one(1, p), pi_value = uni20::pi<R>.at(p);
  C const z = pi_value * u, sin2 = math::sin(2 * z), cos2 = math::cos(2 * z);
  R const q = math::exp(-pi_value * t), q2 = q * q;
  C value(one, zero), log_first(zero, zero), log_second(zero, zero);
  if (kind == 1)
  {
    value = 2 * math::exp(-pi_value * t / 4) * math::sin(z);
    log_first = pi_value * math::cos(z) / math::sin(z);
    log_second = -pi_value * pi_value / (math::sin(z) * math::sin(z));
  }
  if (kind == 2)
  {
    value = 2 * math::exp(-pi_value * t / 4) * math::cos(z);
    log_first = -pi_value * math::sin(z) / math::cos(z);
    log_second = -pi_value * pi_value / (math::cos(z) * math::cos(z));
  }
  R q_even = q2, q_factor = kind < 3 ? q2 : q;
  R sign(kind == 1 || kind == 4 ? -1 : 1, p);
  for (unsigned n = 1; n < 4096; ++n)
  {
    C const factor = one + 2 * sign * q_factor * cos2 + q_factor * q_factor;
    C const d = -4 * sign * pi_value * q_factor * sin2 / factor;
    value *= (one - q_even) * factor;
    log_first += d;
    log_second += -8 * sign * pi_value * pi_value * q_factor * cos2 / factor - d * d;
    if (q_factor < numeric_limits<R>::epsilon(t) * numeric_limits<R>::epsilon(t))
      return {value, value * log_first, value * (log_first * log_first + log_second)};
    q_even *= q2;
    q_factor *= q2;
  }
  throw std::runtime_error("independent theta product did not converge");
}

// Direct differentiated Poisson sum for t<=1/8 near the fundamental rectangle.
// Work at 1280 bits, including the complex exponential and polynomial derivative
// factors. No production shifts, paired sums, phase reduction or jet calls.
// The omitted centers are at least 8 periods away for these inputs; their
// relative contribution is below exp(-500*pi), including the derivative factors.
inline std::array<C, 3> gaussian_reference(unsigned kind, C const& u, R const& t)
{
  auto const p = precision_of(t);
  R const zero(0, p), one(1, p), pi_value = uni20::pi<R>.at(p);
  R const shift = kind == 1 || kind == 4 ? one / 2 : zero;
  R nearest_distance = one;
  for (int n = -1; n <= 1; ++n)
    nearest_distance = std::min(nearest_distance, math::abs(u.real() - (R(n, p) + shift)));
  R const common = pi_value * (u.imag() * u.imag() - nearest_distance * nearest_distance) / t;
  C const cz(zero, zero);
  std::array<C, 3> sum{cz, cz, cz};
  for (int n = -8; n <= 8; ++n)
  {
    C const displacement = u - (R(n, p) + shift);
    C term = math::exp(-pi_value * displacement * displacement / t - common);
    if ((kind == 1 || kind == 2) && n % 2) term = -term;
    C const slope = -2 * pi_value * displacement / t;
    sum[0] += term;
    sum[1] += slope * term;
    sum[2] += (slope * slope - 2 * pi_value / t) * term;
  }
  return sum;
}

} // namespace uni20::test::theta_jet_reference
#endif
