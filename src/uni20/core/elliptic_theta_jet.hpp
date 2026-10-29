// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Ian McCulloch
#pragma once
#include "elliptic_theta.hpp"
#include <algorithm>
#include <array>
#include <cstdint>
#include <utility>

namespace uni20
{
/// \brief Theta value and first two u derivatives with a shared real exponential scale.
/// \details actual derivative[r] = exp(log_scale) * derivative[r], for theta_kind(pi*u | i*t).
///          Every field owns the requested finite precision. Derivatives are with respect to u.
template <Complex C> struct theta_jet_result
{
    std::array<C, 3> derivative;
    make_real_t<C> log_scale;
};
namespace detail::theta_jet_detail
{
template <Complex C> bool finite(C const& z) { return uni20::isfinite(z.real()) && uni20::isfinite(z.imag()); }
template <Complex C> class Sum {
  public:
    explicit Sum(C const& zero) : sum_(zero), correction_(zero) {}
    void add(C const& value)
    {
      C const corrected = value - correction_;
      C const next = sum_ + corrected;
      correction_ = (next - sum_) - corrected;
      sum_ = next;
    }
    C value() const { return sum_; }

  private:
    C sum_, correction_;
};
// Inputs have already been bounded to the signed reduction range (or to the
// much smaller series count). Conversion therefore cannot narrow a live index.
template <ThetaReal R> std::int64_t integral_value(R const& value)
{
#if UNI20_HAS_MPREAL
  if constexpr (std::same_as<R, mpreal>)
    return mpfr_get_sj(value.native_handle(), MPFR_RNDN);
  else
#endif
    return static_cast<std::int64_t>(value);
}
template <ThetaReal R> std::size_t term_count(R const& t, std::size_t max_terms)
{
  R const one = scalar_at_precision<R>(1, precision_of(t));
  R const pi = 4 * math::atan(one);
  R const count = math::ceil(math::sqrt((-math::log(numeric_limits<R>::epsilon(t)) + R{32}) / pi)) + R{3};
  if (count > scalar_at_precision<R>(max_terms, precision_of(t)))
    throw std::runtime_error("theta jet exhausted its term budget");
  return static_cast<std::size_t>(integral_value(count));
}
// Keep phases in cycles until sinpi/cospi reduce the quadrant. Evaluating a
// rounded multiple of pi creates nonzero cos(pi/2) or sin(pi) residuals which
// derivative factors can amplify by 1/t or 1/t^2.
template <ThetaReal R> complex<R> phase_cycles(R const& cycles)
{
  R const twice = 2 * cycles;
  return {math::cospi(twice), math::sinpi(twice)};
}
template <ThetaReal R> complex<R> phase_product(R const& a, R const& b)
{
  R const product = a * b, residual = math::fma(a, b, -product);
  auto phase = phase_cycles(product);
  // Retain fractional cycles lost in a large product, and low bits beside a
  // quarter cycle. Multiplying the phases avoids rounding the residual away.
  if (residual != 0) phase *= phase_cycles(residual);
  return phase;
}
template <ThetaReal R> complex<R> phase_quotient(R const& a, R const& y, R const& t)
{
  R const quotient = y / t, residual = math::fma(-quotient, t, y);
  auto phase = phase_product(a, quotient);
  if (residual != 0) phase *= phase_product(a, residual / t);
  return phase;
}

// Combine Fourier decay and growth before multiplication by pi. This retains
// finite exponents near a half-period as well as avoiding infinity-infinity.
// In the reduced rectangle t +/- 2*y is nonnegative, so an overflow here can
// only mean a vanishing exponential, never a lost cancellation.
template <ThetaReal R> R fourier_exponent(R const& n, bool half_series, R const& t, R const& y, R const& pi)
{
  if (!half_series) return -pi * (n + 1) * (n * t + (t + 2 * y));
  R const edge = math::abs(y) + y;
  if (n == 0) return -pi * edge;
  return -pi * (n * (n * t + (t + 2 * y)) + edge);
}
// Fundamental rectangle |Re(u)|<=1/2, |Im(u)|<=t/2. Both representations
// converge geometrically faster than exp(-pi*n*n) in their selected regimes.
template <Complex C> theta_jet_result<C> theta1_reduced(C u, make_real_t<C> t, bool gaussian, std::size_t max_terms)
{
  using Real = make_real_t<C>;
  using Complex = C;
  auto const p = precision_of(t);
  Real const zero = scalar_at_precision<Real>(0, p), one = scalar_at_precision<Real>(1, p);
  Real const half = one / 2;
  Complex const zero_complex(zero, zero);
  Real const pi = Real{4} * math::atan(one);
  auto const terms = term_count(t, max_terms);
  Real const x = u.real(), y = u.imag();
  theta_jet_result<C> result{{zero_complex, zero_complex, zero_complex}, zero};
  // Isolate the largest exponential before evaluating any series term.
  if (gaussian)
    result.log_scale = pi * (y * y - (half - math::abs(x)) * (half - math::abs(x))) / t - math::log(t) / Real{2};
  else
    result.log_scale = -pi * (t / Real{4} - math::abs(y));
  std::array<Sum<C>, 3> sums{Sum<C>(zero_complex), Sum<C>(zero_complex), Sum<C>(zero_complex)};
  // Finite-precision counters reuse MPFR storage through in-place +=.
  Real a = half, n_real = zero;
  for (unsigned n = 0; n < terms; ++n, a += one, n_real += one)
  {
    Real const sign = n % 2 ? -one : one;
    Complex value, first, second;
    if (!gaussian)
    {
      Real const omega = Real{2} * pi * a;
      if (math::abs(omega * y) <= one)
      {
        Real const fall = (n_real * (n_real + one)) * t * pi;
        Real const weight = math::exp(-fall - pi * math::abs(y));
        auto const phase = phase_product(a, x);
        Real const sh = math::sinh(omega * y), ch = math::cosh(omega * y);
        value = Real{2} * sign * weight * Complex(phase.imag() * ch, phase.real() * sh);
        first = Real{2} * sign * omega * weight * Complex(phase.real() * ch, -phase.imag() * sh);
      }
      else
      {
        Real const ep = fourier_exponent(n_real, true, t, y, pi),
                   em = fourier_exponent(n_real, true, t, -y, pi);
        auto const phase = phase_product(a, x);
        Complex const positive = math::exp(ep) * phase;
        Complex const negative = math::exp(em) * uni20::conj(phase);
        value = Complex(zero, -sign) * (positive - negative);
        first = sign * omega * (positive + negative);
      }
      second = -omega * omega * value;
    }
    else
    {
      Real const omega = Real{2} * pi * a / t, fall = n_real * (n_real + one);
      Complex difference, sum;
      if (math::abs(omega * x) <= one)
      {
        Complex const weight = math::exp(-pi * (fall + math::abs(x)) / t) * phase_quotient(-x, y, t);
        if (weight == zero_complex) continue;
        auto const phase = phase_quotient(a, y, t);
        Real const sh = math::sinh(omega * x), ch = math::cosh(omega * x);
        difference = Real{2} * weight * Complex(sh * phase.real(), ch * phase.imag());
        sum = Real{2} * weight * Complex(ch * phase.real(), sh * phase.imag());
        Complex const center = Real{2} * pi * (u / t);
        value = sign * difference;
        first = sign * (-center * difference + omega * sum);
        // Apply the Gaussian weight before each frequency factor: omega^2 can
        // overflow even when omega*(omega*difference) is representable.
        second = sign * (center * (center * difference) + omega * (omega * difference) -
                         (Real{2} * pi * difference) / t - Real{2} * center * (omega * sum));
      }
      else
      {
        Complex const positive = math::exp(-pi * (fall + math::abs(x) - Real{2} * a * x) / t) *
                                 phase_quotient(a - x, y, t);
        Complex const negative = math::exp(-pi * (fall + math::abs(x) + Real{2} * a * x) / t) *
                                 phase_quotient(-x - a, y, t);
        // Differentiate each centered Gaussian before combining the pair.
        // Expanding (u +/- a)^2 would cancel terms of order 1/t^2 near a center.
        auto derivative = [&](Complex const& term, Complex const& displacement, unsigned order) {
          if (term == zero_complex) return zero_complex;
          Complex const slope = -Real{2} * pi * (displacement / t);
          return order == 1 ? slope * term : slope * (slope * term) - (Real{2} * pi * term) / t;
        };
        value = sign * (positive - negative);
        first = sign * (derivative(positive, u - a, 1) - derivative(negative, u + a, 1));
        second = sign * (derivative(positive, u - a, 2) - derivative(negative, u + a, 2));
      }
    }
    sums[0].add(value);
    sums[1].add(first);
    sums[2].add(second);
  }
  for (unsigned r = 0; r < 3; ++r)
    result.derivative[r] = sums[r].value();
  if (u == zero_complex)
  {
    result.derivative[0] = zero_complex;
    result.derivative[2] = zero_complex;
  }
  return result;
}

// Poisson series centered at the integers for theta2/theta3. Evaluating this
// directly retains tiny u; translating u by 1/2 can round that displacement away.
template <Complex C> theta_jet_result<C> even_gaussian(unsigned kind, C u, make_real_t<C> t, std::size_t max_terms)
{
  using R = make_real_t<C>;
  auto const p = precision_of(t);
  R const zero = scalar_at_precision<R>(0, p), one = scalar_at_precision<R>(1, p);
  R const pi = R(4) * math::atan(one);
  C const cz(zero, zero);
  R const x = u.real(), root_t = math::sqrt(t);
  R const sx = x / root_t, sy = u.imag() / root_t;
  theta_jet_result<C> result{{cz, cz, cz}, pi * (sy * sy - sx * sx) - math::log(t) / 2};
  std::array<Sum<C>, 3> sums{Sum<C>(cz), Sum<C>(cz), Sum<C>(cz)};
  auto add = [&](R const& center, R const& exponent, R const& sign) {
    R const weight = math::exp(exponent);
    if (weight == zero) return;
    C const term = sign * weight * phase_quotient(center - x, u.imag(), t);
    C const slope = -2 * pi * ((u - center) / t);
    sums[0].add(term);
    sums[1].add(slope * term);
    sums[2].add(slope * (slope * term) - (2 * pi * term) / t);
  };
  add(zero, zero, one);
  auto const terms = term_count(t, max_terms);
  R a = one;
  for (unsigned n = 1; n <= terms; ++n, a += one)
  {
    R const sign = kind == 2 && n % 2 ? -one : one;
    add(a, -pi * a * ((a - 2 * x) / t), sign);
    add(-a, -pi * a * ((a + 2 * x) / t), sign);
  }
  for (unsigned d = 0; d < 3; ++d)
    result.derivative[d] = sums[d].value();
  return result;
}

// Translate only after reducing u. Choose the nearest imaginary half-period,
// subtracting it so a displacement from a theta4 zero is not rounded away.
template <Complex C> theta_jet_result<C> theta4_gaussian(C u, make_real_t<C> t, std::size_t max_terms)
{
  using R = make_real_t<C>;
  auto const p = precision_of(t);
  R const zero = scalar_at_precision<R>(0, p), one = scalar_at_precision<R>(1, p);
  R const pi = R(4) * math::atan(one), sign = u.imag() > zero ? -one : one;
  C const shifted(u.real(), u.imag() + sign * (t / 2));
  auto result = theta1_reduced(shifted, t, true, max_terms);
  C const slope(zero, sign * pi);
  result.derivative[2] += R(2) * slope * result.derivative[1] + slope * slope * result.derivative[0];
  result.derivative[1] += slope * result.derivative[0];
  C const phase = C(zero, -sign) * phase_product(sign / 2, u.real());
  for (auto& d : result.derivative)
    d *= phase;
  result.log_scale -= pi * (t / 4 + sign * u.imag());
  if (u == C(zero, zero)) result.derivative[1] = C(zero, zero);
  return result;
}

template <Complex C> theta_jet_result<C> even_reduced(unsigned kind, C u, make_real_t<C> t, std::size_t max_terms)
{
  using Real = make_real_t<C>;
  using Complex = C;
  auto const p = precision_of(t);
  Real const zero = scalar_at_precision<Real>(0, p), one = scalar_at_precision<Real>(1, p);
  Real const half = one / 2;
  Complex const zero_complex(zero, zero);
  Real const pi = Real{4} * math::atan(one);
  theta_jet_result<C> result{{zero_complex, zero_complex, zero_complex}, zero};
  bool const half_series = kind == 2;
  result.log_scale = half_series ? -pi * (t / Real{4} - math::abs(u.imag())) : zero;
  std::array<Sum<C>, 3> sums{Sum<C>(zero_complex), Sum<C>(zero_complex), Sum<C>(zero_complex)};
  if (!half_series) sums[0].add(Complex(one, zero));
  auto const terms = term_count(t, max_terms);
  Real a = half_series ? half : one, n_real = zero;
  for (unsigned n = 0; n < terms; ++n, a += one, n_real += one)
  {
    Real const omega = Real{2} * pi * a;
    Real const sign = kind == 4 && n % 2 == 0 ? -one : one;
    Complex value, first;
    if (math::abs(omega * u.imag()) <= one)
    {
      Real const fall = (half_series ? n_real * (n_real + one) : a * a) * t * pi,
                 shift = half_series ? pi * math::abs(u.imag()) : zero;
      Real const weight = Real{2} * sign * math::exp(-fall - shift);
      auto const phase = phase_product(a, u.real());
      Real const sh = math::sinh(omega * u.imag()), ch = math::cosh(omega * u.imag());
      value = weight * Complex(phase.real() * ch, -phase.imag() * sh);
      first = -weight * omega * Complex(phase.imag() * ch, phase.real() * sh);
    }
    else
    {
      Real const ep = fourier_exponent(n_real, half_series, t, u.imag(), pi),
                 em = fourier_exponent(n_real, half_series, t, -u.imag(), pi);
      auto const phase = phase_product(a, u.real());
      Complex const aterm = math::exp(ep) * phase;
      Complex const bterm = math::exp(em) * uni20::conj(phase);
      value = sign * (aterm + bterm);
      first = Complex(zero, sign * omega) * (aterm - bterm);
    }
    sums[0].add(value);
    sums[1].add(first);
    sums[2].add(-omega * omega * value);
  }
  for (unsigned d = 0; d < 3; ++d)
    result.derivative[d] = sums[d].value();
  if (kind == 4 && u.real() == zero && math::abs(u.imag()) == t / Real{2})
    result.derivative[0] = zero_complex;
  return result;
}

template <Complex C> theta_jet_result<C> theta1(C u, make_real_t<C> t, unsigned kind, std::size_t max_terms)
{
  using Real = make_real_t<C>;
  using Complex = C;
  auto const p = precision_of(t);
  Real const zero = scalar_at_precision<Real>(0, p), one = scalar_at_precision<Real>(1, p);
  Real const half = one / 2;
  Real const pi = Real{4} * math::atan(one);
  Real const eps = numeric_limits<Real>::epsilon(t);
  Real const limit = std::min(one / (Real{16} * eps), Real(std::numeric_limits<std::int64_t>::max() / 4));
  Real const vertical = u.imag() / t;
  if (!uni20::isfinite(vertical) || math::abs(u.real()) > limit || math::abs(vertical) > limit)
    throw std::overflow_error("theta argument reduction exceeds scalar precision");
  auto nearest_integer = [&](Real const& value) {
    Real const lower = math::floor(value);
    return value - lower < half ? lower : lower + one;
  };
  Real const nr = nearest_integer(u.real());
  Real mr = nearest_integer(vertical);
  Real ry = math::fma(-mr, t, u.imag());
  // Division may round across a half-period. Decide from the residual of the
  // original inputs, without rounding away a small displacement from a zero.
  if (ry >= t / 2)
    mr += one;
  else if (ry < -t / 2)
    mr -= one;
  ry = math::fma(-mr, t, u.imag());
  auto const n = integral_value(nr), m = integral_value(mr);
  Complex const reduced(u.real() - nr, ry), cz(zero, zero);
  // Real half-period translations exchange theta1/theta2 and theta3/theta4.
  // Subtract the nearest half-period before evaluation: this preserves small
  // displacements from zeros and stationary points instead of cancelling
  // separately rounded Fourier/Gaussian terms at the edge of the rectangle.
  Complex centered = reduced;
  unsigned centered_kind = kind;
  bool negate = false;
  if (math::abs(reduced.real()) > one / 4)
  {
    bool const positive = reduced.real() > zero;
    centered.real(reduced.real() + (positive ? -half : half));
    centered_kind = kind < 3 ? 3 - kind : 7 - kind;
    negate = (kind == 1 && !positive) || (kind == 2 && positive);
  }
  bool const taylor = centered_kind > 1 && t < one && centered != cz &&
                      math::abs(centered) < math::sqrt(eps) * t / Real(8);
  Complex const at = taylor ? cz : centered;
  auto result = centered_kind == 1 ? theta1_reduced(at, t, t < one, max_terms)
                : t < one ? (centered_kind == 4 ? theta4_gaussian(at, t, max_terms)
                                               : even_gaussian(centered_kind, at, t, max_terms))
                          : even_reduced(centered_kind, at, t, max_terms);
  if (centered_kind > 1 && at == cz) result.derivative[1] = cz;
  if (taylor)
  {
    result.derivative[0] += result.derivative[2] * centered * centered / Real(2);
    result.derivative[1] = result.derivative[2] * centered;
  }
  if (negate)
    for (auto& d : result.derivative)
      d = -d;
  // Quasi-periodicity, differentiated before reconstructing the common scale.
  Complex const slope(zero, -Real{2} * pi * mr);
  result.derivative[2] += Real{2} * slope * result.derivative[1] + slope * slope * result.derivative[0];
  result.derivative[1] += slope * result.derivative[0];
  auto const parity = kind == 1 ? n + m : kind == 2 ? n : kind == 4 ? m : 0;
  Complex phase = phase_product(-mr, reduced.real());
  if (parity % 2) phase = -phase;
  for (auto& d : result.derivative)
    d *= phase;
  result.log_scale = math::fma(pi * mr, mr * t + Real{2} * reduced.imag(), result.log_scale);
  return result;
}
template <Complex C> theta_jet_result<C> evaluate(unsigned kind, C u, make_real_t<C> t, std::size_t max_terms)
{
  using Real = make_real_t<C>;
  auto const p = precision_of(t);
  Real const zero = scalar_at_precision<Real>(0, p);
  if (kind < 1 || kind > 4 || !finite(u) || !uni20::isfinite(t) || !(t > zero))
    throw std::invalid_argument("theta requires kind 1..4, finite u, and finite t>0");
  auto result = theta1(u, t, kind, max_terms);
  if (!uni20::isfinite(result.log_scale)) throw std::overflow_error("theta logarithmic scale exceeds precision");
  for (auto d : result.derivative)
    if (!finite(d)) throw std::overflow_error("theta scaled derivatives exceed precision");
  return result;
}
} // namespace detail::theta_jet_detail
/// \brief Evaluate theta_kind(pi*u | i*t) and its first two u derivatives.
/// \details Rectangular lattice only: finite t>0, kind 1..4. Inputs share one
///          finite precision. The Fourier/Gaussian series is reduced to a
///          fundamental rectangle and differentiated before restoring periods.
///          Native arithmetic remains native; MPC uses 32 guard bits and rounds
///          every field back to the input precision. No ambient state changes.
///          max_terms bounds each reduced series, including its truncation margin.
/// \throws std::invalid_argument for invalid kind/domain/precision or zero budget;
///         std::runtime_error for insufficient terms; std::overflow_error for
///         unrepresentable argument reduction, scale or scaled derivatives.
template <Complex C>
  requires detail::ThetaReal<make_real_t<C>>
theta_jet_result<C> elliptic_theta_jet(unsigned kind, C const& u, make_real_t<C> const& t, std::size_t max_terms = 4096)
{
  auto const p = precision_of(t);
  if (kind < 1 || kind > 4 || max_terms == 0 || p.is_exact() || precision_of(u) != p)
    throw std::invalid_argument("theta jet requires kind 1..4, a positive budget and matching finite precision");
  if (!detail::theta_jet_detail::finite(u) || !uni20::isfinite(t) || !(t > 0))
    throw std::invalid_argument("theta jet requires finite u and finite t>0");
#if UNI20_HAS_MPREAL
  if constexpr (std::same_as<make_real_t<C>, mpreal>)
  {
    if (p.bit_count() > std::numeric_limits<Precision::bit_count_type>::max() - 32)
      throw std::overflow_error("theta jet guard precision exceeds supported range");
    auto const work = Precision::bits(p.bit_count() + 32);
    auto result = detail::theta_jet_detail::evaluate(kind, u.at(work), t.at(work), max_terms);
    for (auto& d : result.derivative)
    {
      d = d.at(p);
      if (!detail::theta_jet_detail::finite(d)) throw std::overflow_error("theta jet derivative exceeds result range");
    }
    result.log_scale = result.log_scale.at(p);
    if (!uni20::isfinite(result.log_scale)) throw std::overflow_error("theta jet scale exceeds result range");
    return result;
  }
  else
#endif
    return detail::theta_jet_detail::evaluate(kind, u, t, max_terms);
}

/// \brief Convert both arguments to p before evaluating the scaled theta jet.
/// \details Native types require their own precision; exact MPC inputs can be
///          rounded to finite p. All result fields retain p.
template <Complex C>
  requires detail::ThetaReal<make_real_t<C>>
theta_jet_result<C> elliptic_theta_jet(unsigned kind, C const& u, make_real_t<C> const& t, Precision p,
                                       std::size_t max_terms = 4096)
{
  if (p.is_exact()) throw std::invalid_argument("theta jet requires explicit finite precision");
  return elliptic_theta_jet(kind, scalar_at_precision<C>(u, p), scalar_at_precision<make_real_t<C>>(t, p), max_terms);
}

/// \brief Ratio of two scaled theta derivatives without reconstructing either jet.
/// \details Derivative orders are 0..2; only the selected components are used.
///          A zero numerator returns zero at the common finite precision.
///          Small denominators and rounded scales can amplify relative error.
/// \throws std::invalid_argument for invalid orders or inconsistent precision;
///         std::domain_error for a zero denominator; std::overflow_error when
///         the ratio magnitude is outside the scalar range.
template <Complex C>
  requires detail::ThetaReal<make_real_t<C>>
C elliptic_theta_ratio(theta_jet_result<C> const& a, unsigned da, theta_jet_result<C> const& b, unsigned db)
{
  if (da > 2 || db > 2) throw std::invalid_argument("theta derivative order must be 0..2");
  using R = make_real_t<C>;
  auto const p = precision_of(a.log_scale);
  if (p.is_exact() || precision_of(b.log_scale) != p || precision_of(a.derivative[da]) != p ||
      precision_of(b.derivative[db]) != p)
    throw std::invalid_argument("theta ratio requires matching finite precision");
  if (!uni20::isfinite(a.log_scale) || !uni20::isfinite(b.log_scale) ||
      !detail::theta_jet_detail::finite(a.derivative[da]) || !detail::theta_jet_detail::finite(b.derivative[db]))
    throw std::invalid_argument("theta ratio requires finite scales and selected derivatives");
  R const zero = scalar_at_precision<R>(0, p), one = scalar_at_precision<R>(1, p);
  R const numerator = math::abs(a.derivative[da]), denominator = math::abs(b.derivative[db]);
  if (!(denominator > zero)) throw std::domain_error("theta ratio has a zero denominator");
  if (numerator == zero) return C(zero, zero);
  R const shift = a.log_scale - b.log_scale;
  if (math::abs(shift) <= one)
  {
    auto const direct = (a.derivative[da] / b.derivative[db]) * math::exp(shift);
    if (detail::theta_jet_detail::finite(direct) && math::abs(direct) > zero) return direct;
  }
  auto log_norm_and_phase = [&](C const& z) -> std::pair<R, C> {
    // Scale components first, both for overflowing norms and for subnormal
    // norms whose rounded value cannot accurately normalize the phase.
    R const scale = std::max(math::abs(z.real()), math::abs(z.imag()));
    C const scaled = z / scale;
    R const scaled_norm = math::abs(scaled);
    return {math::log(scale) + math::log(scaled_norm), scaled / scaled_norm};
  };
  auto const [ln_a, phase_a] = log_norm_and_phase(a.derivative[da]);
  auto const [ln_b, phase_b] = log_norm_and_phase(b.derivative[db]);
  R const magnitude = math::exp(a.log_scale - b.log_scale + ln_a - ln_b);
  if (!uni20::isfinite(magnitude) || !(magnitude > zero))
    throw std::overflow_error("theta ratio magnitude is outside scalar range");
  return magnitude * phase_a * uni20::conj(phase_b);
}
} // namespace uni20
