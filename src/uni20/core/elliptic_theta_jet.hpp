// SPDX-License-Identifier: Apache-2.0
// Copyright (C) 2026 Ian McCulloch
#pragma once
#include "elliptic_theta.hpp"
#include <algorithm>
#include <array>
#include <cstdint>

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
  for (unsigned n = 0; n < terms; ++n)
  {
    Real const a = Real(n) + half, sign = n % 2 ? -one : one;
    Complex value, first, second;
    if (!gaussian)
    {
      Real const omega = Real{2} * pi * a, fall = (Real(n) * Real(n + 1)) * t * pi;
      if (math::abs(omega * y) <= one)
      {
        Real const weight = math::exp(-fall - pi * math::abs(y));
        value = Real{2} * sign * weight * math::sin(omega * u);
        first = Real{2} * sign * omega * weight * math::cos(omega * u);
      }
      else
      {
        Complex const positive = math::exp(Complex(-fall - omega * y - pi * math::abs(y), omega * x));
        Complex const negative = math::exp(Complex(-fall + omega * y - pi * math::abs(y), -omega * x));
        value = Complex(zero, -sign) * (positive - negative);
        first = sign * omega * (positive + negative);
      }
      second = -omega * omega * value;
    }
    else
    {
      Real const omega = Real{2} * pi * a / t, fall = Real(n) * Real(n + 1);
      Complex difference, sum;
      if (math::abs(omega * x) <= one)
      {
        Complex const weight = math::exp(Complex(-pi * (fall + math::abs(x)) / t, -Real{2} * pi * x * (y / t)));
        difference = Real{2} * weight * math::sinh(omega * u);
        sum = Real{2} * weight * math::cosh(omega * u);
      }
      else
      {
        Complex const positive =
            math::exp(Complex(-pi * (fall + math::abs(x) - Real{2} * a * x) / t, -Real{2} * pi * (x - a) * (y / t)));
        Complex const negative =
            math::exp(Complex(-pi * (fall + math::abs(x) + Real{2} * a * x) / t, -Real{2} * pi * (x + a) * (y / t)));
        difference = positive - negative;
        sum = positive + negative;
      }
      Complex const center = Real{2} * pi * (u / t);
      value = sign * difference;
      first = sign * (-center * difference + omega * sum);
      second =
          sign * ((center * center + omega * omega - Real{2} * pi / t) * difference - Real{2} * center * omega * sum);
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

template <Complex C> theta_jet_result<C> even_reduced(unsigned kind, C u, make_real_t<C> t, std::size_t max_terms)
{
  using Real = make_real_t<C>;
  using Complex = C;
  auto const p = precision_of(t);
  Real const zero = scalar_at_precision<Real>(0, p), one = scalar_at_precision<Real>(1, p);
  Real const half = one / 2;
  Complex const zero_complex(zero, zero);
  Real const pi = Real{4} * math::atan(one);
  // Translate theta2 near its real zero without subtracting rounded cosines.
  if (kind == 2 && math::abs(u.real()) > one / 4)
  {
    bool const positive = u.real() > zero;
    auto result = theta1_reduced(u + (positive ? -half : half), t, false, max_terms);
    if (positive)
      for (auto& d : result.derivative)
        d = -d;
    return result;
  }
  theta_jet_result<C> result{{zero_complex, zero_complex, zero_complex}, zero};
  bool const half_series = kind == 2;
  result.log_scale = half_series ? -pi * (t / Real{4} - math::abs(u.imag())) : zero;
  std::array<Sum<C>, 3> sums{Sum<C>(zero_complex), Sum<C>(zero_complex), Sum<C>(zero_complex)};
  if (!half_series) sums[0].add(Complex(one, zero));
  auto const terms = term_count(t, max_terms);
  for (unsigned n = 0; n < terms; ++n)
  {
    Real const a = Real(n) + (half_series ? half : one), omega = Real{2} * pi * a;
    Real const sign = kind == 4 && n % 2 == 0 ? -one : one;
    Real const fall = (half_series ? Real(n) * Real(n + 1) : a * a) * t * pi,
               shift = half_series ? pi * math::abs(u.imag()) : zero;
    Complex value, first;
    if (math::abs(omega * u.imag()) <= one)
    {
      Real const weight = Real{2} * sign * math::exp(-fall - shift);
      value = weight * math::cos(omega * u);
      first = -weight * omega * math::sin(omega * u);
    }
    else
    {
      Complex const aterm = math::exp(Complex(-fall - shift - omega * u.imag(), omega * u.real()));
      Complex const bterm = math::exp(Complex(-fall - shift + omega * u.imag(), -omega * u.real()));
      value = sign * (aterm + bterm);
      first = Complex(zero, sign * omega) * (aterm - bterm);
    }
    sums[0].add(value);
    sums[1].add(first);
    sums[2].add(-omega * omega * value);
  }
  for (unsigned d = 0; d < 3; ++d)
    result.derivative[d] = sums[d].value();
  if (math::abs(u.imag()) == t / Real{2} &&
      ((kind == 3 && math::abs(u.real()) == half) || (kind == 4 && u.real() == zero)))
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
  Real const nr = math::floor(u.real() + half), mr = math::floor(vertical + half);
  auto const n = integral_value(nr), m = integral_value(mr);
  Complex const reduced(u.real() - nr, u.imag() - mr * t);
  auto result = kind == 1 ? theta1_reduced(reduced, t, t < one, max_terms) : even_reduced(kind, reduced, t, max_terms);
  // Quasi-periodicity, differentiated before reconstructing the common scale.
  Complex const slope(zero, -Real{2} * pi * mr);
  result.derivative[2] += Real{2} * slope * result.derivative[1] + slope * slope * result.derivative[0];
  result.derivative[1] += slope * result.derivative[0];
  auto const parity = kind == 1 ? n + m : kind == 2 ? n : kind == 4 ? m : 0;
  Complex const phase = Real(parity % 2 ? -1 : 1) * math::exp(Complex(zero, -Real{2} * pi * mr * reduced.real()));
  for (auto& d : result.derivative)
    d *= phase;
  result.log_scale += pi * mr * (mr * t + Real{2} * reduced.imag());
  return result;
}
template <Complex C> theta_jet_result<C> evaluate(unsigned kind, C u, make_real_t<C> t, std::size_t max_terms)
{
  using Real = make_real_t<C>;
  using Complex = C;
  auto const p = precision_of(t);
  Real const zero = scalar_at_precision<Real>(0, p), one = scalar_at_precision<Real>(1, p);
  Real const half = one / 2;
  Complex const zero_complex(zero, zero);
  if (kind < 1 || kind > 4 || !finite(u) || !uni20::isfinite(t) || !(t > zero))
    throw std::invalid_argument("theta requires kind 1..4, finite u, and finite t>0");
  Real const pi = Real{4} * math::atan(one);
  if (kind > 1 && t < one && u != zero_complex &&
      math::abs(u) < math::sqrt(uni20::numeric_limits<Real>::epsilon(t)) * t / Real{8})
  {
    // Avoid losing tiny arguments when adding a half-period to an even function.
    auto result = evaluate(kind, zero_complex, t, max_terms);
    result.derivative[0] += result.derivative[2] * u * u / Real{2};
    result.derivative[1] = result.derivative[2] * u;
    return result;
  }
  // In the Gaussian regime, translate the other kinds through theta1.
  Complex shifted = u;
  if (t < one && (kind == 2 || kind == 3)) shifted += half;
  if (t < one && (kind == 3 || kind == 4)) shifted += Complex(zero, t / Real{2});
  auto result = theta1(shifted, t, t < one ? 1u : kind, max_terms);
  if (t < one && (kind == 3 || kind == 4))
  {
    Complex const slope(zero, pi);
    result.derivative[2] += Real{2} * slope * result.derivative[1] + slope * slope * result.derivative[0];
    result.derivative[1] += slope * result.derivative[0];
    Complex const phase =
        (kind == 4 ? Complex(zero, -one) : Complex(one, zero)) * math::exp(Complex(zero, pi * u.real()));
    for (auto& d : result.derivative)
      d *= phase;
    result.log_scale -= pi * (t / Real{4} + u.imag());
  }
  if (kind > 1 && u == zero_complex) result.derivative[1] = zero_complex;
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
  using R = make_real_t<C>;
  auto const p = precision_of(t);
  if (kind < 1 || kind > 4 || max_terms == 0 || p.is_exact() || precision_of(u) != p)
    throw std::invalid_argument("theta jet requires kind 1..4, a positive budget and matching finite precision");
  if (!detail::theta_jet_detail::finite(u) || !uni20::isfinite(t) || !(t > 0))
    throw std::invalid_argument("theta jet requires finite u and finite t>0");
#if UNI20_HAS_MPREAL
  if constexpr (std::same_as<R, mpreal>)
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
  R const magnitude = math::exp(a.log_scale - b.log_scale + math::log(numerator) - math::log(denominator));
  if (!uni20::isfinite(magnitude) || !(magnitude > zero))
    throw std::overflow_error("theta ratio magnitude is outside scalar range");
  return magnitude * (a.derivative[da] / numerator) * uni20::conj(b.derivative[db] / denominator);
}
} // namespace uni20
