#pragma once

#include "math.hpp"
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace uni20
{
/// \brief A positive theta constant represented as scaled_value * exp(log_scale).
/// \details Keeping the factors separate preserves exponentially small constants.
///          Reconstructing the value or a ratio can still underflow or amplify the
///          absolute rounding error in log_scale. Both fields have the input precision.
template <class R> struct theta_constant_result
{
    R scaled_value;
    R log_scale;
};

namespace detail
{
template <class R>
concept ThetaReal = native_math::NativeReal<R>
#if UNI20_HAS_MPREAL
                    || std::same_as<R, mpreal>
#endif
    ;

template <ThetaReal R> theta_constant_result<R> theta_constant_series(unsigned kind, R const& t, std::size_t max_terms)
{
  auto const p = precision_of(t);
  auto number = [p](auto n) { return scalar_at_precision<R>(n, p); };
  R const zero = number(0), one = number(1), two = number(2);
  R const pi_value = uni20::pi<R> * one;
  bool const dual = t < one;
  // S-transform: theta2 <-> theta4, theta3 -> theta3, with factor 1/sqrt(t).
  unsigned const series_kind = dual && kind != 3 ? 6 - kind : kind;
  R scale = dual ? -math::log(t) / two : zero;
  if (series_kind == 2)
  {
    // Divide pi first: pi*t can overflow while pi*t/4 is representable.
    R const leading = dual ? (pi_value / number(4)) / t : (pi_value / number(4)) * t;
    scale -= leading;
  }
  if (!uni20::isfinite(scale)) throw std::overflow_error("theta constant logarithmic scale exceeds scalar range");

  // An infinite decay is harmless: every nonconstant series term then rounds
  // to zero. Never form exp(+decay) or subtract nearly equal theta constants.
  R const decay = dual ? pi_value / t : pi_value * t;
  R const target = numeric_limits<R>::epsilon(t) / number(16);
  if (!(target > zero)) throw std::underflow_error("theta constant series tolerance exceeds scalar range");
  R sum = series_kind == 2 ? two : one, correction = zero;
  for (std::size_t index = 0; index < max_terms; ++index)
  {
    R const n = number(index) + one;
    R const power = series_kind == 2 ? n * (n + one) : n * n;
    R const term = two * math::exp(-decay * power);
    // In the reduced domain t>=1 successive absolute terms have ratio at
    // most exp(-3*pi)<1/2. The entire omitted tail is less than twice term.
    // This bounds truncation in exact arithmetic, not floating-point rounding.
    if (term <= target) return {sum + correction, scale};
    R const signed_term = series_kind == 4 && index % 2 == 0 ? -term : term;
    R const next = sum + signed_term;
    correction += (sum - next) + signed_term; // |sum| > |signed_term|.
    sum = next;
  }
  throw std::runtime_error("theta constant series exhausted its term budget");
}
} // namespace detail

/// \brief Evaluate theta_kind(0 | i*t), kind=2,3,4 and finite t>0, in scaled form.
/// \details Uses a rapidly convergent Fourier series after a modular transformation.
///          Native arithmetic stays native. MPFR uses 32 guard bits and rounds both
///          result fields to t's precision, without changing ambient precision.
///          max_terms bounds the number of inspected nonconstant terms, including
///          the first omitted term used to bound the tail. The default permits
///          substantially more precision than ordinary numerical applications need.
/// \throws std::invalid_argument for an invalid kind/domain/budget or exact precision;
///         std::overflow_error if the logarithmic scale is unrepresentable;
///         std::runtime_error if the term budget is exhausted.
template <detail::ThetaReal R>
theta_constant_result<R> elliptic_theta_constant(unsigned kind, R const& t, std::size_t max_terms = 4096)
{
  if (kind < 2 || kind > 4 || max_terms == 0)
    throw std::invalid_argument("theta constant requires kind 2..4 and a positive term budget");
  auto const p = precision_of(t);
  if (p.is_exact()) throw std::invalid_argument("theta constant requires explicit finite precision");
  if (!uni20::isfinite(t) || !(t > 0)) throw std::invalid_argument("theta constant requires finite t>0");
#if UNI20_HAS_MPREAL
  if constexpr (std::same_as<R, mpreal>)
  {
    if (p.bit_count() > std::numeric_limits<Precision::bit_count_type>::max() - 32)
      throw std::overflow_error("theta constant guard precision exceeds supported range");
    auto const work = Precision::bits(p.bit_count() + 32);
    auto result = detail::theta_constant_series(kind, t.at(work), max_terms);
    theta_constant_result<R> rounded{result.scaled_value.at(p), result.log_scale.at(p)};
    if (!uni20::isfinite(rounded.log_scale))
      throw std::overflow_error("theta constant logarithmic scale exceeds result range");
    return rounded;
  }
  else
#endif
    return detail::theta_constant_series(kind, t, max_terms);
}

/// \brief Convert t to explicit precision p, then evaluate its scaled theta constant.
/// \details Native types require their own precision; MPFR accepts finite p and
///          can round exact input. This overload does not infer a precision from defaults.
template <detail::ThetaReal R>
theta_constant_result<R> elliptic_theta_constant(unsigned kind, R const& t, Precision p, std::size_t max_terms = 4096)
{
  if (p.is_exact()) throw std::invalid_argument("theta constant requires explicit finite precision");
  return elliptic_theta_constant(kind, scalar_at_precision<R>(t, p), max_terms);
}
} // namespace uni20
