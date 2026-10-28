#pragma once

#include "detail/stirling_coefficients.hpp"
#include "math.hpp"
#if UNI20_HAS_MPREAL
#include "detail/exact_bernoulli.hpp"
#endif
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace uni20
{
/// \brief Work limits for recurrence and Stirling/logarithm series evaluation.
struct log_gamma_options
{
    std::size_t max_shifts = 4096;
    std::size_t max_terms = 512;
};

/// \brief Analytic log-Gamma value, truncation bound and heuristic roundoff indicator.
/// \details The bound is evaluated in ordinary arithmetic, not interval arithmetic.
///          It bounds omitted series in exact arithmetic; the roundoff indicator
///          is a separate heuristic, not a correctness certificate. All scalar
///          fields have the context's requested precision. terms counts included
///          Stirling coefficients; shifts counts common recurrence steps.
template <class R> struct log_gamma_result
{
    complex<R> value;
    R truncation_bound;
    R roundoff_indicator;
    std::size_t terms;
    std::size_t shifts;
};

namespace detail
{
template <class R>
concept LogGammaReal = native_math::NativeReal<R>
#if UNI20_HAS_MPREAL
                       || std::same_as<R, mpreal>
#endif
    ;
} // namespace detail

/// \brief Immutable reusable right-half-plane log-Gamma and real-shift log-ratio context.
/// \details The branch is analytic for finite Re(z)>0 and real on the positive
///          axis, not the principal logarithm of Gamma(z). MPFR uses 64 guard
///          bits; fixed types retain native arithmetic. No ambient precision or
///          exponent range is changed. Inputs must be exact or have the context's
///          precision. Concurrent calls own their scratch and share coefficients.
/// \throws std::invalid_argument for invalid precision, domain or term budget;
///         std::runtime_error if the recurrence/series budget is exhausted;
///         std::overflow_error for unrepresentable arithmetic or diagnostics.
template <detail::LogGammaReal R> class log_gamma_context {
  public:
    using complex_type = complex<R>;

    /// \brief Construct at the fixed scalar's native precision.
    explicit log_gamma_context(log_gamma_options options = {})
      requires(!has_runtime_precision_v<R>)
        : log_gamma_context(native_precision<R>(), options)
    {}

    /// \brief Prepare coefficients at explicit finite precision p.
    explicit log_gamma_context(Precision p, log_gamma_options options = {})
        : precision_(p), work_(this->working_precision(p)), options_(options)
    {
      if (options.max_terms < 2) throw std::invalid_argument("log-Gamma needs at least two coefficient slots");
      std::size_t count = 16;
#if UNI20_HAS_MPREAL
      if constexpr (std::same_as<R, mpreal>)
      {
        auto const wanted = static_cast<std::uint64_t>(work_.bit_count() / 6 + 8);
        if (wanted > options.max_terms || wanted > (std::numeric_limits<std::size_t>::max() - 1) / 2)
          throw std::invalid_argument("log-Gamma coefficient budget is too small for requested precision");
        count = static_cast<std::size_t>(wanted);
        coefficients_.reserve(count);
        detail::exact_bernoulli_sequence sequence(2 * count);
        for (std::size_t n = 0; n <= 2 * count; ++n)
        {
          auto b = sequence.next();
          if (n >= 2 && n % 2 == 0)
            coefficients_.push_back(this->number(b / (exact_rational{n} * exact_rational{n - 1})));
        }
      }
      else
#endif
      {
        count = std::min(count, options.max_terms);
        auto table = detail::native_stirling_coefficients<R>();
        coefficients_.assign(table.begin(), table.begin() + count);
      }
      log_coefficient_ = math::log(math::abs(coefficients_.back()));
      log_target_ = math::log(numeric_limits<R>::epsilon(work_) / this->number(32));
      small_angle_ = math::sqrt(numeric_limits<R>::epsilon(work_)) / this->number(2);
      half_log_two_pi_ = math::log(this->number(2) * (pi<R> * this->number(1))) / this->number(2);
    }

    /// \brief Requested output precision.
    Precision precision() const { return precision_; }

    /// \brief Evaluate analytic log Gamma(z) on finite Re(z)>0.
    log_gamma_result<R> log_gamma_right(complex_type const& input) const
    {
      complex_type z = this->argument(input);
      accumulator sum(this->number(0));
      auto const one = this->number(1), half = one / this->number(2);
      std::size_t shifts = 0;
      R log_tail = this->log_remainder(z);
      while (log_tail > log_target_)
      {
        this->check_shift(shifts, z);
        sum.add(-this->logarithm(z));
        z += one;
        ++shifts;
        log_tail = this->log_remainder(z);
      }
      auto const log_z = this->logarithm(z);
      // Fuse the cancellation in the real component before the product rounds.
      // Near the real axis, x*atan(y/x) rounds to y at working precision;
      // evaluating the ratio first can underflow and lose that contribution.
      R const leading_real = math::fma(z.real(), log_z.real() - one, -z.imag() * log_z.imag());
      R const leading_imag = this->near_real(z) ? z.imag() * log_z.real()
                                                : math::fma(z.imag(), log_z.real() - one, z.real() * log_z.imag());
      sum.add(complex_type(leading_real, leading_imag));
      sum.add(-half * log_z);
      sum.add(complex_type(half_log_two_pi_));
      complex_type inverse = this->reciprocal(z), power = inverse;
      complex_type const square = inverse * inverse;
      for (std::size_t k = 0; k + 1 < coefficients_.size(); ++k)
      {
        sum.add(coefficients_[k] * power);
        power *= square;
      }
      return this->finish(sum, math::exp(log_tail), shifts);
    }

    /// \brief Evaluate log Gamma(z+h)-log Gamma(z), h real and both real parts positive.
    /// \details Forms paired differences before rounding, retaining a small shift
    ///          even when z+h rounds back to z. Does not subtract two log-Gamma
    ///          results. A zero shift returns exact zero at the output precision.
    log_gamma_result<R> log_gamma_ratio_right(complex_type const& input, R const& shift) const
    {
      complex_type z = this->argument(input);
      this->check_input_precision(precision_of(shift));
      R h = this->number(shift);
      R const zero = this->number(0), one = this->number(1), half = one / this->number(2);
      if (!uni20::isfinite(h)) throw std::invalid_argument("log-Gamma ratio requires finite shift");
      complex_type upper(z.real() + h, z.imag());
      if (!uni20::isfinite(upper)) throw std::overflow_error("log-Gamma shifted argument exceeds scalar range");
      if (!(upper.real() > zero)) throw std::invalid_argument("log-Gamma ratio requires Re(z+h)>0");
      if (h == zero) return {complex_type(this->output(zero)), this->output(zero), this->output(zero), 0, 0};

      // Recur from the smaller real part. The larger argument can already
      // satisfy the asymptotic bound while addition of one rounds away there.
      // Retaining h separately accounts for that common shift to working precision.
      bool const reverse = h < zero;
      if (reverse)
      {
        std::swap(z, upper);
        h = -h;
      }

      accumulator sum(zero);
      R logarithm_tail = zero;
      std::size_t shifts = 0;
      // Each remainder <= target/2, so their sum <= target. The factor two
      // is handled in logs to avoid underflow changing the admission decision.
      R const threshold = log_target_ - math::log(this->number(2));
      R tail0 = this->log_remainder(z), tail1 = zero;
      while (true)
      {
        // A failing first bound already requires another shift. Evaluate the
        // second bound only when it can change that decision.
        if (tail0 <= threshold)
        {
          tail1 = this->log_remainder(upper);
          if (tail1 <= threshold) break;
        }
        this->check_shift(shifts, z);
        sum.add(-this->log_increment(z, h));
        z += one;
        upper = complex_type(z.real() + h, z.imag());
        ++shifts;
        tail0 = this->log_remainder(z);
      }
      auto const delta = this->log_difference(z, h);
      bool const near_real = this->near_real(z);
      auto add_leading = [&](complex_type const& term) { sum.add(near_real ? complex_type(term.real(), zero) : term); };
      if (delta.small)
      {
        add_leading(h * this->logarithm(z));
        add_leading((h - half) * delta.log1p);
        add_leading(h * delta.relative_remainder);
        logarithm_tail += math::abs(h - half) * delta.log_error + math::abs(h) * delta.relative_error;
      }
      else
      {
        add_leading(h * (this->logarithm(upper) - one));
        add_leading((z - half) * delta.log1p);
      }
      if (near_real)
      {
        // First-order imaginary part of the paired leading term:
        // y*[log(1+h/x) + h/(2*x*(x+h))]. The omitted relative correction
        // is O((y/x)^2), below working roundoff in this branch. Preserve y*h/x
        // even when either y/x or h/x cannot be represented on its own.
        R const product = this->product_quotient(z.imag(), h, z.real());
        R const phase = delta.small ? product * (one + delta.relative_remainder.real()) : z.imag() * delta.log1p.real();
        sum.add(complex_type(zero, phase + (half * product) / upper.real()));
      }
      complex_type const inv0 = this->reciprocal(z), inv1 = this->reciprocal(upper);
      complex_type const square0 = inv0 * inv0, square1 = inv1 * inv1;
      complex_type const delta_inv = (-h * inv0) * inv1;
      complex_type const delta_square = delta_inv * (inv0 + inv1);
      complex_type power = inv0, delta_power = delta_inv;
      for (std::size_t k = 0; k + 1 < coefficients_.size(); ++k)
      {
        sum.add(coefficients_[k] * delta_power);
        delta_power = delta_power * square1 + power * delta_square;
        power *= square0;
      }
      auto result = this->finish(sum, math::exp(tail0) + math::exp(tail1) + logarithm_tail, shifts);
      if (reverse) result.value = -result.value;
      return result;
    }

  private:
    struct accumulator
    {
        complex_type sum, correction;
        R scaled_mass, epsilon;
        explicit accumulator(R zero)
            : sum(zero), correction(zero), scaled_mass(zero), epsilon(numeric_limits<R>::epsilon(zero))
        {}
        static R error(R const& a, R const& b, R const& next)
        {
          return math::abs(a) >= math::abs(b) ? (a - next) + b : (b - next) + a;
        }
        void add(complex_type const& term)
        {
          complex_type const next = sum + term;
          correction +=
              complex_type(error(sum.real(), term.real(), next.real()), error(sum.imag(), term.imag(), next.imag()));
          sum = next;
          // Scale before taking a magnitude or summing. A finite complex
          // value can have a magnitude outside R's range, yet finite error.
          scaled_mass += math::hypot(epsilon * term.real(), epsilon * term.imag());
        }
    };

    struct logarithm_difference
    {
        complex_type log1p, relative_remainder;
        R log_error, relative_error;
        bool small;
    };

    static Precision working_precision(Precision p)
    {
      if (p.is_exact()) throw std::invalid_argument("log-Gamma requires explicit finite precision");
      require_precision<R>(p);
      if constexpr (has_runtime_precision_v<R>)
      {
        if (p.bit_count() > std::numeric_limits<Precision::bit_count_type>::max() - 64)
          throw std::overflow_error("log-Gamma guard precision exceeds supported range");
        return Precision::bits(p.bit_count() + 64);
      }
      return p;
    }
    template <class T> R number(T const& x) const { return scalar_at_precision<R>(x, work_); }
    template <class T> T output(T const& x) const { return scalar_at_precision<T>(x, precision_); }
    void check_input_precision(Precision p) const
    {
      if (!p.is_exact() && p != precision_) throw std::invalid_argument("log-Gamma input precision mismatch");
    }
    complex_type argument(complex_type const& z) const
    {
      this->check_input_precision(precision_of(z));
      if (!uni20::isfinite(z) || !(z.real() > 0)) throw std::invalid_argument("log-Gamma requires finite Re(z)>0");
      return scalar_at_precision<complex_type>(z, work_);
    }
    // Scaling avoids overflow of abs(z), and avoids squaring tiny components.
    R scale(complex_type const& z) const { return std::max(math::abs(z.real()), math::abs(z.imag())); }
    bool near_real(complex_type const& z) const { return math::abs(z.imag()) <= z.real() * small_angle_; }
    // Only the final product/quotient is rounded into the scalar exponent range.
    R product_quotient(R const& a, R const& b, R const& c) const
    {
      auto const aa = math::frexp(a), bb = math::frexp(b), cc = math::frexp(c);
      auto const exponent = static_cast<__int128_t>(aa.exponent) + bb.exponent - cc.exponent;
      // MPFR permits exponent ranges whose intermediate sum exceeds int64.
      // Beyond these limits the final scaling necessarily overflows/underflows.
      auto const limited = std::clamp(exponent, static_cast<__int128_t>(std::numeric_limits<std::int64_t>::min()),
                                      static_cast<__int128_t>(std::numeric_limits<std::int64_t>::max()));
      return math::ldexp((aa.fraction * bb.fraction) / cc.fraction, static_cast<std::int64_t>(limited));
    }
    complex_type logarithm(complex_type const& z) const
    {
      R const scale = this->scale(z), re = z.real() / scale, im = z.imag() / scale;
      return {math::log(scale) + math::log(math::hypot(re, im)), math::atan2(z.imag(), z.real())};
    }
    complex_type reciprocal(complex_type const& z) const
    {
      R const scale = this->scale(z), re = z.real() / scale, im = z.imag() / scale;
      R const norm = re * re + im * im;
      return {(re / norm) / scale, (-im / norm) / scale};
    }
    R log_remainder(complex_type const& z) const
    {
      R const scale = this->scale(z), re = z.real() / scale, im = z.imag() / scale;
      R const norm = math::hypot(re, im), n = this->number(coefficients_.size());
      R const sec2 = this->number(2) / (this->number(1) + re / norm);
      // DLMF 5.11(ii): first omitted term times sec^(2n)(arg(z)/2).
      return log_coefficient_ + n * math::log(sec2) -
             (this->number(2) * n - this->number(1)) * (math::log(scale) + math::log(norm));
    }
    // Recurrence needs only log(1+h/z), whereas the leading Stirling term
    // also needs log(1+h/z)/(h/z)-1 without cancellation. Avoid evaluating
    // that separate power series at every recurrence step. Here h>0; when
    // h/scale<=1/8 the normalized denominators are in [1, 2+1/8], so the
    // log1p/atan2 arguments cannot overflow or hide a representable quotient.
    complex_type log_increment(complex_type const& z, R const& h) const
    {
      R const scale = this->scale(z);
      if (h > scale / this->number(8)) return this->log_difference(z, h).log1p;
      R const re = z.real() / scale, im = z.imag() / scale, delta = h / scale;
      R const norm = re * re + im * im;
      R const real = math::log1p(delta * (this->number(2) * re + delta) / norm) / this->number(2);
      R const imag = math::atan2(-delta * im, norm + delta * re);
      return {real, imag};
    }
    logarithm_difference log_difference(complex_type const& z, R const& h) const
    {
      R const zero = this->number(0), one = this->number(1), scale = this->scale(z);
      if (math::abs(h) <= scale / this->number(8))
      {
        // R=log(1+w)/w-1 is evaluated directly. In particular w==0
        // from underflow must not turn z*log(1+w)-h into -h.
        complex_type const w = complex_type(h / scale) / complex_type(z.real() / scale, z.imag() / scale);
        complex_type power = -w;
        accumulator remainder(zero);
        R const target = numeric_limits<R>::epsilon(work_) / this->number(32);
        R denominator = this->number(2);
        for (std::size_t k = 0; k < options_.max_terms; ++k)
        {
          complex_type const term = power / denominator;
          remainder.add(term);
          power *= -w;
          // The next term and all successors are <= |power|/(k+3)/(1-1/8).
          if (math::abs(power) <= target)
          {
            complex_type const r = remainder.sum + remainder.correction;
            R const error = (math::abs(power) / (denominator + one)) * (this->number(8) / this->number(7));
            return {w * (one + r), r, math::abs(w) * error, error, true};
          }
          denominator += one;
        }
        throw std::runtime_error("log-Gamma ratio exhausted its logarithm term budget");
      }
      complex_type const upper(z.real() + h, z.imag());
      R const upper_scale = this->scale(upper);
      auto const lower_binary = math::frexp(scale), upper_binary = math::frexp(upper_scale);
      R const lower_magnitude = lower_binary.fraction * math::hypot(z.real() / scale, z.imag() / scale);
      R const upper_magnitude =
          upper_binary.fraction * math::hypot(upper.real() / upper_scale, upper.imag() / upper_scale);
      // Cancel the common binary scale before taking logarithms. Subtracting
      // log magnitudes separately loses digits when both arguments are tiny.
      R const real = math::log(upper_magnitude / lower_magnitude) +
                     this->number(upper_binary.exponent - lower_binary.exponent) * math::log(this->number(2));
      R const imag = math::atan2(upper.imag(), upper.real()) - math::atan2(z.imag(), z.real());
      return {complex_type(real, imag), complex_type(zero), zero, zero, false};
    }
    void check_shift(std::size_t shifts, complex_type const& z) const
    {
      if (shifts == options_.max_shifts) throw std::runtime_error("log-Gamma exhausted its recurrence budget");
      R const next = z.real() + this->number(1);
      if (!uni20::isfinite(next) || !(next > z.real()))
        throw std::overflow_error("log-Gamma recurrence cannot advance at working precision");
    }
    log_gamma_result<R> finish(accumulator const& sum, R const& tail, std::size_t shifts) const
    {
      auto const value = this->output(sum.sum + sum.correction);
      R const roundoff = this->number(64) * (numeric_limits<R>::epsilon(work_) + sum.scaled_mass);
      R const output_epsilon = numeric_limits<R>::epsilon(precision_);
      R const rounded_error = this->output(roundoff) + output_epsilon +
                              math::hypot(output_epsilon * value.real(), output_epsilon * value.imag());
      if (!uni20::isfinite(value) || !uni20::isfinite(rounded_error) || !uni20::isfinite(tail))
        throw std::overflow_error("log-Gamma value or diagnostic exceeds scalar range");
      return {value, this->output(tail), rounded_error, coefficients_.size() - 1, shifts};
    }

    Precision precision_, work_;
    log_gamma_options options_;
    std::vector<R> coefficients_;
    R log_coefficient_, log_target_, small_angle_, half_log_two_pi_;
};
} // namespace uni20
