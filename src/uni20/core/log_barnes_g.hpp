#pragma once

#include "complex_log_gamma.hpp"

namespace uni20
{
/// \brief Work limits for Barnes recurrence, coefficients and small-shift series.
struct log_barnes_g_options
{
    std::size_t max_shifts = 4096;
    std::size_t max_terms = 512;
};

/// \brief Real logarithm of a Barnes magnitude, with separate numerical diagnostics.
/// \details truncation_bound bounds omitted series in exact arithmetic, evaluated
///          without outward rounding. roundoff_indicator is a heuristic. All scalar
///          fields have the requested precision; neither diagnostic is an interval certificate.
template <class R> struct log_barnes_g_result
{
    R value;
    R truncation_bound;
    R roundoff_indicator;
    std::size_t terms;
    std::size_t shifts;
};

/// \brief Immutable right-half-plane Barnes magnitude and real-shift log-ratio context.
/// \details Finite Re(z)>0 is required. No branch of complex log G is exposed.
///          MPFR uses internal guard precision; fixed types remain native. Inputs
///          must be exact or match the requested precision. Construction prepares
///          coefficients and the G(1)=1 normalization within the supplied budgets.
///          Each call owns its scratch; concurrent readers share only immutable data.
/// \throws std::invalid_argument for invalid domain, precision or coefficient budget;
///         std::runtime_error for exhausted recurrence or logarithm budgets;
///         std::overflow_error for unrepresentable arithmetic or diagnostics.
template <detail::LogGammaReal R> class log_barnes_g_context {
  public:
    using complex_type = complex<R>;

    /// \brief Prepare at the fixed scalar type's native precision.
    explicit log_barnes_g_context(log_barnes_g_options options = {})
      requires(!has_runtime_precision_v<R>)
        : log_barnes_g_context(native_precision<R>(), options)
    {}

    /// \brief Prepare coefficients and normalization at explicit finite precision.
    explicit log_barnes_g_context(Precision p, log_barnes_g_options options = {})
        : precision_(p), work_(this->working_precision(p)), options_(options),
          gamma_(work_, {.max_shifts = options.max_shifts, .max_terms = options.max_terms})
    {
      if (options.max_terms < 3) throw std::invalid_argument("Barnes G needs at least three coefficient slots");
      std::size_t count = 16;
#if UNI20_HAS_MPREAL
      if constexpr (std::same_as<R, mpreal>)
      {
        auto const wanted = static_cast<std::uint64_t>(work_.bit_count() / 6 + 8);
        if (wanted > options.max_terms || wanted > (std::numeric_limits<std::size_t>::max() - 1) / 2)
          throw std::invalid_argument("Barnes coefficient budget is too small for requested precision");
        count = static_cast<std::size_t>(wanted);
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
      log_target_ = math::log(numeric_limits<R>::epsilon(work_) / this->number(64));
      half_log_two_pi_ = math::log(this->number(2) * (pi<R> * this->number(1))) / this->number(2);
      normalization_ = this->absolute_core(complex_type(this->number(1), this->number(0)));
    }

    /// \brief Requested precision of values and diagnostics.
    Precision precision() const { return precision_; }

    /// \brief Evaluate log(abs(G(z))) for finite Re(z)>0.
    log_barnes_g_result<R> log_abs_barnes_g_right(complex_type const& input) const
    {
      auto z = this->argument(input);
      auto const zero = this->number(0), one = this->number(1);
      if (z.imag() == zero && (z.real() == one || z.real() == this->number(2)))
        return {this->output(zero), this->output(zero), this->output(zero), 0, 0};
      auto result = this->absolute_core(z);
      result.value -= normalization_.value;
      result.truncation_bound += normalization_.truncation_bound;
      result.roundoff_indicator += normalization_.roundoff_indicator;
      return this->finish(result);
    }

    /// \brief Evaluate log(abs(G(z+h)))-log(abs(G(z))) for a finite real shift.
    /// \details Both real parts must be positive. The paired expansion retains h
    ///          even when adding it rounds back to z. No separate Barnes values
    ///          are subtracted; the normalization cancels algebraically.
    log_barnes_g_result<R> log_abs_barnes_g_ratio_right(complex_type const& input, R const& shift) const
    {
      complex_type z = this->argument(input);
      this->check_precision(precision_of(shift));
      R h = this->number(shift), zero = this->number(0), one = this->number(1);
      if (!uni20::isfinite(h)) throw std::invalid_argument("Barnes shift must be finite");
      complex_type upper(z.real() + h, z.imag());
      if (!uni20::isfinite(upper)) throw std::overflow_error("Barnes shifted argument exceeds scalar range");
      if (!(upper.real() > zero)) throw std::invalid_argument("Barnes ratio requires Re(z+h)>0");
      if (h == zero) return {this->output(zero), this->output(zero), this->output(zero), 0, 0};
      bool const reverse = h < zero;
      if (reverse)
      {
        z = upper;
        h = -h;
      }
      accumulator sum(zero);
      complex_type w = z;
      std::size_t shifts = 0;
      R tail0, tail1;
      R const threshold = log_target_ - math::log(this->number(2));
      while (true)
      {
        auto const v = w - one;
        upper = complex_type(v.real() + h, v.imag());
        if (this->asymptotic_sector(v) && (tail0 = this->log_remainder(v)) <= threshold &&
            (tail1 = this->log_remainder(upper)) <= threshold)
          break;
        this->check_shift(shifts, w);
        sum.add(this->number(shifts + 1) * this->log_increment_real(w, h));
        w += one;
        ++shifts;
      }
      complex_type v = w - one;
      R tail = math::exp(tail0) + math::exp(tail1);
      tail += this->paired_leading(sum, v, h);
      auto const inv0 = this->reciprocal(v), inv1 = this->reciprocal(upper);
      auto const square0 = inv0 * inv0, square1 = inv1 * inv1;
      auto const difference_inverse = (-h * inv0) * inv1;
      auto const difference_square = difference_inverse * (inv0 + inv1);
      complex_type power = square0, difference_power = difference_square;
      for (std::size_t k = 1; k + 1 < coefficients_.size(); ++k)
      {
        sum.add(this->coefficient(k) * difference_power);
        difference_power = difference_power * square1 + power * difference_square;
        power *= square0;
      }
      R extra_round = zero;
      if (shifts)
      {
        auto const gamma = gamma_.log_gamma_ratio_right(w, h);
        sum.add(-this->number(shifts) * gamma.value.real());
        tail += this->number(shifts) * gamma.truncation_bound;
        extra_round += this->number(shifts) * gamma.roundoff_indicator;
      }
      auto result = this->evaluate(sum, tail, extra_round, shifts);
      if (reverse) result.value = -result.value;
      return this->finish(result);
    }

  private:
    struct accumulator
    {
        R sum, correction, scaled_mass, epsilon;
        explicit accumulator(R zero)
            : sum(zero), correction(zero), scaled_mass(zero), epsilon(numeric_limits<R>::epsilon(zero))
        {}
        void account(complex_type const& term)
        {
          scaled_mass += math::hypot(epsilon * term.real(), epsilon * term.imag());
        }
        void add(R const& term)
        {
          R const next = sum + term;
          correction += math::abs(sum) >= math::abs(term) ? (sum - next) + term : (term - next) + sum;
          sum = next;
          scaled_mass += math::abs(epsilon * term);
        }
        void add(complex_type const& term)
        {
          this->account(term);
          this->add(term.real());
        }
    };

    static Precision working_precision(Precision p)
    {
      if (p.is_exact()) throw std::invalid_argument("Barnes G requires finite precision");
      require_precision<R>(p);
      if constexpr (has_runtime_precision_v<R>)
      {
        if (p.bit_count() > std::numeric_limits<Precision::bit_count_type>::max() - 128)
          throw std::overflow_error("Barnes guard precision exceeds supported range");
        return Precision::bits(p.bit_count() + 64);
      }
      return p;
    }
    template <class T> R number(T const& value) const { return scalar_at_precision<R>(value, work_); }
    R output(R const& value) const { return scalar_at_precision<R>(value, precision_); }
    void check_precision(Precision p) const
    {
      if (!p.is_exact() && p != precision_) throw std::invalid_argument("Barnes input precision mismatch");
    }
    complex_type argument(complex_type const& z) const
    {
      this->check_precision(precision_of(z));
      if (!uni20::isfinite(z) || !(z.real() > 0)) throw std::invalid_argument("Barnes G requires finite Re(z)>0");
      return scalar_at_precision<complex_type>(z, work_);
    }
    R scale(complex_type const& z) const { return std::max(math::abs(z.real()), math::abs(z.imag())); }
    complex_type logarithm(complex_type const& z) const
    {
      R const s = this->scale(z);
      return {math::log(s) + math::log(math::hypot(z.real() / s, z.imag() / s)), math::atan2(z.imag(), z.real())};
    }
    complex_type reciprocal(complex_type const& z) const
    {
      R const s = this->scale(z), re = z.real() / s, im = z.imag() / s, norm = re * re + im * im;
      return {(re / norm) / s, (-im / norm) / s};
    }
    R log_increment_real(complex_type const& z, R const& h) const
    {
      R const s = this->scale(z), two = this->number(2);
      if (h <= s / this->number(8))
      {
        R const re = z.real() / s, im = z.imag() / s, delta = h / s;
        return math::log1p(delta * (two * re + delta) / (re * re + im * im)) / two;
      }
      complex_type const upper(z.real() + h, z.imag());
      R const u = this->scale(upper);
      auto const a = math::frexp(s), b = math::frexp(u);
      return math::log((b.fraction * math::hypot(upper.real() / u, upper.imag() / u)) /
                       (a.fraction * math::hypot(z.real() / s, z.imag() / s))) +
             (this->number(b.exponent) - this->number(a.exponent)) * math::log(two);
    }
    R coefficient(std::size_t k) const
    {
      // c[k]=B_(2k+2)/((2k+2)(2k+1)); the combined Barnes series
      // has B_(2k+2)/(4k(k+1)). The last slot is reserved for the remainder.
      return coefficients_[k] * this->number(2 * k + 1) / this->number(2 * k);
    }
    bool asymptotic_sector(complex_type const& v) const
    {
      // The remainder holds for |arg(v)|<pi, including Re(v)<0. Keep a
      // margin from the cut, but do not force a Gamma recurrence for large
      // imaginary arguments simply because v=z-1 has a negative real part.
      return v.real() > 0 || (v.imag() != 0 && math::abs(v.imag()) >= math::abs(v.real()));
    }
    R log_remainder(complex_type const& v) const
    {
      R const s = this->scale(v), re = v.real() / s, im = v.imag() / s;
      R const norm = math::hypot(re, im), n = this->number(coefficients_.size() - 1);
      R const sec2 = this->number(2) / (this->number(1) + re / norm);
      // Combine |v| times the log-Gamma remainder with Nemes (2014),
      // Eq. (1.10), for the Gamma-containing Barnes expansion. Both use
      // the same first omitted Bernoulli coefficient and powers |v|^(-2n).
      return log_coefficient_ - this->number(2) * n * (math::log(s) + math::log(norm)) +
             (n + this->number(1)) * math::log(sec2) +
             math::log1p(this->number(1) / (this->number(2) * n * math::sqrt(sec2)));
    }
    void check_shift(std::size_t n, complex_type const& z) const
    {
      if (n == options_.max_shifts) throw std::runtime_error("Barnes recurrence budget exhausted");
      R const next = z.real() + this->number(1);
      if (!uni20::isfinite(next) || !(next > z.real()))
        throw std::overflow_error("Barnes recurrence cannot advance at working precision");
    }
    void leading(accumulator& sum, complex_type const& v) const
    {
      auto const log = this->logarithm(v);
      auto const square = v * v;
      sum.account(square);
      sum.add((square / this->number(2)) * (log - this->number(3) / this->number(2)));
      sum.add(-log / this->number(12));
      sum.add(half_log_two_pi_ * v);
    }
    R paired_leading(accumulator& sum, complex_type const& v, R const& h) const
    {
      R const one = this->number(1), two = this->number(2), zero = this->number(0);
      auto const log = this->logarithm(v);
      if (h <= this->scale(v) / this->number(8))
      {
        // F(v+h)-F(v) with F(v)=v^2(log(v)/2-3/4):
        // h*v*(log(v)-1)+h^2*log(v)/2+h^2*T(h/v),
        // T(w)=sum_{k>=1} (-1)^(k+1)w^k/[k(k+1)(k+2)].
        // This avoids v^2*log1p(h/v), even when v^2 would overflow.
        R const s = this->scale(v);
        complex_type const w = complex_type(h / s) / complex_type(v.real() / s, v.imag() / s);
        complex_type power = w, log1p(zero), remainder(zero);
        R log_tail = zero, remainder_tail = zero;
        bool done = false;
        auto const target = numeric_limits<R>::epsilon(work_) / this->number(64);
        for (std::size_t k = 1; k <= options_.max_terms; ++k)
        {
          log1p += power / this->number(k);
          remainder += power / (this->number(k) * this->number(k + 1) * this->number(k + 2));
          power *= -w;
          if (math::abs(power) <= target)
          {
            log_tail = math::abs(power) * this->number(8) / (this->number(7) * this->number(k + 1));
            remainder_tail = log_tail / (this->number(k + 2) * this->number(k + 3));
            done = true;
            break;
          }
        }
        if (!done) throw std::runtime_error("Barnes small-shift series budget exhausted");
        auto const hv = h * v;
        R const square = h * h;
        sum.account(hv);
        sum.add(hv * (log - one));
        sum.add((square / two) * log);
        sum.add(square * remainder);
        sum.add(-log1p / this->number(12));
        sum.add(h * half_log_two_pi_);
        return square * remainder_tail + log_tail / this->number(12);
      }
      complex_type const upper(v.real() + h, v.imag());
      auto const log_upper = this->logarithm(upper);
      complex_type const delta(this->log_increment_real(v, h), log_upper.imag() - log.imag());
      auto const product = h * (v + h / two);
      sum.account(product);
      sum.add(product * (log_upper - this->number(3) / two));
      sum.add((v * v / two - one / this->number(12)) * delta);
      sum.add(h * half_log_two_pi_);
      return zero;
    }
    log_barnes_g_result<R> absolute_core(complex_type z) const
    {
      R const zero = this->number(0), one = this->number(1);
      accumulator sum(zero);
      complex_type w = z;
      std::size_t shifts = 0;
      R log_tail;
      while (!this->asymptotic_sector(w - one) || (log_tail = this->log_remainder(w - one)) > log_target_)
      {
        this->check_shift(shifts, w);
        sum.add(this->number(shifts + 1) * this->logarithm(w).real());
        w += one;
        ++shifts;
      }
      complex_type const v = w - one, inverse = this->reciprocal(v), square = inverse * inverse;
      this->leading(sum, v);
      complex_type power = square;
      for (std::size_t k = 1; k + 1 < coefficients_.size(); ++k)
      {
        sum.add(this->coefficient(k) * power);
        power *= square;
      }
      R tail = math::exp(log_tail), rounding = zero;
      if (shifts)
      {
        auto const gamma = gamma_.log_gamma_right(w);
        sum.add(-this->number(shifts) * gamma.value.real());
        tail += this->number(shifts) * gamma.truncation_bound;
        rounding += this->number(shifts) * gamma.roundoff_indicator;
      }
      return this->evaluate(sum, tail, rounding, shifts);
    }
    log_barnes_g_result<R> evaluate(accumulator const& sum, R const& tail, R const& extra_round,
                                    std::size_t shifts) const
    {
      R const value = sum.sum + sum.correction;
      R const rounding = extra_round + this->number(128) * (numeric_limits<R>::epsilon(work_) + sum.scaled_mass);
      if (!uni20::isfinite(value) || !uni20::isfinite(tail) || !uni20::isfinite(rounding))
        throw std::overflow_error("Barnes value or diagnostic exceeds scalar range");
      return {value, tail, rounding, coefficients_.size() - 2, shifts};
    }
    log_barnes_g_result<R> finish(log_barnes_g_result<R> const& result) const
    {
      R const value = this->output(result.value), eps = numeric_limits<R>::epsilon(precision_);
      R const rounding = this->output(result.roundoff_indicator) + eps + math::abs(eps * value);
      R const tail = this->output(result.truncation_bound);
      if (!uni20::isfinite(value) || !uni20::isfinite(rounding) || !uni20::isfinite(tail))
        throw std::overflow_error("Barnes output or diagnostic exceeds scalar range");
      return {value, tail, rounding, result.terms, result.shifts};
    }

    Precision precision_, work_;
    log_barnes_g_options options_;
    log_gamma_context<R> gamma_;
    std::vector<R> coefficients_;
    R log_coefficient_, log_target_, half_log_two_pi_;
    log_barnes_g_result<R> normalization_{};
};
} // namespace uni20
