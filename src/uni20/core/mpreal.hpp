#pragma once

#include "detail/scaled_rational.hpp"
#include "exact_constant.hpp"
#include "math_constants.hpp"
#include "scalar_traits.hpp"
#include <memory>
#include <ostream>
#include <uni20/common/initialization.hpp>
#include <variant>

namespace uni20
{
namespace detail
{
struct mpreal_access;
struct mpcomplex_access;
inline void require_mpfr_tls()
{
  static bool const supported = mpfr_buildopt_tls_p() != 0;
  if (!supported) throw std::runtime_error("uni20::mpreal requires a thread-safe MPFR build");
}

// Only the active variant alternative owns a numerical payload. MPFR retains
// finite precision; the variant discriminator records exact/approximate/unset.
struct mpfr_value
{
    mpfr_t value{};
    bool initialized = false;
    explicit mpfr_value(Precision p)
    {
      require_mpfr_tls();
      mpfr_init2(value, p.bit_count());
      initialized = true;
      mpfr_set_zero(value, 1);
    }
    mpfr_value(mpfr_value const& other) : mpfr_value(Precision::bits(mpfr_get_prec(other.value)))
    {
      mpfr_set(value, other.value, MPFR_RNDN);
    }
    mpfr_value(mpfr_value&& other) noexcept
    {
      std::swap(value[0], other.value[0]);
      std::swap(initialized, other.initialized);
    }
    mpfr_value& operator=(mpfr_value other) noexcept
    {
      std::swap(value[0], other.value[0]);
      std::swap(initialized, other.initialized);
      return *this;
    }
    ~mpfr_value()
    {
      if (initialized) mpfr_clear(value);
    }
};
} // namespace detail

/// \brief Owning exact rational or finite-precision MPFR real, with eager arithmetic.
/// \details Default construction is exact zero. Integers and exact literals need
///          no working precision. Approximate arithmetic rounds to nearest, ties
///          to even, and unequal finite precisions require explicit conversion.
///          Assignment adopts the source state and precision. Native floating
///          inputs require explicit finite precision.
class mpreal {
  public:
    /// \brief Construct exact zero for ordinary generic accumulators.
    mpreal() : value_(std::in_place_type<exact_constant>) {}
    /// \brief Construct an unset placeholder for overwrite storage.
    explicit mpreal(uninitialized_t) noexcept : value_(std::monostate{}) {}
    /// \brief Construct zero at finite precision, or exact zero.
    explicit mpreal(Precision p) : mpreal(uninitialized)
    {
      if (p.is_exact())
        value_.emplace<exact_constant>();
      else
        value_.emplace<detail::mpfr_value>(p);
    }
    /// \brief Retain an exact rational without rounding.
    /// \throws std::logic_error If the source is unset.
    explicit mpreal(exact_constant value) : mpreal(uninitialized)
    {
      (void)value.native_handle();
      value_.emplace<exact_constant>(std::move(value));
    }
    explicit mpreal(decimal_literal value) : mpreal(exact_constant(value)) {}
    template <std::integral I>
      requires(!std::same_as<I, bool>)
    explicit mpreal(I value) : mpreal(exact_constant(value))
    {}

    /// \brief Parse a complete decimal or integer fraction at the requested precision.
    mpreal(std::string_view text, Precision p) : mpreal(p)
    {
      if (p.is_exact() || text.find('/') != std::string_view::npos)
      {
        *this = mpreal(exact_constant(text), p);
        return;
      }
      if (text.empty() || text.find('\0') != std::string_view::npos ||
          text.find_first_of(" \t\r\n\f\v") != std::string_view::npos)
        throw std::invalid_argument("mpreal: expected a complete decimal value");
      std::string owned(text);
      char* end = nullptr;
      mpfr_strtofr(this->approximate(), owned.c_str(), &end, 10, MPFR_RNDN);
      if (end != owned.c_str() + owned.size() || end == owned.c_str())
        throw std::invalid_argument("mpreal: invalid decimal value");
    }
    mpreal(exact_constant const& value, Precision p) : mpreal(p)
    {
      auto rational = value.native_handle();
      if (p.is_exact())
        value_ = value;
      else
        mpfr_set_q(this->approximate(), rational, MPFR_RNDN);
    }
    mpreal(decimal_literal value, Precision p) : mpreal(exact_constant(value), p) {}
    template <std::integral I>
      requires(!std::same_as<I, bool>)
    mpreal(I value, Precision p) : mpreal(exact_constant(value), p)
    {}

    /// \brief Copy a valid borrowed MPFR value, rounding at finite precision.
    /// \pre value points to an initialized MPFR object during this call.
    explicit mpreal(mpfr_srcptr value, Precision p) : mpreal(p) { mpfr_set(this->approximate(), value, MPFR_RNDN); }
    /// \brief Import a native floating approximation; precision must be finite.
    template <std::floating_point F> mpreal(F value, Precision p) : mpreal(p)
    {
      mpfr_set_ld(this->approximate(), static_cast<long double>(value), MPFR_RNDN);
    }
    /// \brief Convert to finite precision, or retain an already-exact value in exact mode.
    /// \throws std::invalid_argument If asked to reclassify an approximation as exact.
    mpreal(mpreal const& other, Precision p) : mpreal(p)
    {
      other.require_value();
      if (p.is_exact())
      {
        if (!other.is_exact()) throw std::invalid_argument("mpreal: an approximation cannot be reclassified as exact");
        value_ = other.exact_value();
      }
      else
        other.copy_to(this->approximate());
    }
    mpreal(mpreal const&) = default;
    /// \brief Transfer ownership, leaving the source unset.
    mpreal(mpreal&& other) noexcept : value_(std::move(other.value_)) { other.value_.emplace<std::monostate>(); }
    mpreal& operator=(mpreal other) noexcept
    {
      this->swap(other);
      return *this;
    }
    void swap(mpreal& other) noexcept { value_.swap(other.value_); }
    /// \brief Whether this object holds an exact or approximate numerical value.
    bool initialized() const noexcept { return !std::holds_alternative<std::monostate>(value_); }
    /// \brief Whether the active payload is an exact rational.
    bool is_exact() const noexcept { return std::holds_alternative<exact_constant>(value_); }
    /// \brief Return exact state or the working precision stored by MPFR.
    Precision precision() const
    {
      this->require_value();
      return this->is_exact() ? Precision::exact() : Precision::bits(mpfr_get_prec(this->native_handle()));
    }
    /// \brief Borrow an approximate MPFR value; exact and unset states require explicit materialization first.
    mpfr_srcptr native_handle() const
    {
      this->require_value();
      if (this->is_exact()) throw std::logic_error("mpreal: exact value needs finite precision before MPFR access");
      return std::get<detail::mpfr_value>(value_).value;
    }
    /// \brief Borrow the exact rational; valid while this object retains its state.
    exact_constant const& exact_value() const
    {
      this->require_value();
      if (!this->is_exact()) throw std::logic_error("mpreal: value is approximate");
      return std::get<exact_constant>(value_);
    }
    /// \brief Round into an initialized MPFR destination at its existing precision.
    void copy_to(mpfr_ptr destination) const
    {
      this->require_value();
      if (this->is_exact())
        mpfr_set_q(destination, this->exact_value().native_handle(), MPFR_RNDN);
      else
        mpfr_set(destination, this->native_handle(), MPFR_RNDN);
    }
    mpreal at(Precision p) const { return mpreal(*this, p); }
    /// \brief Explicitly round to a native format, nearest with ties to even.
    explicit operator float() const { return this->to_native<float, mpfr_get_flt>(); }
    explicit operator double() const { return this->to_native<double, mpfr_get_d>(); }
    explicit operator long double() const { return this->to_native<long double, mpfr_get_ld>(); }

    /// \brief Format exact rationals as numerator/denominator and approximations as decimal scientific notation.
    /// \details Exact values retain their rational spelling regardless of digits.
    ///          For approximations, zero digits requests a round-trip representation.
    std::string to_string(std::size_t digits = 0) const
    {
      if (this->is_exact()) return this->exact_value().to_string();
      auto value = this->native_handle();
      if (mpfr_nan_p(value)) return "nan";
      if (mpfr_inf_p(value)) return mpfr_signbit(value) ? "-inf" : "inf";
      if (mpfr_zero_p(value)) return mpfr_signbit(value) ? "-0" : "0";
      mpfr_exp_t exponent;
      std::unique_ptr<char, decltype(&mpfr_free_str)> text(
          mpfr_get_str(nullptr, &exponent, 10, digits, value, MPFR_RNDN), &mpfr_free_str);
      if (!text) throw std::bad_alloc();
      std::string_view mantissa(text.get());
      std::string result;
      if (mantissa.front() == '-')
      {
        result += '-';
        mantissa.remove_prefix(1);
      }
      result += mantissa.front();
      if (mantissa.size() > 1)
      {
        result += '.';
        result += mantissa.substr(1);
      }
      result += 'e';
      result += std::to_string(exponent - 1);
      return result;
    }

    mpreal& operator+=(mpreal const& rhs);
    mpreal& operator-=(mpreal const& rhs);
    mpreal& operator*=(mpreal const& rhs);
    mpreal& operator/=(mpreal const& rhs);
    mpreal& operator+=(exact_constant const& rhs);
    mpreal& operator-=(exact_constant const& rhs);
    mpreal& operator*=(exact_constant const& rhs);
    mpreal& operator/=(exact_constant const& rhs);

  private:
    friend mpreal epsilon(Precision precision);
    friend struct detail::mpreal_access;
    friend struct detail::mpcomplex_access;
    template <class F, auto Convert> F to_native() const
    {
      this->require_value();
      if (!this->is_exact()) return Convert(this->native_handle(), MPFR_RNDN);
      // Enclose the rational and refine until both bounds round identically.
      // This avoids double rounding, including at native subnormal boundaries.
      auto bits = static_cast<mpfr_prec_t>(std::numeric_limits<F>::digits + 16);
      for (;;)
      {
        detail::mpfr_value lower(Precision::bits(bits)), upper(Precision::bits(bits));
        mpfr_set_q(lower.value, this->exact_value().native_handle(), MPFR_RNDD);
        mpfr_set_q(upper.value, this->exact_value().native_handle(), MPFR_RNDU);
        F lo = Convert(lower.value, MPFR_RNDN), hi = Convert(upper.value, MPFR_RNDN);
        if (lo == hi) return lo;
        if (bits > MPFR_PREC_MAX / 2) throw std::overflow_error("mpreal: native conversion precision exhausted");
        bits *= 2;
      }
    }
    void require_value() const
    {
      if (!this->initialized()) throw std::logic_error("mpreal: numerical use of an unset value");
    }
    mpfr_ptr approximate()
    {
      (void)this->native_handle();
      return std::get<detail::mpfr_value>(value_).value;
    }
    std::variant<std::monostate, exact_constant, detail::mpfr_value> value_;
};

/// \brief Type properties and explicit runtime limits for exact-or-approximate reals.
/// \details Precision-dependent queries require a finite Precision or an
///          approximate exemplar. There is no type-wide epsilon or digit count.
///          Unimplemented limits are intentionally absent, rather than inheriting
///          the zero-valued defaults of an unspecialized std::numeric_limits.
template <> struct numeric_limits<mpreal>
{
    static constexpr bool is_specialized = true;
    static constexpr bool is_signed = true;
    static constexpr bool is_integer = false;
    static constexpr bool is_exact = false;
    static constexpr bool is_bounded = false;
    static constexpr bool is_iec559 = false;
    static constexpr int radix = 2;
    static constexpr bool has_infinity = true;
    static constexpr bool has_quiet_NaN = true;

    /// \brief Significand bits at a finite working precision.
    static mpfr_prec_t digits(Precision p) { return p.bit_count(); }
    static mpfr_prec_t digits(mpreal const& x) { return digits(x.precision()); }

    /// \brief Spacing above one at a finite working precision.
    /// \throws std::logic_error If the precision is exact or the exemplar is unset.
    static mpreal epsilon(Precision p);
    static mpreal epsilon(mpreal const& x) { return epsilon(x.precision()); }
    static mpreal epsilon() = delete;
};

namespace detail
{
struct mpreal_access
{
    // Construct results without exposing mutable provider handles on the scalar.
    template <class F> static mpreal finite_result(Precision p, F&& operation)
    {
      (void)p.bit_count();
      mpreal result(p);
      operation(result.approximate());
      return result;
    }
    template <class F> static exact_constant rational_result(F&& operation)
    {
      exact_constant result;
      operation(result.value_);
      mpq_canonicalize(result.value_);
      return result;
    }

    static bool can_expand_rational(mpreal const& x)
    {
      if (x.is_exact() || mpfr_zero_p(x.native_handle())) return true;
      auto bound = std::max<mpfr_prec_t>(4096, mpfr_get_prec(x.native_handle()));
      auto exponent = mpfr_get_exp(x.native_handle());
      return exponent >= -bound && exponent <= bound;
    }
    static scaled_rational stored_scaled(mpreal const& x)
    {
      if (x.is_exact()) return {x.exact_value(), 0};
      scaled_rational result;
      if (!mpfr_zero_p(x.native_handle()))
        result.exponent = mpfr_get_z_2exp(mpq_numref(result.coefficient.value_), x.native_handle());
      return result;
    }
    // Internal arithmetic representation of a finite stored value. This does
    // not reclassify the user's approximation as an exact scalar.
    static exact_constant stored_rational(mpreal const& x)
    {
      if (x.is_exact()) return x.exact_value();
      exact_constant result;
      mpfr_get_q(result.value_, x.native_handle());
      return result;
    }

    template <auto Operation> static mpreal mixed(mpreal const& a, mpreal const& b, Precision p)
    {
      auto const& real = a.is_exact() ? b : a;
      auto const& rational = a.is_exact() ? a.exact_value() : b.exact_value();
      auto q = rational.native_handle();
      mpreal result(p);
      if (rational == 0)
      {
        // MPFR's _q helpers may copy their floating operand for q == 0.
        // Use floating +0 to preserve binary-operation signed-zero semantics.
        auto r = real.native_handle();
        auto zero = result.native_handle();
        Operation(result.approximate(), a.is_exact() ? zero : r, a.is_exact() ? r : zero, MPFR_RNDN);
        return result;
      }
      if constexpr (Operation == mpfr_add)
        mpfr_add_q(result.approximate(), real.native_handle(), q, MPFR_RNDN);
      else if constexpr (Operation == mpfr_mul)
        mpfr_mul_q(result.approximate(), real.native_handle(), q, MPFR_RNDN);
      else if constexpr (Operation == mpfr_sub)
      {
        if (a.is_exact())
        {
          // Negation is exact, including signed zero; adding q then rounds once.
          mpfr_neg(result.approximate(), real.native_handle(), MPFR_RNDN);
          mpfr_add_q(result.approximate(), result.native_handle(), q, MPFR_RNDN);
        }
        else
          mpfr_sub_q(result.approximate(), real.native_handle(), q, MPFR_RNDN);
      }
      else if constexpr (Operation == mpfr_div)
      {
        if (!a.is_exact())
          mpfr_div_q(result.approximate(), real.native_handle(), q, MPFR_RNDN);
        else if (mpfr_number_p(real.native_handle()) && !mpfr_zero_p(real.native_handle()) && rational != 0)
        {
          if (can_expand_rational(real))
            result = mpreal(rational / stored_rational(real), p);
          else
            scaled_rational_ratio({stored_scaled(a), {}, stored_scaled(b), {}}).round_to(result.approximate());
        }
        else
        {
          // Only signs and zero matter here. Do not overflow/underflow q while
          // preparing zero, infinity or NaN arithmetic.
          mpfr_set_si(result.approximate(), mpq_sgn(q), MPFR_RNDN);
          mpfr_div(result.approximate(), result.native_handle(), real.native_handle(), MPFR_RNDN);
        }
      }
      return result;
    }
    // For rational x and base 2 or 10, a rational logarithm is an integer.
    // Check numerator and denominator independently without any floating conversion.
    static mpreal exact_logarithm(exact_constant const& x, unsigned long base)
    {
      auto q = x.native_handle();
      if (mpq_sgn(q) > 0)
      {
        mp_integer factor, remainder;
        mpz_set_ui(factor.value, base);
        auto numerator_power = mpz_remove(remainder.value, mpq_numref(q), factor.value);
        if (mpz_cmp_ui(remainder.value, 1) == 0)
        {
          auto denominator_power = mpz_remove(remainder.value, mpq_denref(q), factor.value);
          if (mpz_cmp_ui(remainder.value, 1) == 0)
            return mpreal(exact_constant(numerator_power) - exact_constant(denominator_power));
        }
      }
      throw std::logic_error("mpreal: logarithm requires finite working precision");
    }

    // Explicit operation precision means convert the inputs, then evaluate.
    template <auto Operation> static mpreal apply(mpreal const& x, Precision p)
    {
      (void)p.bit_count();
      return apply<Operation>(x.at(p));
    }
    template <auto Operation> static mpreal apply(mpreal const& x, mpreal const& y, Precision p)
    {
      (void)p.bit_count();
      return apply<Operation>(x.at(p), y.at(p));
    }

    template <auto Operation> static mpreal apply(mpreal const& x)
    {
      auto p = x.precision();
      if (p.is_exact())
      {
        if constexpr (Operation == mpfr_neg)
          return mpreal(-x.exact_value());
        else if constexpr (Operation == mpfr_abs)
          return mpreal(x.exact_value() < 0 ? -x.exact_value() : x.exact_value());
        else if constexpr (Operation == mpfr_sqrt)
          return mpreal(uni20::sqrt(x.exact_value()));
        else if constexpr (Operation == mpfr_cbrt)
        {
          if (auto root = rational_root(x.exact_value(), 3)) return mpreal(*root);
          throw std::logic_error("mpreal: cube root requires finite working precision");
        }
        else if constexpr (Operation == mpfr_exp2)
          return mpreal(uni20::pow(exact_constant{2}, x.exact_value()));
        else if constexpr (Operation == mpfr_log2)
          return exact_logarithm(x.exact_value(), 2);
        else if constexpr (Operation == mpfr_log10)
          return exact_logarithm(x.exact_value(), 10);
        else
        {
          auto const& q = x.exact_value();
          if constexpr (Operation == mpfr_exp || Operation == mpfr_cos || Operation == mpfr_cosh)
          {
            if (q == 0) return mpreal{1};
          }
          else if constexpr (Operation == mpfr_log || Operation == mpfr_acos || Operation == mpfr_acosh)
          {
            if (q == 1) return mpreal{};
          }
          else if constexpr (Operation == mpfr_sin || Operation == mpfr_tan || Operation == mpfr_atan ||
                             Operation == mpfr_expm1 || Operation == mpfr_log1p || Operation == mpfr_asin ||
                             Operation == mpfr_sinh || Operation == mpfr_tanh || Operation == mpfr_asinh ||
                             Operation == mpfr_atanh)
          {
            if (q == 0) return mpreal{};
          }
          throw std::logic_error("mpreal: elementary function requires finite working precision");
        }
      }
      mpreal result(p);
      Operation(result.approximate(), x.native_handle(), MPFR_RNDN);
      return result;
    }
    template <auto Operation> static mpreal apply(mpreal const& a, mpreal const& b)
    {
      auto p = common_precision(a.precision(), b.precision());
      if (p.is_exact())
      {
        if constexpr (Operation == mpfr_add)
          return mpreal(a.exact_value() + b.exact_value());
        else if constexpr (Operation == mpfr_sub)
          return mpreal(a.exact_value() - b.exact_value());
        else if constexpr (Operation == mpfr_mul)
          return mpreal(a.exact_value() * b.exact_value());
        else if constexpr (Operation == mpfr_div)
          return mpreal(a.exact_value() / b.exact_value());
        else if constexpr (Operation == mpfr_pow)
          return mpreal(uni20::pow(a.exact_value(), b.exact_value()));
        else if constexpr (Operation == mpfr_hypot)
          return mpreal(uni20::sqrt(a.exact_value() * a.exact_value() + b.exact_value() * b.exact_value()));
        else
        {
          if constexpr (Operation == mpfr_atan2)
            if (a.exact_value() == 0 && b.exact_value() > 0) return mpreal{};
          throw std::logic_error("mpreal: elementary function requires finite working precision");
        }
      }
      if constexpr (Operation == mpfr_add || Operation == mpfr_sub || Operation == mpfr_mul || Operation == mpfr_div)
        if (a.is_exact() != b.is_exact()) return mixed<Operation>(a, b, p);
      mpreal av(uninitialized), bv(uninitialized);
      if (a.is_exact()) av = a.at(p);
      if (b.is_exact()) bv = b.at(p);
      mpreal result(p);
      Operation(result.approximate(), a.is_exact() ? av.native_handle() : a.native_handle(),
                b.is_exact() ? bv.native_handle() : b.native_handle(), MPFR_RNDN);
      return result;
    }
    template <auto Operation> static mpreal& assign(mpreal& a, mpreal const& b)
    {
      if (!a.is_exact() && !b.is_exact())
      {
        (void)common_precision(a.precision(), b.precision());
        Operation(a.approximate(), a.native_handle(), b.native_handle(), MPFR_RNDN);
      }
      else
        a = apply<Operation>(a, b);
      return a;
    }
    template <auto Operation> static mpreal constant(Precision p)
    {
      (void)p.bit_count();
      mpreal result(p);
      Operation(result.approximate(), MPFR_RNDN);
      return result;
    }
};
} // namespace detail

inline mpreal exact_constant::at(Precision precision) const { return mpreal(*this, precision); }
inline mpreal decimal_literal::at(Precision precision) const { return mpreal(*this, precision); }

inline mpreal operator+(mpreal const& a, mpreal const& b) { return detail::mpreal_access::apply<mpfr_add>(a, b); }
inline mpreal operator-(mpreal const& a, mpreal const& b) { return detail::mpreal_access::apply<mpfr_sub>(a, b); }
inline mpreal operator*(mpreal const& a, mpreal const& b) { return detail::mpreal_access::apply<mpfr_mul>(a, b); }
inline mpreal operator/(mpreal const& a, mpreal const& b) { return detail::mpreal_access::apply<mpfr_div>(a, b); }
inline mpreal operator-(mpreal const& a) { return detail::mpreal_access::apply<mpfr_neg>(a); }
inline mpreal operator+(mpreal const& a)
{
  (void)a.precision();
  return a;
}

// Basic arithmetic retains the exact operand until rounding the result.
inline mpreal operator+(mpreal const& a, exact_constant const& b) { return a + mpreal(b); }
inline mpreal operator-(mpreal const& a, exact_constant const& b) { return a - mpreal(b); }
inline mpreal operator*(mpreal const& a, exact_constant const& b) { return a * mpreal(b); }
inline mpreal operator/(mpreal const& a, exact_constant const& b) { return a / mpreal(b); }
inline mpreal operator+(exact_constant const& a, mpreal const& b) { return mpreal(a) + b; }
inline mpreal operator-(exact_constant const& a, mpreal const& b) { return mpreal(a) - b; }
inline mpreal operator*(exact_constant const& a, mpreal const& b) { return mpreal(a) * b; }
inline mpreal operator/(exact_constant const& a, mpreal const& b) { return mpreal(a) / b; }

inline mpreal& mpreal::operator+=(mpreal const& rhs) { return detail::mpreal_access::assign<mpfr_add>(*this, rhs); }
inline mpreal& mpreal::operator-=(mpreal const& rhs) { return detail::mpreal_access::assign<mpfr_sub>(*this, rhs); }
inline mpreal& mpreal::operator*=(mpreal const& rhs) { return detail::mpreal_access::assign<mpfr_mul>(*this, rhs); }
inline mpreal& mpreal::operator/=(mpreal const& rhs) { return detail::mpreal_access::assign<mpfr_div>(*this, rhs); }
inline mpreal& mpreal::operator+=(exact_constant const& rhs) { return *this += mpreal(rhs); }
inline mpreal& mpreal::operator-=(exact_constant const& rhs) { return *this -= mpreal(rhs); }
inline mpreal& mpreal::operator*=(exact_constant const& rhs) { return *this *= mpreal(rhs); }
inline mpreal& mpreal::operator/=(exact_constant const& rhs) { return *this /= mpreal(rhs); }

/// \brief Compare exact or stored approximate values without rounding either operand.
inline std::partial_ordering operator<=>(mpreal const& a, mpreal const& b)
{
  (void)a.precision();
  (void)b.precision();
  if (a.is_exact() && b.is_exact()) return a.exact_value() <=> b.exact_value();
  if (a.is_exact())
  {
    if (mpfr_nan_p(b.native_handle())) return std::partial_ordering::unordered;
    return 0 <=> mpfr_cmp_q(b.native_handle(), a.exact_value().native_handle());
  }
  if (b.is_exact())
  {
    if (mpfr_nan_p(a.native_handle())) return std::partial_ordering::unordered;
    return mpfr_cmp_q(a.native_handle(), b.exact_value().native_handle()) <=> 0;
  }
  if (mpfr_unordered_p(a.native_handle(), b.native_handle())) return std::partial_ordering::unordered;
  return mpfr_cmp(a.native_handle(), b.native_handle()) <=> 0;
}
inline bool operator==(mpreal const& a, mpreal const& b) { return (a <=> b) == 0; }
inline std::partial_ordering operator<=>(mpreal const& a, exact_constant const& b) { return a <=> mpreal(b); }
inline bool operator==(mpreal const& a, exact_constant const& b) { return (a <=> b) == 0; }

namespace detail
{
template <std::integral I> int compare_integer(mpreal const& a, I b)
{
  if constexpr (std::is_signed_v<I> && std::numeric_limits<I>::digits <= std::numeric_limits<long>::digits)
    return mpfr_cmp_si(a.native_handle(), static_cast<long>(b));
  else if constexpr (!std::is_signed_v<I> &&
                     std::numeric_limits<I>::digits <= std::numeric_limits<unsigned long>::digits)
    return mpfr_cmp_ui(a.native_handle(), static_cast<unsigned long>(b));
  else
    return mpfr_cmp_q(a.native_handle(), exact_constant(b).native_handle());
}
} // namespace detail

template <std::integral I>
  requires(!std::same_as<I, bool>)
inline bool operator==(mpreal const& a, I b)
{
  if (a.is_exact()) return a.exact_value() == exact_constant(b);
  if (b == 0) return mpfr_zero_p(a.native_handle());
  return mpfr_number_p(a.native_handle()) && detail::compare_integer(a, b) == 0;
}
template <std::integral I>
  requires(!std::same_as<I, bool>)
inline std::partial_ordering operator<=>(mpreal const& a, I b)
{
  if (a.is_exact()) return a.exact_value() <=> exact_constant(b);
  if (mpfr_nan_p(a.native_handle())) return std::partial_ordering::unordered;
  return detail::compare_integer(a, b) <=> 0;
}
inline bool isfinite(mpreal const& x) { return x.is_exact() || mpfr_number_p(x.native_handle()); }
inline bool isnan(mpreal const& x) { return !x.is_exact() && mpfr_nan_p(x.native_handle()); }
inline bool isinf(mpreal const& x) { return !x.is_exact() && mpfr_inf_p(x.native_handle()); }
inline bool signbit(mpreal const& x)
{
  return x.is_exact() ? mpq_sgn(x.exact_value().native_handle()) < 0 : mpfr_signbit(x.native_handle());
}
inline mpreal conj(mpreal const& x)
{
  (void)x.precision();
  return x;
}
inline mpreal abs(mpreal const& x) { return detail::mpreal_access::apply<mpfr_abs>(x); }
inline mpreal sqrt(mpreal const& x) { return detail::mpreal_access::apply<mpfr_sqrt>(x); }
inline mpreal exp(mpreal const& x) { return detail::mpreal_access::apply<mpfr_exp>(x); }
inline mpreal log(mpreal const& x) { return detail::mpreal_access::apply<mpfr_log>(x); }
inline mpreal sin(mpreal const& x) { return detail::mpreal_access::apply<mpfr_sin>(x); }
inline mpreal cos(mpreal const& x) { return detail::mpreal_access::apply<mpfr_cos>(x); }
inline mpreal tan(mpreal const& x) { return detail::mpreal_access::apply<mpfr_tan>(x); }
inline mpreal atan(mpreal const& x) { return detail::mpreal_access::apply<mpfr_atan>(x); }
inline mpreal atan2(mpreal const& y, mpreal const& x) { return detail::mpreal_access::apply<mpfr_atan2>(y, x); }
inline mpreal hypot(mpreal const& x, mpreal const& y) { return detail::mpreal_access::apply<mpfr_hypot>(x, y); }
inline mpreal pow(mpreal const& x, mpreal const& y) { return detail::mpreal_access::apply<mpfr_pow>(x, y); }

/// \brief Evaluate `abs(x.at(p))`, returning an approximation at finite precision p.
/// \throws std::logic_error If p is exact or x is unset.
inline mpreal abs(mpreal const& x, Precision p) { return detail::mpreal_access::apply<mpfr_abs>(x, p); }
/// \brief Evaluate `sqrt(x.at(p))`, returning an approximation at finite precision p.
/// \throws std::logic_error If p is exact or x is unset.
inline mpreal sqrt(mpreal const& x, Precision p) { return detail::mpreal_access::apply<mpfr_sqrt>(x, p); }
/// \brief Evaluate `exp(x.at(p))`, returning an approximation at finite precision p.
/// \throws std::logic_error If p is exact or x is unset.
inline mpreal exp(mpreal const& x, Precision p) { return detail::mpreal_access::apply<mpfr_exp>(x, p); }
/// \brief Evaluate `log(x.at(p))`, returning an approximation at finite precision p.
/// \throws std::logic_error If p is exact or x is unset.
inline mpreal log(mpreal const& x, Precision p) { return detail::mpreal_access::apply<mpfr_log>(x, p); }
/// \brief Evaluate `sin(x.at(p))`, returning an approximation at finite precision p.
/// \throws std::logic_error If p is exact or x is unset.
inline mpreal sin(mpreal const& x, Precision p) { return detail::mpreal_access::apply<mpfr_sin>(x, p); }
/// \brief Evaluate `cos(x.at(p))`, returning an approximation at finite precision p.
/// \throws std::logic_error If p is exact or x is unset.
inline mpreal cos(mpreal const& x, Precision p) { return detail::mpreal_access::apply<mpfr_cos>(x, p); }
/// \brief Evaluate `tan(x.at(p))`, returning an approximation at finite precision p.
/// \throws std::logic_error If p is exact or x is unset.
inline mpreal tan(mpreal const& x, Precision p) { return detail::mpreal_access::apply<mpfr_tan>(x, p); }
/// \brief Evaluate `atan(x.at(p))`, returning an approximation at finite precision p.
/// \throws std::logic_error If p is exact or x is unset.
inline mpreal atan(mpreal const& x, Precision p) { return detail::mpreal_access::apply<mpfr_atan>(x, p); }
/// \brief Evaluate `atan2(y.at(p), x.at(p))`, overriding both operands' precisions.
/// \throws std::logic_error If p is exact or either operand is unset.
inline mpreal atan2(mpreal const& y, mpreal const& x, Precision p)
{
  return detail::mpreal_access::apply<mpfr_atan2>(y, x, p);
}
/// \brief Evaluate `hypot(x.at(p), y.at(p))`, overriding both operands' precisions.
/// \throws std::logic_error If p is exact or either operand is unset.
inline mpreal hypot(mpreal const& x, mpreal const& y, Precision p)
{
  return detail::mpreal_access::apply<mpfr_hypot>(x, y, p);
}
/// \brief Evaluate `pow(x.at(p), y.at(p))`, overriding both operands' precisions.
/// \throws std::logic_error If p is exact or either operand is unset.
inline mpreal pow(mpreal const& x, mpreal const& y, Precision p)
{
  return detail::mpreal_access::apply<mpfr_pow>(x, y, p);
}

/// \brief Real cube root, including negative arguments; retain input precision or a supported exact result.
/// \throws std::logic_error If an exact input requires approximation, or x is unset.
inline mpreal cbrt(mpreal const& x) { return detail::mpreal_access::apply<mpfr_cbrt>(x); }
/// \brief Evaluate `cbrt(x.at(p))`, returning an approximation at finite precision p.
/// \throws std::logic_error If p is exact or x is unset.
inline mpreal cbrt(mpreal const& x, Precision p) { return detail::mpreal_access::apply<mpfr_cbrt>(x, p); }

/// \brief Base-two exponential; retain input precision or a supported exact result.
/// \throws std::logic_error If an exact input requires approximation, or x is unset.
inline mpreal exp2(mpreal const& x) { return detail::mpreal_access::apply<mpfr_exp2>(x); }
/// \brief Evaluate `exp2(x.at(p))`, returning an approximation at finite precision p.
/// \throws std::logic_error If p is exact or x is unset.
inline mpreal exp2(mpreal const& x, Precision p) { return detail::mpreal_access::apply<mpfr_exp2>(x, p); }

/// \brief Exponential minus one without cancellation; retain input precision or exact zero.
/// \throws std::logic_error If an exact input requires approximation, or x is unset.
inline mpreal expm1(mpreal const& x) { return detail::mpreal_access::apply<mpfr_expm1>(x); }
/// \brief Evaluate `expm1(x.at(p))`, returning an approximation at finite precision p.
/// \throws std::logic_error If p is exact or x is unset.
inline mpreal expm1(mpreal const& x, Precision p) { return detail::mpreal_access::apply<mpfr_expm1>(x, p); }

/// \brief Base-two logarithm; retain input precision or a supported exact result.
/// \throws std::logic_error If an exact input requires approximation, or x is unset.
inline mpreal log2(mpreal const& x) { return detail::mpreal_access::apply<mpfr_log2>(x); }
/// \brief Evaluate `log2(x.at(p))`, returning an approximation at finite precision p.
/// \throws std::logic_error If p is exact or x is unset.
inline mpreal log2(mpreal const& x, Precision p) { return detail::mpreal_access::apply<mpfr_log2>(x, p); }

/// \brief Base-ten logarithm; retain input precision or a supported exact result.
/// \throws std::logic_error If an exact input requires approximation, or x is unset.
inline mpreal log10(mpreal const& x) { return detail::mpreal_access::apply<mpfr_log10>(x); }
/// \brief Evaluate `log10(x.at(p))`, returning an approximation at finite precision p.
/// \throws std::logic_error If p is exact or x is unset.
inline mpreal log10(mpreal const& x, Precision p) { return detail::mpreal_access::apply<mpfr_log10>(x, p); }

/// \brief Logarithm of one plus x without cancellation; retain input precision or exact zero.
/// \throws std::logic_error If an exact input requires approximation, or x is unset.
inline mpreal log1p(mpreal const& x) { return detail::mpreal_access::apply<mpfr_log1p>(x); }
/// \brief Evaluate `log1p(x.at(p))`, returning an approximation at finite precision p.
/// \throws std::logic_error If p is exact or x is unset.
inline mpreal log1p(mpreal const& x, Precision p) { return detail::mpreal_access::apply<mpfr_log1p>(x, p); }

/// \brief Inverse sine in radians; retain input precision or a supported exact result.
/// \throws std::logic_error If an exact input requires approximation, or x is unset.
inline mpreal asin(mpreal const& x) { return detail::mpreal_access::apply<mpfr_asin>(x); }
/// \brief Evaluate `asin(x.at(p))`, returning an approximation at finite precision p.
/// \throws std::logic_error If p is exact or x is unset.
inline mpreal asin(mpreal const& x, Precision p) { return detail::mpreal_access::apply<mpfr_asin>(x, p); }

/// \brief Inverse cosine in radians; retain input precision or a supported exact result.
/// \throws std::logic_error If an exact input requires approximation, or x is unset.
inline mpreal acos(mpreal const& x) { return detail::mpreal_access::apply<mpfr_acos>(x); }
/// \brief Evaluate `acos(x.at(p))`, returning an approximation at finite precision p.
/// \throws std::logic_error If p is exact or x is unset.
inline mpreal acos(mpreal const& x, Precision p) { return detail::mpreal_access::apply<mpfr_acos>(x, p); }

/// \brief Hyperbolic sine; retain input precision or a supported exact result.
/// \throws std::logic_error If an exact input requires approximation, or x is unset.
inline mpreal sinh(mpreal const& x) { return detail::mpreal_access::apply<mpfr_sinh>(x); }
/// \brief Evaluate `sinh(x.at(p))`, returning an approximation at finite precision p.
/// \throws std::logic_error If p is exact or x is unset.
inline mpreal sinh(mpreal const& x, Precision p) { return detail::mpreal_access::apply<mpfr_sinh>(x, p); }

/// \brief Hyperbolic cosine; retain input precision or a supported exact result.
/// \throws std::logic_error If an exact input requires approximation, or x is unset.
inline mpreal cosh(mpreal const& x) { return detail::mpreal_access::apply<mpfr_cosh>(x); }
/// \brief Evaluate `cosh(x.at(p))`, returning an approximation at finite precision p.
/// \throws std::logic_error If p is exact or x is unset.
inline mpreal cosh(mpreal const& x, Precision p) { return detail::mpreal_access::apply<mpfr_cosh>(x, p); }

/// \brief Hyperbolic tangent; retain input precision or a supported exact result.
/// \throws std::logic_error If an exact input requires approximation, or x is unset.
inline mpreal tanh(mpreal const& x) { return detail::mpreal_access::apply<mpfr_tanh>(x); }
/// \brief Evaluate `tanh(x.at(p))`, returning an approximation at finite precision p.
/// \throws std::logic_error If p is exact or x is unset.
inline mpreal tanh(mpreal const& x, Precision p) { return detail::mpreal_access::apply<mpfr_tanh>(x, p); }

/// \brief Inverse hyperbolic sine; retain input precision or a supported exact result.
/// \throws std::logic_error If an exact input requires approximation, or x is unset.
inline mpreal asinh(mpreal const& x) { return detail::mpreal_access::apply<mpfr_asinh>(x); }
/// \brief Evaluate `asinh(x.at(p))`, returning an approximation at finite precision p.
/// \throws std::logic_error If p is exact or x is unset.
inline mpreal asinh(mpreal const& x, Precision p) { return detail::mpreal_access::apply<mpfr_asinh>(x, p); }

/// \brief Inverse hyperbolic cosine; retain input precision or a supported exact result.
/// \throws std::logic_error If an exact input requires approximation, or x is unset.
inline mpreal acosh(mpreal const& x) { return detail::mpreal_access::apply<mpfr_acosh>(x); }
/// \brief Evaluate `acosh(x.at(p))`, returning an approximation at finite precision p.
/// \throws std::logic_error If p is exact or x is unset.
inline mpreal acosh(mpreal const& x, Precision p) { return detail::mpreal_access::apply<mpfr_acosh>(x, p); }

/// \brief Inverse hyperbolic tangent; retain input precision or a supported exact result.
/// \throws std::logic_error If an exact input requires approximation, or x is unset.
inline mpreal atanh(mpreal const& x) { return detail::mpreal_access::apply<mpfr_atanh>(x); }
/// \brief Evaluate `atanh(x.at(p))`, returning an approximation at finite precision p.
/// \throws std::logic_error If p is exact or x is unset.
inline mpreal atanh(mpreal const& x, Precision p) { return detail::mpreal_access::apply<mpfr_atanh>(x, p); }

/// \brief Spacing above one for a specified binary working precision.
inline mpreal epsilon(Precision precision)
{
  // MPFR can construct this exact power of two without ambient precision changes.
  (void)precision.bit_count();
  mpreal result(precision);
  mpfr_set_ui_2exp(result.approximate(), 1, 1 - precision.bit_count(), MPFR_RNDN);
  return result;
}

inline mpreal numeric_limits<mpreal>::epsilon(Precision p) { return uni20::epsilon(p); }

namespace detail
{
template <auto Provider> struct mp_constant
{
    mpreal at(Precision p) const { return mpreal_access::constant<Provider>(p); }
    friend mpreal operator+(mpreal const& x, mp_constant c) { return x + c.at(x.precision()); }
    friend mpreal operator-(mpreal const& x, mp_constant c) { return x - c.at(x.precision()); }
    friend mpreal operator*(mpreal const& x, mp_constant c) { return x * c.at(x.precision()); }
    friend mpreal operator/(mpreal const& x, mp_constant c) { return x / c.at(x.precision()); }
    friend mpreal operator+(mp_constant c, mpreal const& x) { return c.at(x.precision()) + x; }
    friend mpreal operator-(mp_constant c, mpreal const& x) { return c.at(x.precision()) - x; }
    friend mpreal operator*(mp_constant c, mpreal const& x) { return c.at(x.precision()) * x; }
    friend mpreal operator/(mp_constant c, mpreal const& x) { return c.at(x.precision()) / x; }
};
template <> struct math_constants<mpreal>
{
    static constexpr mp_constant<mpfr_const_pi> pi{};
    static constexpr mp_constant<mpfr_const_log2> log_two{};
    static constexpr mp_constant<mpfr_const_euler> euler_gamma{};
    static constexpr mp_constant<mpfr_const_catalan> catalan{};
};
} // namespace detail

/// \brief Stream the round-trip decimal representation, independent of stream precision.
inline std::ostream& operator<<(std::ostream& os, mpreal const& value) { return os << value.to_string(); }
inline std::ostream& operator<<(std::ostream& os, exact_constant const& value) { return os << value.to_string(); }

} // namespace uni20

#include "mpreal_functions.hpp"

#include "mpreal_special.hpp"

#include "mpreal_extensions.hpp"
