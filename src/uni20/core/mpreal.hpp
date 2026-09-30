#pragma once

#include "exact_constant.hpp"
#include "scalar_traits.hpp"
#include <memory>
#include <ostream>

namespace uni20
{

namespace detail
{
struct mpreal_access;
inline void require_mpfr_tls()
{
  static bool const supported = mpfr_buildopt_tls_p() != 0;
  if (!supported) throw std::runtime_error("uni20::mpreal requires a thread-safe MPFR build");
}
} // namespace detail

/// \brief Owning MPFR real scalar with explicit precision and eager arithmetic.
/// \details Arithmetic rounds to nearest, ties to even. Binary real arithmetic
///          requires equal precision. Copy and move assignment adopt the source
///          precision. There is no default construction or implicit float conversion.
class mpreal {
  public:
    /// \brief Construct positive zero at the specified precision.
    explicit mpreal(Precision precision)
    {
      detail::require_mpfr_tls();
      mpfr_init2(value_, precision.bit_count());
      mpfr_set_zero(value_, 1);
    }

    /// \brief Parse a complete base-ten value, rounding once at explicit precision.
    /// \details Also accepts MPFR's inf/nan spellings. No leading/trailing whitespace
    ///          or embedded NUL is accepted. Invalid input throws invalid_argument.
    mpreal(std::string_view text, Precision precision) : mpreal(precision)
    {
      if (text.empty() || text.find('\0') != std::string_view::npos ||
          text.find_first_of(" \t\r\n\f\v") != std::string_view::npos)
        throw std::invalid_argument("mpreal: expected a complete decimal value");
      std::string owned(text);
      char* end = nullptr;
      mpfr_strtofr(value_, owned.c_str(), &end, 10, MPFR_RNDN);
      if (end != owned.c_str() + owned.size() || end == owned.c_str())
        throw std::invalid_argument("mpreal: invalid decimal value");
    }

    /// \brief Round an exact constant once at explicit precision.
    mpreal(exact_constant const& value, Precision precision) : mpreal(precision)
    {
      mpfr_set_q(value_, value.native_handle(), MPFR_RNDN);
    }
    mpreal(decimal_literal value, Precision precision) : mpreal(exact_constant(value), precision) {}

    template <std::integral I>
      requires(!std::same_as<I, bool>)
    mpreal(I value, Precision precision) : mpreal(exact_constant(value), precision)
    {}

    /// \brief Import an already-rounded native floating value at explicit precision.
    /// \details Higher precision cannot recover digits lost before this conversion.
    template <std::floating_point F> mpreal(F value, Precision precision) : mpreal(precision)
    {
      mpfr_set_ld(value_, static_cast<long double>(value), MPFR_RNDN);
    }

    /// \brief Explicitly round or promote an existing value to a new precision.
    mpreal(mpreal const& other, Precision precision) : mpreal(precision) { mpfr_set(value_, other.value_, MPFR_RNDN); }
    mpreal(mpreal const& other) : mpreal(other, other.precision()) {}
    mpreal(mpreal&& other) : mpreal(other.precision()) { mpfr_swap(value_, other.value_); }
    mpreal& operator=(mpreal other) noexcept
    {
      mpfr_swap(value_, other.value_);
      return *this;
    }
    ~mpreal() { mpfr_clear(value_); }

    /// \brief The stored value's working precision.
    Precision precision() const noexcept { return Precision(mpfr_get_prec(value_)); }
    /// \brief Borrow the read-only MPFR value; valid for this object's lifetime.
    mpfr_srcptr native_handle() const noexcept { return value_; }
    /// \brief Return a copy rounded to an explicitly chosen precision.
    mpreal at(Precision precision) const { return mpreal(*this, precision); }

    /// \brief Convert explicitly to a native float, with nearest-even rounding.
    explicit operator double() const { return mpfr_get_d(value_, MPFR_RNDN); }
    explicit operator long double() const { return mpfr_get_ld(value_, MPFR_RNDN); }

    /// \brief Format locale-independent decimal scientific notation.
    /// \details Zero digits selects enough significant decimal digits to round-trip
    ///          at this value's precision. A positive count requests that many digits.
    std::string to_string(std::size_t digits = 0) const
    {
      if (mpfr_nan_p(value_)) return "nan";
      if (mpfr_inf_p(value_)) return mpfr_signbit(value_) ? "-inf" : "inf";
      if (mpfr_zero_p(value_)) return mpfr_signbit(value_) ? "-0" : "0";
      mpfr_exp_t exponent;
      std::unique_ptr<char, decltype(&mpfr_free_str)> text(
          mpfr_get_str(nullptr, &exponent, 10, digits, value_, MPFR_RNDN), &mpfr_free_str);
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
    mpfr_t value_;
};

namespace detail
{
struct mpreal_access
{
    // All writes stay behind the scalar's precision invariant.
    template <auto Operation> static mpreal apply(mpreal const& x)
    {
      mpreal result(x.precision());
      Operation(result.value_, x.value_, MPFR_RNDN);
      return result;
    }
    template <auto Operation> static mpreal apply(mpreal const& a, mpreal const& b)
    {
      if (a.precision() != b.precision())
        throw std::invalid_argument("mpreal: mixed precisions require explicit conversion");
      mpreal result(a.precision());
      Operation(result.value_, a.value_, b.value_, MPFR_RNDN);
      return result;
    }
    template <auto Operation> static mpreal& assign(mpreal& a, mpreal const& b)
    {
      if (a.precision() != b.precision())
        throw std::invalid_argument("mpreal: mixed precisions require explicit conversion");
      // MPFR permits the destination to alias either or both operands.
      Operation(a.value_, a.value_, b.value_, MPFR_RNDN);
      return a;
    }
    /// \brief Evaluate a provider constant directly at explicit precision.
    template <auto Operation> static mpreal constant(Precision precision)
    {
      mpreal result(precision);
      Operation(result.value_, MPFR_RNDN);
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
inline mpreal operator+(mpreal const& a) { return a; }

// An exact operand is rounded at the real operand's precision before arithmetic.
inline mpreal operator+(mpreal const& a, exact_constant const& b) { return a + b.at(a.precision()); }
inline mpreal operator-(mpreal const& a, exact_constant const& b) { return a - b.at(a.precision()); }
inline mpreal operator*(mpreal const& a, exact_constant const& b) { return a * b.at(a.precision()); }
inline mpreal operator/(mpreal const& a, exact_constant const& b) { return a / b.at(a.precision()); }
inline mpreal operator+(exact_constant const& a, mpreal const& b) { return a.at(b.precision()) + b; }
inline mpreal operator-(exact_constant const& a, mpreal const& b) { return a.at(b.precision()) - b; }
inline mpreal operator*(exact_constant const& a, mpreal const& b) { return a.at(b.precision()) * b; }
inline mpreal operator/(exact_constant const& a, mpreal const& b) { return a.at(b.precision()) / b; }

inline mpreal& mpreal::operator+=(mpreal const& rhs) { return detail::mpreal_access::assign<mpfr_add>(*this, rhs); }
inline mpreal& mpreal::operator-=(mpreal const& rhs) { return detail::mpreal_access::assign<mpfr_sub>(*this, rhs); }
inline mpreal& mpreal::operator*=(mpreal const& rhs) { return detail::mpreal_access::assign<mpfr_mul>(*this, rhs); }
inline mpreal& mpreal::operator/=(mpreal const& rhs) { return detail::mpreal_access::assign<mpfr_div>(*this, rhs); }
inline mpreal& mpreal::operator+=(exact_constant const& rhs) { return *this += rhs.at(this->precision()); }
inline mpreal& mpreal::operator-=(exact_constant const& rhs) { return *this -= rhs.at(this->precision()); }
inline mpreal& mpreal::operator*=(exact_constant const& rhs) { return *this *= rhs.at(this->precision()); }
inline mpreal& mpreal::operator/=(exact_constant const& rhs) { return *this /= rhs.at(this->precision()); }

/// \brief Compare stored values without rounding; precision need not match.
inline bool operator==(mpreal const& a, mpreal const& b) { return mpfr_equal_p(a.native_handle(), b.native_handle()); }
inline std::partial_ordering operator<=>(mpreal const& a, mpreal const& b)
{
  if (mpfr_unordered_p(a.native_handle(), b.native_handle())) return std::partial_ordering::unordered;
  int const cmp = mpfr_cmp(a.native_handle(), b.native_handle());
  return cmp < 0   ? std::partial_ordering::less
         : cmp > 0 ? std::partial_ordering::greater
                   : std::partial_ordering::equivalent;
}
/// \brief Compare a stored real to an exact rational without rounding the rational.
inline bool operator==(mpreal const& a, exact_constant const& b)
{
  return mpfr_number_p(a.native_handle()) && mpfr_cmp_q(a.native_handle(), b.native_handle()) == 0;
}
inline std::partial_ordering operator<=>(mpreal const& a, exact_constant const& b)
{
  if (mpfr_nan_p(a.native_handle())) return std::partial_ordering::unordered;
  int const cmp = mpfr_cmp_q(a.native_handle(), b.native_handle());
  return cmp < 0   ? std::partial_ordering::less
         : cmp > 0 ? std::partial_ordering::greater
                   : std::partial_ordering::equivalent;
}

namespace detail
{
// The caller handles NaN. Native MPFR comparisons do not round the integer.
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

/// \brief Compare with an integer exactly, without constructing a real approximation.
/// \details Integers fitting MPFR's long/unsigned long interfaces avoid rational
///          temporaries. Zero equality uses a direct classification check.
template <std::integral I>
  requires(!std::same_as<I, bool>)
inline bool operator==(mpreal const& a, I b)
{
  if (b == 0) return mpfr_zero_p(a.native_handle());
  return mpfr_number_p(a.native_handle()) && detail::compare_integer(a, b) == 0;
}
/// \brief Order against an exact integer; NaN is unordered.
template <std::integral I>
  requires(!std::same_as<I, bool>)
inline std::partial_ordering operator<=>(mpreal const& a, I b)
{
  if (mpfr_nan_p(a.native_handle())) return std::partial_ordering::unordered;
  int const cmp = detail::compare_integer(a, b);
  return cmp < 0   ? std::partial_ordering::less
         : cmp > 0 ? std::partial_ordering::greater
                   : std::partial_ordering::equivalent;
}

inline bool isfinite(mpreal const& x) noexcept { return mpfr_number_p(x.native_handle()); }
inline bool isnan(mpreal const& x) noexcept { return mpfr_nan_p(x.native_handle()); }
inline bool isinf(mpreal const& x) noexcept { return mpfr_inf_p(x.native_handle()); }
inline bool signbit(mpreal const& x) noexcept { return mpfr_signbit(x.native_handle()); }
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

/// \brief Spacing above one for a specified binary working precision.
inline mpreal epsilon(Precision precision)
{
  // MPFR can construct this exact power of two without ambient precision changes.
  mpreal result(precision);
  mpfr_set_ui_2exp(result.value_, 1, 1 - precision.bit_count(), MPFR_RNDN);
  return result;
}

/// \brief Descriptor for pi, evaluated only when working precision is known.
struct mp_pi_constant
{
    mpreal at(Precision precision) const { return detail::mpreal_access::constant<mpfr_const_pi>(precision); }
};

template <typename Real> struct pi_constant;
template <> struct pi_constant<mpreal> : mp_pi_constant
{};
/// \brief Precision-aware pi descriptor; this slice supplies the mpreal specialization.
template <typename Real> inline constexpr pi_constant<Real> pi{};
inline mpreal operator+(mpreal const& x, mp_pi_constant c) { return x + c.at(x.precision()); }
inline mpreal operator-(mpreal const& x, mp_pi_constant c) { return x - c.at(x.precision()); }
inline mpreal operator*(mpreal const& x, mp_pi_constant c) { return x * c.at(x.precision()); }
inline mpreal operator/(mpreal const& x, mp_pi_constant c) { return x / c.at(x.precision()); }
inline mpreal operator+(mp_pi_constant c, mpreal const& x) { return c.at(x.precision()) + x; }
inline mpreal operator-(mp_pi_constant c, mpreal const& x) { return c.at(x.precision()) - x; }
inline mpreal operator*(mp_pi_constant c, mpreal const& x) { return c.at(x.precision()) * x; }
inline mpreal operator/(mp_pi_constant c, mpreal const& x) { return c.at(x.precision()) / x; }

/// \brief Stream the round-trip decimal representation, independent of stream precision.
inline std::ostream& operator<<(std::ostream& os, mpreal const& value) { return os << value.to_string(); }
inline std::ostream& operator<<(std::ostream& os, exact_constant const& value) { return os << value.to_string(); }

} // namespace uni20
