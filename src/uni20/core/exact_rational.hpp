#pragma once

#include "precision.hpp"
#include <charconv>
#include <concepts>
#include <gmp.h>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace uni20
{
class mpreal;
class exact_rational;
class decimal_literal;
namespace literals
{
constexpr decimal_literal operator""_mp(char const* text);
}

/// \brief A precisionless decimal spelling produced by the _mp literal operator.
/// \details Literal text has static lifetime. Arithmetic materializes an owning
///          exact rational, never an expression referring to other operands.
class decimal_literal {
  public:
    /// \brief Materialize the literal with one rounding to the requested precision.
    mpreal at(Precision precision) const;
    constexpr bool is_exact() const noexcept { return true; }
    constexpr std::string_view spelling() const noexcept { return spelling_; }

  private:
    friend constexpr decimal_literal literals::operator""_mp(char const* text);
    explicit constexpr decimal_literal(char const* text) : spelling_(text) {}
    std::string_view spelling_;
};

namespace detail
{
struct mpreal_access;
// Own temporary GMP integers, including on parser exceptions.
struct mp_integer
{
    mpz_t value;
    mp_integer() { mpz_init(value); }
    ~mp_integer() { mpz_clear(value); }
    mp_integer(mp_integer const&) = delete;
    mp_integer& operator=(mp_integer const&) = delete;
};

inline void parse_decimal_rational(mpq_ptr result, std::string_view text)
{
  if (text.empty()) throw std::invalid_argument("exact_rational: empty decimal");
  std::string unseparated;
  if (text.find('\'') != std::string_view::npos)
  {
    for (std::size_t i = 0; i < text.size(); ++i)
    {
      if (text[i] != '\'')
        unseparated += text[i];
      else if (i == 0 || i + 1 == text.size() || text[i - 1] < '0' || text[i - 1] > '9' || text[i + 1] < '0' ||
               text[i + 1] > '9')
        throw std::invalid_argument("exact_rational: misplaced digit separator");
    }
    text = unseparated;
  }
  std::string digits;
  std::size_t pos = 0;
  bool const negative = text[pos] == '-';
  if (negative || text[pos] == '+') ++pos;
  bool dot = false;
  std::size_t fraction_digits = 0;
  for (; pos < text.size(); ++pos)
  {
    char c = text[pos];
    if (c == '.')
    {
      if (dot) throw std::invalid_argument("exact_rational: repeated decimal point");
      dot = true;
    }
    else if (c >= '0' && c <= '9')
    {
      digits += c;
      if (dot) ++fraction_digits;
    }
    else
      break;
  }
  if (digits.empty()) throw std::invalid_argument("exact_rational: expected decimal digits");
  // Decimal literals are deliberate: hexadecimal/binary spellings are not accepted.
  bool exponent_negative = false;
  unsigned long exponent = 0;
  if (pos < text.size() && (text[pos] == 'e' || text[pos] == 'E'))
  {
    ++pos;
    if (pos < text.size() && (text[pos] == '+' || text[pos] == '-')) exponent_negative = text[pos++] == '-';
    auto const first = text.data() + pos;
    auto const last = text.data() + text.size();
    auto const parsed = std::from_chars(first, last, exponent);
    if (parsed.ec == std::errc::result_out_of_range)
      throw std::out_of_range("exact_rational: decimal exponent is too large");
    if (parsed.ec != std::errc{} || parsed.ptr != last)
      throw std::invalid_argument("exact_rational: invalid decimal exponent");
    pos = text.size();
  }
  if (pos != text.size()) throw std::invalid_argument("exact_rational: invalid decimal spelling");
  if (fraction_digits > std::numeric_limits<unsigned long>::max())
    throw std::out_of_range("exact_rational: too many fractional digits");
  auto const fraction = static_cast<unsigned long>(fraction_digits);
  if (exponent_negative && exponent > std::numeric_limits<unsigned long>::max() - fraction)
    throw std::out_of_range("exact_rational: decimal scale is too large");
  bool const denominator = exponent_negative || fraction > exponent;
  auto const scale =
      exponent_negative ? fraction + exponent : (denominator ? fraction - exponent : exponent - fraction);
  // Bound expansion of short inputs such as 1e999999999 before allocating 10^scale.
  constexpr unsigned long max_decimal_scale = 1'000'000;
  if (scale > max_decimal_scale) throw std::out_of_range("exact_rational: decimal scale exceeds 1000000");
  mpz_set_str(mpq_numref(result), digits.c_str(), 10);
  mpz_set_ui(mpq_denref(result), 1);
  if (mpz_sgn(mpq_numref(result)) == 0) return;
  mp_integer factor;
  mpz_ui_pow_ui(factor.value, 10, scale);
  if (denominator)
    mpz_set(mpq_denref(result), factor.value);
  else
    mpz_mul(mpq_numref(result), mpq_numref(result), factor.value);
  if (negative) mpz_neg(mpq_numref(result), mpq_numref(result));
  mpq_canonicalize(result);
}
// Fractions use signed base-ten integers, with no whitespace or separators.
inline void parse_rational(mpq_ptr result, std::string_view text)
{
  auto slash = text.find('/');
  if (slash == std::string_view::npos) return parse_decimal_rational(result, text);
  auto integer = [](mpz_ptr out, std::string_view part) {
    if (part.empty()) throw std::invalid_argument("exact_rational: missing fraction component");
    bool negative = part.front() == '-';
    if (negative || part.front() == '+') part.remove_prefix(1);
    if (part.empty() || part.find_first_not_of("0123456789") != std::string_view::npos)
      throw std::invalid_argument("exact_rational: expected integer fraction components");
    std::string owned(part);
    mpz_set_str(out, owned.c_str(), 10);
    if (negative) mpz_neg(out, out);
  };
  integer(mpq_numref(result), text.substr(0, slash));
  integer(mpq_denref(result), text.substr(slash + 1));
  if (mpz_sgn(mpq_denref(result)) == 0) throw std::domain_error("exact_rational: zero denominator");
  mpq_canonicalize(result);
}
} // namespace detail

/// \brief Owning arbitrary-size rational with exact, eager arithmetic.
/// \details Prefer mpreal for general numerical code. Use this type when the
///          value must remain rational, independent of any working precision.
///          Text accepts finite decimals and signed
///          integer fractions; division can produce rationals such as 1/3.
///          Default construction produces zero; moving leaves the source unset.
///          Copies and assignments preserve unset state, but numerical use throws.
class exact_rational {
  public:
    exact_rational()
    {
      mpq_init(value_);
      initialized_ = true;
    }
    /// \brief Whether this object owns a rational value rather than being unset.
    constexpr bool initialized() const noexcept { return initialized_; }
    /// \brief Whether this object carries an exact value; false when unset.
    constexpr bool is_exact() const noexcept { return this->initialized(); }
    explicit exact_rational(std::string_view text) : exact_rational() { detail::parse_rational(value_, text); }
    exact_rational(decimal_literal literal) : exact_rational(literal.spelling()) {}

    template <std::integral I>
      requires(!std::same_as<I, bool>)
    exact_rational(I value) : exact_rational()
    {
      char text[std::numeric_limits<I>::digits10 + 4];
      auto const result = std::to_chars(text, text + sizeof(text), value);
      *result.ptr = '\0';
      mpz_set_str(mpq_numref(value_), text, 10);
    }

    exact_rational(exact_rational const& other)
    {
      if (other.initialized_)
      {
        mpq_init(value_);
        initialized_ = true;
        mpq_set(value_, other.value_);
      }
    }
    /// \brief Transfer the payload without allocating, leaving the source unset.
    exact_rational(exact_rational&& other) noexcept { this->swap(other); }
    exact_rational& operator=(exact_rational other) noexcept
    {
      this->swap(other);
      return *this;
    }
    ~exact_rational()
    {
      if (initialized_) mpq_clear(value_);
    }
    /// \brief Exchange payload ownership, including unset states, without allocating.
    void swap(exact_rational& other) noexcept
    {
      // Swap descriptors, not GMP values: an unset descriptor is not an mpq.
      std::swap(value_[0], other.value_[0]);
      std::swap(initialized_, other.initialized_);
    }

    /// \brief Convert to a real value with one rounding at explicit precision.
    mpreal at(Precision precision) const;

    /// \brief Borrow the read-only GMP rational while this object retains its payload.
    /// \throws std::logic_error If this object is unset.
    mpq_srcptr native_handle() const
    {
      if (!this->initialized()) throw std::logic_error("exact_rational: numerical use of an unset value");
      return value_;
    }

    /// \brief Return the canonical numerator/denominator, or an integer when exact.
    std::string to_string() const
    {
      auto value = this->native_handle();
      std::string text(mpz_sizeinbase(mpq_numref(value), 10) + mpz_sizeinbase(mpq_denref(value), 10) + 4, '\0');
      mpq_get_str(text.data(), 10, value);
      text.resize(std::char_traits<char>::length(text.c_str()));
      return text;
    }

    friend exact_rational operator+(exact_rational const& a, exact_rational const& b);
    friend std::optional<exact_rational> rational_root(exact_rational const&, unsigned long);
    friend exact_rational operator-(exact_rational const& a, exact_rational const& b);
    friend exact_rational operator*(exact_rational const& a, exact_rational const& b);
    friend exact_rational operator/(exact_rational const& a, exact_rational const& b);
    friend exact_rational operator-(exact_rational const& a);

  private:
    friend struct detail::mpreal_access;
    mpq_t value_{};
    bool initialized_ = false;
};

inline exact_rational operator+(exact_rational const& a, exact_rational const& b)
{
  exact_rational result;
  mpq_add(result.value_, a.native_handle(), b.native_handle());
  return result;
}
inline exact_rational operator-(exact_rational const& a, exact_rational const& b)
{
  exact_rational result;
  mpq_sub(result.value_, a.native_handle(), b.native_handle());
  return result;
}
inline exact_rational operator*(exact_rational const& a, exact_rational const& b)
{
  exact_rational result;
  mpq_mul(result.value_, a.native_handle(), b.native_handle());
  return result;
}
inline exact_rational operator/(exact_rational const& a, exact_rational const& b)
{
  auto divisor = b.native_handle();
  if (mpq_sgn(divisor) == 0) throw std::domain_error("exact_rational: division by zero");
  exact_rational result;
  mpq_div(result.value_, a.native_handle(), divisor);
  return result;
}
inline exact_rational operator-(exact_rational const& a)
{
  exact_rational result;
  mpq_neg(result.value_, a.native_handle());
  return result;
}
inline exact_rational operator+(exact_rational const& a)
{
  (void)a.native_handle();
  return a;
}
inline bool operator==(exact_rational const& a, exact_rational const& b)
{
  return mpq_equal(a.native_handle(), b.native_handle()) != 0;
}
inline std::strong_ordering operator<=>(exact_rational const& a, exact_rational const& b)
{
  return mpq_cmp(a.native_handle(), b.native_handle()) <=> 0;
}

/// \brief Return a rational nth root when it exists, without choosing a precision.
/// \details Negative inputs have a real root only for odd degrees. A missing
///          result means that the root cannot be represented as a rational.
inline std::optional<exact_rational> rational_root(exact_rational const& x, unsigned long degree)
{
  if (degree == 0) throw std::domain_error("exact_rational: zeroth root");
  auto q = x.native_handle();
  bool negative = mpq_sgn(q) < 0;
  if (negative && degree % 2 == 0) return std::nullopt;
  detail::mp_integer magnitude;
  mpz_abs(magnitude.value, mpq_numref(q));
  exact_rational result;
  if (!mpz_root(mpq_numref(result.value_), magnitude.value, degree) ||
      !mpz_root(mpq_denref(result.value_), mpq_denref(q), degree))
    return std::nullopt;
  if (negative) mpz_neg(mpq_numref(result.value_), mpq_numref(result.value_));
  return result;
}

namespace detail
{
// Repeated squaring also accepts exponents wider than an unsigned long.
template <class T> T integer_power(T base, mpz_srcptr exponent)
{
  mp_integer remaining;
  mpz_abs(remaining.value, exponent);
  T result{1};
  if (mpz_sgn(exponent) < 0) base = result / base;
  while (mpz_sgn(remaining.value) != 0)
  {
    if (mpz_odd_p(remaining.value)) result = result * base;
    mpz_fdiv_q_2exp(remaining.value, remaining.value, 1);
    if (mpz_sgn(remaining.value) != 0) base = base * base;
  }
  return result;
}
} // namespace detail

/// \brief Evaluate a rational power exactly when its real result is rational.
/// \details Uses the real root for negative bases with odd-denominator powers;
///          zero to the zeroth power is one. No approximate result is inferred.
template <class X, class Y>
  requires((std::same_as<X, exact_rational> || std::same_as<X, decimal_literal>) &&
           (std::same_as<Y, exact_rational> || std::same_as<Y, decimal_literal>))
inline exact_rational pow(X const& base, Y const& power)
{
  exact_rational const x(base), y(power);
  (void)x.native_handle();
  auto exponent = y.native_handle();
  auto degree = mpq_denref(exponent);
  if (mpz_cmp_ui(degree, 1) == 0) return detail::integer_power(x, mpq_numref(exponent));
  if (x == 0 || x == 1 || (x == -1 && mpz_odd_p(degree))) return detail::integer_power(x, mpq_numref(exponent));
  // A nontrivial integer nth power needs at least n+1 bits. An unrepresentably
  // large degree cannot have a rational root in an in-memory GMP operand.
  if (mpz_fits_ulong_p(degree))
    if (auto root = rational_root(x, mpz_get_ui(degree))) return detail::integer_power(*root, mpq_numref(exponent));
  throw std::logic_error("exact_rational: power requires finite working precision");
}

/// \brief Return the nonnegative rational square root, or require finite precision.
template <class X>
  requires(std::same_as<X, exact_rational> || std::same_as<X, decimal_literal>)
inline exact_rational sqrt(X const& value)
{
  exact_rational const x(value);
  if (auto root = rational_root(x, 2)) return *root;
  throw std::logic_error("exact_rational: square root requires finite working precision");
}

/// \brief Exact numeric sources accepted without an approximation precision.
template <class T>
concept ExactRationalSource =
    (std::integral<std::remove_cvref_t<T>> && !std::same_as<std::remove_cvref_t<T>, bool>) ||
    std::same_as<std::remove_cvref_t<T>, exact_rational> || std::same_as<std::remove_cvref_t<T>, decimal_literal>;

namespace literals
{
/// \brief Preserve decimal source digits without conversion through a machine float.
/// \details Include core/mpreal.hpp to materialize or combine these literals.
constexpr decimal_literal operator""_mp(char const* text) { return decimal_literal(text); }
} // namespace literals

} // namespace uni20
