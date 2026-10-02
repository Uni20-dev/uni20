#pragma once

#include "precision.hpp"
#include <charconv>
#include <concepts>
#include <limits>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>

namespace uni20
{
class mpreal;
class exact_constant;
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
  if (text.empty()) throw std::invalid_argument("exact_constant: empty decimal");
  std::string unseparated;
  if (text.find('\'') != std::string_view::npos)
  {
    for (std::size_t i = 0; i < text.size(); ++i)
    {
      if (text[i] != '\'')
        unseparated += text[i];
      else if (i == 0 || i + 1 == text.size() || text[i - 1] < '0' || text[i - 1] > '9' || text[i + 1] < '0' ||
               text[i + 1] > '9')
        throw std::invalid_argument("exact_constant: misplaced digit separator");
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
      if (dot) throw std::invalid_argument("exact_constant: repeated decimal point");
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
  if (digits.empty()) throw std::invalid_argument("exact_constant: expected decimal digits");
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
      throw std::out_of_range("exact_constant: decimal exponent is too large");
    if (parsed.ec != std::errc{} || parsed.ptr != last)
      throw std::invalid_argument("exact_constant: invalid decimal exponent");
    pos = text.size();
  }
  if (pos != text.size()) throw std::invalid_argument("exact_constant: invalid decimal spelling");
  if (fraction_digits > std::numeric_limits<unsigned long>::max())
    throw std::out_of_range("exact_constant: too many fractional digits");
  auto const fraction = static_cast<unsigned long>(fraction_digits);
  if (exponent_negative && exponent > std::numeric_limits<unsigned long>::max() - fraction)
    throw std::out_of_range("exact_constant: decimal scale is too large");
  bool const denominator = exponent_negative || fraction > exponent;
  auto const scale =
      exponent_negative ? fraction + exponent : (denominator ? fraction - exponent : exponent - fraction);
  // Bound expansion of short inputs such as 1e999999999 before allocating 10^scale.
  constexpr unsigned long max_decimal_scale = 1'000'000;
  if (scale > max_decimal_scale) throw std::out_of_range("exact_constant: decimal scale exceeds 1000000");
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
    if (part.empty()) throw std::invalid_argument("exact_constant: missing fraction component");
    bool negative = part.front() == '-';
    if (negative || part.front() == '+') part.remove_prefix(1);
    if (part.empty() || part.find_first_not_of("0123456789") != std::string_view::npos)
      throw std::invalid_argument("exact_constant: expected integer fraction components");
    std::string owned(part);
    mpz_set_str(out, owned.c_str(), 10);
    if (negative) mpz_neg(out, out);
  };
  integer(mpq_numref(result), text.substr(0, slash));
  integer(mpq_denref(result), text.substr(slash + 1));
  if (mpz_sgn(mpq_denref(result)) == 0) throw std::domain_error("exact_constant: zero denominator");
  mpq_canonicalize(result);
}
} // namespace detail

/// \brief Owning exact rational for precisionless literal arithmetic.
/// \details All arithmetic is eager. Text accepts finite decimals and signed
///          integer fractions; division can produce rationals such as 1/3.
class exact_constant {
  public:
    exact_constant() { mpq_init(value_); }
    /// \brief Exact constants always carry a rational value.
    constexpr bool is_exact() const noexcept { return true; }
    explicit exact_constant(std::string_view text) : exact_constant() { detail::parse_rational(value_, text); }
    exact_constant(decimal_literal literal) : exact_constant(literal.spelling()) {}

    template <std::integral I>
      requires(!std::same_as<I, bool>)
    exact_constant(I value) : exact_constant()
    {
      char text[std::numeric_limits<I>::digits10 + 4];
      auto const result = std::to_chars(text, text + sizeof(text), value);
      *result.ptr = '\0';
      mpz_set_str(mpq_numref(value_), text, 10);
    }

    exact_constant(exact_constant const& other) : exact_constant() { mpq_set(value_, other.value_); }
    exact_constant(exact_constant&& other) noexcept : exact_constant() { mpq_swap(value_, other.value_); }
    exact_constant& operator=(exact_constant other) noexcept
    {
      mpq_swap(value_, other.value_);
      return *this;
    }
    ~exact_constant() { mpq_clear(value_); }

    /// \brief Convert to a real value with one rounding at explicit precision.
    mpreal at(Precision precision) const;

    /// \brief Borrow the read-only GMP rational; valid for this object's lifetime.
    mpq_srcptr native_handle() const noexcept { return value_; }

    /// \brief Return the canonical numerator/denominator, or an integer when exact.
    std::string to_string() const
    {
      std::string text(mpz_sizeinbase(mpq_numref(value_), 10) + mpz_sizeinbase(mpq_denref(value_), 10) + 4, '\0');
      mpq_get_str(text.data(), 10, value_);
      text.resize(std::char_traits<char>::length(text.c_str()));
      return text;
    }

    friend exact_constant operator+(exact_constant const& a, exact_constant const& b);
    friend std::optional<exact_constant> rational_root(exact_constant const&, unsigned long);
    friend exact_constant operator-(exact_constant const& a, exact_constant const& b);
    friend exact_constant operator*(exact_constant const& a, exact_constant const& b);
    friend exact_constant operator/(exact_constant const& a, exact_constant const& b);
    friend exact_constant operator-(exact_constant const& a);

  private:
    friend struct detail::mpreal_access;
    mpq_t value_;
};

inline exact_constant operator+(exact_constant const& a, exact_constant const& b)
{
  exact_constant result;
  mpq_add(result.value_, a.value_, b.value_);
  return result;
}
inline exact_constant operator-(exact_constant const& a, exact_constant const& b)
{
  exact_constant result;
  mpq_sub(result.value_, a.value_, b.value_);
  return result;
}
inline exact_constant operator*(exact_constant const& a, exact_constant const& b)
{
  exact_constant result;
  mpq_mul(result.value_, a.value_, b.value_);
  return result;
}
inline exact_constant operator/(exact_constant const& a, exact_constant const& b)
{
  if (mpq_sgn(b.value_) == 0) throw std::domain_error("exact_constant: division by zero");
  exact_constant result;
  mpq_div(result.value_, a.value_, b.value_);
  return result;
}
inline exact_constant operator-(exact_constant const& a)
{
  exact_constant result;
  mpq_neg(result.value_, a.value_);
  return result;
}
inline exact_constant operator+(exact_constant const& a) { return a; }
inline bool operator==(exact_constant const& a, exact_constant const& b)
{
  return mpq_equal(a.native_handle(), b.native_handle()) != 0;
}
inline std::strong_ordering operator<=>(exact_constant const& a, exact_constant const& b)
{
  return mpq_cmp(a.native_handle(), b.native_handle()) <=> 0;
}

/// \brief Return a rational nth root when it exists, without choosing a precision.
/// \details Negative inputs have a real root only for odd degrees. A missing
///          result means that the root cannot be represented as a rational.
inline std::optional<exact_constant> rational_root(exact_constant const& x, unsigned long degree)
{
  if (degree == 0) throw std::domain_error("exact_constant: zeroth root");
  auto q = x.native_handle();
  bool negative = mpq_sgn(q) < 0;
  if (negative && degree % 2 == 0) return std::nullopt;
  detail::mp_integer magnitude;
  mpz_abs(magnitude.value, mpq_numref(q));
  exact_constant result;
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
  requires((std::same_as<X, exact_constant> || std::same_as<X, decimal_literal>) &&
           (std::same_as<Y, exact_constant> || std::same_as<Y, decimal_literal>))
inline exact_constant pow(X const& base, Y const& power)
{
  exact_constant const x(base), y(power);
  auto exponent = y.native_handle();
  auto degree = mpq_denref(exponent);
  if (mpz_cmp_ui(degree, 1) == 0) return detail::integer_power(x, mpq_numref(exponent));
  if (x == 0 || x == 1 || (x == -1 && mpz_odd_p(degree))) return detail::integer_power(x, mpq_numref(exponent));
  // A nontrivial integer nth power needs at least n+1 bits. An unrepresentably
  // large degree cannot have a rational root in an in-memory GMP operand.
  if (mpz_fits_ulong_p(degree))
    if (auto root = rational_root(x, mpz_get_ui(degree))) return detail::integer_power(*root, mpq_numref(exponent));
  throw std::logic_error("exact_constant: power requires finite working precision");
}

/// \brief Return the nonnegative rational square root, or require finite precision.
template <class X>
  requires(std::same_as<X, exact_constant> || std::same_as<X, decimal_literal>)
inline exact_constant sqrt(X const& value)
{
  exact_constant const x(value);
  if (auto root = rational_root(x, 2)) return *root;
  throw std::logic_error("exact_constant: square root requires finite working precision");
}

/// \brief Exact numeric sources accepted without an approximation precision.
template <class T>
concept ExactRationalSource =
    (std::integral<std::remove_cvref_t<T>> && !std::same_as<std::remove_cvref_t<T>, bool>) ||
    std::same_as<std::remove_cvref_t<T>, exact_constant> || std::same_as<std::remove_cvref_t<T>, decimal_literal>;

namespace literals
{
/// \brief Preserve decimal source digits without conversion through a machine float.
/// \details Include core/mpreal.hpp to materialize or combine these literals.
constexpr decimal_literal operator""_mp(char const* text) { return decimal_literal(text); }
} // namespace literals

} // namespace uni20
