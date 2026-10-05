#pragma once

#include "detail/math_integer.hpp"
#include "math_results.hpp"
#include "mpreal.hpp"
#include <vector>

namespace uni20
{
namespace detail
{
inline bool rational_integer(exact_constant const& x) { return mpz_cmp_ui(mpq_denref(x.native_handle()), 1) == 0; }
inline unsigned long exact_nonnegative_order(exact_constant const& x)
{
  auto q = x.native_handle();
  if (!rational_integer(x) || mpq_sgn(q) < 0)
    throw std::domain_error("special function requires a nonnegative integer order");
  if (!mpz_fits_slong_p(mpq_numref(q))) throw std::out_of_range("special function order does not fit int64_t");
  return mpz_get_ui(mpq_numref(q));
}
inline exact_constant exact_factorial(unsigned long n)
{
  return mpreal_access::rational_result([&](mpq_ptr out) { mpz_fac_ui(mpq_numref(out), n); });
}
inline mpreal exact_gamma(exact_constant const& x)
{
  if (x > 0 && rational_integer(x)) return mpreal(exact_factorial(exact_nonnegative_order(x - exact_constant{1})));
  throw std::logic_error("Gamma requires finite precision unless its input is a positive integer");
}
inline mpreal exact_zeta(exact_constant const& x)
{
  if (x == 0) return mpreal(exact_constant{-1} / exact_constant{2});
  if (x < 0 && rational_integer(x))
  {
    if (mpz_even_p(mpq_numref(x.native_handle()))) return mpreal{};
    // Zeta(-m) = -B_(m+1)/(m+1) for odd positive m. This exact
    // Bernoulli recurrence has no floating-point approximation or ambient precision.
    auto order = exact_nonnegative_order(-x) + 1;
    std::vector<exact_constant> row(order + 1);
    for (unsigned long m = 0; m <= order; ++m)
    {
      row[m] = exact_constant{1} / exact_constant{m + 1};
      for (unsigned long j = m; j > 0; --j)
        row[j - 1] = exact_constant{j} * (row[j - 1] - row[j]);
    }
    return mpreal(-row[0] / exact_constant{order});
  }
  throw std::logic_error("Zeta requires finite precision except at nonpositive integers");
}

template <auto Operation> mpreal special_unary(mpreal const& x)
{
  auto p = x.precision();
  if (p.is_exact())
  {
    auto const& q = x.exact_value();
    if constexpr (Operation == mpfr_gamma)
      return exact_gamma(q);
    else if constexpr (Operation == mpfr_zeta)
      return exact_zeta(q);
    else if constexpr (Operation == mpfr_erf || Operation == mpfr_li2)
    {
      if (q == 0) return mpreal{};
    }
    else if constexpr (Operation == mpfr_erfc)
    {
      if (q == 0) return mpreal{1};
    }
    throw std::logic_error("special function requires finite working precision");
  }
  return mpreal_access::finite_result(p, [&](mpfr_ptr out) { Operation(out, x.native_handle(), MPFR_RNDN); });
}
template <auto Operation> mpreal special_unary(mpreal const& x, Precision p)
{
  (void)p.bit_count();
  return special_unary<Operation>(x.at(p));
}

inline mpreal exact_beta(exact_constant const& a, exact_constant const& b)
{
  if (a > 0 && b > 0 && (rational_integer(a) || rational_integer(b)))
  {
    // If both are integers, use the shorter recurrence. In particular B(a,1)
    // needs no large-order conversion even when a is an arbitrary-size integer.
    bool const use_a = rational_integer(a) && (!rational_integer(b) || a <= b);
    auto n = exact_nonnegative_order(use_a ? a : b);
    auto x = use_a ? b : a;
    // B(x,n) = (n-1)! / (x (x+1) ... (x+n-1)). The recurrence
    // avoids computing three enormous factorials for small rational arguments.
    exact_constant result = exact_constant{1} / x;
    for (unsigned long k = 1; k < n; ++k)
      result = result * exact_constant{k} / (x + exact_constant{k});
    return mpreal(result);
  }
  throw std::logic_error("Beta requires finite precision except for positive rational/integer pairs");
}
template <auto Operation> mpreal special_binary(mpreal const& a, mpreal const& b)
{
  auto p = common_precision(a.precision(), b.precision());
  if (p.is_exact())
  {
    if constexpr (Operation == mpfr_beta)
      return exact_beta(a.exact_value(), b.exact_value());
    else if constexpr (Operation == mpfr_gamma_inc)
    {
      if (b == 0) return exact_gamma(a.exact_value());
    }
    else if constexpr (Operation == mpfr_agm)
    {
      if (a >= 0 && b >= 0)
      {
        if (a == b) return a;
        if (a == 0 || b == 0) return mpreal{};
      }
    }
    throw std::logic_error("special function requires finite working precision");
  }
  auto av = a.at(p), bv = b.at(p);
  return mpreal_access::finite_result(
      p, [&](mpfr_ptr out) { Operation(out, av.native_handle(), bv.native_handle(), MPFR_RNDN); });
}
template <auto Operation> mpreal special_binary(mpreal const& a, mpreal const& b, Precision p)
{
  (void)p.bit_count();
  return special_binary<Operation>(a.at(p), b.at(p));
}
} // namespace detail

/// \brief Real Gamma function; exact positive integers produce exact factorials.
inline mpreal tgamma(mpreal const& x) { return detail::special_unary<mpfr_gamma>(x); }
/// \brief Convert the input to finite precision p, then evaluate tgamma at p.
inline mpreal tgamma(mpreal const& x, Precision p) { return detail::special_unary<mpfr_gamma>(x, p); }

/// \brief Logarithmic derivative of Gamma; finite working precision is required.
inline mpreal digamma(mpreal const& x) { return detail::special_unary<mpfr_digamma>(x); }
/// \brief Convert the input to finite precision p, then evaluate digamma at p.
inline mpreal digamma(mpreal const& x, Precision p) { return detail::special_unary<mpfr_digamma>(x, p); }

/// \brief Error function, retaining exact zero or the input precision.
inline mpreal erf(mpreal const& x) { return detail::special_unary<mpfr_erf>(x); }
/// \brief Convert the input to finite precision p, then evaluate erf at p.
inline mpreal erf(mpreal const& x, Precision p) { return detail::special_unary<mpfr_erf>(x, p); }

/// \brief Complementary error function, evaluated without subtracting erf from one.
inline mpreal erfc(mpreal const& x) { return detail::special_unary<mpfr_erfc>(x); }
/// \brief Convert the input to finite precision p, then evaluate erfc at p.
inline mpreal erfc(mpreal const& x, Precision p) { return detail::special_unary<mpfr_erfc>(x, p); }

/// \brief Riemann zeta; nonpositive integer arguments retain exact rational results.
inline mpreal zeta(mpreal const& x) { return detail::special_unary<mpfr_zeta>(x); }
/// \brief Convert the input to finite precision p, then evaluate zeta at p.
inline mpreal zeta(mpreal const& x, Precision p) { return detail::special_unary<mpfr_zeta>(x, p); }

/// \brief Real exponential integral Ei, with its real principal-value meaning.
inline mpreal expint(mpreal const& x) { return detail::special_unary<mpfr_eint>(x); }
/// \brief Convert the input to finite precision p, then evaluate expint at p.
inline mpreal expint(mpreal const& x, Precision p) { return detail::special_unary<mpfr_eint>(x, p); }

/// \brief Real part of the dilogarithm; this is not a complex branch-valued API.
inline mpreal dilog_real(mpreal const& x) { return detail::special_unary<mpfr_li2>(x); }
/// \brief Convert the input to finite precision p, then evaluate dilog_real at p.
inline mpreal dilog_real(mpreal const& x, Precision p) { return detail::special_unary<mpfr_li2>(x, p); }

/// \brief Real Airy Ai; MPFR recommends arguments with magnitude below about 500.
inline mpreal airy_ai(mpreal const& x) { return detail::special_unary<mpfr_ai>(x); }
/// \brief Convert the input to finite precision p, then evaluate airy_ai at p.
inline mpreal airy_ai(mpreal const& x, Precision p) { return detail::special_unary<mpfr_ai>(x, p); }

/// \brief Real Beta function; positive rational/integer pairs retain exact values.
inline mpreal beta(mpreal const& a, mpreal const& x) { return detail::special_binary<mpfr_beta>(a, x); }
/// \brief Convert both inputs to finite precision p, then evaluate beta at p.
inline mpreal beta(mpreal const& a, mpreal const& x, Precision p) { return detail::special_binary<mpfr_beta>(a, x, p); }

/// \brief Upper incomplete Gamma, with shape a and lower integration endpoint x.
inline mpreal upper_gamma(mpreal const& a, mpreal const& x) { return detail::special_binary<mpfr_gamma_inc>(a, x); }
/// \brief Convert both inputs to finite precision p, then evaluate upper_gamma at p.
inline mpreal upper_gamma(mpreal const& a, mpreal const& x, Precision p)
{
  return detail::special_binary<mpfr_gamma_inc>(a, x, p);
}

/// \brief Nonnegative real arithmetic-geometric mean.
inline mpreal agm(mpreal const& a, mpreal const& x) { return detail::special_binary<mpfr_agm>(a, x); }
/// \brief Convert both inputs to finite precision p, then evaluate agm at p.
inline mpreal agm(mpreal const& a, mpreal const& x, Precision p) { return detail::special_binary<mpfr_agm>(a, x, p); }

/// \brief Log(abs(Gamma(x))) and the sign of Gamma, as owning values.
/// \details Undefined Gamma signs are reported as zero (NaN, negative infinity
///          or negative integer poles). At signed zero the sign follows x.
inline lgamma_result<mpreal> lgamma_sign(mpreal const& x)
{
  auto p = x.precision();
  if (p.is_exact())
  {
    if (x == 1 || x == 2) return {mpreal{}, 1};
    throw std::logic_error("log-Gamma requires finite precision except at one and two");
  }
  int sign = 0;
  auto value = detail::mpreal_access::finite_result(
      p, [&](mpfr_ptr out) { mpfr_lgamma(out, &sign, x.native_handle(), MPFR_RNDN); });
  if (isnan(x) || (x < 0 && (isinf(x) || mpfr_integer_p(x.native_handle())))) sign = 0;
  return {std::move(value), sign};
}
/// \brief Convert x to p, then evaluate log-absolute-Gamma and its sign.
inline lgamma_result<mpreal> lgamma_sign(mpreal const& x, Precision p)
{
  (void)p.bit_count();
  return lgamma_sign(x.at(p));
}
/// \brief Natural logarithm of abs(Gamma(x)), including negative Gamma values.
inline mpreal lgamma(mpreal const& x) { return lgamma_sign(x).value; }
/// \brief Convert x to p, then evaluate log(abs(Gamma(x))) at p.
inline mpreal lgamma(mpreal const& x, Precision p) { return lgamma_sign(x, p).value; }

namespace detail
{
template <auto Operation> mpreal integer_bessel(mpreal const& x, std::int64_t order)
{
  auto p = x.precision();
  if (p.is_exact())
  {
    if constexpr (Operation == mpfr_jn)
      if (x == 0) return mpreal{order == 0 ? 1 : 0};
    throw std::logic_error("Bessel function requires finite working precision");
  }
  static_assert(sizeof(long) >= sizeof(std::int64_t));
  return mpreal_access::finite_result(
      p, [&](mpfr_ptr out) { Operation(out, static_cast<long>(order), x.native_handle(), MPFR_RNDN); });
}
} // namespace detail

/// \brief Bessel function of the first kind, with a checked signed integer order.
template <detail::MathInteger I> inline mpreal bessel_j(I n, mpreal const& x)
{
  return detail::integer_bessel<mpfr_jn>(x, detail::math_integer(n));
}
/// \brief Convert x to p, then evaluate bessel_j(n,x) at p.
template <detail::MathInteger I> inline mpreal bessel_j(I n, mpreal const& x, Precision p)
{
  (void)p.bit_count();
  return bessel_j(n, x.at(p));
}

/// \brief Bessel function of the second kind, with a checked signed integer order.
template <detail::MathInteger I> inline mpreal bessel_y(I n, mpreal const& x)
{
  return detail::integer_bessel<mpfr_yn>(x, detail::math_integer(n));
}
/// \brief Convert x to p, then evaluate bessel_y(n,x) at p.
template <detail::MathInteger I> inline mpreal bessel_y(I n, mpreal const& x, Precision p)
{
  (void)p.bit_count();
  return bessel_y(n, x.at(p));
}

/// \brief Factorial of a nonnegative integer, at an explicitly chosen arithmetic state.
/// \details Precision::exact() constructs the full exact integer; finite precision
///          uses MPFR's factorial routine. Exact values may require large storage.
template <detail::MathInteger I> inline mpreal factorial(I n, Precision p)
{
  auto order = detail::math_integer(n);
  if (order < 0) throw std::domain_error("factorial requires a nonnegative order");
  if (p.is_exact()) return mpreal(detail::exact_factorial(static_cast<unsigned long>(order)));
  return detail::mpreal_access::finite_result(
      p, [&](mpfr_ptr out) { mpfr_fac_ui(out, static_cast<unsigned long>(order), MPFR_RNDN); });
}
} // namespace uni20
