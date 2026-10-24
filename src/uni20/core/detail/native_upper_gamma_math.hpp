#pragma once

#include "native_upper_gamma_guard.hpp"
#include <uni20/core/detail/native_gamma_math.hpp>
#include <uni20/core/math_constants.hpp>

namespace uni20::detail::native_math::upper_gamma_detail
{
template <NativeReal R> struct compensated_log
{
    R sum = 0, low = 0;
    void add(R v)
    {
      R n = sum + v;
      low += std::abs(sum) >= std::abs(v) ? (sum - n) + v : (v - n) + sum;
      sum = n;
    }
    void product(R a, R b)
    {
      R v = a * b;
      this->add(v);
      if (std::isfinite(v)) this->add(std::fma(a, b, -v));
    }
    R value() const { return sum + low; }
    R exp() const
    {
      if (std::isinf(sum)) return sum > 0 ? sum : R(0);
      R v = sum + low;
      R e = std::abs(sum) >= std::abs(low) ? (sum - v) + low : (low - v) + sum;
      R h = std::exp(v / 2);
      if (h == 0 || std::isinf(h)) return h;
      return h * (h * std::exp(e));
    }
};
template <NativeReal R> R log1p_ratio(R x)
{
  return std::abs(x) < numeric_limits<R>::epsilon() / 2 ? R(1) : std::log1p(x) / x;
}
template <NativeReal R> R expm1_ratio(R x)
{
  return std::abs(x) < numeric_limits<R>::epsilon() / 2 ? R(1) : std::expm1(x) / x;
}
// Evaluate log(Gamma(1+a))/a without first rounding 1+a. The shifted
// Stirling divided difference has a bounded derivative remainder.
template <NativeReal R> R gamma1p_log_quotient(R a)
{
  if (a == 0) return -euler_gamma<R>;
  constexpr int p = numeric_limits<R>::digits(), N = p <= 24 ? 16 : p <= 64 ? 32 : 64;
  constexpr int terms = p <= 24 ? 4 : p <= 53 ? 8 : p <= 64 ? 10 : 16;
  constexpr R c[] = {R(1) / 12,
                     R(-1) / 360,
                     R(1) / 1260,
                     R(-1) / 1680,
                     R(1) / 1188,
                     R(-691) / 360360,
                     R(1) / 156,
                     R(-3617) / 122400,
                     R(43867) / 244188,
                     R(-174611) / 125400,
                     R(77683) / 5796,
                     R(-236364091LL) / 1506960,
                     R(657931) / 300,
                     R(-3392780147LL) / 93960,
                     R(1723168255201LL) / 2492028,
                     R(-7709321041217LL) / 505920};
  R r = a / N, l = std::log1p(r), q = log1p_ratio(r), power = R(1) / N;
  compensated_log<R> s;
  s.add(std::log(R(N)));
  s.add(std::fma((a - R(.5)) / N, q, q - 1));
  for (int k = 1; k < N; ++k)
    s.add(-log1p_ratio(a / R(k)) / R(k));
  for (int k = 0; k < terms; ++k)
  {
    R m = R(2 * k + 1);
    s.add(c[k] * power * (-m / R(N)) * q * expm1_ratio(-m * l));
    power /= R(N * N);
  }
  return s.value();
}
template <NativeReal R> R small_scaled(R a, R x)
{
  using L = numeric_limits<R>;
  R d = gamma1p_log_quotient(a) - std::log(x);
  compensated_log<R> s;
  s.add(d * expm1_ratio(a * d));
  R power = 1;
  for (int k = 1; k <= 2 * L::digits() + 32; ++k)
  {
    power *= -x / R(k);
    R term = power / (a + R(k));
    s.add(-term);
    if (std::abs(term) < L::epsilon() * std::abs(s.sum) / 16) return std::exp(x) * s.value();
  }
  return L::quiet_NaN();
}
// Neighboring positive Stieltjes truncations bracket the exact scaled
// value. Rounded endpoints are an empirical convergence test, not intervals.
template <NativeReal R> R positive_fraction(R a, R x)
{
  using L = numeric_limits<R>;
  for (int depth = 16; depth <= 64 * L::digits(); depth *= 2)
  {
    R lo = (R(depth) - a) / (1 + R(depth) / x), hi = R(depth) - a;
    for (int k = depth - 1; k >= 1; --k)
    {
      lo = (R(k) - a) / (1 + R(k) / (x + lo));
      hi = (R(k) - a) / (1 + R(k) / (x + hi));
    }
    lo = 1 / (x + lo);
    hi = 1 / (x + hi);
    if (std::abs(hi - lo) <= 4 * L::epsilon() * lo) return lo + (hi - lo) / 2;
  }
  return L::quiet_NaN();
}
template <NativeReal R> R tail_fraction(R a, R x)
{
  using L = numeric_limits<R>;
  R previous = 0;
  for (int depth = 8; depth <= 64 * L::digits(); depth *= 2)
  {
    R value = 0;
    for (int k = depth; k >= 1; --k)
      value = (a - R(k)) * (R(k) / (x - a + R(2 * k + 1) + value));
    value = 1 / (x - a + 1 + value);
    if (std::abs(value - previous) <= 2 * L::epsilon() * value) return value;
    previous = value;
  }
  return L::quiet_NaN();
}
template <NativeReal R> compensated_log<R> lgamma(R a)
{
  constexpr int cutoff = numeric_limits<R>::digits() <= 24 ? 8 : numeric_limits<R>::digits() <= 64 ? 16 : 32;
  compensated_log<R> s;
  if (a < cutoff)
  {
    s.add(std::log(std::tgamma(a)));
    return s;
  }
  R l = std::log(a);
  s.product(a, l);
  s.add(-l / 2);
  s.add(-a);
  s.add(std::log(2 * pi<R>) / 2);
  s.add(detail::native_math::stirling_correction(R(1) / a));
  return s;
}
template <NativeReal R> R scaled_result(R a, R x, R u)
{
  if (std::isnan(u)) return u;
  using L = numeric_limits<R>;
  R estimate = std::fma(a, std::log(x), -x) + std::log(u);
  if (L::epsilon() * std::abs(a) * std::abs(std::log(x)) + L::epsilon() * x > R(1) / 32 ||
      std::abs(estimate - std::log(L::max())) < 4 || std::abs(estimate - std::log(L::denorm_min())) < 4)
    return upper_gamma_guard::upper_gamma(a, x);
  if (a <= 0)
  {
    R half = std::pow(x, a / 2) * std::exp(-x / 2);
    return half * (half * u);
  }
  compensated_log<R> s;
  R lx = std::log(x);
  if (std::isfinite(a * lx))
  {
    s.product(a, lx);
    s.add(-x);
  }
  else
  {
    s.add(std::fma(a, lx, -x));
  }
  if (std::isinf(s.sum)) return s.sum > 0 ? s.sum : R(0);
  s.add(std::log(u));
  return s.exp();
}
template <NativeReal R> R evaluate(R a, R x)
{
  using L = numeric_limits<R>;
  if (std::isnan(a) || std::isnan(x)) return L::quiet_NaN();
  if (a == L::infinity()) return std::isfinite(x) ? a : L::quiet_NaN();
  if (a == -L::infinity()) return x > 0 ? (x < 1 ? L::infinity() : R(0)) : L::quiet_NaN();
  if (x < 0) return L::quiet_NaN();
  if (x == 0) return std::tgamma(a);
  if (x == L::infinity()) return R(0);
  if (a <= 1 && x > R(2 * L::max_exponent)) return R(0);
  // Integrating over [max(a,x),max(a,x)+1] gives an overflowing lower
  // bound here. Avoid an unnecessarily difficult transition iteration.
  if (a > R(2 * L::max_exponent) && x / 2 <= a) return L::infinity();
  if (a == 1) return std::exp(-x);
  if (a >= R(.5) && x <= a)
  {
    auto g = lgamma(a);
    compensated_log<R> lb;
    lb.product(a, std::log(a));
    lb.add(-std::log(a));
    lb.add(-a);
    lb.add(-R(1));
    if (a >= 1 && lb.value() > std::log(L::max()) + 1) return L::infinity();
    compensated_log<R> s;
    R term = 1;
    s.add(term);
    for (int k = 1; k <= 64 * L::digits(); ++k)
    {
      term *= x / (a + R(k));
      s.add(term);
      if (term < L::epsilon() * s.sum / 16)
      {
        compensated_log<R> l;
        l.product(a, std::log(x));
        l.add(-x);
        l.add(-std::log(a));
        l.add(-g.sum);
        l.add(-g.low);
        R lower = l.exp() * s.value();
        g.add(std::log1p(-lower));
        if (std::abs(g.value() - std::log(L::max())) < 4) return upper_gamma_guard::lower_complement(a, x);
        return g.exp();
      }
    }
    return L::quiet_NaN();
  }
  constexpr int cutoff = L::digits() <= 24 ? 8 : L::digits() <= 64 ? 16 : 32;
  if (x <= 1 && a > -R(cutoff) && a < R(.5))
  {
    int count = int(std::ceil(-a - R(.5)));
    if (count < 0) count = 0;
    R b = a + R(count);
    R u = small_scaled(b, x);
    for (int k = 1; k <= count; ++k)
      u = (std::fma(x, u, -R(1))) / (b - R(k));
    return scaled_result(a, x, u);
  }
  R u = a <= 0 && x >= 1 ? positive_fraction(a, x) : tail_fraction(a, x);
  return scaled_result(a, x, u);
}
} // namespace uni20::detail::native_math::upper_gamma_detail

namespace uni20::detail::native_math
{
/// \brief Upper unregularized incomplete Gamma at native precision.
/// \details For x>0, all finite real shapes are supported. See
///          docs/tensor/native_upper_gamma.md for endpoint conventions,
///          numerical targets and guarded range evaluation.
template <NativeReal R> R upper_gamma(R a, R x) { return upper_gamma_detail::evaluate(a, x); }
} // namespace uni20::detail::native_math
