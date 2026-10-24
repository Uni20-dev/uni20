#pragma once

#include "native_gamma_math.hpp"
#include "native_pi_math.hpp"

namespace uni20::detail::native_math
{
template <NativeReal R> R zeta_coefficient(int k)
{
  // B_(2k)/(2k)!, generated from exact rationals. Binary chunks preserve
  // native precision without first rounding a coefficient to long double.
  constexpr R coefficients[] = {
      (R(0xaaaaaaaaaaaaaaaaULL) * R(0x1p-63L) + R(0xaaaaaaaaaaaaaaaaULL) * R(0x1p-127L)) * R(0x1p-4L),
      -(R(0xb60b60b60b60b60bULL) * R(0x1p-63L) + R(0x60b60b60b60b60b6ULL) * R(0x1p-127L)) * R(0x1p-10L),
      (R(0x8ab355e008ab355eULL) * R(0x1p-63L) + R(0x008ab355e008ab35ULL) * R(0x1p-127L)) * R(0x1p-15L),
      -(R(0xddebbc99a7785563ULL) * R(0x1p-63L) + R(0x34111eefccdaab88ULL) * R(0x1p-127L)) * R(0x1p-21L),
      (R(0xb354795fb875ed14ULL) * R(0x1p-63L) + R(0xb089f4cc1f12f738ULL) * R(0x1p-127L)) * R(0x1p-26L),
      -(R(0x91402eb22133f7baULL) * R(0x1p-63L) + R(0x56337ff27cb7da47ULL) * R(0x1p-127L)) * R(0x1p-31L),
      (R(0xeb6d9627048b0fdcULL) * R(0x1p-63L) + R(0x257768c274f0337cULL) * R(0x1p-127L)) * R(0x1p-37L),
      -(R(0xbed270fbccaae25fULL) * R(0x1p-63L) + R(0xf129a06f42d0dc42ULL) * R(0x1p-127L)) * R(0x1p-42L),
      (R(0x9aac38eb2974eceeULL) * R(0x1p-63L) + R(0x56665d7d7bc1dbfaULL) * R(0x1p-127L)) * R(0x1p-47L),
      -(R(0xfabecb4655678663ULL) * R(0x1p-63L) + R(0xce0f50adaa0b8212ULL) * R(0x1p-127L)) * R(0x1p-53L),
      (R(0xcb3f0f84e1bb77d8ULL) * R(0x1p-63L) + R(0x2b4adb048eaa5e7bULL) * R(0x1p-127L)) * R(0x1p-58L),
      -(R(0xa4bec819d15ae370ULL) * R(0x1p-63L) + R(0x87e6555a6c876af6ULL) * R(0x1p-127L)) * R(0x1p-63L),
      (R(0x858996be35683203ULL) * R(0x1p-63L) + R(0xa8a4d91cebb6a6c7ULL) * R(0x1p-127L)) * R(0x1p-68L),
      -(R(0xd87b96acf8e0b3e6ULL) * R(0x1p-63L) + R(0x16e913f6cf2056caULL) * R(0x1p-127L)) * R(0x1p-74L),
      (R(0xaf796d2665136ad7ULL) * R(0x1p-63L) + R(0x3275bfba8ce6d0edULL) * R(0x1p-127L)) * R(0x1p-79L),
      -(R(0x8e3befcb6f1c57e2ULL) * R(0x1p-63L) + R(0x53a622f2cc53f6c7ULL) * R(0x1p-127L)) * R(0x1p-84L),
  };
  return coefficients[k];
}

// Euler-Maclaurin tail for s>=1/2, using the integral-remainder bounds
// generated in scripts/generate-zeta-coefficients.py.
template <NativeReal R> R positive_zeta(R s)
{
  using L = numeric_limits<R>;
  constexpr int p = L::digits(), N = p <= 24 ? 8 : p <= 64 ? 16 : 32;
  constexpr int M = p <= 24 ? 4 : p <= 53 ? 8 : p <= 64 ? 10 : 16;
  if (s == 1) return L::infinity();
  if (s >= R(p + 1)) return R(1);
  R sum = 0, low = 0;
  auto add = [&](R v) {
    R next = sum + v;
    low += std::abs(sum) >= std::abs(v) ? (sum - next) + v : (v - next) + sum;
    sum = next;
  };
  R logn = std::log(R(N));
  for (int n = 1; n < N; ++n)
    add(std::pow(R(n), -s));
  add(std::exp((R(1) - s) * logn) / (s - R(1)));
  add(std::exp(-s * logn) / 2);
  R power = std::exp((-s - R(1)) * logn) * s;
  for (int k = 1; k <= M; ++k)
  {
    add(native_math::zeta_coefficient<R>(k - 1) * power);
    power *= (s + R(2 * k - 1)) / R(N);
    power *= (s + R(2 * k)) / R(N);
  }
  return sum + low;
}
// Compute delta*zeta(1+delta) without constructing its pole. The exact
// delta enters the singular tail even if 1+delta rounds to one.
template <NativeReal R> R zeta_residue(R delta)
{
  using L = numeric_limits<R>;
  constexpr int p = L::digits(), N = p <= 24 ? 8 : p <= 64 ? 16 : 32;
  constexpr int M = p <= 24 ? 4 : p <= 53 ? 8 : p <= 64 ? 10 : 16;
  R sum = 0, low = 0;
  auto add = [&](R v) {
    R next = sum + v;
    low += std::abs(sum) >= std::abs(v) ? (sum - next) + v : (v - next) + sum;
    sum = next;
  };
  for (int n = 1; n < N; ++n)
    add(std::exp(-delta * std::log(R(n))) / R(n));
  R tail = std::exp(-delta * std::log(R(N)));
  add(tail / (2 * R(N)));
  R power = tail * (R(1) + delta) / (R(N) * R(N));
  for (int k = 1; k <= M; ++k)
  {
    add(native_math::zeta_coefficient<R>(k - 1) * power);
    power *= (delta + R(2 * k)) / R(N);
    power *= (delta + R(2 * k + 1)) / R(N);
  }
  return std::fma(delta, sum, tail) + delta * low;
}
template <NativeReal R> R zeta(R s)
{
  using L = numeric_limits<R>;
  if (std::isnan(s) || s == -L::infinity()) return L::quiet_NaN();
  if (s == L::infinity()) return R(1);
  if (s == 0) return -R(.5);
  if (std::abs(s) < L::epsilon() / 2) return -R(.5) - s * std::log(R(2) * pi<R>) / 2;
  if (s >= R(.5)) return native_math::positive_zeta(s);
  if (std::abs(s) < R(.5))
  {
    R t = -s;
    return native_math::sinpi(s / 2) / t * (std::tgamma(R(1) + t) / pi<R>) * std::exp(-t * std::log(R(2) * pi<R>)) *
           native_math::zeta_residue(t);
  }
  // Exact phase reduction precedes any amplitude that might overflow.
  R phase = native_math::sinpi(s / 2);
  if (phase == 0) return R(0);
  if (s < -R(2 * L::max_exponent)) return std::copysign(L::infinity(), phase);
  R t = -s, c = std::log(R(2) * pi<R>);
  constexpr int cutoff = L::digits() <= 24 ? 8 : L::digits() <= 64 ? 16 : 32;
  if (t < cutoff)
    return (std::tgamma(t) * std::pow(R(2) * pi<R>, -t)) * (t / pi<R>) * native_math::positive_zeta(R(1) + t) * phase;
  else
  {
    R sum = 0, low = 0;
    auto add = [&](R v) {
      R next = sum + v;
      low += std::abs(sum) >= std::abs(v) ? (sum - next) + v : (v - next) + sum;
      sum = next;
    };
    R lt = std::log(t), v = t * lt;
    add(v);
    add(std::fma(t, lt, -v));
    v = -t * c;
    add(v);
    add(std::fma(-t, c, -v));
    add(-t);
    add(lt / 2);
    add(-c / 2);
    add(log_two<R>);
    add(native_math::stirling_correction(R(1) / t));
    add(std::log(native_math::positive_zeta(R(1) + t)));
    // Keep both the small sine factor and the logarithm residual outside
    // the rounded exponent. Logging the sine loses accuracy near zeros.
    R half = std::exp(sum / 2);
    return half * (half * (phase * std::exp(low)));
  }
}
} // namespace uni20::detail::native_math
