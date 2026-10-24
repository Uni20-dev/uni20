#include "precision_registry.hpp"
#include <cstdint>
#include <vector>

namespace uni20::test
{
UNI20_PRECISION_TEST(NumericalScalar, UpperGammaIdentities)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  for (int n : {1, 2, 4})
  {
    auto x = C::real(n) / 2;
    auto value = math::upper_gamma(C::real(1), x);
    C::expect_precision(value);
    expect_error_at_most(value, math::exp(-x), C::epsilon() * 8);
    auto reference = math::exp(-x) * (x * x + 2 * x + 2);
    expect_error_at_most(math::upper_gamma(C::real(3), x), reference, C::epsilon() * 64);
    auto half = C::real(1) / 2;
    reference = math::sqrt(pi<typename C::real_type> * C::real(1)) * math::erfc(math::sqrt(x));
    expect_error_at_most(math::upper_gamma(half, x), reference, C::epsilon() * 64);
    // Integration by parts, including a negative shape with no shape pole.
    auto lower = math::upper_gamma(-half, x);
    reference = 2 * (math::exp(-x) / math::sqrt(x) - math::upper_gamma(half, x));
    expect_error_at_most(lower, reference, C::epsilon() * 64);
  }
}
#if UNI20_HAS_MPREAL
namespace
{
// Independent finite sums / E1 / erfc identities for range sweeps. Recurrence
// cancellation is paid for in oracle bits, not charged to the tested function.
mpreal identity_reference(mpreal const& a, mpreal const& x)
{
  auto p = precision_of(a);
  bool integer = math::trunc(a) == a && a >= -100 && a <= 2000;
  int n = integer ? static_cast<int>(static_cast<double>(a)) : 0;
  if (!integer && abs(a) != mpreal(1, p) / 2) return math::upper_gamma(a, x);
  int exponent = static_cast<int>(math::frexp(x).exponent);
  int extra = 64 + (n < 0 ? -n : 1) * std::max(0, exponent + 2);
  auto work = Precision::bits(p.bit_count() + extra);
  auto xx = x.at(work), one = mpreal(1, work), result = one;
  if (integer && n > 0)
  {
    auto term = one;
    for (int k = 1; k < n; ++k)
    {
      term *= (n - k) / xx;
      result += term;
    }
    return (math::exp((n - 1) * math::log(xx) - xx) * result).at(p);
  }
  if (integer)
  {
    result = -math::expint(-xx);
    auto term = math::exp(-xx);
    for (int k = 1; k <= -n; ++k)
    {
      term /= xx;
      result = (term - result) / k;
    }
  }
  else
  {
    auto root = math::sqrt(xx);
    result = math::sqrt(pi<mpreal>.at(work)) * math::erfc(root);
    if (a < 0) result = 2 * (math::exp(-xx) / root - result);
  }
  return result.at(p);
}

// Positive integer a, x>4*a: integration by parts gives a positive
// falling-factorial sum. Its remaining tail is bounded by the first omitted
// term divided by 1-(a-N-1)/x. Extra argument-exponent bits protect a*log(x)-x.
mpreal large_reference(mpreal const& a, mpreal const& x)
{
  auto p = precision_of(a);
  auto work = Precision::bits(p.bit_count() + std::max(math::frexp(a).exponent, math::frexp(x).exponent) + 32);
  auto aa = a.at(work), xx = x.at(work), one = mpreal(1, work), sum = one, term = one;
  EXPECT_GT(x, 4 * a);
  bool converged = false;
  for (int k = 1; k <= 2048; ++k)
  {
    term *= (aa - k) / xx;
    sum += term;
    if (term == 0 || term < sum * math::ldexp(one, -p.bit_count() - 16))
    {
      converged = true;
      break;
    }
  }
  EXPECT_TRUE(converged);
  return math::exp((aa - 1) * math::log(xx) - xx + math::log(sum)).at(p);
}

template <class C> void upper_gamma_accuracy()
{
  using R = typename C::real_type;
  auto p = Precision::bits(512);
  auto zero = mpreal(0, p), one = mpreal(1, p);
  auto eps = math::ldexp(one, 1 - C::digits()), half_subnormal = zero;
  if constexpr (!C::runtime) half_subnormal = widen(numeric_limits<R>::denorm_min()) / 2;
  auto maximum = zero, worst_a = one, worst_x = one;
  auto check = [&](R const& a, R const& x, mpreal const& reference, mpreal const& uncertainty) {
    SCOPED_TRACE("a=" + format_scalar(a) + " x=" + format_scalar(x));
    auto result = math::upper_gamma(a, x);
    C::expect_precision(result);
    R rounded = scalar_at_precision<R>(reference, C::precision());
    ASSERT_FALSE(math::isnan(result));
    if (math::isinf(rounded) || rounded == 0 || result == 0)
    {
      EXPECT_EQ(result, rounded);
      EXPECT_EQ(math::signbit(result), math::signbit(rounded));
      return;
    }
    ASSERT_TRUE(math::isfinite(result));
    auto scale = std::max({one, abs(math::log(reference)), abs(widen(a) * math::log(widen(x))), widen(x)});
    auto ratio = (abs(widen(result) - reference) + uncertainty) / (64 * eps * scale * reference + half_subnormal);
    EXPECT_LE(ratio, 1);
    if (ratio > maximum)
    {
      maximum = ratio;
      worst_a = widen(a);
      worst_x = widen(x);
    }
  };
  auto evaluate = [&](R const& a, R const& x) {
    auto aa = widen(a), xx = widen(x);
    if (a != 0 && abs(aa) < math::ldexp(one, -256) && x > C::real(1) / 2 && x < C::real(3))
      check(a, x, math::upper_gamma(zero, xx), 10 * abs(aa));
    else
      check(a, x, identity_reference(aa, xx), zero);
  };
  // Cross-check the independent identities before relying on them at range edges.
  for (int n : {-4, -1, 0, 1, 2, 20})
    for (int k : {1, 4, 20})
    {
      auto a = mpreal(n, p), x = mpreal(k, p) / 2;
      auto ref = math::upper_gamma(a, x);
      expect_error_at_most(identity_reference(a, x), ref, abs(ref) * math::ldexp(one, -500));
    }
  for (int n = -32; n <= 48; ++n)
    for (int k : {1, 2, 4, 16, 64})
      evaluate(C::real(n) / 2, C::real(k) / 4);
  // This modest negative shape previously exhausted the guarded fraction
  // near float32 overflow, although the result is finite. Retain the exact
  // represented endpoint for every configured scalar, including MPFR.
  evaluate(-C::real(7) / 2, C::real(0x1.5fd7fep-37));
  auto neighbors = [&](R a, R x) {
    for (int k = 0; k < 8; ++k)
    {
      evaluate(a, x);
      a = math::next_up(a);
      x = math::next_down(x);
    }
  };
  for (int n : {-64, -32, -16, -2, -1, 1, 2, 16, 32, 64})
  {
    neighbors(C::real(n) / 2, C::real(1));
    neighbors(C::real(n) / 2, C::real(1) / 2);
    if (n > 0) neighbors(C::real(n) / 2, C::real(n) / 2);
  }
  std::uint64_t state = 0x67616d6d6132ULL;
  auto next = [&]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  for (int k = 0; k < 64; ++k)
  {
    auto f = math::ldexp(scalar_at_precision<R>(next(), C::precision()), -64) +
             math::ldexp(scalar_at_precision<R>(next(), C::precision()), -128);
    auto a = -C::real(1) / 2 + f;
    auto x = math::ldexp(C::real(1) / 2 + f / 2, -int(next() % C::digits()));
    evaluate(a, x);
  }
  if constexpr (!C::runtime)
  {
    using L = numeric_limits<R>;
    for (R a : {R(-.5), R(0), R(.5)})
      evaluate(a, L::denorm_min());
    for (R a : {L::denorm_min(), -L::denorm_min()})
      evaluate(a, R(1));
    auto neighbors_x = [&](R a, R x) {
      R lo = x, hi = x;
      for (int k = 0; k < 8; ++k)
      {
        evaluate(a, lo);
        evaluate(a, hi);
        lo = math::next_down(lo);
        hi = math::next_up(hi);
      }
    };
    // Locate boundaries with independent identities, without consulting the
    // tested implementation. Search in log(x) to span tiny overflow endpoints.
    for (R a : {R(-100), R(-8), R(-4), math::next_down(R(-4)), math::next_up(R(-4)), R(-3.5), R(-3.25), R(-3),
                R(-.5), R(0), R(.5), R(2), R(2000)})
      for (bool overflow : {false, true})
      {
        // The added modest negative shapes exercise the small-x overflow
        // guard. Retain the original underflow shapes: direct MPFR evaluation
        // at large x is much more expensive for nearby noninteger shapes.
        if (!overflow && a > R(-100) && a < R(-.5)) continue;
        auto target = overflow ? widen(L::max()) + math::ldexp(one, L::max_exponent - L::digits() - 1) : half_subnormal;
        auto lo = math::log(widen(L::denorm_min())), hi = math::log(mpreal(64000, p));
        auto aa = widen(a);
        if (identity_reference(aa, math::exp(lo)) < target) continue;
        for (int k = 0; k < C::digits() + 20; ++k)
        {
          auto mid = (lo + hi) / 2;
          if (identity_reference(aa, math::exp(mid)) > target)
            lo = mid;
          else
            hi = mid;
        }
        neighbors_x(a, scalar_at_precision<R>(math::exp((lo + hi) / 2), C::precision()));
      }
    // Shape-varying overflow of the lower-complement path needs guard bits
    // in log Gamma and Q, not only in the upper-tail continued fraction.
    for (R ratio : {R(.75), R(1)})
    {
      auto target = widen(L::max()) + math::ldexp(one, L::max_exponent - L::digits() - 1);
      auto lo = mpreal(2, p), hi = mpreal(2000, p), rr = widen(ratio);
      for (int k = 0; k < C::digits() + 8; ++k)
      {
        auto mid = (lo + hi) / 2;
        if (math::upper_gamma(mid, mid * rr) > target)
          hi = mid;
        else
          lo = mid;
      }
      R a = scalar_at_precision<R>((lo + hi) / 2, C::precision());
      for (int k = 0; k < 4; ++k)
        a = math::next_down(a);
      for (int k = 0; k < 8; ++k)
      {
        evaluate(a, a * ratio);
        a = math::next_up(a);
      }
    }
    // Large a*log(x)-x cancellation away from a range boundary. A positive
    // integer finite sum is independent of the production fraction/log guard.
    R a = 1024, x = a * (math::log(a) + math::log(math::log(a)));
    for (int k = 0; k < 16; ++k)
      x = math::fma(a, math::log(x), -math::log(x - a));
    neighbors_x(a, x);
    // At a=2^p the phase contains cancellation of terms of size about p*2^p.
    // A native log alone cannot resolve the final exponent at these inputs.
    a = math::ldexp(R(1), C::digits());
    x = a * (math::log(a) + math::log(math::log(a)));
    for (int k = 0; k < 16; ++k)
      x = math::fma(a, math::log(x), -math::log(x - a));
    for (int k = 0; k < 4; ++k)
      x = math::next_down(x);
    for (int k = 0; k < 8; ++k)
    {
      check(a, x, large_reference(widen(a), widen(x)), zero);
      x = math::next_up(x);
    }
  }
  testing::Test::RecordProperty("upper_gamma_max_bound_fraction", format_scalar(maximum));
  testing::Test::RecordProperty("upper_gamma_worst_shape", format_scalar(worst_a));
  testing::Test::RecordProperty("upper_gamma_worst_endpoint", format_scalar(worst_x));
  auto reference_function = worst_a > 2000 && worst_x > 4 * worst_a ? large_reference : identity_reference;
  auto reference = reference_function(worst_a, worst_x);
  auto wider = reference_function(worst_a.at(Precision::bits(1024)), worst_x.at(Precision::bits(1024))).at(p);
  expect_error_at_most(reference, wider, abs(wider) * math::ldexp(one, -500));
  auto ref = math::sqrt(pi<mpreal>.at(p)) * math::erfc(one);
  auto error = abs(widen(math::upper_gamma(C::real(1) / 2, C::real(1))) - ref);
  constexpr int bits = C::digits();
  constexpr int previous = bits <= 24    ? 12
                           : bits <= 53  ? 24
                           : bits <= 64  ? 53
                           : bits <= 113 ? 64
                           : bits <= 128 ? 113
                                         : 128;
  auto previous_error = abs(ref.at(Precision::bits(previous)).at(p) - ref);
  EXPECT_LT(error * 8, previous_error);
}
} // namespace
UNI20_PRECISION_TEST(NumericalScalar, UpperGammaAccuracy)
{
  this->RecordProperty("backend", "scalar_math");
  upper_gamma_accuracy<TypeParam>();
}
#endif
} // namespace uni20::test
