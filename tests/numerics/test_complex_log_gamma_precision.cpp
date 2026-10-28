#include "../core/complex_log_gamma_references.hpp"
#include "precision_registry.hpp"
#include <uni20/core/complex_log_gamma.hpp>

namespace uni20::test
{
UNI20_PRECISION_TEST(NumericalScalar, ComplexLogGammaIdentities)
{
  using C = TypeParam;
  using R = typename C::real_type;
  this->RecordProperty("backend", "scalar_math");
  log_gamma_context<R> context(C::precision());
  auto z = C::scalar(C::real(1) / C::real(4), C::real(1) / C::real(2));
  auto a = context.log_gamma_right(z), b = context.log_gamma_right(z + C::real(1));
  C::expect_precision(a.value);
  C::expect_precision(a.truncation_bound);
  C::expect_precision(a.roundoff_indicator);
  auto const tolerance = C::epsilon() * C::real(256);
  expect_error_at_most(b.value - a.value, math::log(z), tolerance);
  expect_error_at_most(context.log_gamma_ratio_right(z, C::real(1)).value, math::log(z), tolerance);
  expect_error_at_most(context.log_gamma_ratio_right(z + C::real(1), -C::real(1)).value, -math::log(z), tolerance);
  expect_equal(context.log_gamma_right(uni20::conj(z)).value, uni20::conj(a.value));
}

UNI20_PRECISION_TEST(NumericalScalar, ComplexLogGammaScaledComponents)
{
  using C = TypeParam;
  using R = typename C::real_type;
  this->RecordProperty("backend", "scalar_math");
  log_gamma_context<R> context(C::precision());
  auto const range = [] {
    if constexpr (C::runtime) return 8 * numeric_limits<R>::digits(C::precision());
    else return numeric_limits<R>::max_exponent;
  }();
  auto const zero = C::real(0), one = C::real(1);
  auto const tolerance = C::epsilon() * C::real(32);
  auto const small = math::ldexp(one, -3 * range / 4);
  // Gamma(z) ~ 1/z: the discarded O(small) correction is far below epsilon.
  auto tiny = context.log_gamma_ratio_right(C::scalar(small, zero), small / C::real(4));
  expect_error_at_most(tiny.value.real(), -math::log1p(one / C::real(4)), tolerance);

  auto const x = math::ldexp(one, 3 * range / 4), y = math::ldexp(one, -range / 2);
  auto gamma = context.log_gamma_right(C::scalar(x, y));
  // The asymptotic phase is y*log(x); 1/x and (y/x)^2 are below epsilon.
  // Compare the component after scaling: a complex norm would hide its loss.
  expect_error_at_most(gamma.value.imag() / y, math::log(x), tolerance * math::log(x));
  for (int denominator : {8, 4, -8, -4})
  {
    auto ratio = context.log_gamma_ratio_right(C::scalar(x, y), x / C::real(denominator));
    expect_error_at_most(ratio.value.imag() / y, math::log1p(one / C::real(denominator)), tolerance);
    C::expect_precision(ratio.value);
  }

  // h/x underflows for native types, while y*h/x remains representable.
  auto const h = math::ldexp(one, -range / 2), larger_y = math::ldexp(one, range / 2);
  auto ratio = context.log_gamma_ratio_right(C::scalar(x, larger_y), h);
  expect_error_at_most(ratio.value.imag() / small, one, tolerance);

  // Check both sides of the native small-angle boundary against the exact
  // finite Gamma recurrence, for both logarithm-difference evaluation paths.
  for (int divisor : {1, 4})
    for (int shift : {1, 8})
    {
      auto const boundary_y = C::real(32) * math::sqrt(C::epsilon()) / C::real(divisor);
      auto phase = zero;
      for (int k = 0; k < shift; ++k) phase += math::atan2(boundary_y, C::real(32 + k));
      auto boundary = context.log_gamma_ratio_right(C::scalar(C::real(32), boundary_y), C::real(shift));
      expect_error_at_most(boundary.value.imag() / boundary_y, phase / boundary_y, tolerance);
    }
}

#if UNI20_HAS_MPREAL
UNI20_PRECISION_TEST(NumericalScalar, ComplexLogGammaReferenceAccuracy)
{
  using C = TypeParam;
  using R = typename C::real_type;
  this->RecordProperty("backend", "scalar_math");
  log_gamma_context<R> context(C::precision());
  auto oracle = Precision::bits(1400);
  auto eps = mpreal(C::epsilon(), oracle);
  if constexpr (!C::runtime)
  {
    auto const one = C::real(1), eighth = one / C::real(8);
    auto const maximum = numeric_limits<R>::max();
    auto const target = maximum * (one - C::epsilon() * C::real(64));
    auto const angular = math::log1p(eighth * eighth) / C::real(2) - eighth * math::atan(eighth);
    auto x = maximum / math::log(maximum);
    // Locate a finite result close enough to max that its unfused x*log term
    // exceeds the range, for each native exponent range and significand size.
    for (int i = 0; i < 24; ++i) x = target / (math::log(x) - one + angular);
    auto const y = x * eighth;
    complex<mpreal> const z(mpreal(x, oracle), mpreal(y, oracle));
    auto const unit = mpreal(1, oracle), half = unit / 2;
    // Independent leading Stirling expression: the O(1/z) correction is far
    // below an output ULP here. MPFR is an oracle, never the native evaluator.
    auto const log_z = math::log(z);
    auto const expected = (z - half) * log_z - z + math::log((pi<mpreal> * unit) * 2) / 2;
    ASSERT_GT(mpreal(x, oracle) * (log_z.real() - unit), mpreal(maximum, oracle));
    ASSERT_LT(expected.real(), mpreal(maximum, oracle));
    auto const actual = context.log_gamma_right(C::scalar(x, y));
    EXPECT_LE(math::abs(mpreal(actual.value.real(), oracle) / expected.real() - unit), eps * 32);
    EXPECT_LE(math::abs(mpreal(actual.value.imag(), oracle) / expected.imag() - unit), eps * 32);
  }
  for (auto const& ref : complex_log_gamma_references)
  {
    SCOPED_TRACE(ref.re);
    SCOPED_TRACE(ref.im);
    SCOPED_TRACE(ref.shift);
    // float32 cannot represent 2^256 or 2^-256. The ordinary cases and a
    // small retained shift remain covered, and the wider families cover both.
    if constexpr (C::kind == PrecisionKind::float32)
      if (math::abs(mpreal(ref.im, oracle)) > mpreal("1e30", oracle) ||
          math::abs(mpreal(ref.im, oracle)) < mpreal("1e-30", oracle))
        continue;
    auto convert = [&](char const* text) { return scalar_at_precision<R>(mpreal(text, oracle), C::precision()); };
    auto z = C::scalar(convert(ref.re), convert(ref.im));
    auto g = context.log_gamma_right(z), ratio = context.log_gamma_ratio_right(z, convert(ref.shift));
    auto measure = [&](auto const& result, char const* re, char const* im, bool relative) {
      mpreal const rr(re, oracle), ri(im, oracle);
      auto error = math::hypot(mpreal(result.value.real(), oracle) - rr, mpreal(result.value.imag(), oracle) - ri);
      auto scale = math::hypot(rr, ri);
      if (!relative) scale += mpreal(1, oracle);
      EXPECT_LE(error, eps * scale * 256);
      EXPECT_LE(error, mpreal(result.truncation_bound, oracle) + mpreal(result.roundoff_indicator, oracle));
      C::expect_precision(result.value);
    };
    measure(g, ref.gamma_re, ref.gamma_im, false);
    measure(ratio, ref.ratio_re, ref.ratio_im, true);
  }
}
#endif
} // namespace uni20::test
