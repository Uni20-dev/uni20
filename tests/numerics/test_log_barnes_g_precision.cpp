#include "../core/log_barnes_g_references.hpp"
#include "precision_registry.hpp"
#include <uni20/core/log_barnes_g.hpp>

namespace uni20::test
{
UNI20_PRECISION_TEST(NumericalScalar, LogBarnesGIdentities)
{
  using C = TypeParam;
  using R = typename C::real_type;
  this->RecordProperty("backend", "scalar_math");
  log_barnes_g_context<R> context(C::precision());
  log_gamma_context<R> gamma(C::precision());
  auto const one = C::real(1), zero = C::real(0);
  for (auto const& z : {C::scalar(one / C::real(4), one / C::real(2)), C::scalar(C::real(2), C::real(3)),
                        C::scalar(one / C::real(2), C::real(100))})
  {
    auto a = context.log_abs_barnes_g_right(z), b = context.log_abs_barnes_g_right(z + one);
    auto ratio = context.log_abs_barnes_g_ratio_right(z, one);
    auto expected = gamma.log_gamma_right(z).value.real();
    auto scale = one + math::abs(a.value) + math::abs(b.value);
    expect_error_at_most(b.value - a.value, expected, C::epsilon() * scale * C::real(4096));
    expect_error_at_most(ratio.value, expected, C::epsilon() * (one + math::abs(expected)) * C::real(4096));
    expect_equal(context.log_abs_barnes_g_ratio_right(z + one, -one).value, -ratio.value);
    expect_equal(context.log_abs_barnes_g_right(uni20::conj(z)).value, a.value);
    C::expect_precision(a.value);
    C::expect_precision(a.truncation_bound);
    C::expect_precision(a.roundoff_indicator);
    EXPECT_GE(a.truncation_bound, zero);
    EXPECT_GE(a.roundoff_indicator, zero);
  }
  // Same formula through each native exponent range; no separate G value fits.
  auto const exponent = [] {
    if constexpr (C::runtime)
      return 8 * numeric_limits<R>::digits(C::precision());
    else
      return numeric_limits<R>::max_exponent;
  }();
  auto const y = math::ldexp(one, 3 * exponent / 4);
  auto r = context.log_abs_barnes_g_ratio_right(C::scalar(one / C::real(2), y), one / C::real(2));
  expect_error_at_most(r.value / y, -(pi<R> * one) / C::real(4), C::epsilon() * C::real(32));
}

#if UNI20_HAS_MPREAL
UNI20_PRECISION_TEST(NumericalScalar, LogBarnesGReferenceAccuracy)
{
  using C = TypeParam;
  using R = typename C::real_type;
  this->RecordProperty("backend", "scalar_math");
  log_barnes_g_context<R> context(C::precision());
  auto const oracle = Precision::bits(1400);
  auto const eps = mpreal(C::epsilon(), oracle);
  for (auto const& ref : log_barnes_g_references)
  {
    SCOPED_TRACE(ref.re);
    SCOPED_TRACE(ref.im);
    SCOPED_TRACE(ref.shift);
    if constexpr (C::kind == PrecisionKind::float32)
      if (math::abs(mpreal(ref.im, oracle)) > mpreal("1e30", oracle) ||
          math::abs(mpreal(ref.shift, oracle)) < mpreal("1e-30", oracle))
        continue;
    auto convert = [&](char const* text) { return scalar_at_precision<R>(mpreal(text, oracle), C::precision()); };
    auto z = C::scalar(convert(ref.re), convert(ref.im));
    auto measure = [&](auto const& result, char const* text, bool relative) {
      auto const expected = mpreal(text, oracle);
      auto const error = math::abs(mpreal(result.value, oracle) - expected);
      auto scale = math::abs(expected);
      if (!relative) scale += mpreal(1, oracle);
      EXPECT_LE(error, eps * scale * 4096);
      EXPECT_LE(error, mpreal(result.truncation_bound, oracle) + mpreal(result.roundoff_indicator, oracle));
      C::expect_precision(result.value);
    };
    bool absolute_fits = true;
    if constexpr (!C::runtime)
      absolute_fits = math::abs(mpreal(ref.value, oracle)) < mpreal(numeric_limits<R>::max(), oracle) / 16;
    if (absolute_fits) measure(context.log_abs_barnes_g_right(z), ref.value, false);
    measure(context.log_abs_barnes_g_ratio_right(z, convert(ref.shift)), ref.ratio, true);
  }
}
#endif
} // namespace uni20::test
