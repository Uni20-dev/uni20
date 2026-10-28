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

#if UNI20_HAS_MPREAL
UNI20_PRECISION_TEST(NumericalScalar, ComplexLogGammaReferenceAccuracy)
{
  using C = TypeParam;
  using R = typename C::real_type;
  this->RecordProperty("backend", "scalar_math");
  log_gamma_context<R> context(C::precision());
  auto oracle = Precision::bits(1400);
  auto eps = mpreal(C::epsilon(), oracle);
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
