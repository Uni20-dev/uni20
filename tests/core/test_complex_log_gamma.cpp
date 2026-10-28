#include "complex_log_gamma_references.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <uni20/async/async.hpp>
#include <uni20/async/tbb_scheduler.hpp>
#include <uni20/core/complex_log_gamma.hpp>

namespace
{
using uni20::complex;
using uni20::log_gamma_context;
using uni20::Precision;

TEST(ComplexLogGamma, DomainAndWorkBudgets)
{
  log_gamma_context<double> context;
  EXPECT_THROW(context.log_gamma_right({0, 1}), std::invalid_argument);
  EXPECT_THROW(context.log_gamma_right({1, std::numeric_limits<double>::infinity()}), std::invalid_argument);
  EXPECT_THROW(context.log_gamma_ratio_right({0.5, 1}, -0.5), std::invalid_argument);
  EXPECT_THROW((log_gamma_context<double>(Precision::exact())), std::invalid_argument);
  EXPECT_THROW((log_gamma_context<double>(Precision::bits(24))), std::invalid_argument);
  log_gamma_context<double> no_shifts({.max_shifts = 0});
  EXPECT_THROW(no_shifts.log_gamma_right({0.5, 0}), std::runtime_error);
  EXPECT_THROW(no_shifts.log_gamma_ratio_right({0.5, 0}, 0.5), std::runtime_error);
  EXPECT_NO_THROW(no_shifts.log_gamma_right({100, 0}));
}

TEST(ComplexLogGamma, BranchRecurrenceAndConjugation)
{
  log_gamma_context<double> context;
  for (auto z : {complex<double>(0.25, 0.5), {2, 7}, {0.5, 80}})
  {
    auto a = context.log_gamma_right(z);
    auto b = context.log_gamma_right(z + 1.0);
    EXPECT_LT(std::abs(b.value - a.value - std::log(z)), 2e-12);
    EXPECT_EQ(context.log_gamma_right(std::conj(z)).value, std::conj(a.value));
    auto ratio = context.log_gamma_ratio_right(z, 1.0);
    EXPECT_LT(std::abs(ratio.value - std::log(z)), 3e-14);
  }
  EXPECT_GT(context.log_gamma_right({2, 7}).value.imag(), 3.14159);
  EXPECT_NEAR(context.log_gamma_right({0.5, 0}).value.real(), std::log(std::sqrt(std::acos(-1.0))), 2e-14);
}

TEST(ComplexLogGamma, PairedRatioRetainsLargeArgumentPhase)
{
  log_gamma_context<double> context;
  double const pi = std::acos(-1.0);
  for (double y : {1e20, 1e100, 1e300, std::numeric_limits<double>::max()})
  {
    auto result = context.log_gamma_ratio_right({0.5, y}, 0.5);
    EXPECT_NEAR(result.value.imag(), pi / 4, 2e-15);
    EXPECT_NEAR(result.value.real(), std::log(y) / 2, 2e-13);
    EXPECT_TRUE(uni20::isfinite(result.roundoff_indicator));
    EXPECT_GT(result.roundoff_indicator, 0);
  }
  // Separate Gamma evaluations have lost this O(1) phase entirely.
  auto a = context.log_gamma_right({0.5, 1e100}).value;
  auto b = context.log_gamma_right({1, 1e100}).value;
  EXPECT_GT(std::abs((b - a).imag() - pi / 4), 0.5);
  auto tiny = context.log_gamma_ratio_right({0.5, 1e-100}, 0.5);
  EXPECT_NEAR(tiny.value.imag() / 1e-100, 2 * std::log(2.0), 2e-14);
}

TEST(ComplexLogGamma, TinyArgumentsShiftsAndZero)
{
  log_gamma_context<double> context;
  for (double x : {1e-300, std::numeric_limits<double>::denorm_min()})
  {
    auto value = context.log_gamma_right({x, 0});
    EXPECT_NEAR(value.value.real(), -std::log(x), 3e-12);
    auto ratio = context.log_gamma_ratio_right({x, 0}, 0.5);
    EXPECT_NEAR(ratio.value.real(), std::log(x) + std::log(std::sqrt(std::acos(-1.0))), 3e-12);
  }
  auto zero = context.log_gamma_ratio_right({0.5, 2}, 0.0);
  EXPECT_EQ(zero.value, complex<double>(0, 0));
  EXPECT_EQ(zero.terms, 0);
  EXPECT_THROW(context.log_gamma_right({1e308, 1e308}), std::overflow_error);
}

TEST(ComplexLogGamma, FiniteAnswersNearNativeRangeLimit)
{
  log_gamma_context<double> context;
  // Independent mpmath loggamma, 100 dps, evaluated at this exact binary64 x.
  double const x = 2.557426983037746e305;
  double const reference = 1.7958954417274532642231584817155455266521079030845348061535409011e308;
  auto g = context.log_gamma_right({x, 0});
  EXPECT_NEAR(g.value.real() / reference, 1, 2e-15);
  EXPECT_TRUE(uni20::isfinite(g.roundoff_indicator));
  auto ratio = context.log_gamma_ratio_right({0.5, 0}, x);
  EXPECT_NEAR(ratio.value.real() / reference, 1, 2e-15);
  EXPECT_TRUE(uni20::isfinite(ratio.roundoff_indicator));
  // Both components fit although their Euclidean magnitude exceeds DBL_MAX.
  auto c = context.log_gamma_right({0.5, 2.5599807715086384e305});
  EXPECT_NEAR(c.value.real() / -4.0212083925513347123742936457694947152681474132195e305, 1, 2e-15);
  EXPECT_NEAR(c.value.imag() / 1.7976913371691809441651532767710184005460994156441e308, 1, 2e-15);
  EXPECT_TRUE(uni20::isfinite(c.roundoff_indicator));
}

#if UNI20_HAS_MPREAL
TEST(ComplexLogGamma, IndependentReferencesThrough1024Bits)
{
  using uni20::mpreal;
  auto oracle = Precision::bits(1400);
  for (int bits : {128, 256, 333, 1024})
  {
    SCOPED_TRACE(bits);
    auto p = Precision::bits(bits);
    log_gamma_context<mpreal> context(p);
    auto eps = mpreal(uni20::numeric_limits<mpreal>::epsilon(p), oracle);
    for (auto const& ref : uni20::test::complex_log_gamma_references)
    {
      SCOPED_TRACE(ref.im);
      auto z = complex<mpreal>(ref.re, ref.im, p);
      auto g = context.log_gamma_right(z);
      auto r = context.log_gamma_ratio_right(z, mpreal(ref.shift, p));
      auto compare = [&](auto const& value, char const* re, char const* im) {
        auto rr = mpreal(re, oracle), ri = mpreal(im, oracle);
        auto error = uni20::math::hypot(value.real().at(oracle) - rr, value.imag().at(oracle) - ri);
        EXPECT_LE(error, eps * uni20::math::hypot(rr, ri) * 16);
        EXPECT_EQ(uni20::precision_of(value), p);
      };
      compare(g.value, ref.gamma_re, ref.gamma_im);
      compare(r.value, ref.ratio_re, ref.ratio_im);
    }
  }
}

TEST(ComplexLogGamma, ExplicitPrecisionAndExactCoefficients)
{
  using uni20::mpreal;
  auto p = Precision::bits(333);
  log_gamma_context<mpreal> context(p);
  auto result = context.log_gamma_right(complex<mpreal>(mpreal(1)));
  EXPECT_EQ(uni20::precision_of(result.value), p);
  EXPECT_EQ(uni20::precision_of(result.truncation_bound), p);
  EXPECT_EQ(uni20::precision_of(result.roundoff_indicator), p);
  EXPECT_LT(uni20::math::abs(result.value), uni20::numeric_limits<mpreal>::epsilon(p));
  EXPECT_THROW(context.log_gamma_right(complex<mpreal>(1, Precision::bits(128))), std::invalid_argument);
  EXPECT_THROW(context.log_gamma_right(complex<mpreal>(uni20::uninitialized)), std::logic_error);
  EXPECT_THROW((log_gamma_context<mpreal>(p, {.max_terms = 16})), std::invalid_argument);
  EXPECT_THROW((log_gamma_context<mpreal>(Precision::bits(std::numeric_limits<std::int64_t>::max()))),
               std::overflow_error);
  EXPECT_EQ(uni20::detail::exact_bernoulli(0), uni20::exact_constant(1));
  EXPECT_EQ(uni20::detail::exact_bernoulli(1), uni20::exact_constant(-1) / uni20::exact_constant(2));
  EXPECT_EQ(uni20::detail::exact_bernoulli(12), uni20::exact_constant(-691) / uni20::exact_constant(2730));
  EXPECT_EQ(uni20::detail::exact_bernoulli(19), uni20::exact_constant(0));
}
TEST(ComplexLogGamma, SharedContextsRetainPrecisionThroughAsyncCompletion)
{
  using uni20::mpreal;
  using namespace uni20::async;
  TbbScheduler scheduler{4};
  ScopedScheduler scope(&scheduler);
  std::vector<Async<complex<mpreal>>> outputs;
  std::vector<complex<mpreal>> expected;
  auto const ambient = mpfr_get_default_prec();
  for (int bits : {128, 333})
  {
    auto p = Precision::bits(bits);
    auto context = std::make_shared<log_gamma_context<mpreal> const>(p);
    for (int i = 1; i <= 12; ++i)
    {
      complex<mpreal> z(mpreal(1, p) / 2, mpreal(i, p));
      expected.push_back(context->log_gamma_ratio_right(z, mpreal(1, p) / 2).value);
      outputs.emplace_back(complex<mpreal>(p));
      scheduler.schedule([](WriteBuffer<complex<mpreal>> output,
                            std::shared_ptr<log_gamma_context<mpreal> const> context,
                            complex<mpreal> z) static -> AsyncTask {
        auto destination = co_await output;
        auto result = context->log_gamma_ratio_right(z, mpreal(1, context->precision()) / 2);
        destination = std::move(result.value);
      }(outputs.back().write(), context, z));
    }
    // Only the admitted operations own this context after the loop iteration.
  }
  for (std::size_t i = 0; i < outputs.size(); ++i)
  {
    auto const& value = outputs[i].get_wait();
    EXPECT_EQ(value, expected[i]);
    EXPECT_EQ(uni20::precision_of(value), uni20::precision_of(expected[i]));
  }
  EXPECT_EQ(mpfr_get_default_prec(), ambient);
}
#endif
} // namespace
