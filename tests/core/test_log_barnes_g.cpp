#include "log_barnes_g_references.hpp"
#include <gtest/gtest.h>
#include <memory>
#include <uni20/async/async.hpp>
#include <uni20/async/tbb_scheduler.hpp>
#include <uni20/core/log_barnes_g.hpp>

namespace
{
using uni20::complex;
using uni20::log_barnes_g_context;
using uni20::Precision;

TEST(LogBarnesG, DomainPrecisionAndWorkBudgets)
{
  log_barnes_g_context<double> context;
  EXPECT_THROW(context.log_abs_barnes_g_right({0, 1}), std::invalid_argument);
  EXPECT_THROW(context.log_abs_barnes_g_right({1, std::numeric_limits<double>::infinity()}), std::invalid_argument);
  EXPECT_THROW(context.log_abs_barnes_g_ratio_right({0.5, 1}, -0.5), std::invalid_argument);
  EXPECT_THROW(context.log_abs_barnes_g_ratio_right({1, 0}, std::numeric_limits<double>::quiet_NaN()),
               std::invalid_argument);
  EXPECT_THROW((log_barnes_g_context<double>(Precision::exact())), std::invalid_argument);
  EXPECT_THROW((log_barnes_g_context<double>(Precision::bits(24))), std::invalid_argument);
  EXPECT_THROW((log_barnes_g_context<double>({.max_terms = 2})), std::invalid_argument);
  // Normalization is prepared at construction and shares the explicit work budget.
  EXPECT_THROW((log_barnes_g_context<double>({.max_shifts = 0})), std::runtime_error);
  for (double x : {1., 2.})
  {
    auto result = context.log_abs_barnes_g_right({x, 0});
    EXPECT_EQ(result.value, 0);
    EXPECT_EQ(result.truncation_bound, 0);
    EXPECT_EQ(result.shifts, 0);
  }
  auto zero = context.log_abs_barnes_g_ratio_right({0.5, 2}, 0.0);
  EXPECT_EQ(zero.value, 0);
  EXPECT_EQ(zero.roundoff_indicator, 0);
}

TEST(LogBarnesG, PairedDifferenceSurvivesLossAndOverflowOfSeparateValues)
{
  log_barnes_g_context<double> context;
  double const pi = std::acos(-1.0);
  for (double y : {1e20, 1e100, 1e200})
  {
    auto result = context.log_abs_barnes_g_ratio_right({0.5, y}, 0.5);
    // The next terms are O(log(y)), negligible relative to this leading term.
    EXPECT_NEAR(result.value / y, -pi / 4, 3e-15);
    EXPECT_TRUE(uni20::isfinite(result.roundoff_indicator));
  }
  auto const a = context.log_abs_barnes_g_right({0.5, 1e100});
  auto const b = context.log_abs_barnes_g_right({1, 1e100});
  EXPECT_GT(std::abs((b.value - a.value) / 1e100 + pi / 4), 0.5);
  EXPECT_THROW(context.log_abs_barnes_g_right({0.5, 1e200}), std::overflow_error);
  auto tiny = context.log_abs_barnes_g_ratio_right({2, 1}, 0x1p-100);
  auto ordinary = context.log_abs_barnes_g_ratio_right({2, 1}, 0x1p-20);
  EXPECT_NEAR(tiny.value / 0x1p-100, ordinary.value / 0x1p-20, 1e-6);
}

#if UNI20_HAS_MPREAL
TEST(LogBarnesG, IndependentReferencesThrough1024Bits)
{
  using uni20::mpreal;
  auto const oracle = Precision::bits(1400);
  for (int bits : {128, 256, 333, 1024})
  {
    SCOPED_TRACE(bits);
    auto p = Precision::bits(bits);
    log_barnes_g_context<mpreal> context(p);
    auto eps = mpreal(uni20::numeric_limits<mpreal>::epsilon(p), oracle);
    for (auto const& ref : uni20::test::log_barnes_g_references)
    {
      SCOPED_TRACE(ref.re);
      SCOPED_TRACE(ref.im);
      SCOPED_TRACE(ref.shift);
      auto z = complex<mpreal>(ref.re, ref.im, p);
      auto a = context.log_abs_barnes_g_right(z);
      auto r = context.log_abs_barnes_g_ratio_right(z, mpreal(ref.shift, p));
      auto compare = [&](auto const& result, char const* text, bool relative) {
        auto const expected = mpreal(text, oracle);
        auto const error = uni20::math::abs(result.value.at(oracle) - expected);
        auto scale = uni20::math::abs(expected);
        if (!relative) scale += mpreal(1, oracle);
        EXPECT_LE(error, eps * scale * 16);
        EXPECT_LE(error, result.truncation_bound.at(oracle) + result.roundoff_indicator.at(oracle));
        EXPECT_EQ(uni20::precision_of(result.value), p);
        EXPECT_EQ(uni20::precision_of(result.truncation_bound), p);
        EXPECT_EQ(uni20::precision_of(result.roundoff_indicator), p);
      };
      compare(a, ref.value, false);
      compare(r, ref.ratio, true);
    }
  }
}

TEST(LogBarnesG, ExplicitPrecisionAndImmutableAsyncContexts)
{
  using uni20::mpreal;
  using namespace uni20::async;
  auto const p = Precision::bits(333);
  log_barnes_g_context<mpreal> context(p);
  auto a = context.log_abs_barnes_g_right(complex<mpreal>(mpreal(3)));
  EXPECT_EQ(uni20::precision_of(a.value), p);
  EXPECT_THROW(context.log_abs_barnes_g_right(complex<mpreal>(1, Precision::bits(128))), std::invalid_argument);
  EXPECT_THROW(context.log_abs_barnes_g_ratio_right(complex<mpreal>(1, p), mpreal(1, Precision::bits(128))),
               std::invalid_argument);
  EXPECT_THROW(context.log_abs_barnes_g_right(complex<mpreal>(uni20::uninitialized)), std::logic_error);
  EXPECT_THROW((log_barnes_g_context<mpreal>(p, {.max_terms = 16})), std::invalid_argument);
  EXPECT_THROW((log_barnes_g_context<mpreal>(Precision::bits(std::numeric_limits<std::int64_t>::max()))),
               std::overflow_error);
  auto const ambient = mpfr_get_default_prec();
  auto const emin = mpfr_get_emin(), emax = mpfr_get_emax();
  TbbScheduler scheduler{4};
  ScopedScheduler scope(&scheduler);
  std::vector<Async<mpreal>> outputs;
  std::vector<mpreal> expected;
  for (int bits : {128, 333})
  {
    auto precision = Precision::bits(bits);
    auto shared = std::make_shared<log_barnes_g_context<mpreal> const>(precision);
    for (int i = 1; i <= 8; ++i)
    {
      complex<mpreal> z(mpreal(1, precision) / 2, mpreal(i, precision));
      expected.push_back(shared->log_abs_barnes_g_ratio_right(z, mpreal(1, precision) / 2).value);
      outputs.emplace_back(mpreal(precision));
      scheduler.schedule([](WriteBuffer<mpreal> output, std::shared_ptr<log_barnes_g_context<mpreal> const> shared,
                            complex<mpreal> z) static -> AsyncTask {
        auto destination = co_await output;
        auto result = shared->log_abs_barnes_g_ratio_right(z, mpreal(1, shared->precision()) / 2);
        destination = std::move(result.value);
      }(outputs.back().write(), shared, z));
    }
  }
  for (std::size_t i = 0; i < outputs.size(); ++i)
  {
    EXPECT_EQ(outputs[i].get_wait(), expected[i]);
    EXPECT_EQ(uni20::precision_of(outputs[i].get_wait()), uni20::precision_of(expected[i]));
  }
  EXPECT_EQ(mpfr_get_default_prec(), ambient);
  EXPECT_EQ(mpfr_get_emin(), emin);
  EXPECT_EQ(mpfr_get_emax(), emax);
}
#endif
} // namespace
