#include <uni20/random/transforms.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <numbers>

namespace uni20::random::test
{
namespace
{
constexpr Philox4x32Block mixed{{0x12345aff, 0xabcdef81, 0x76543a7f, 0xfedcba98}};
static_assert(Uniform<float>{}({mixed}).values[0] == 0x1.2345ap-4f);
static_assert(Uniform<double>{}({mixed}).values[1] == 0x1.fdb97530eca87p-1);

TEST(RandomTransformTest, UnitUniformsMatchRecordedWords)
{
  constexpr Philox4x32Block extremes{{0, 0, ~0u, ~0u}};
  EXPECT_TRUE(
      std::ranges::equal(Uniform<float>{}({extremes}).values, std::array{0.f, 0.f, 0x1.fffffep-1f, 0x1.fffffep-1f}));
  EXPECT_TRUE(std::ranges::equal(Uniform<double>{}({extremes}).values, std::array{0., 0x1.fffffffffffffp-1}));
  EXPECT_TRUE(std::ranges::equal(Uniform<float>{}({mixed}).values,
                                 std::array{0x1.2345ap-4f, 0x1.579bdep-1f, 0x1.d950e8p-2f, 0x1.fdb974p-1f}));
  EXPECT_TRUE(
      std::ranges::equal(Uniform<double>{}({mixed}).values, std::array{0x1.579bdf022468bp-1, 0x1.fdb97530eca87p-1}));
}

TEST(RandomTransformTest, ScalingAndBoxMullerFollowAnalyticValues)
{
  constexpr long double eps = std::numeric_limits<double>::epsilon();
  // Exercise each quarter turn away from its axis, where a reflection or a lane swap changes the result.
  auto const at_sixteenth_turn = [](std::uint32_t n) { return Philox4x32Block{{~0u, 0x7fffffff, 0, n << 28}}; };
  for (auto words : {Philox4x32Block{}, Philox4x32Block{{~0u, ~0u, ~0u, ~0u}}, mixed,
                     Philox4x32Block{{0, 0x80000000, 0, 0x40000000}}, Philox4x32Block{{~0u, 0, 0, 0x80000000}},
                     at_sixteenth_turn(1), at_sixteenth_turn(5), at_sixteenth_turn(9), at_sixteenth_turn(13)})
  {
    auto const unit = Uniform<double>{}({words});
    SCOPED_TRACE(::testing::Message() << "angle in turns = " << unit.values[1]);
    long double const radius = std::sqrt(-2 * std::log(static_cast<long double>(unit.values[0]) + 0x1p-53L));
    long double const angle = 2 * std::numbers::pi_v<long double> * unit.values[1];
    std::array ref{radius * std::cos(angle), radius * std::sin(angle)};
    auto const scaled = Uniform{-3.0, 5.0}({words});
    auto const normal = Normal{0.25, 1.5}({words});
    complex<double> const mean{-0.25, 0.5};
    auto const z = ComplexNormal{mean, 2.0}({words}).values[0];
    for (unsigned i = 0; i < 2; ++i)
    {
      EXPECT_NEAR(scaled.values[i], -3.L + 8.L * unit.values[i], 4 * eps * (3 + 8 * unit.values[i]));
      // The independent reference reduces an angle formed in long double.
      long double const phase_error = 8 * std::numeric_limits<long double>::epsilon() * radius;
      EXPECT_NEAR(normal.values[i], 0.25L + 1.5L * ref[i],
                  6 * eps * (0.25L + 1.5L * std::abs(ref[i])) + 2 * phase_error);
      long double const part = i == 0 ? mean.real() : mean.imag();
      EXPECT_NEAR(i == 0 ? z.real() : z.imag(), part + std::numbers::sqrt2_v<long double> * ref[i],
                  6 * eps * (std::abs(part) + 2 * std::abs(ref[i])) + 2 * phase_error);
    }
  }
}

// For a fixed sample, check its first two moments and 16 equal-probability bins.
struct Sample
{
    double sum = 0, squares = 0;
    unsigned count = 0;
    std::array<unsigned, 16> bins{};
    void add(double value, double cdf)
    {
      ++count;
      sum += value;
      squares += value * value;
      ++bins[std::min(15u, static_cast<unsigned>(cdf * 16))];
    }
    void check(double mean, double variance) const
    {
      auto const observed_mean = sum / count;
      EXPECT_NEAR(observed_mean, mean, 6 * std::sqrt(variance / count));
      EXPECT_NEAR(squares / count - observed_mean * observed_mean, variance, 10 * variance / std::sqrt(count));
      double chi2 = 0, expected = count / 16.;
      for (auto n : bins)
        chi2 += (n - expected) * (n - expected) / expected;
      EXPECT_LT(chi2, 52.) << "15 degrees of freedom";
    }
};

TEST(RandomTransformTest, DistributionMomentsAndEqualProbabilityBins)
{
  Sample uniform_float, uniform_double, real_normal, complex_power;
  complex<double> const mu{0.5, -3};
  complex<double> sum{}, pseudo{};
  constexpr unsigned n = 32768;
  for (unsigned i = 0; i < n; ++i)
  {
    auto const words = philox4x32_10({{i, 0, 0, 0}}, {{0x4635, 0xac9d}});
    for (auto u : Uniform<float>{}({words}).values)
      uniform_float.add(u, u);
    for (auto u : Uniform<double>{}({words}).values)
      uniform_double.add(u, u);
    for (auto x : Normal{}({words}).values)
      real_normal.add(x, std::erfc(-x / std::numbers::sqrt2) * 0.5);
    auto const z = ComplexNormal{mu, 2}({words}).values[0] - mu;
    sum += z;
    pseudo += z * z;
    double const power = std::norm(z);
    complex_power.add(power, -std::expm1(-power / 4));
  }
  uniform_float.check(0.5, 1. / 12);
  uniform_double.check(0.5, 1. / 12);
  real_normal.check(0, 1);
  complex_power.check(4, 16);
  EXPECT_LT(std::abs(sum) / n, 0.06);
  EXPECT_LT(std::abs(pseudo) / n, 0.19);
}
} // namespace
} // namespace uni20::random::test
