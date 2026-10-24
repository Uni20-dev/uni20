#include "precision_registry.hpp"
#include <cstdint>
#include <vector>

namespace uni20::test
{
UNI20_PRECISION_TEST(NumericalScalar, ZetaIdentities)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  auto value = math::zeta(C::real(-1));
  C::expect_precision(value);
  expect_error_at_most(value, -C::real(1) / 12, C::epsilon() * 4);
  expect_error_at_most(math::zeta(C::real(-3)), C::real(1) / 120, C::epsilon() * 4);
  EXPECT_EQ(math::zeta(C::real(0)), -C::real(1) / 2);
  EXPECT_EQ(math::zeta(C::real(-2)), 0);
}
#if UNI20_HAS_MPREAL
namespace
{
template <class C> void zeta_accuracy()
{
  using R = typename C::real_type;
  auto p = Precision::bits(512);
  auto one = mpreal(1, p), eps = math::ldexp(one, 1 - C::digits()), half_subnormal = mpreal(0, p);
  if constexpr (!C::runtime) half_subnormal = widen(numeric_limits<R>::denorm_min()) / 2;
  std::vector<R> inputs;
  auto add = [&](R x) {
    if (math::isfinite(x) && math::abs(x) < 24000) inputs.push_back(x);
  };
  for (int k = -512; k <= 512; ++k)
    add(C::real(k) / 7);
  auto neighbors = [&](R center) {
    R lo = center, hi = center;
    for (int k = 0; k < 16; ++k)
    {
      add(lo);
      add(hi);
      lo = math::next_down(lo);
      hi = math::next_up(hi);
    }
  };
  for (int n : {-34, -32, -16, -8, -2, 0, 1, 2, C::digits() + 1})
    neighbors(C::real(n));
  neighbors(C::real(-1) / 2);
  neighbors(C::real(1) / 2);
  // More neighbors at the small-Gamma/Stirling joins expose cancellation and
  // precision lost if the small phase is placed inside the logarithm.
  for (int n : {-8, -16, -32})
  {
    R lo = C::real(n), hi = lo;
    for (int k = 0; k < 64; ++k)
    {
      add(lo);
      add(hi);
      lo = math::next_down(lo);
      hi = math::next_up(hi);
    }
  }
  std::uint64_t state = 0x7a657461ULL;
  auto next = [&]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  for (int k = 0; k < 128; ++k)
  {
    auto fraction = math::ldexp(scalar_at_precision<R>(next(), C::precision()), -64) +
                    math::ldexp(scalar_at_precision<R>(next(), C::precision()), -128);
    add(fraction - C::real(1) / 2);
    add(-fraction * 128);
    add(fraction * C::digits());
  }
  if constexpr (!C::runtime)
  {
    using L = numeric_limits<R>;
    add(L::denorm_min());
    add(-L::denorm_min());
    for (int e = L::min_exponent - L::digits(); e < 16; e += 503)
    {
      auto x = math::ldexp(R(1), e);
      add(x);
      add(-x);
    }
    // Locate the first overflowing odd argument independently, then bisect
    // both transitions in that lobe and two subsequent lobes.
    auto target = widen(L::max()) + math::ldexp(one, L::max_exponent - C::digits() - 1);
    int lower = 0, upper = L::max_exponent;
    while (lower + 1 < upper)
    {
      int mid = lower + (upper - lower) / 2;
      if (abs(math::zeta(mpreal(-2 * mid - 1, p))) > target)
        upper = mid;
      else
        lower = mid;
    }
    for (int n = 2 * upper + 1; n < 2 * upper + 6; n += 2)
      for (int direction : {-1, 1})
      {
        auto lo = mpreal(-n, p), hi = mpreal(-n + direction, p);
        for (int k = 0; k < 2 * C::digits(); ++k)
        {
          auto mid = (lo + hi) / 2;
          if (abs(math::zeta(mid)) > target)
            lo = mid;
          else
            hi = mid;
        }
        neighbors(scalar_at_precision<R>((lo + hi) / 2, C::precision()));
      }
  }
  auto maximum = mpreal(0, p), worst = one;
  auto check = [&](R const& x, mpreal const& reference) {
    SCOPED_TRACE(format_scalar(x));
    auto result = math::zeta(x);
    C::expect_precision(result);
    R rounded = scalar_at_precision<R>(reference, C::precision());
    if (math::isinf(rounded))
    {
      EXPECT_EQ(result, rounded);
      return;
    }
    ASSERT_TRUE(math::isfinite(result));
    if (rounded == 0 || result == 0)
    {
      EXPECT_EQ(result, rounded);
      EXPECT_EQ(math::signbit(result), math::signbit(rounded));
      return;
    }
    auto scale = abs(reference), wide = widen(x);
    if (x < 0)
    {
      auto phase = math::sinpi(wide / 2);
      scale *= std::max(one, abs(math::log(abs(reference / phase))));
      EXPECT_EQ(math::signbit(result), math::signbit(rounded));
    }
    auto ratio = abs(widen(result) - reference) / (32 * eps * scale + half_subnormal);
    EXPECT_LE(ratio, 1);
    if (ratio > maximum)
    {
      maximum = ratio;
      worst = wide;
    }
  };
  for (auto const& x : inputs)
    check(x, math::zeta(widen(x)));
  testing::Test::RecordProperty("zeta_max_bound_fraction", format_scalar(maximum));
  testing::Test::RecordProperty("zeta_worst_input", format_scalar(worst));
  EXPECT_EQ(math::zeta(worst), math::zeta(worst.at(Precision::bits(1024))).at(p));
  // Independent Bernoulli identities at negative odd arguments.
  for (auto [n, numerator, denominator] : {std::tuple{1, -1, 12}, std::tuple{3, 1, 120}, std::tuple{5, -1, 252},
                                           std::tuple{7, 1, 240}, std::tuple{9, -1, 132}})
    check(C::real(-n), mpreal(numerator, p) / denominator);
  auto pi_ref = pi<mpreal>.at(p), reference = pi_ref * pi_ref / 6;
  check(C::real(2), reference);
  check(C::real(4), math::pown(pi_ref, 4) / 90);
  auto error = abs(widen(math::zeta(C::real(2))) - reference);
  constexpr int bits = C::digits();
  constexpr int previous = bits <= 24    ? 12
                           : bits <= 53  ? 24
                           : bits <= 64  ? 53
                           : bits <= 113 ? 64
                           : bits <= 128 ? 113
                                         : 128;
  auto previous_error = abs(reference.at(Precision::bits(previous)).at(p) - reference);
  EXPECT_LT(error * 16, previous_error);
}
} // namespace
UNI20_PRECISION_TEST(NumericalScalar, ZetaAccuracy)
{
  this->RecordProperty("backend", "scalar_math");
  zeta_accuracy<TypeParam>();
}
#endif
} // namespace uni20::test
