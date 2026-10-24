#include "precision_registry.hpp"
#include <cstdint>
#include <vector>

namespace uni20::test
{
UNI20_PRECISION_TEST(NumericalScalar, ExpintIdentities)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  auto x = C::real(1) / 4 + C::gap(), term = x, sum = x;
  for (int k = 3; k < 101; k += 2)
  {
    term *= x * x / C::real(k * (k - 1));
    sum += term / C::real(k);
  }
  auto result = math::expint(x);
  C::expect_precision(result);
  expect_error_at_most(result - math::expint(-x), 2 * sum, C::epsilon() * 32);
}
#if UNI20_HAS_MPREAL
namespace
{
mpreal const& expint_positive_root()
{
  static auto const root = [] {
    auto p = Precision::bits(512);
    auto lo = mpreal(0, p), hi = mpreal(1, p);
    for (int k = 0; k < 510; ++k)
    {
      auto mid = (lo + hi) / 2;
      if (math::expint(mid) < 0)
        lo = mid;
      else
        hi = mid;
    }
    return (lo + hi) / 2;
  }();
  return root;
}
template <class C> void expint_accuracy()
{
  using R = typename C::real_type;
  auto p = Precision::bits(512);
  auto one = mpreal(1, p), eps = math::ldexp(one, 1 - C::digits()), half_subnormal = mpreal(0, p);
  if constexpr (!C::runtime) half_subnormal = widen(numeric_limits<R>::denorm_min()) / 2;
  auto evaluate = math::expint;
  std::vector<R> inputs;
  auto add = [&](R x) {
    // Extreme native inputs have known range results, tested separately. Avoid
    // asking MPFR to evaluate exp of an astronomically large native argument.
    if (math::isfinite(x) && math::abs(x) < 24000) inputs.push_back(x);
  };
  for (int k = -1024; k <= 1024; ++k)
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
  for (int n : {-1, 1, 2 * C::digits()})
    neighbors(C::real(n));
  neighbors(scalar_at_precision<R>(expint_positive_root(), C::precision()));
  constexpr int min_exp = [] {
    if constexpr (C::runtime)
      return -256;
    else
      return numeric_limits<R>::min_exponent - C::digits();
  }();
  constexpr int max_exp = [] {
    if constexpr (C::runtime)
      return 256;
    else
      return numeric_limits<R>::max_exponent;
  }();
  for (int e = min_exp; e < max_exp; e += 97)
  {
    auto x = math::ldexp(C::real(1), e);
    add(x);
    add(-x);
  }
  std::uint64_t state = 0x4569ULL;
  auto next = [&]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  for (int k = 0; k < 256; ++k)
  {
    R fraction = math::ldexp(scalar_at_precision<R>(next(), C::precision()), -64) +
                 math::ldexp(scalar_at_precision<R>(next(), C::precision()), -128);
    auto x = math::ldexp(fraction, static_cast<int>(next() % (max_exp - min_exp)) + min_exp);
    add(x);
    add(-x);
    add(fraction * C::real(2 * C::digits()));
    add(-C::real(1) - fraction * 3);
  }
  if constexpr (!C::runtime)
  {
    using L = numeric_limits<R>;
    neighbors(L::min());
    add(L::denorm_min());
    add(-L::denorm_min());
    for (int region = 0; region < 3; ++region)
    {
      bool positive = region == 2;
      auto target = region == 0 ? widen(L::min()) : half_subnormal;
      if (positive)
        target = widen(L::max()) + math::ldexp(one, L::max_exponent - C::digits() - 1);
      else
        target = -target;
      auto lo = mpreal(positive ? 1 : 2 * (L::min_exponent - C::digits()), p);
      auto hi = mpreal(positive ? 2 * L::max_exponent : -1, p);
      for (int k = 0; k < C::digits() + 24; ++k)
      {
        auto mid = (lo + hi) / 2;
        if ((math::expint(mid) < target) == positive)
          lo = mid;
        else
          hi = mid;
      }
      neighbors(scalar_at_precision<R>((lo + hi) / 2, C::precision()));
    }
  }
  auto maximum = mpreal(0, p), worst = one;
  for (auto const& x : inputs)
  {
    SCOPED_TRACE(format_scalar(x));
    auto wide = widen(x), reference = math::expint(wide);
    auto result = evaluate(x);
    C::expect_precision(result);
    auto rounded = scalar_at_precision<R>(reference, C::precision());
    if (math::isinf(rounded))
    {
      EXPECT_EQ(result, rounded);
      continue;
    }
    ASSERT_TRUE(math::isfinite(result));
    if (x < 0 && (rounded == 0 || result == 0))
    {
      EXPECT_EQ(result, rounded);
      EXPECT_EQ(math::signbit(result), math::signbit(rounded));
    }
    auto scale = abs(reference);
    if (wide > 0 && wide < 1) scale = std::max(one, scale);
    auto ratio = abs(widen(result) - reference) / (32 * eps * scale + half_subnormal);
    EXPECT_LE(ratio, 1);
    if (ratio > maximum)
    {
      maximum = ratio;
      worst = wide;
    }
  }
  testing::Test::RecordProperty("expint_max_bound_fraction", format_scalar(maximum));
  testing::Test::RecordProperty("expint_worst_input", format_scalar(worst));
  EXPECT_EQ(math::expint(worst), math::expint(worst.at(Precision::bits(1024))).at(p));
  // Independent small-argument series and precision ladder. The 400-term
  // tail at +/-1/4 is far below 2^-512; constants are evaluated at 512 bits.
  auto positive_reference = one;
  for (int sign : {-1, 1})
  {
    auto x = sign * one / 4, power = x, sum = x;
    for (int k = 2; k <= 400; ++k)
    {
      power *= x;
      power /= k;
      sum += power / k;
    }
    auto reference = euler_gamma<mpreal>.at(p) + math::log(abs(x)) + sum;
    expect_error_at_most(widen(evaluate(C::real(sign) / 4)), reference, eps * 32 * std::max(one, abs(reference)));
    if (sign == 1) positive_reference = reference;
  }
  // Independent upper-incomplete-Gamma identity on the negative real line.
  for (int n : {2, 7, 32, 100})
  {
    auto reference = -math::upper_gamma(mpreal(0, p), mpreal(n, p));
    expect_error_at_most(widen(evaluate(C::real(-n))), reference, eps * 32 * abs(reference) + half_subnormal);
  }
  auto error = abs(widen(evaluate(C::real(1) / 4)) - positive_reference);
  constexpr int bits = C::digits();
  constexpr int previous = bits <= 24    ? 12
                           : bits <= 53  ? 24
                           : bits <= 64  ? 53
                           : bits <= 113 ? 64
                           : bits <= 128 ? 113
                                         : 128;
  auto previous_error = abs(positive_reference.at(Precision::bits(previous)).at(p) - positive_reference);
  EXPECT_LT(error * 16, previous_error);
}
} // namespace
UNI20_PRECISION_TEST(NumericalScalar, ExpintAccuracy)
{
  this->RecordProperty("backend", "scalar_math");
  expint_accuracy<TypeParam>();
}
#endif
} // namespace uni20::test
