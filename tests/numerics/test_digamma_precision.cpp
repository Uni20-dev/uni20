#include "precision_registry.hpp"
#include <cstdint>
#include <vector>

namespace uni20::test
{
UNI20_PRECISION_TEST(NumericalScalar, DigammaIdentities)
{
  using C = TypeParam;
  using R = typename C::real_type;
  this->RecordProperty("backend", "scalar_math");
  auto x = C::real(3) / 4, result = math::digamma(x);
  C::expect_precision(result);
  expect_error_at_most(math::digamma(x + 1) - result, C::real(1) / x, C::epsilon() * 16);
  auto value = [](auto constant) {
    if constexpr (C::runtime)
      return constant.at(C::precision());
    else
      return constant;
  };
  expect_error_at_most(math::digamma(C::real(1)), -value(euler_gamma<R>), C::epsilon() * 8);
  expect_error_at_most(math::digamma(C::real(1) / 2), -value(euler_gamma<R>) - 2 * value(log_two<R>),
                       C::epsilon() * 16);
}
#if UNI20_HAS_MPREAL
namespace
{
std::vector<mpreal> const& digamma_roots()
{
  static auto const roots = [] {
    auto p = Precision::bits(512);
    std::vector<mpreal> result;
    for (int n : {-1, 0, 1, 15, 999})
    {
      auto lo = mpreal(n == -1 ? 1 : -n - 1, p), hi = mpreal(n == -1 ? 2 : -n, p);
      for (int k = 0; k < 500; ++k)
      {
        auto mid = (lo + hi) / 2;
        if (math::digamma(mid) < 0)
          lo = mid;
        else
          hi = mid;
      }
      result.push_back((lo + hi) / 2);
    }
    return result;
  }();
  return roots;
}
template <class C> void digamma_accuracy()
{
  using R = typename C::real_type;
  auto p = Precision::bits(512);
  std::vector<R> inputs;
  for (int i = -1024; i <= 1024; ++i)
    inputs.push_back(C::real(i) / 97);
  auto neighbors = [&](R center) {
    R up = center, down = center;
    for (int i = 0; i < 16; ++i)
    {
      inputs.push_back(up);
      inputs.push_back(down);
      up = math::next_up(up);
      down = math::next_down(down);
    }
  };
  for (int n : {-1024, -16, -2, -1, 0, 1, 2, 8, 16, 32})
    neighbors(C::real(n));
  neighbors(C::epsilon());
  neighbors(-C::epsilon());
  for (auto const& root : digamma_roots())
    neighbors(scalar_at_precision<R>(root, C::precision()));
  if constexpr (!C::runtime)
  {
    using L = numeric_limits<R>;
    neighbors(R(1) / L::max());
    neighbors(-R(1) / L::max());
    for (int e = L::min_exponent - C::digits(); e < L::max_exponent; e += 127)
    {
      R x = math::ldexp(R(1), e);
      inputs.push_back(x);
      inputs.push_back(-x);
    }
    inputs.push_back(L::max());
    inputs.push_back(-L::max());
  }
  std::uint64_t state = 0x446967616d6d61ULL;
  auto next = [&]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  constexpr int min_exp = [] {
    if constexpr (C::runtime)
      return -256;
    else
      return numeric_limits<R>::min_exponent;
  }();
  constexpr int max_exp = [] {
    if constexpr (C::runtime)
      return 256;
    else
      return numeric_limits<R>::max_exponent;
  }();
  for (int i = 0; i < 512; ++i)
  {
    R fraction = math::ldexp(scalar_at_precision<R>(next(), C::precision()), -64) +
                 math::ldexp(scalar_at_precision<R>(next(), C::precision()), -128);
    int e = static_cast<int>(next() % (max_exp - min_exp + C::digits())) + min_exp - C::digits();
    R x = math::ldexp(fraction, e);
    inputs.push_back(x);
    inputs.push_back(-x);
  }
  auto maximum = mpreal(0, p), worst = mpreal(1, p), one = mpreal(1, p);
  for (auto const& x : inputs)
  {
    SCOPED_TRACE(format_scalar(x));
    auto wide = widen(x), reference = math::digamma(wide);
    auto result = math::digamma(x);
    C::expect_precision(result);
    R rounded = scalar_at_precision<R>(reference, C::precision());
    if (math::isnan(rounded))
    {
      EXPECT_TRUE(math::isnan(result));
      continue;
    }
    if (math::isinf(rounded))
    {
      EXPECT_TRUE(math::isinf(result));
      EXPECT_EQ(math::signbit(result), math::signbit(rounded));
      continue;
    }
    ASSERT_TRUE(math::isfinite(result));
    auto scale = std::max(one, abs(reference));
    if (x < 0) scale = std::max(scale, math::log1p(-wide));
    auto error = abs(widen(result) - reference) / (scale * math::ldexp(one, 1 - C::digits()));
    EXPECT_LE(error, 8);
    if (error > maximum)
    {
      maximum = error;
      worst = wide;
    }
  }
  testing::Test::RecordProperty("digamma_max_scaled_error", format_scalar(maximum));
  testing::Test::RecordProperty("digamma_worst_input", format_scalar(worst));
  EXPECT_EQ(math::digamma(worst), math::digamma(worst.at(Precision::bits(1024))).at(p));
  // Independent integer and half-integer identities, including finite harmonic sums.
  auto gamma = euler_gamma<mpreal>.at(p), ln2 = log_two<mpreal>.at(p);
  for (int n : {1, 2, 7, 31})
  {
    auto integer_reference = -gamma, half_reference = -gamma - 2 * ln2;
    for (int k = 1; k < n; ++k)
      integer_reference += one / k;
    for (int k = 0; k < n; ++k)
      half_reference += 2 * one / (2 * k + 1);
    expect_error_at_most(widen(math::digamma(C::real(n))), integer_reference,
                         math::ldexp(one, 1 - C::digits()) * 8 * std::max(one, abs(integer_reference)));
    expect_error_at_most(widen(math::digamma(C::real(n) + C::real(1) / 2)), half_reference,
                         math::ldexp(one, 1 - C::digits()) * 8 * std::max(one, abs(half_reference)));
  }
  auto reference = -gamma - 2 * ln2;
  auto error = abs(widen(math::digamma(C::real(1) / 2)) - reference);
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
UNI20_PRECISION_TEST(NumericalScalar, DigammaAccuracy)
{
  this->RecordProperty("backend", "scalar_math");
  digamma_accuracy<TypeParam>();
}
#endif
} // namespace uni20::test
