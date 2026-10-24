#include "precision_registry.hpp"
#include <cstdint>
#include <vector>

namespace uni20::test
{
UNI20_PRECISION_TEST(NumericalScalar, DilogIdentities)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  auto x = C::real(1) / 4 + C::gap();
  auto result = math::dilog_real(x);
  C::expect_precision(result);
  // The duplication identity also exercises the negative real continuation.
  expect_error_at_most(result + math::dilog_real(-x), math::dilog_real(x * x) / 2, C::epsilon() * 8);
  EXPECT_GT(math::dilog_real(C::real(2)), 0);
  EXPECT_LT(math::dilog_real(C::real(13)), 0);
}
#if UNI20_HAS_MPREAL
namespace
{
mpreal const& dilog_positive_root()
{
  static auto const root = [] {
    auto p = Precision::bits(512);
    auto lo = mpreal(12, p), hi = mpreal(13, p);
    for (int k = 0; k < 510; ++k)
    {
      auto mid = (lo + hi) / 2;
      if (math::dilog_real(mid) > 0)
        lo = mid;
      else
        hi = mid;
    }
    return (lo + hi) / 2;
  }();
  return root;
}
template <class C> void dilog_accuracy()
{
  using R = typename C::real_type;
  auto p = Precision::bits(512);
  auto one = mpreal(1, p);
  auto eps = math::ldexp(one, 1 - C::digits()), half_subnormal = mpreal(0, p);
  if constexpr (!C::runtime) half_subnormal = widen(numeric_limits<R>::denorm_min()) / 2;
  std::vector<R> inputs;
  for (int k = -1024; k <= 1024; ++k)
    inputs.push_back(C::real(k) / 61);
  auto neighbors = [&](R center) {
    R lo = center, hi = center;
    for (int k = 0; k < 16; ++k)
    {
      inputs.push_back(lo);
      inputs.push_back(hi);
      lo = math::next_down(lo);
      hi = math::next_up(hi);
    }
  };
  // MPFR nextbelow(0) uses its ambient exponent floor, not a useful test scale.
  for (R x : {C::real(-1), C::real(1) / 2, C::real(1), C::real(2)})
    neighbors(x);
  neighbors(scalar_at_precision<R>(dilog_positive_root(), C::precision()));
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
  for (int e = min_exp; e < max_exp; e += 31)
  {
    auto x = math::ldexp(C::real(1), e);
    inputs.push_back(x);
    inputs.push_back(-x);
  }
  if constexpr (!C::runtime)
  {
    neighbors(numeric_limits<R>::min());
    inputs.push_back(numeric_limits<R>::max());
    inputs.push_back(-numeric_limits<R>::max());
    inputs.push_back(numeric_limits<R>::denorm_min());
    inputs.push_back(-numeric_limits<R>::denorm_min());
  }
  std::uint64_t state = 0x4c6932ULL;
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
    int exponent = static_cast<int>(next() % (max_exp - min_exp)) + min_exp;
    auto x = math::ldexp(fraction, exponent);
    inputs.push_back(x);
    inputs.push_back(-x);
  }
  auto maximum = mpreal(0, p), worst = one;
  for (auto const& x : inputs)
  {
    SCOPED_TRACE(format_scalar(x));
    auto wide = widen(x), reference = math::dilog_real(wide);
    auto result = math::dilog_real(x);
    C::expect_precision(result);
    ASSERT_TRUE(math::isfinite(result));
    auto scale = abs(reference);
    if (abs(wide) > one / 2) scale = std::max(one, scale);
    if (x == 0)
    {
      EXPECT_EQ(result, 0);
      EXPECT_EQ(math::signbit(result), math::signbit(x));
      continue;
    }
    auto ratio = abs(widen(result) - reference) / (16 * eps * scale + half_subnormal);
    EXPECT_LE(ratio, 1);
    if (ratio > maximum)
    {
      maximum = ratio;
      worst = wide;
    }
  }
  testing::Test::RecordProperty("dilog_max_bound_fraction", format_scalar(maximum));
  testing::Test::RecordProperty("dilog_worst_input", format_scalar(worst));
  EXPECT_EQ(math::dilog_real(worst), math::dilog_real(worst.at(Precision::bits(1024))).at(p));
  // Independent constants, not calls to the function being tested.
  auto pi_value = pi<mpreal>.at(p), ln2 = log_two<mpreal>.at(p);
  auto sixth = pi_value * pi_value / 6;
  for (auto const& pair : std::vector<std::pair<R, mpreal>>{{C::real(-1), -sixth / 2},
                                                            {C::real(1) / 2, sixth / 2 - ln2 * ln2 / 2},
                                                            {C::real(1), sixth},
                                                            {C::real(2), 3 * sixth / 2}})
    expect_error_at_most(widen(math::dilog_real(pair.first)), pair.second, eps * 16 * abs(pair.second));
  // Defining series at +/-1/4, with tail < 2^-800; no continuation formula.
  auto positive_reference = one;
  for (int sign : {-1, 1})
  {
    auto x = sign * one / 4, power = x, sum = x;
    for (int k = 2; k <= 400; ++k)
    {
      power *= x;
      sum += power / (k * k);
    }
    expect_error_at_most(widen(math::dilog_real(C::real(sign) / 4)), sum, eps * 16 * abs(sum));
    if (sign == 1) positive_reference = sum;
  }
  auto error = abs(widen(math::dilog_real(C::real(1) / 4)) - positive_reference);
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
UNI20_PRECISION_TEST(NumericalScalar, DilogAccuracy)
{
  this->RecordProperty("backend", "scalar_math");
  dilog_accuracy<TypeParam>();
}
#endif
} // namespace uni20::test
