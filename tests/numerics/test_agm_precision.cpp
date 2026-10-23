#include "precision_registry.hpp"
#include <array>
#include <cstdint>
#include <vector>

namespace uni20::test
{
UNI20_PRECISION_TEST(NumericalScalar, AgmIdentities)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  auto a = C::real(1), b = a / 2, result = math::agm(a, b);
  C::expect_precision(result);
  EXPECT_EQ(result, math::agm(b, a));
  EXPECT_EQ(math::agm(a, a), a);
  EXPECT_EQ(math::agm(a, C::real(0)), 0);
  expect_error_at_most(math::agm(a * 8, b * 8), result * 8, C::epsilon() * 16);
  expect_error_at_most(math::agm((a + b) / 2, math::sqrt(a * b)), result, C::epsilon() * 8);
  EXPECT_GT(result, b);
  EXPECT_LT(result, a);
}

#if UNI20_HAS_MPREAL
namespace
{
template <class C> mpreal agm_error(typename C::real_type const& result, mpreal const& reference)
{
  using R = typename C::real_type;
  C::expect_precision(result);
  EXPECT_TRUE(math::isfinite(result));
  if (!math::isfinite(result)) return mpreal(100, reference.precision());
  EXPECT_GT(result, 0);
  auto exponent = math::frexp(reference).exponent;
  if constexpr (!C::runtime) exponent = std::max<std::int64_t>(exponent, numeric_limits<R>::min_exponent);
  auto error = abs(widen(result) - reference) / math::ldexp(mpreal(1, reference.precision()), exponent - C::digits());
  EXPECT_LE(error, 8);
  return error;
}
template <class C> void agm_accuracy()
{
  using R = typename C::real_type;
  auto p = Precision::bits(512);
  std::vector<std::pair<R, R>> inputs;
  for (int i = 1; i <= 16; ++i)
    for (int j = 1; j <= 16; ++j)
      inputs.emplace_back(C::real(i) / 13, C::real(j) / 17);
  for (int e = -4; e <= 4; ++e)
  {
    R a = math::ldexp(C::real(1), e), b = a;
    for (int i = 0; i < 32; ++i)
    {
      inputs.emplace_back(a, b);
      a = math::next_up(a);
      b = math::next_down(b);
    }
  }
  if constexpr (!C::runtime)
  {
    using L = numeric_limits<R>;
    for (int e = L::min_exponent - C::digits(); e < L::max_exponent; e += 127)
    {
      R a = math::ldexp(C::real(1), e);
      for (R b : {L::denorm_min(), L::min(), R(1), L::max()})
        inputs.emplace_back(a, b);
    }
    for (int i = 1; i <= 32; ++i)
      for (int j = 1; j <= 16; ++j)
        inputs.emplace_back(L::denorm_min() * i, L::denorm_min() * j);
  }
  // Both sides of the separate-scale/common-scale join.
  for (int gap = C::digits(); gap <= C::digits() + 4; ++gap)
  {
    R b = math::ldexp(C::real(1), -gap);
    for (R a : {C::real(1), C::real(3) / 2})
      for (R small : {b, math::next_up(b), math::next_down(b)})
        inputs.emplace_back(a, small);
  }
  std::uint64_t state = 0x41676d5363616eULL;
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
  auto draw = [&]() {
    R fraction = math::ldexp(scalar_at_precision<R>(next(), C::precision()), -64) +
                 math::ldexp(scalar_at_precision<R>(next(), C::precision()), -128);
    int e = static_cast<int>(next() % (max_exp - min_exp + C::digits())) + min_exp - C::digits();
    return math::ldexp(fraction, e);
  };
  for (int i = 0; i < 2048; ++i)
  {
    // Do not depend on the evaluation order of function arguments.
    R a = draw(), b = draw();
    if (a != 0 && b != 0) inputs.emplace_back(a, b);
  }
  auto maximum = mpreal(0, p), worst_a = mpreal(1, p), worst_b = worst_a;
  for (auto const& [a, b] : inputs)
  {
    SCOPED_TRACE(format_scalar(a));
    SCOPED_TRACE(format_scalar(b));
    auto reference = math::agm(widen(a), widen(b));
    auto result = math::agm(a, b);
    EXPECT_GE(result, std::min(a, b));
    EXPECT_LE(result, std::max(a, b));
    EXPECT_EQ(result, math::agm(b, a));
    auto error = agm_error<C>(result, reference);
    if (error > maximum)
    {
      maximum = error;
      worst_a = widen(a);
      worst_b = widen(b);
    }
  }
  testing::Test::RecordProperty("agm_max_grid_error", format_scalar(maximum));
  testing::Test::RecordProperty("agm_worst_a", format_scalar(worst_a));
  testing::Test::RecordProperty("agm_worst_b", format_scalar(worst_b));
  auto wider = Precision::bits(1024);
  EXPECT_EQ(math::agm(worst_a, worst_b), math::agm(worst_a.at(wider), worst_b.at(wider)).at(p));

  // K(sqrt(3/4)) = pi/2 * sum [(binomial(2n,n)/4^n)^2 * (3/4)^n].
  // Hence AGM(1,1/2)=1/sum. The omitted tail after 2000 terms is
  // bounded by 4*(3/4)^2000 < 2^-828; compute at 1024 bits first.
  auto term = mpreal(1, wider), sum = term;
  for (int n = 1; n < 2000; ++n)
  {
    auto ratio = mpreal(2 * n - 1, wider) / (2 * n);
    term *= ratio * ratio * (mpreal(3, wider) / 4);
    sum += term;
  }
  auto reference = (mpreal(1, wider) / sum).at(p);
  expect_error_at_most(reference, math::agm(mpreal(1, p), mpreal("0.5", p)), math::ldexp(mpreal(1, p), -500));
  auto result = math::agm(C::real(1), C::real(1) / 2);
  agm_error<C>(result, reference);
  constexpr int bits = C::digits();
  constexpr int previous = bits <= 24    ? 12
                           : bits <= 53  ? 24
                           : bits <= 64  ? 53
                           : bits <= 113 ? 64
                           : bits <= 128 ? 113
                                         : 128;
  auto error = abs(widen(result) - reference);
  auto previous_error = abs(reference.at(Precision::bits(previous)).at(p) - reference);
  EXPECT_LT(error * 16, previous_error);
}
} // namespace
UNI20_PRECISION_TEST(NumericalScalar, AgmAccuracy)
{
  this->RecordProperty("backend", "scalar_math");
  agm_accuracy<TypeParam>();
}
#endif
} // namespace uni20::test
