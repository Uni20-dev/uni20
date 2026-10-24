#include "precision_registry.hpp"
#include <array>
#include <cstdint>
#include <vector>

namespace uni20::test
{
UNI20_PRECISION_TEST(NumericalScalar, ReciprocalFunctionIdentities)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  auto x = C::real(1) / 4 + C::gap(), one = C::real(1);
  for (auto value : {math::sec(x), math::csc(x), math::cot(x), math::sech(x), math::csch(x), math::coth(x)})
    C::expect_precision(value);
  expect_error_at_most(math::sec(x) * math::cos(x), one, C::epsilon() * 4);
  expect_error_at_most(math::csc(x) * math::sin(x), one, C::epsilon() * 4);
  expect_error_at_most(math::cot(x) * math::tan(x), one, C::epsilon() * 4);
  expect_error_at_most(math::sech(x) * math::cosh(x), one, C::epsilon() * 4);
  expect_error_at_most(math::csch(x) * math::sinh(x), one, C::epsilon() * 4);
  expect_error_at_most(math::coth(x) * math::tanh(x), one, C::epsilon() * 4);
}

#if UNI20_HAS_MPREAL
namespace
{
constexpr std::array names{"sec", "csc", "cot", "sech", "csch", "coth"};
template <class R> R reciprocal(int operation, R const& x)
{
  switch (operation)
  {
    case 0:
      return math::sec(x);
    case 1:
      return math::csc(x);
    case 2:
      return math::cot(x);
    case 3:
      return math::sech(x);
    case 4:
      return math::csch(x);
    default:
      return math::coth(x);
  }
}
template <class C> mpreal reciprocal_error(typename C::real_type const& result, mpreal const& reference)
{
  using R = typename C::real_type;
  auto p = reference.precision();
  C::expect_precision(result);
  R rounded = scalar_at_precision<R>(reference, C::precision());
  if (math::isinf(rounded))
  {
    EXPECT_TRUE(math::isinf(result));
    EXPECT_EQ(math::signbit(result), math::signbit(rounded));
    return mpreal(0, p);
  }
  EXPECT_TRUE(math::isfinite(result));
  if (!math::isfinite(result)) return mpreal(100, p);
  EXPECT_EQ(math::signbit(result), math::signbit(rounded));
  if (rounded == 0)
  {
    EXPECT_EQ(result, 0);
  }
  if (reference == 0) return abs(widen(result));
  int exponent = static_cast<int>(math::frexp(reference).exponent);
  if constexpr (!C::runtime) exponent = std::max(exponent, numeric_limits<R>::min_exponent);
  auto error = abs(widen(result) - reference) / math::ldexp(mpreal(1, p), exponent - C::digits());
  EXPECT_LE(error, 4);
  return error;
}

template <class C> void reciprocal_accuracy()
{
  using R = typename C::real_type;
  auto p = Precision::bits(512);
  auto one = mpreal(1, p);
  std::vector<R> inputs;
  for (int i = -512; i <= 512; ++i)
    inputs.push_back(C::real(i) / 29);
  auto add_neighbors = [&](R const& center) {
    R up = center, down = center;
    for (int i = 0; i < 32; ++i)
    {
      inputs.push_back(up);
      inputs.push_back(down);
      up = math::next_up(up);
      down = math::next_down(down);
    }
  };
  for (auto center : {C::real(-1), C::real(0), C::real(1), C::real(C::digits()),
                      scalar_at_precision<R>(pi<mpreal>.at(p) / 2, C::precision()),
                      scalar_at_precision<R>(pi<mpreal>.at(p), C::precision())})
    add_neighbors(center);
  if constexpr (!C::runtime)
  {
    using L = numeric_limits<R>;
    for (int e = L::min_exponent - C::digits(); e < L::max_exponent; e += 127)
    {
      inputs.push_back(math::ldexp(C::real(1), e));
      inputs.push_back(-math::ldexp(C::real(1), e));
    }
    for (R center : {R(-L::min_exponent) * log_two<R>, R(L::max_exponent) * log_two<R>,
                     R(C::digits() - L::min_exponent + 2) * log_two<R>})
    {
      add_neighbors(center);
      add_neighbors(-center);
    }
    inputs.push_back(L::max());
    inputs.push_back(-L::max());
  }
  // Regression where a half-exponential formula near x=1 exceeded the target
  // for binary80. The selected direct-provider region avoids that amplification.
  inputs.push_back(
      scalar_at_precision<R>(mpreal("-1.00000000000000000477048955893621950963279232", p), C::precision()));
  std::uint64_t state = 0x5265636970726f63ULL;
  auto next = [&]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  for (int i = 0; i < 512; ++i)
  {
    R fraction = math::ldexp(scalar_at_precision<R>(next(), C::precision()), -64) +
                 math::ldexp(scalar_at_precision<R>(next(), C::precision()), -128);
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
    int exponent = static_cast<int>(next() % (max_exp - min_exp + C::digits())) + min_exp - C::digits();
    R x = math::ldexp(fraction, exponent);
    inputs.push_back(x);
    inputs.push_back(-x);
  }
  for (int operation = 0; operation < 6; ++operation)
  {
    auto maximum = mpreal(0, p), worst = mpreal(0, p);
    for (auto const& x : inputs)
    {
      SCOPED_TRACE(names[operation]);
      SCOPED_TRACE(format_scalar(x));
      auto reference = reciprocal(operation, widen(x));
      auto error = reciprocal_error<C>(reciprocal(operation, x), reference);
      if (error > maximum)
      {
        maximum = error;
        worst = widen(x);
      }
    }
    testing::Test::RecordProperty(std::string(names[operation]) + "_max_grid_error", format_scalar(maximum));
    testing::Test::RecordProperty(std::string(names[operation]) + "_worst_input", format_scalar(worst));
  }
  // Independent power series for the six denominators at x=1/4.
  auto x = one / 4, xx = x * x, sin_term = x, sinh_term = x, cos_term = one, cosh_term = one;
  auto sine = x, hyperbolic_sine = x, cosine = one, hyperbolic_cosine = one;
  for (int k = 1; k < 120; ++k)
  {
    sin_term *= -xx / ((2 * k) * (2 * k + 1));
    sine += sin_term;
    sinh_term *= xx / ((2 * k) * (2 * k + 1));
    hyperbolic_sine += sinh_term;
    cos_term *= -xx / ((2 * k - 1) * (2 * k));
    cosine += cos_term;
    cosh_term *= xx / ((2 * k - 1) * (2 * k));
    hyperbolic_cosine += cosh_term;
  }
  std::array references{one / cosine,          one / sine,
                        cosine / sine,         one / hyperbolic_cosine,
                        one / hyperbolic_sine, hyperbolic_cosine / hyperbolic_sine};
  constexpr int bits = C::digits();
  constexpr int previous = bits <= 24    ? 12
                           : bits <= 53  ? 24
                           : bits <= 64  ? 53
                           : bits <= 113 ? 64
                           : bits <= 128 ? 113
                                         : 128;
  for (int operation = 0; operation < 6; ++operation)
  {
    SCOPED_TRACE(names[operation]);
    auto result = reciprocal(operation, C::real(1) / 4);
    auto const& reference = references[operation];
    reciprocal_error<C>(result, reference);
    auto error = abs(widen(result) - reference);
    auto previous_error = abs(reference.at(Precision::bits(previous)).at(p) - reference);
    EXPECT_LT(error * 16, previous_error);
  }
}
} // namespace
UNI20_PRECISION_TEST(NumericalScalar, ReciprocalFunctionAccuracy)
{
  this->RecordProperty("backend", "scalar_math");
  reciprocal_accuracy<TypeParam>();
}
#endif
} // namespace uni20::test
