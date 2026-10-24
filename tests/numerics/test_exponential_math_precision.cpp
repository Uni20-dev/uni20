#include "precision_registry.hpp"

#include <array>
#include <cstdint>
#include <vector>

namespace uni20::test
{
UNI20_PRECISION_TEST(NumericalScalar, ExponentialExtensionIdentities)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  auto one = C::real(1), x = one / 8 + C::gap();
  for (auto result : {math::exp10(x), math::exp2m1(x), math::exp10m1(x), math::log2p1(x), math::log10p1(x)})
    C::expect_precision(result);
  expect_error_at_most(math::log2p1(math::exp2m1(x)), x, C::epsilon() * 8);
  expect_error_at_most(math::log10p1(math::exp10m1(x)), x, C::epsilon() * 8);
  auto tiny = math::ldexp(one, -C::digits() - 8);
  expect_equal(one + tiny, one);
  EXPECT_GT(math::exp2m1(tiny), C::real(0));
  EXPECT_GT(math::exp10m1(tiny), C::real(0));
  EXPECT_GT(math::log2p1(tiny), C::real(0));
  EXPECT_GT(math::log10p1(tiny), C::real(0));
  // The leading small-x terms retain the tested precision without subtracting
  // nearly equal numbers in either the result or the reference.
  auto bound = C::epsilon() * tiny * 4;
  expect_error_at_most(math::exp2m1(tiny), tiny * log_two<typename C::real_type>, bound);
  expect_error_at_most(math::log2p1(tiny), tiny / log_two<typename C::real_type>, bound);
}

#if UNI20_HAS_MPREAL
namespace
{
struct PublicExponentialMath
{
    static constexpr auto exp10 = math::exp10;
    static constexpr auto exp2m1 = math::exp2m1;
    static constexpr auto exp10m1 = math::exp10m1;
    static constexpr auto log2p1 = math::log2p1;
    static constexpr auto log10p1 = math::log10p1;
};
constexpr std::array names{"exp10", "exp2m1", "exp10m1", "log2p1", "log10p1"};
template <class F, class R> R evaluate(int operation, R const& x)
{
  switch (operation)
  {
    case 0:
      return F::exp10(x);
    case 1:
      return F::exp2m1(x);
    case 2:
      return F::exp10m1(x);
    case 3:
      return F::log2p1(x);
    default:
      return F::log10p1(x);
  }
}

template <class C> mpreal grid_error(typename C::real_type const& result, mpreal const& reference)
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
  if (rounded == 0)
  {
    EXPECT_EQ(result, 0);
    EXPECT_EQ(math::signbit(result), math::signbit(rounded));
  }
  if (reference == 0) return abs(widen(result));
  // Measure spacing from the reference exponent. Unlike next_up(max), this
  // remains finite at the largest native number and clamps subnormal spacing.
  int exponent = static_cast<int>(math::frexp(reference).exponent);
  if constexpr (!C::runtime) exponent = std::max(exponent, numeric_limits<R>::min_exponent);
  auto spacing = math::ldexp(mpreal(1, p), exponent - C::digits());
  auto error = abs(widen(result) - reference) / spacing;
  EXPECT_LE(error, 4);
  return error;
}

template <class C, class F> void sample_accuracy()
{
  using R = typename C::real_type;
  auto one = C::real(1);
  auto p = Precision::bits(512);
  std::vector<R> inputs;
  for (int i = -256; i <= 256; ++i)
    inputs.push_back(C::real(i) / 97);
  for (R center : {one / 4, -one / 4, one / 2, -one / 2, one, -one})
  {
    inputs.push_back(center);
    auto up = center, down = center;
    for (int i = 0; i < 32; ++i)
    {
      up = math::next_up(up);
      down = math::next_down(down);
      inputs.push_back(up);
      inputs.push_back(down);
    }
  }
  // Regression inputs from the wider prototype scan, transferred without a
  // double intermediate. These are binary80 inputs, not high-precision constants.
  inputs.push_back(scalar_at_precision<R>(0.3229116324144490426137181493526639997071L, C::precision()));
  inputs.push_back(scalar_at_precision<R>(0.293864621518922183613254511813650537988L, C::precision()));
  std::uint64_t state = 0x4a6f792d6d617468ULL;
  auto next = [&]() {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
  };
  for (int i = 0; i < 1024; ++i)
  {
    R x = math::ldexp(scalar_at_precision<R>(next(), C::precision()), -64) +
          math::ldexp(scalar_at_precision<R>(next(), C::precision()), -128);
    inputs.push_back(x);
    inputs.push_back(-x);
  }
  if constexpr (!C::runtime)
  {
    using L = numeric_limits<R>;
    for (int exponent = L::min_exponent - C::digits(); exponent < L::max_exponent; exponent += 127)
    {
      R x = math::ldexp(one, exponent);
      inputs.push_back(x);
      inputs.push_back(-x);
    }
    for (R unit : {L::min(), L::denorm_min()})
      for (int n : {-31, -7, -2, -1, 1, 2, 7, 31})
        inputs.push_back(unit * R(n));
    // Locate both decimal range boundaries with typed constants, then examine
    // enough adjacent inputs to include any rounding in the boundary estimate.
    for (R center : {R(L::max_exponent) * log_two<R> / std::numbers::ln10_v<R>,
                     R(L::min_exponent - C::digits() - 1) * log_two<R> / std::numbers::ln10_v<R>, R(L::max_exponent)})
    {
      auto up = center, down = center;
      for (int i = 0; i < 32; ++i)
      {
        inputs.push_back(up);
        inputs.push_back(down);
        up = math::next_up(up);
        down = math::next_down(down);
      }
    }
    inputs.push_back(L::max());
  }
  for (int op = 0; op < 5; ++op)
  {
    auto maximum = mpreal(0, p), worst = mpreal(0, p);
    for (R const& x : inputs)
    {
      if (op >= 3 && x <= -one) continue; // Core tests cover the domain endpoints.
      // Exponential arguments beyond this already overflow/underflow all native
      // types. Avoid asking MPFR to represent an astronomically scaled output.
      if (op <= 2 && math::abs(x) > C::real(20000)) continue;
      SCOPED_TRACE(names[op]);
      SCOPED_TRACE(format_scalar(x));
      auto wide = widen(x), reference = evaluate<PublicExponentialMath>(op, wide);
      auto error = grid_error<C>(evaluate<F>(op, x), reference);
      if (error > maximum)
      {
        maximum = error;
        worst = wide;
      }
    }
    testing::Test::RecordProperty(std::string(names[op]) + "_max_grid_error", format_scalar(maximum));
    testing::Test::RecordProperty(std::string(names[op]) + "_worst_input", format_scalar(worst));
  }
}

mpreal log_ratio(int numerator, int denominator)
{
  auto p = Precision::bits(512);
  auto z = mpreal(numerator - denominator, p) / (numerator + denominator);
  auto term = z, result = z;
  for (int k = 1; k < 180; ++k)
  {
    term *= z * z;
    result += term / (2 * k + 1);
  }
  return 2 * result;
}
template <class C, class F> void accuracy_improvement()
{
  auto p = Precision::bits(512);
  auto ln2 = log_ratio(2, 1), ln10 = log_ratio(5, 4) + 3 * ln2;
  auto binary = math::sqrt(math::sqrt(math::sqrt(mpreal(2, p))));
  auto decimal = math::sqrt(math::sqrt(math::sqrt(mpreal(10, p))));
  std::array<mpreal, 5> references{decimal, binary - 1, decimal - 1, log_ratio(9, 8) / ln2, log_ratio(9, 8) / ln10};
  constexpr int bits = C::digits();
  constexpr int previous = bits <= 24    ? 12
                           : bits <= 53  ? 24
                           : bits <= 64  ? 53
                           : bits <= 113 ? 64
                           : bits <= 128 ? 113
                                         : 128;
  for (int op = 0; op < 5; ++op)
  {
    SCOPED_TRACE(names[op]);
    auto result = evaluate<F>(op, C::real(1) / 8);
    grid_error<C>(result, references[op]);
    auto error = abs(widen(result) - references[op]);
    auto previous_error = abs(references[op].at(Precision::bits(previous)).at(p) - references[op]);
    EXPECT_LT(error * 16, previous_error);
  }
}
} // namespace

UNI20_PRECISION_TEST(NumericalScalar, ExponentialExtensionAccuracy)
{
  this->RecordProperty("backend", "scalar_math");
  sample_accuracy<TypeParam, PublicExponentialMath>();
  accuracy_improvement<TypeParam, PublicExponentialMath>();
}
UNI20_PRECISION_TEST(NumericalScalar, ExponentialExtensionPortableAccuracy)
{
  this->RecordProperty("backend", "portable_native_math");
  using F = detail::native_math::portable_exponential_math<typename TypeParam::real_type>;
  sample_accuracy<TypeParam, F>();
  accuracy_improvement<TypeParam, F>();
}
#endif
} // namespace uni20::test
