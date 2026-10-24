#include "precision_registry.hpp"
#include <cstdint>
#include <vector>

namespace uni20::test
{
UNI20_PRECISION_TEST(NumericalScalar, AiryIdentities)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  auto one = C::real(1), zero = C::real(0);
  auto value = math::airy_ai(zero);
  C::expect_precision(value);
  auto reference = one / (math::cbrt(C::real(9)) * math::tgamma(C::real(2) / 3));
  expect_error_at_most(value, reference, C::epsilon() * 16);
  // Differential equation check with balanced truncation and roundoff costs.
  auto h = math::ldexp(one, -C::digits() / 4);
  auto middle = math::airy_ai(one);
  auto derivative = (math::airy_ai(one + h) - 2 * middle + math::airy_ai(one - h)) / (h * h);
  expect_error_at_most(derivative, middle, 64 * math::sqrt(C::epsilon()));
}
#if UNI20_HAS_MPREAL
namespace
{
mpreal airy_reference(mpreal const& input)
{
  if (abs(input) <= 128) return math::airy_ai(input);
  auto p = precision_of(input), work = Precision::bits(p.bit_count() + 64);
  auto x = input.at(work), v = abs(x), one = mpreal(1, work);
  auto zeta = 2 * v * math::sqrt(v) / 3, term = one, even = one, odd = mpreal(0, work);
  bool positive = x > 0, done = false;
  auto threshold = math::ldexp(one, -p.bit_count() - 40);
  // For |x|>128, the first omitted asymptotic term reaches this tolerance
  // well before the least term, for the 512/1024-bit reference precisions.
  for (std::int64_t k = 1; k < 8 * work.bit_count(); ++k)
  {
    term *= (6 * k - 5) * (6 * k - 1);
    term /= 72 * k;
    term /= zeta;
    if (positive)
      even += k % 2 ? -term : term;
    else if (k % 2)
      odd += (k / 2) % 2 ? -term : term;
    else
      even += (k / 2) % 2 ? -term : term;
    if (term < threshold)
    {
      done = true;
      break;
    }
  }
  EXPECT_TRUE(done);
  auto amplitude = one / (math::sqrt(pi<mpreal>.at(work)) * math::sqrt(math::sqrt(v)));
  if (positive) return (math::exp(-zeta) * amplitude * even / 2).at(p);
  // Keep the fractional phase before conversion back to the reference result
  // precision. A lower-precision sin override would first round the input.
  auto phase_precision = Precision::bits(work.bit_count() + 3 * math::frexp(v).exponent / 2);
  auto wide = v.at(phase_precision), wide_pi = pi<mpreal>.at(phase_precision);
  auto phase = 2 * wide * math::sqrt(wide) / 3 + wide_pi / 4;
  auto angle = math::remainder(phase, 2 * wide_pi).at(work);
  return (amplitude * (math::sin(angle) * even - math::cos(angle) * odd)).at(p);
}

template <class C> void airy_accuracy()
{
  using R = typename C::real_type;
  auto p = Precision::bits(512);
  auto zero = mpreal(0, p), one = mpreal(1, p), pi_ref = pi<mpreal>.at(p);
  auto eps = math::ldexp(one, 1 - C::digits()), half_subnormal = zero;
  if constexpr (!C::runtime) half_subnormal = widen(numeric_limits<R>::denorm_min()) / 2;
  auto maximum = zero, worst = zero;
  auto check = [&](R const& x) {
    SCOPED_TRACE(format_scalar(x));
    auto wide = widen(x), reference = airy_reference(wide);
    auto value = math::airy_ai(x);
    C::expect_precision(value);
    ASSERT_TRUE(math::isfinite(value));
    if (x > 0)
    {
      R rounded = scalar_at_precision<R>(reference, C::precision());
      if (value == 0 || rounded == 0)
      {
        EXPECT_EQ(value, rounded);
        EXPECT_FALSE(math::signbit(value));
        return;
      }
    }
    auto scale = x <= 0 ? one / (math::sqrt(pi_ref) * math::sqrt(math::sqrt(one - wide)))
                        : std::max(one, 2 * wide * math::sqrt(wide) / 3) * reference;
    auto bound = 64 * eps * scale;
    if (x > 0) bound += half_subnormal;
    auto ratio = abs(widen(value) - reference) / bound;
    EXPECT_LE(ratio, 1);
    if (ratio > maximum)
    {
      maximum = ratio;
      worst = wide;
    }
  };
  // Cross-check the independent asymptotic reference where MPFR Ai remains
  // practical; do not use direct mpfr_ai as the huge-argument oracle.
  for (int n : {-256, -129, 129, 256})
  {
    auto x = mpreal(n, p), direct = math::airy_ai(x);
    expect_error_at_most(airy_reference(x), direct, abs(direct) * math::ldexp(one, -500));
  }
  for (int k = -128; k <= 128; ++k)
    check(C::real(k) / 4);
  for (int center : {-20, -14, -12, -8, 8, 12, 14, 20})
  {
    auto lo = C::real(center), hi = lo;
    for (int k = 0; k < 4; ++k)
    {
      check(lo);
      check(hi);
      lo = math::next_down(lo);
      hi = math::next_up(hi);
    }
  }
  // Locate the first zero independently, then probe adjacent native values.
  auto lo = mpreal(-3, p), hi = mpreal(-2, p);
  for (int k = 0; k < C::digits() + 16; ++k)
  {
    auto mid = (lo + hi) / 2;
    if (math::airy_ai(mid) < 0)
      lo = mid;
    else
      hi = mid;
  }
  auto root = scalar_at_precision<R>((lo + hi) / 2, C::precision());
  for (int k = 0; k < 4; ++k)
    root = math::next_down(root);
  for (int k = 0; k < 8; ++k)
  {
    check(root);
    root = math::next_up(root);
  }
  if constexpr (!C::runtime)
  {
    using L = numeric_limits<R>;
    check(L::denorm_min());
    check(-L::denorm_min());
    check(-L::max());
    for (int exponent : {C::digits(), 2 * C::digits(), L::max_exponent / 2, L::max_exponent - 2})
      for (R mantissa : {R(.5), R(.625), R(.875)})
      {
        R x = -math::ldexp(mantissa, exponent);
        check(x);
        check(math::next_down(x));
      }
    lo = mpreal(1, p);
    hi = mpreal(2000, p);
    for (int k = 0; k < C::digits() + 16; ++k)
    {
      auto mid = (lo + hi) / 2;
      if (airy_reference(mid) > half_subnormal)
        lo = mid;
      else
        hi = mid;
    }
    R below = scalar_at_precision<R>((lo + hi) / 2, C::precision()), above = below;
    for (int k = 0; k < 16; ++k)
    {
      check(below);
      check(above);
      below = math::next_down(below);
      above = math::next_up(above);
    }
  }
  testing::Test::RecordProperty("airy_max_bound_fraction", format_scalar(maximum));
  testing::Test::RecordProperty("airy_worst_input", format_scalar(worst));
  auto reference = airy_reference(worst), wider = airy_reference(worst.at(Precision::bits(1024))).at(p);
  expect_error_at_most(reference, wider, abs(wider) * math::ldexp(one, -500));
  reference = math::airy_ai(zero);
  auto error = abs(widen(math::airy_ai(C::real(0))) - reference);
  constexpr int bits = C::digits();
  constexpr int previous = bits <= 24    ? 12
                           : bits <= 53  ? 24
                           : bits <= 64  ? 53
                           : bits <= 113 ? 64
                           : bits <= 128 ? 113
                                         : 128;
  auto previous_error = abs(reference.at(Precision::bits(previous)).at(p) - reference);
  EXPECT_LT(error * 8, previous_error);
}
} // namespace
UNI20_PRECISION_TEST(NumericalScalar, AiryAccuracy)
{
  this->RecordProperty("backend", "scalar_math");
  airy_accuracy<TypeParam>();
}
#endif
} // namespace uni20::test
