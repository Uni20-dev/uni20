#include "precision_registry.hpp"

namespace uni20::test
{
UNI20_PRECISION_TEST(NumericalScalar, EpsilonAndRetainedIncrement)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar");
  using R = typename C::real_type;
  R const one = C::real(1), eps = C::epsilon();
  // Negative control: these increments disappear when narrowed. The same
  // narrowing in a tested kernel must therefore fail its precision probe.
  if constexpr (C::digits() > 53)
  {
    EXPECT_EQ(static_cast<double>(one + C::gap()), 1.0);
  }
  else if constexpr (C::digits() > 24)
  {
    EXPECT_EQ(static_cast<float>(one + C::gap()), 1.0f);
  }
  EXPECT_EQ(eps, C::power_of_two(1 - C::digits()));
  EXPECT_GT(one + eps, one);
  EXPECT_EQ(one + eps / C::real(4), one);
  auto z = C::scalar(one + C::gap(), -one);
  auto difference = z - C::scalar(1, -1);
  expect_equal(difference, C::scalar(C::gap(), C::real(0)));
  C::expect_precision(z);
}

#if UNI20_ENABLE_MPFR
UNI20_PRECISION_TEST(NumericalScalar, ReciprocalAccuracyImprovesWithPrecision)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar");
  // Identical rational problem at every precision, with an analytic reference
  // evaluated at 512 bits and no input rounded through a native literal.
  auto result = C::scalar(1) / C::scalar(3, 2);
  if constexpr (C::is_complex)
  {
    expect_rational_accuracy<C>(result.real(), 3, 13);
    expect_rational_accuracy<C>(result.imag(), -2, 13);
  }
  else
    expect_rational_accuracy<C>(result, 1, 3);
  C::expect_precision(result);
}
#endif

UNI20_PRECISION_TEST(NumericalScalar, MathDispatchRetainsWorkingPrecision)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  using R = typename C::real_type;
  using S = typename C::scalar_type;
  R const one = C::real(1), zero = C::real(0), gap = C::gap();
  S const expected = C::scalar(one + gap, C::is_complex ? one / C::real(2) : zero);
  auto root = uni20::math::sqrt(S(expected * expected));
  static_assert(std::same_as<decltype(root), S>);
  expect_error_at_most(root, expected, C::real(4) * C::epsilon());
  C::expect_precision(root);
  auto magnitude = uni20::math::abs(C::scalar(one + gap, zero));
  static_assert(std::same_as<decltype(magnitude), R>);
  expect_equal(magnitude, R(one + gap));
  C::expect_precision(magnitude);
  // These exact identities cover the other shared transcendental dispatches.
  expect_equal(uni20::math::exp(C::scalar(0)), C::scalar(1));
  expect_equal(uni20::math::sin(C::scalar(0)), C::scalar(0));
  expect_equal(uni20::math::cos(C::scalar(0)), C::scalar(1));
  expect_error_at_most(uni20::math::pow(C::scalar(2), C::scalar(3)), C::scalar(8), C::real(32) * C::epsilon());
  expect_equal(uni20::math::log2(C::real(8)), C::real(3));
  if constexpr (!C::runtime)
  {
    expect_equal(uni20::math::ceil(C::real(3) / C::real(2)), C::real(2));
    expect_equal(uni20::math::ldexp(one + gap, -3), R((one + gap) / C::real(8)));
  }
}

UNI20_PRECISION_TEST(NumericalScalar, ElementarySmallArguments)
{
  using C = TypeParam;
  using R = typename C::real_type;
  this->RecordProperty("backend", "scalar_math");
  R const one = C::real(1);
  for (int sign : {-1, 1})
  {
    R const x = C::real(sign) * C::power_of_two(-C::digits() - 2);
    // At this scale both true results round to x. The naive compositions
    // lose the entire result, even when evaluated at the tested precision.
    expect_equal(R(one + x), one);
    expect_equal(R(math::exp(x) - one), C::real(0));
    auto exponential = math::expm1(x);
    auto logarithm = math::log1p(x);
    static_assert(std::same_as<decltype(exponential), R>);
    static_assert(std::same_as<decltype(logarithm), R>);
    expect_equal(exponential, x);
    expect_equal(logarithm, x);
    C::expect_precision(exponential);
    C::expect_precision(logarithm);
  }
}

UNI20_PRECISION_TEST(NumericalScalar, ElementaryRetainsWorkingPrecision)
{
  using C = TypeParam;
  using R = typename C::real_type;
  this->RecordProperty("backend", "scalar_math");
  R const one = C::real(1), two = C::real(2);
  R const x = one / two + C::gap(), tolerance = C::epsilon() * C::real(8);
  auto check = [&](auto result, R expected) {
    static_assert(std::same_as<decltype(result), R>);
    expect_error_at_most(result, expected, tolerance);
    C::expect_precision(result);
  };
  // The retained increment exceeds the error allowance; evaluating these
  // identities after narrowing the input cannot pass at higher precisions.
  check(math::cbrt(R(-x * x * x)), R(-x));
  check(math::exp2(math::log2(x)), x);
  check(math::pow(C::real(10), math::log10(x)), x);
  check(math::sin(math::asin(x)), x);
  check(math::cos(math::acos(x)), x);
  check(math::tan(math::atan(x)), x);
  check(math::sinh(math::asinh(x)), x);
  check(math::cosh(math::acosh(R(one + x))), R(one + x));
  check(math::tanh(math::atanh(x)), x);
  check(math::tan(math::atan2(x, one)), x);
  check(math::hypot(R(C::real(3) * x), R(C::real(4) * x)), R(C::real(5) * x));
}

#if UNI20_ENABLE_MPFR
UNI20_PRECISION_TEST(NumericalScalar, ElementaryAccuracyImprovesWithPrecision)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  auto reference_precision = Precision::bits(512);
  auto x = mpreal(1, reference_precision) / 4;
  mpreal exponential(0, reference_precision), logarithm(0, reference_precision);
  mpreal exponential_term(1, reference_precision), power(1, reference_precision);
  // Independent series at x=1/4. After 300 terms the logarithm tail is
  // below 2^-600; 512-bit accumulation error is negligible for these probes.
  for (int k = 1; k <= 300; ++k)
  {
    exponential_term *= x;
    exponential_term /= k;
    exponential += exponential_term;
    power *= x;
    logarithm += (k % 2 == 1 ? power : -power) / k;
  }
  constexpr int bits = C::digits();
  constexpr int previous = bits <= 24    ? 12
                           : bits <= 53  ? 24
                           : bits <= 64  ? 53
                           : bits <= 113 ? 64
                           : bits <= 128 ? 113
                                         : 128;
  auto argument = C::real(1) / C::real(4);
  auto check = [&](auto result, mpreal const& reference) {
    C::expect_precision(result);
    auto error = abs(widen(result) - reference);
    auto lower_error = abs(reference.at(Precision::bits(previous)).at(reference_precision) - reference);
    EXPECT_GT(error, 0);
    EXPECT_LE(error, widen(C::epsilon()) * abs(reference));
    EXPECT_LT(error * 16, lower_error);
  };
  check(math::expm1(argument), exponential);
  check(math::log1p(argument), logarithm);
}
#endif

UNI20_PRECISION_TEST(NumericalScalar, UtilitiesRetainWorkingPrecision)
{
  using C = TypeParam;
  using R = typename C::real_type;
  this->RecordProperty("backend", "scalar_math");
  auto zero = C::real(0), one = C::real(1), two = C::real(2);
  R x = one + C::gap();
  auto binary = math::frexp(x);
  static_assert(std::same_as<decltype(binary.fraction), R>);
  expect_equal(math::ldexp(binary.fraction, binary.exponent), x);
  C::expect_precision(binary.fraction);
  auto parts = math::modf(x);
  expect_equal(parts.fraction, C::gap());
  expect_equal(parts.integer, one);
  C::expect_precision(parts.fraction);
  auto remainder = math::remquo(x, one);
  expect_equal(remainder.remainder, C::gap());
  EXPECT_EQ(remainder.quotient, 1);
  expect_equal(math::fmod(x, one), C::gap());
  expect_equal(math::remainder(x, one), C::gap());
  expect_equal(math::fdim(x, one), C::gap());
  expect_equal(math::fmin(x, one), one);
  expect_equal(math::fmax(x, one), x);
  expect_equal(math::round_even(R(C::real(5) / two)), two);
  expect_equal(math::round(R(C::real(5) / two)), C::real(3));
  expect_equal(math::floor(x), one);
  expect_equal(math::ceil(x), two);
  expect_equal(math::trunc(R(-x)), R(-one));
  expect_equal(math::scalbn(x, -3), R(x / C::real(8)));
  expect_equal(math::next_up(one), R(one + C::epsilon()));
  expect_equal(math::next_down(one), R(one - C::epsilon() / two));
  expect_equal(math::nextafter(one, x), R(one + C::epsilon()));
  expect_equal(math::copysign(x, R(-one)), R(-x));
  EXPECT_EQ(math::ilogb(x), 0);
  EXPECT_TRUE(math::isfinite(x));
  EXPECT_FALSE(math::isnan(x));
  EXPECT_FALSE(math::isinf(x));
  EXPECT_TRUE(math::signbit(R(-zero)));
  EXPECT_TRUE(math::isinf(R(one / zero)));
  EXPECT_TRUE(math::isnan(R(zero / zero)));
  auto trig = math::sincos(x);
  expect_equal(trig.sin, math::sin(x));
  expect_equal(trig.cos, math::cos(x));
  auto hyperbolic = math::sinhcosh(x);
  expect_equal(hyperbolic.sinh, math::sinh(x));
  expect_equal(hyperbolic.cosh, math::cosh(x));
  expect_error_at_most(math::rootn(math::pown(x, 3), 3), x, C::real(8) * C::epsilon());
}

UNI20_PRECISION_TEST(NumericalScalar, FusedArithmeticRoundsOnce)
{
  using C = TypeParam;
  using R = typename C::real_type;
  this->RecordProperty("backend", "scalar_math");
  auto one = C::real(1);
  auto gap = C::power_of_two(-(C::digits() + 2) / 2);
  R a = one + gap, b = one - gap;
  R product = a * b;
  expect_equal(R(product - one), C::real(0));
  auto fused = math::fma(a, b, R(-one));
  static_assert(std::same_as<decltype(fused), R>);
  expect_equal(fused, R(-gap * gap));
  C::expect_precision(fused);
}
UNI20_PRECISION_TEST(NumericalScalar, UtilitiesBoundarySemantics)
{
  using C = TypeParam;
  using R = typename C::real_type;
  this->RecordProperty("backend", "scalar_math");
  R zero = C::real(0), one = C::real(1), two = C::real(2), nz = -zero;
  R inf = one / zero, nan = zero / zero;
  EXPECT_TRUE(math::signbit(math::round_even(R(-one / two))));
  EXPECT_TRUE(math::signbit(math::round(nz)));
  EXPECT_TRUE(math::signbit(math::floor(nz)));
  EXPECT_TRUE(math::signbit(math::ceil(nz)));
  EXPECT_TRUE(math::signbit(math::trunc(nz)));
  EXPECT_TRUE(math::signbit(math::modf(nz).fraction));
  EXPECT_TRUE(math::signbit(math::modf(nz).integer));
  EXPECT_TRUE(math::signbit(math::frexp(nz).fraction));
  EXPECT_EQ(math::frexp(nz).exponent, 0);
  for (auto a : {zero, nz})
    for (auto b : {zero, nz})
    {
      EXPECT_EQ(math::signbit(math::fmin(a, b)), math::signbit(a) || math::signbit(b));
      EXPECT_EQ(math::signbit(math::fmax(a, b)), math::signbit(a) && math::signbit(b));
      EXPECT_EQ(math::signbit(math::nextafter(a, b)), math::signbit(b));
    }
  expect_equal(math::fmin(nan, one), one);
  expect_equal(math::fmax(one, nan), one);
  EXPECT_TRUE(math::isnan(math::fdim(one, nan)));
  expect_equal(math::fdim(inf, inf), zero);
  EXPECT_TRUE(math::isnan(math::fma(zero, inf, one)));
  auto remainder = math::remquo(C::real(-23), two);
  expect_equal(remainder.remainder, one);
  EXPECT_EQ(remainder.quotient, -4);
  EXPECT_TRUE(math::signbit(math::remainder(C::real(-2), one)));
  EXPECT_TRUE(math::signbit(math::fmod(C::real(-2), one)));
  EXPECT_TRUE(math::isnan(math::remquo(inf, one).remainder));
  EXPECT_EQ(math::remquo(inf, one).quotient, 0);
  expect_equal(math::next_up(inf), inf);
  expect_equal(math::next_down(R(-inf)), R(-inf));
  EXPECT_TRUE(math::isfinite(math::next_down(inf)));
  expect_equal(math::next_up(math::next_down(inf)), inf);
  EXPECT_GT(math::next_up(zero), zero);
  EXPECT_LT(math::next_down(zero), zero);
  EXPECT_TRUE(math::isfinite(math::next_up(zero)));
  EXPECT_EQ(math::frexp(inf).exponent, 0);
  expect_equal(math::frexp(inf).fraction, inf);
  EXPECT_TRUE(math::signbit(math::modf(R(-inf)).fraction));
  EXPECT_THROW(math::ilogb(zero), std::domain_error);
  EXPECT_THROW(math::ilogb(inf), std::domain_error);
  EXPECT_THROW(math::ilogb(nan), std::domain_error);
  EXPECT_TRUE(math::isnan(math::rootn(one, 0)));
  EXPECT_TRUE(math::isnan(math::rootn(R(-one), 2)));
  expect_equal(math::rootn(C::real(-8), -3), R(-one / two));
  expect_equal(math::pown(R(-one), numeric_limits<std::int64_t>::min()), one);
  expect_equal(math::rootn(R(-one), numeric_limits<std::int64_t>::max()), R(-one));
}

#if UNI20_ENABLE_MPFR
UNI20_PRECISION_TEST(NumericalScalar, UtilityAccuracyImprovesWithPrecision)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  auto ref = Precision::bits(512);
  // Rational power is analytic. The fifth-root oracle is Newton iteration,
  // independent of rootn/pow providers, from an exactly represented input.
  mpreal power_reference("16384/78125", ref); // (5/4)^-7
  mpreal root_reference(1, ref);
  for (int i = 0; i < 30; ++i)
  {
    auto square = root_reference * root_reference;
    root_reference = (4 * root_reference + mpreal(2, ref) / (square * square)) / 5;
  }
  constexpr int bits = C::digits();
  constexpr int previous = bits <= 24 ? 12 : bits <= 53 ? 24 : bits <= 64 ? 53 :
                           bits <= 113 ? 64 : bits <= 128 ? 113 : 128;
  auto check = [&](auto value, mpreal const& reference) {
    C::expect_precision(value);
    auto error = abs(widen(value) - reference);
    auto lower_error = abs(reference.at(Precision::bits(previous)).at(ref) - reference);
    EXPECT_LE(error, abs(reference) * widen(C::epsilon()) * 8);
    EXPECT_LT(error * 16, lower_error);
  };
  check(math::pown(typename C::real_type(C::real(5) / C::real(4)), -7), power_reference);
  check(math::rootn(C::real(2), 5), root_reference);
}
#endif

} // namespace uni20::test
