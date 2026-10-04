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
} // namespace uni20::test
