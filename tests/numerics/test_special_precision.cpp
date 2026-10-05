#include "precision_registry.hpp"

namespace uni20::test
{
UNI20_PRECISION_TEST(NumericalScalar, StandardProviderIdentities)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  auto one = C::real(1), x = one / 4 + C::gap();
  auto check = [&](auto const& result, auto const& expected) {
    C::expect_precision(result);
    expect_error_at_most(result, expected, C::epsilon() * C::real(32) * (one + math::abs(expected)));
  };
  check(math::beta(x, one), one / x);
  check(math::beta(x, C::real(2)), one / (x * (x + one)));
  check(math::zeta(C::real(-1)), -one / 12);
  check(math::zeta(C::real(0)), -one / 2);
  // Ei(x)-Ei(-x) cancels the logarithm and Euler constant. The odd series
  // converges rapidly at x near 1/4, with its tail below the tested precisions.
  auto term = x, sum = x;
  for (int k = 3; k < 101; k += 2)
  {
    term *= x * x / C::real(k * (k - 1));
    sum += term / C::real(k);
  }
  check(math::expint(x) - math::expint(-x), 2 * sum);
}
} // namespace uni20::test

#if UNI20_ENABLE_MPFR
namespace uni20::test
{
namespace
{
// Reference series at 512 bits. None calls the function whose accuracy is
// measured. Small dyadic arguments keep both input and tail errors controlled.
mpreal erf_series(mpreal const& x)
{
  auto p = x.precision();
  auto term = x, sum = x;
  for (int k = 1; k < 200; ++k)
  {
    term *= -x * x;
    term /= k;
    sum += term / (2 * k + 1);
  }
  return 2 * sum / math::sqrt(pi<mpreal>.at(p));
}
mpreal dilog_series(mpreal const& x)
{
  auto term = x, sum = x;
  for (int k = 2; k <= 300; ++k)
  {
    term *= x;
    sum += term / (k * k);
  }
  return sum;
}
mpreal bessel_j_series(int n, mpreal const& x)
{
  auto half = x / 2;
  auto term = math::pown(half, n);
  for (int k = 2; k <= n; ++k)
    term /= k;
  auto sum = term;
  for (int k = 1; k < 150; ++k)
  {
    term *= -half * half;
    term /= k * (n + k);
    sum += term;
  }
  return sum;
}
template <class C> void accuracy(typename C::real_type const& result, mpreal const& reference, int epsilon_multiple = 1)
{
  C::expect_precision(result);
  constexpr int bits = C::digits();
  constexpr int previous = bits <= 24    ? 12
                           : bits <= 53  ? 24
                           : bits <= 64  ? 53
                           : bits <= 113 ? 64
                           : bits <= 128 ? 113
                                         : 128;
  auto error = abs(widen(result) - reference);
  auto low_error = abs(reference.at(Precision::bits(previous)).at(reference.precision()) - reference);
  EXPECT_GT(error, 0);
  EXPECT_LE(error, widen(C::epsilon()) * abs(reference) * epsilon_multiple);
  EXPECT_LT(error * 16, low_error);
}
} // namespace

UNI20_PRECISION_TEST(NumericalScalar, StandardSpecialAccuracy)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  auto p = Precision::bits(512);
  auto x = mpreal("1/4", p);
  auto erf_reference = erf_series(x);
  accuracy<C>(math::erf(C::real(1) / C::real(4)), erf_reference);
  accuracy<C>(math::erfc(C::real(1) / C::real(4)), 1 - erf_reference);
  // Gamma(1/2) = sqrt(pi); no Gamma function is used by the oracle.
  accuracy<C>(math::tgamma(C::real(1) / C::real(2)), math::sqrt(pi<mpreal>.at(p)));
}

UNI20_PRECISION_TEST(NumericalScalar, StandardProviderAccuracy)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  auto p = Precision::bits(512);
  auto wide_pi = pi<mpreal>.at(p);
  auto half = C::real(1) / 2;
  accuracy<C>(math::beta(half, half), wide_pi, 32);
  accuracy<C>(math::zeta(C::real(2)), wide_pi * wide_pi / 6, 32);
  for (int sign : {-1, 1})
  {
    auto x = mpreal(sign, p) / 4;
    auto term = x, series = x;
    for (int k = 2; k < 200; ++k)
    {
      term *= x;
      term /= k;
      series += term / k;
    }
    // Ei(x) = gamma + log(abs(x)) + sum x^k/(k*k!).
    accuracy<C>(math::expint(C::real(sign) / 4), euler_gamma<mpreal>.at(p) + math::log(abs(x)) + series, 32);
  }
}

UNI20_PRECISION_TEST(NumericalScalar, MathConstantsAccuracy)
{
  using C = TypeParam;
  using R = typename C::real_type;
  this->RecordProperty("backend", "scalar_constants");
  auto value = [](auto constant) {
    if constexpr (C::runtime)
      return constant.at(Precision::bits(C::digits()));
    else
      return constant;
  };
  auto p = Precision::bits(512);
  accuracy<C>(value(pi<R>), pi<mpreal>.at(p));
  accuracy<C>(value(log_two<R>), log_two<mpreal>.at(p));
  accuracy<C>(value(euler_gamma<R>), euler_gamma<mpreal>.at(p));
}

UNI20_PRECISION_TEST(NumericalScalar, LogGammaAccuracy)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  auto p = Precision::bits(512);
  auto half_log_pi = math::log(pi<mpreal>.at(p)) / 2;
  auto positive = math::lgamma_sign(C::real(1) / 2);
  auto negative = math::lgamma_sign(-C::real(1) / 2);
  accuracy<C>(positive.value, half_log_pi, 8);
  accuracy<C>(negative.value, half_log_pi + math::log(mpreal(2, p)), 8);
  EXPECT_EQ(positive.sign, 1);
  EXPECT_EQ(negative.sign, -1);
  EXPECT_EQ(math::lgamma(C::real(1) / 2), positive.value);
}

UNI20_PRECISION_TEST(NumericalScalar, BesselAccuracy)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  auto p = Precision::bits(512);
  auto x = mpreal("1/4", p);
  auto argument = C::real(1) / 4;
  for (int n : {0, 1, 3})
  {
    // The J oracle is an independent convergent series. MPFR supplies a
    // wider reference for Y; native calls never use MPFR as their backend.
    accuracy<C>(math::bessel_j(n, argument), bessel_j_series(n, x), 16);
    accuracy<C>(math::bessel_y(n, argument), math::bessel_y(n, x), 16);
  }
}

UNI20_PRECISION_TEST(NumericalScalar, MpfrSpecialIdentities)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  auto p = Precision::bits(C::digits());
  auto x = C::real(1) / 4 + C::gap(), one = C::real(1);
  auto check = [&](mpreal const& result, mpreal const& expected) {
    C::expect_precision(result);
    expect_error_at_most(result, expected, C::epsilon() * 32 * (1 + abs(expected)));
  };
  check(math::tgamma(x + 1), x * math::tgamma(x));
  check(math::digamma(x + 1) - math::digamma(x), one / x);
  check(math::digamma(one), -euler_gamma<mpreal>.at(p));
  check(math::beta(x, one), one / x);
  check(math::beta(x, C::real(2)), one / (x * (x + 1)));
  check(math::upper_gamma(C::real(3), x), math::exp(-x) * (x * x + 2 * x + 2));
  check(math::zeta(C::real(2)), math::pown(pi<mpreal>.at(p), 2) / 6);
  check(math::lgamma(x), math::log(math::tgamma(x)));
  auto j0 = math::bessel_j(0, x), j1 = math::bessel_j(1, x);
  auto y0 = math::bessel_y(0, x), y1 = math::bessel_y(1, x);
  check(j0 * y1 - j1 * y0, -2 / (pi<mpreal>.at(p) * x));
  check(math::airy_ai(C::real(0)), one / (math::cbrt(C::real(9)) * math::tgamma(C::real(2) / 3)));
  // Tiny erfc tail survives after erf has rounded all the way to one.
  EXPECT_EQ(math::erf(C::real(16)), 1);
  EXPECT_GT(math::erfc(C::real(16)), 0);
  EXPECT_GT(math::dilog_real(C::real(2)), 0); // Real-part continuation across x=1.
}

UNI20_PRECISION_TEST(NumericalScalar, MpfrSpecialAccuracy)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  auto p = Precision::bits(512);
  auto x = mpreal("1/4", p), argument = C::real(1) / 4;
  accuracy<C>(math::dilog_real(argument), dilog_series(x));
  for (int n : {0, 1, 3})
    accuracy<C>(math::bessel_j(n, argument), bessel_j_series(n, x));
  // Ei(x) = gamma + log(x) + sum x^k/(k*k!).
  auto term = x, series = x;
  for (int k = 2; k < 200; ++k)
  {
    term *= x;
    term /= k;
    series += term / k;
  }
  accuracy<C>(math::expint(argument), euler_gamma<mpreal>.at(p) + math::log(x) + series);
  auto a = mpreal(1, p), b = x;
  for (int k = 0; k < 16; ++k)
  {
    auto next_a = (a + b) / 2;
    b = math::sqrt(a * b);
    a = next_a;
  }
  accuracy<C>(math::agm(C::real(1), argument), a);
  // Ai series coefficients obey a_(n+3)=a_n/((n+3)(n+2)).
  auto a0 = 1 / (math::cbrt(mpreal(9, p)) * math::tgamma(mpreal(2, p) / 3));
  auto a1 = -1 / (math::cbrt(mpreal(3, p)) * math::tgamma(mpreal(1, p) / 3));
  auto airy = a0 + a1 * x;
  auto power0 = mpreal(1, p), power1 = x;
  for (int n = 3; n < 150; n += 3)
  {
    a0 /= n * (n - 1);
    a1 /= (n + 1) * n;
    power0 *= x * x * x;
    power1 *= x * x * x;
    airy += a0 * power0 + a1 * power1;
  }
  accuracy<C>(math::airy_ai(argument), airy);
}

UNI20_PRECISION_TEST(NumericalScalar, MpfrExtensionsStableArithmetic)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  auto p = Precision::bits(C::digits()), high = Precision::bits(512);
  auto ln2 = log_two<mpreal>.at(high), ln10 = math::log(mpreal(10, high));
  for (int sign : {-1, 1})
  {
    auto x = C::real(sign) * C::power_of_two(-C::digits() - 8);
    EXPECT_EQ(C::real(1) + x, 1);
    // Higher terms are below 1/100 ULP of these references. A rounded
    // exp-minus-one or log-of-sum implementation loses the entire answer.
    auto check = [&](mpreal const& result, mpreal const& reference) {
      C::expect_precision(result);
      EXPECT_GT(abs(result), 0);
      EXPECT_LE(abs(result.at(high) - reference), C::epsilon().at(high) * abs(reference) * 2);
    };
    check(math::exp2m1(x), x.at(high) * ln2);
    check(math::exp10m1(x), x.at(high) * ln10);
    check(math::log2p1(x), x.at(high) / ln2);
    check(math::log10p1(x), x.at(high) / ln10);
  }
  auto large = C::power_of_two(C::digits() - 4);
  EXPECT_EQ(math::sinpi(large), 0);
  EXPECT_EQ(math::cospi(large + C::real(1) / 2), 0);
  EXPECT_EQ(math::tanpi(large + C::real(1) / 4), 1);
  EXPECT_GT(abs(math::sin(large * pi<mpreal>.at(p))), C::epsilon());
  auto x = C::real(1) / 4 + C::gap();
  expect_error_at_most(math::sinpi(math::asinpi(x)), x, C::epsilon() * 4);
  expect_error_at_most(math::cospi(math::acospi(x)), x, C::epsilon() * 4);
  expect_error_at_most(math::tanpi(math::atanpi(x)), x, C::epsilon() * 4);
  expect_error_at_most(math::tanpi(math::atan2pi(x, C::real(1))), x, C::epsilon() * 4);
  expect_error_at_most(math::exp10(math::log10(x)), x, C::epsilon() * 4);
  expect_error_at_most(math::sec(x) * math::cos(x), C::real(1), C::epsilon() * 4);
  expect_error_at_most(math::csc(x) * math::sin(x), C::real(1), C::epsilon() * 4);
  expect_error_at_most(math::cot(x) * math::tan(x), C::real(1), C::epsilon() * 4);
  expect_error_at_most(math::sech(x) * math::cosh(x), C::real(1), C::epsilon() * 4);
  expect_error_at_most(math::csch(x) * math::sinh(x), C::real(1), C::epsilon() * 4);
  expect_error_at_most(math::coth(x) * math::tanh(x), C::real(1), C::epsilon() * 4);
  auto tiny = C::power_of_two(-C::digits() - 2);
  EXPECT_EQ(C::real(1) + tiny, 1);
  auto compound = math::compound(tiny, 16);
  C::expect_precision(compound);
  EXPECT_EQ(compound, 1 + 16 * tiny);
}
} // namespace uni20::test
#endif
