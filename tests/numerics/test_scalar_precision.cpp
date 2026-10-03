#include "precision_cases.hpp"

namespace uni20::test
{
template <class C> using NumericalScalar = PrecisionTest<C>;
TYPED_TEST_SUITE(NumericalScalar, PrecisionCases, PrecisionCaseNames);

TYPED_TEST(NumericalScalar, EpsilonAndRetainedIncrement)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar");
  if constexpr (!C::available)
    GTEST_SKIP() << "unavailable: scalar dependency or native format not configured";
  else
  {
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
}

TYPED_TEST(NumericalScalar, ReciprocalAccuracyImprovesWithPrecision)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar");
  if constexpr (!C::available) GTEST_SKIP() << "unavailable: scalar dependency or native format not configured";
#if !UNI20_ENABLE_MPFR
  else
    GTEST_SKIP() << "unavailable: MPFR required for independent 512-bit error measurement";
#else
  else
  {
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
}

TYPED_TEST(NumericalScalar, MathDispatchRetainsWorkingPrecision)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  if constexpr (!C::available)
    GTEST_SKIP() << "unavailable: scalar dependency or native format not configured";
  else
  {
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
    if constexpr (!C::runtime)
    {
      expect_equal(uni20::math::log2(C::real(8)), C::real(3));
      expect_equal(uni20::math::ceil(C::real(3) / C::real(2)), C::real(2));
      expect_equal(uni20::math::ldexp(one + gap, -3), R((one + gap) / C::real(8)));
    }
  }
}
} // namespace uni20::test
