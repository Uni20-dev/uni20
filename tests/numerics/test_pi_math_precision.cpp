#include "precision_registry.hpp"

#include <uni20/common/gtest.hpp>

namespace uni20::test
{
UNI20_PRECISION_TEST(NumericalScalar, PiScaledIdentities)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  auto one = C::real(1), x = one / 8 + C::gap();
  auto check = [&](auto const& actual, auto const& expected) {
    C::expect_precision(actual);
    expect_error_at_most(actual, expected, C::epsilon() * C::real(8));
  };
  check(math::sinpi(x) * math::sinpi(x) + math::cospi(x) * math::cospi(x), one);
  check(math::sinpi(-x), -math::sinpi(x));
  check(math::cospi(-x), math::cospi(x));
  check(math::tanpi(-x), -math::tanpi(x));
  check(math::asinpi(math::sinpi(x)), x);
  check(math::acospi(math::cospi(x)), x);
  check(math::atanpi(math::tanpi(x)), x);
  check(math::atan2pi(math::sinpi(x), math::cospi(x)), x);
  // These quarter/half-integer offsets survive at every tested precision.
  // Multiplying the original large argument by pi would lose their phase.
  auto huge = C::power_of_two(C::digits() - 3);
  expect_equal(math::sinpi(huge + one / 2), one);
  expect_equal(math::cospi(huge + one), -one);
  expect_equal(math::tanpi(huge + one / 4), one);
}

#if UNI20_HAS_MPREAL
namespace
{
struct PublicPiMath
{
    static constexpr auto sinpi = math::sinpi;
    static constexpr auto cospi = math::cospi;
    static constexpr auto tanpi = math::tanpi;
    static constexpr auto asinpi = math::asinpi;
    static constexpr auto acospi = math::acospi;
    static constexpr auto atanpi = math::atanpi;
    static constexpr auto atan2pi = math::atan2pi;
};

template <class C> void oracle_check(typename C::real_type const& result, mpreal const& reference)
{
  C::expect_precision(result);
  if (isnan(reference))
  {
    EXPECT_TRUE(math::isnan(result));
    return;
  }
  if (isinf(reference))
  {
    EXPECT_TRUE(math::isinf(result));
    EXPECT_EQ(math::signbit(result), signbit(reference));
    return;
  }
  ASSERT_TRUE(math::isfinite(result));
  if (reference == 0)
  {
    EXPECT_EQ(result, 0);
    EXPECT_EQ(math::signbit(result), signbit(reference));
    return;
  }
  auto error = abs(widen(result) - reference);
  // Count output-grid units, including native subnormals. This compares with
  // the unrounded 512-bit oracle, rather than rounding the oracle down first.
  auto magnitude = math::abs(result);
  auto spacing = widen(math::next_up(magnitude)) - widen(magnitude);
  EXPECT_LE(error, 4 * spacing);
}

template <class C, class F> void sample_accuracy()
{
  using R = typename C::real_type;
  auto one = C::real(1);
  auto check = [&](R const& x) {
    SCOPED_TRACE(format_scalar(x));
    auto wide = widen(x);
    oracle_check<C>(F::sinpi(x), math::sinpi(wide));
    oracle_check<C>(F::cospi(x), math::cospi(wide));
    auto tangent_reference = math::tanpi(wide);
    // MPFR 4.2.1 gives tanpi an alternating zero sign at odd integers.
    // Native IEEE tanPi uses the input sign; its zero contract is checked
    // independently in the core tests. Keep MPFR's own provider behavior.
    if constexpr (!C::runtime)
      if (tangent_reference == 0) tangent_reference = math::copysign(mpreal(0, wide.precision()), wide);
    oracle_check<C>(F::tanpi(x), tangent_reference);
    oracle_check<C>(F::atanpi(x), math::atanpi(wide));
    if (math::abs(x) <= one)
    {
      oracle_check<C>(F::asinpi(x), math::asinpi(wide));
      oracle_check<C>(F::acospi(x), math::acospi(wide));
    }
    for (R denominator : {one, -one, C::real(3), -C::real(3)})
      oracle_check<C>(F::atan2pi(x, denominator), math::atan2pi(wide, widen(denominator)));
  };
  // A deterministic non-dyadic grid exercises ordinary arguments; neighbors
  // of every quarter integer exercise reduction ties, zeros and poles.
  for (int i = -128; i <= 128; ++i)
    check(C::real(i) / 37);
  for (int i = -16; i <= 16; ++i)
  {
    R x = C::real(i) / 4;
    check(x);
    // MPFR's neighbor of zero is near its enormous exponent limit; the native
    // subnormal tests below cover zero-adjacent values without that allocation.
    if (i != 0)
    {
      check(math::next_up(x));
      check(math::next_down(x));
    }
  }
  for (int exponent : {C::digits() / 2, C::digits() - 3})
  {
    auto huge = C::power_of_two(exponent);
    for (int i = -8; i <= 8; ++i)
    {
      check(huge + C::real(i) / 4);
      check(-huge + C::real(i) / 4);
    }
  }
  if constexpr (!C::runtime)
  {
    for (R unit : {numeric_limits<R>::denorm_min(), numeric_limits<R>::min()})
      for (int n : {-31, -7, -2, -1, 1, 2, 7, 31})
        check(unit * R(n));
    // Extended exponent range, independent of significand retention.
    check(numeric_limits<R>::max());
    check(-numeric_limits<R>::max());
  }
}

template <class C, class F> void accuracy_improvement()
{
  auto p = Precision::bits(512);
  auto one = C::real(1);
  auto two = mpreal(2, p), root_two = math::sqrt(two);
  auto quarter = mpreal(1, p) / 4;
  auto term = quarter, atan_series = quarter;
  for (int k = 1; k < 180; ++k)
  {
    term *= -quarter * quarter;
    atan_series += term / (2 * k + 1);
  }
  atan_series /= pi<mpreal>.at(p);
  auto check = [&](auto const& actual, mpreal const& reference) {
    oracle_check<C>(actual, reference);
    constexpr int bits = C::digits();
    constexpr int previous = bits <= 24    ? 12
                             : bits <= 53  ? 24
                             : bits <= 64  ? 53
                             : bits <= 113 ? 64
                             : bits <= 128 ? 113
                                           : 128;
    auto error = abs(widen(actual) - reference);
    auto lower_error = abs(reference.at(Precision::bits(previous)).at(p) - reference);
    EXPECT_LT(error * 16, lower_error);
  };
  // Independent analytic identities and an atan series, rather than calls
  // back to the function being measured, establish precision improvement.
  check(F::sinpi(one / 8), math::sqrt(two - root_two) / 2);
  check(F::cospi(one / 8), math::sqrt(two + root_two) / 2);
  check(F::tanpi(one / 8), root_two - 1);
  check(F::asinpi(one / 2), mpreal(1, p) / 6);
  check(F::acospi(one / 2), mpreal(1, p) / 3);
  check(F::atanpi(one / 4), atan_series);
  check(F::atan2pi(one / 4, one), atan_series);
}
} // namespace

UNI20_PRECISION_TEST(NumericalScalar, PiScaledAccuracy)
{
  this->RecordProperty("backend", "scalar_math");
  sample_accuracy<TypeParam, PublicPiMath>();
  accuracy_improvement<TypeParam, PublicPiMath>();
}
UNI20_PRECISION_TEST(NumericalScalar, PiScaledPortableAccuracy)
{
  this->RecordProperty("backend", "portable_native_math");
  using F = detail::native_math::portable_pi_math<typename TypeParam::real_type>;
  sample_accuracy<TypeParam, F>();
  accuracy_improvement<TypeParam, F>();
}
#endif
} // namespace uni20::test
