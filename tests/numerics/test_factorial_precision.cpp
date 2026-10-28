#include "precision_registry.hpp"

namespace uni20::test
{
UNI20_PRECISION_TEST(NumericalScalar, FactorialIdentities)
{
  using C = TypeParam;
  using R = typename C::real_type;
  this->RecordProperty("backend", "scalar_math");
  auto result = uni20::factorial<R>(10, C::precision());
  C::expect_precision(result);
  EXPECT_EQ(result, C::real(3628800));
  EXPECT_EQ(uni20::factorial<R>(0, C::precision()), C::real(1));
  EXPECT_EQ(uni20::factorial<R>(1, C::precision()), C::real(1));
  auto previous = uni20::factorial<R>(15, C::precision());
  auto next = uni20::factorial<R>(16, C::precision());
  expect_error_at_most(next / 16, previous, math::abs(previous) * C::epsilon() * 2);
}
#if UNI20_HAS_MPREAL
UNI20_PRECISION_TEST(NumericalScalar, FactorialAccuracy)
{
  using C = TypeParam;
  using R = typename C::real_type;
  this->RecordProperty("backend", "scalar_math");
  auto p = Precision::bits(512);
  constexpr int count = [] {
    if constexpr (C::runtime)
      return 4 * C::digits();
    else if constexpr (numeric_limits<R>::max_exponent == 128)
      return 35;
    else if constexpr (numeric_limits<R>::max_exponent == 1024)
      return 171;
    else
      return 1755;
  }();
  exact_rational integer{1};
  auto maximum = mpreal(0, p);
  int worst = 0;
  for (int n = 0; n <= count; ++n)
  {
    SCOPED_TRACE(n);
    if (n > 1) integer = integer * exact_rational(n);
    auto reference = mpreal(integer, p);
    // Convert the full exact integer directly at the target precision; do not
    // infer correct rounding from an intermediate fixed-width oracle.
    auto rounded = scalar_at_precision<R>(mpreal(integer), C::precision());
    auto result = uni20::factorial<R>(n, C::precision());
    C::expect_precision(result);
    if (math::isinf(rounded))
    {
      EXPECT_EQ(result, rounded);
      break;
    }
    ASSERT_TRUE(math::isfinite(result));
    auto exponent = math::frexp(reference).exponent;
    auto error = math::abs(widen(result) - reference) / math::ldexp(mpreal(1, p), exponent - C::digits());
    EXPECT_LE(error, 1);
    if (error > maximum)
    {
      maximum = error;
      worst = n;
    }
    auto numerator = mpq_numref(integer.native_handle());
    auto significant_bits = mpz_sizeinbase(numerator, 2) - mpz_scan1(numerator, 0);
    if (significant_bits <= C::digits())
    {
      EXPECT_EQ(result, rounded);
    }
  }
  this->RecordProperty("factorial_max_grid_error", format_scalar(maximum));
  this->RecordProperty("factorial_worst_order", worst);
  // Inexact at every requested precision; use an independently accumulated
  // exact integer, also for the lower-precision comparison.
  constexpr int n = C::digits() / 4 + 16;
  integer = exact_rational{1};
  for (int k = 2; k <= n; ++k)
    integer = integer * exact_rational(k);
  auto reference = mpreal(integer, p);
  auto error = math::abs(widen(uni20::factorial<R>(n, C::precision())) - reference);
  constexpr int bits = C::digits();
  constexpr int previous = bits <= 24    ? 12
                           : bits <= 53  ? 24
                           : bits <= 64  ? 53
                           : bits <= 113 ? 64
                           : bits <= 128 ? 113
                                         : 128;
  auto previous_error = math::abs(mpreal(integer, Precision::bits(previous)).at(p) - reference);
  EXPECT_LT(error * 16, previous_error);
}
#endif
} // namespace uni20::test
