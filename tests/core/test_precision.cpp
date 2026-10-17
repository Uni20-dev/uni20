#include <uni20/core/runtime_precision.hpp>
#if UNI20_ENABLE_MPFR
#include <uni20/core/mpreal.hpp>
#endif

#include <gtest/gtest.h>
#include <limits>
#include <string>

namespace
{
using namespace uni20;

static_assert(Precision::bits(53).bit_count() == 53);
static_assert(Precision::decimal_digits(80).bit_count() == 266);
static_assert(precision_of(1.0) == Precision::bits(53));
static_assert(precision_of(1) == Precision::exact());
static_assert(native_precision<complex<double>>() == Precision::bits(53));
static_assert(numeric_limits<double>::digits(precision_of(1.0)) == 53);
static_assert(numeric_limits<double>::digits10(1.0) == 15);
static_assert(numeric_limits<double>::max_digits10(precision_of(1.0)) == 17);
static_assert(numeric_limits<double>::epsilon(precision_of(1.0)) == numeric_limits<double>::epsilon());
static_assert(scalar_at_precision<double>(3, precision_of(1.0)) == 3.0);

template <class T>
concept HasNativePrecision = requires { native_precision<T>(); };
template <class T>
concept ConvertsToDouble = requires(T const& value) { scalar_at_precision<double>(value, Precision::bits(53)); };
static_assert(!HasNativePrecision<std::string>);
static_assert(!ConvertsToDouble<Precision>);

TEST(PrecisionTest, ConcretePrecisionAndInference)
{
  auto p = Precision::bits(128);
  EXPECT_EQ(common_precision(Precision::exact(), p), p);
  EXPECT_EQ(common_precision(p, Precision::exact()), p);
  EXPECT_EQ(common_precision(p, p), p);
  EXPECT_TRUE(common_precision(Precision::exact(), Precision::exact()).is_exact());
  EXPECT_THROW(common_precision(p, Precision::bits(256)), std::invalid_argument);
  EXPECT_THROW(Precision::bits(0), std::invalid_argument);
  EXPECT_THROW(Precision::bits(-1), std::invalid_argument);
  EXPECT_THROW(Precision::decimal_digits(0), std::invalid_argument);
  EXPECT_THROW(Precision::decimal_digits(std::numeric_limits<std::int64_t>::max()), std::invalid_argument);
  EXPECT_THROW(Precision::exact().bit_count(), std::logic_error);
}

TEST(PrecisionTest, FixedScalarRequiresMatchingPrecision)
{
  for (auto p : {Precision::exact(), Precision::bits(24), Precision::bits(128)})
  {
    EXPECT_THROW((void)scalar_at_precision<double>(3, p), std::invalid_argument);
    EXPECT_THROW(numeric_limits<double>::digits(p), std::invalid_argument);
    EXPECT_THROW(numeric_limits<double>::digits10(p), std::invalid_argument);
    EXPECT_THROW(numeric_limits<double>::max_digits10(p), std::invalid_argument);
    EXPECT_THROW(numeric_limits<double>::epsilon(p), std::invalid_argument);
  }
  EXPECT_EQ(scalar_at_precision<int>(3, Precision::exact()), 3);
  EXPECT_THROW((void)scalar_at_precision<int>(3, Precision::bits(32)), std::invalid_argument);
}

#if UNI20_ENABLE_MPFR
static_assert(!HasNativePrecision<mpreal>);
TEST(PrecisionTest, RuntimePrecisionIsExplicitAndProviderChecksItsRange)
{
  EXPECT_TRUE(precision_of(mpreal(3)).is_exact());
  EXPECT_EQ(precision_of(scalar_at_precision<mpreal>(3, Precision::bits(256))), Precision::bits(256));
  auto large = Precision::bits(std::numeric_limits<Precision::bit_count_type>::max());
  EXPECT_THROW(mpreal{large}, std::invalid_argument);
  EXPECT_THROW((void)precision_of(mpreal(uninitialized)), std::logic_error);
}
#endif
} // namespace
