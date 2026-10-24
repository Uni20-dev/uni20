#include <gtest/gtest.h>
#include <uni20/core/factorial.hpp>
#include <uni20/core/math.hpp>

namespace
{
using namespace uni20;
template <class R, class I>
concept Factory = requires(I n) { factorial<R>(n); };
template <class R, class I>
concept PrecisionFactory = requires(I n, Precision p) { factorial<R>(n, p); };
struct ConversionOnly
{
    operator int() const;
};
static_assert(Factory<double, int> && PrecisionFactory<double, unsigned>);
static_assert(!Factory<int, int> && !Factory<complex<double>, int>);
static_assert(!Factory<double, bool> && !Factory<double, double> && !Factory<double, ConversionOnly>);
static_assert(!PrecisionFactory<double, bool> && !PrecisionFactory<double, double>);
#if UNI20_HAS_MPREAL
static_assert(!Factory<mpreal, int> && PrecisionFactory<mpreal, int>);
#endif

template <class R> class NativeFactorial : public testing::Test {};
using NativeReals = testing::Types<float, double, long double
#if UNI20_HAS_FLOAT128
                                   ,
                                   float128
#endif
                                   >;
TYPED_TEST_SUITE(NativeFactorial, NativeReals);
TYPED_TEST(NativeFactorial, CheckedFactoryAndPrecision)
{
  using R = TypeParam;
  EXPECT_EQ(factorial<R>(0), R(1));
  EXPECT_EQ(factorial<R>(1), R(1));
  EXPECT_EQ(factorial<R>(10u), R(3628800));
  EXPECT_EQ(factorial<R>(10, native_precision<R>()), factorial<R>(10));
  EXPECT_THROW(factorial<R>(-1), std::domain_error);
  EXPECT_THROW(factorial<R>(numeric_limits<std::int64_t>::min()), std::domain_error);
  EXPECT_THROW(factorial<R>(numeric_limits<std::uint64_t>::max()), std::out_of_range);
  EXPECT_THROW(factorial<R>(10, Precision::exact()), std::invalid_argument);
  EXPECT_THROW(factorial<R>(10, Precision::bits(numeric_limits<R>::digits() + 1)), std::invalid_argument);
}
TYPED_TEST(NativeFactorial, OverflowAndHugeOrders)
{
  using R = TypeParam;
  using L = numeric_limits<R>;
  static_assert(L::max_exponent == 128 || L::max_exponent == 1024 || L::max_exponent == 16384);
  constexpr int last = L::max_exponent == 128 ? 34 : L::max_exponent == 1024 ? 170 : 1754;
  EXPECT_TRUE(math::isfinite(factorial<R>(last)));
  EXPECT_GT(factorial<R>(last), R(0));
  EXPECT_EQ(factorial<R>(last + 1), L::infinity());
  EXPECT_EQ(factorial<R>(L::max_exponent), L::infinity());
  EXPECT_EQ(factorial<R>(numeric_limits<std::int64_t>::max()), L::infinity());
}
#if UNI20_HAS_MPREAL
TEST(FactorialFactory, RuntimePrecisionAndExactState)
{
  auto p = Precision::bits(173);
  auto value = factorial<mpreal>(40, p);
  EXPECT_EQ(value.precision(), p);
  EXPECT_EQ(value, math::factorial(40, p));
  auto exact = factorial<mpreal>(20, Precision::exact());
  EXPECT_TRUE(exact.is_exact());
  EXPECT_EQ(exact, mpreal(exact_constant("2432902008176640000")));
  EXPECT_THROW(factorial<mpreal>(-1, p), std::domain_error);
  EXPECT_THROW(factorial<mpreal>(numeric_limits<std::uint64_t>::max(), p), std::out_of_range);
}
#endif
} // namespace
