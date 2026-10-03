#include <gtest/gtest.h>
#include <limits>
#include <uni20/core/numeric_limits.hpp>

namespace
{
struct CustomScalar
{};
struct StandardOnlyScalar
{};
} // namespace

template <> struct std::numeric_limits<StandardOnlyScalar> : std::numeric_limits<double>
{};

namespace uni20
{
template <> struct numeric_limits<CustomScalar>
{
    static constexpr bool is_specialized = true;
    static constexpr int digits = 37;

    static constexpr CustomScalar epsilon() noexcept { return {}; }
};
} // namespace uni20

TEST(NumericLimitsTest, DelegatesToStdNumericLimitsForBuiltins)
{
  static_assert(uni20::numeric_limits<float>::is_specialized);
  static_assert(uni20::numeric_limits<double const>::digits == std::numeric_limits<double>::digits);
  static_assert(uni20::numeric_limits<int const volatile>::max() == std::numeric_limits<int>::max());
  static_assert(uni20::numeric_limits<double>::is_specialized);
  static_assert(uni20::numeric_limits<long double>::is_specialized);

  static_assert(uni20::numeric_limits<float>::digits == std::numeric_limits<float>::digits);
  static_assert(uni20::numeric_limits<double>::max_digits10 == std::numeric_limits<double>::max_digits10);
  EXPECT_EQ(uni20::numeric_limits<float>::epsilon(), std::numeric_limits<float>::epsilon());
  EXPECT_EQ(uni20::numeric_limits<double>::infinity(), std::numeric_limits<double>::infinity());
}

TEST(NumericLimitsTest, SupportsUni20CustomSpecializations)
{
  static_assert(uni20::has_numeric_limits_v<CustomScalar>);
  static_assert(uni20::numeric_limits<CustomScalar>::digits == 37);
}

TEST(NumericLimitsTest, ForwardsCvQualifiedTypesToUni20Specialization)
{
  static_assert(uni20::has_numeric_limits_v<CustomScalar const>);
  static_assert(uni20::has_numeric_limits_v<CustomScalar volatile>);
  static_assert(uni20::numeric_limits<CustomScalar const>::digits == 37);
  static_assert(uni20::numeric_limits<CustomScalar volatile>::digits == 37);
}

template <class T>
concept HasEpsilon = requires { uni20::numeric_limits<T>::epsilon(); };
template <class T>
concept HasMinimum = requires { uni20::numeric_limits<T>::min(); };
template <class T>
concept HasDigits = requires { uni20::numeric_limits<T>::digits; };

TEST(NumericLimitsTest, ReportsMissingLimits)
{
  struct NoLimits
  {};

  static_assert(!uni20::has_numeric_limits_v<NoLimits>);
  static_assert(!uni20::has_numeric_limits_v<NoLimits const&>);
  static_assert(!uni20::has_numeric_limits_v<void>);
  static_assert(!HasEpsilon<NoLimits>);
  static_assert(!HasEpsilon<NoLimits const>);
  static_assert(!HasEpsilon<NoLimits volatile>);
  static_assert(!HasEpsilon<NoLimits const volatile>);
  static_assert(!HasMinimum<NoLimits>);
  static_assert(!HasDigits<NoLimits>);
  static_assert(!HasEpsilon<void>);
  static_assert(!uni20::has_numeric_limits_v<StandardOnlyScalar>);
  static_assert(!HasEpsilon<StandardOnlyScalar>);
}

TEST(NumericLimitsTest, ExemplarEpsilonPreservesNativeTypeAndPrecision)
{
  static_assert(uni20::numeric_limits<float>::epsilon(0.0f) == std::numeric_limits<float>::epsilon());
  static_assert(uni20::numeric_limits<double>::epsilon(1.0) == std::numeric_limits<double>::epsilon());
  static_assert(uni20::numeric_limits<long double>::epsilon(2.0L) == std::numeric_limits<long double>::epsilon());
}
