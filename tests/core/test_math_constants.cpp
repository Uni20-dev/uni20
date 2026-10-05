// The native constants are available without the math dispatcher or MPFR headers.
#include <uni20/core/math_constants.hpp>
#include <uni20/core/numeric_limits.hpp>
#include <gtest/gtest.h>

namespace
{
template <class R> concept HasPi = requires { uni20::pi<R>; };
template <class R> concept HasCatalan = requires { uni20::catalan<R>; };
static_assert(!HasPi<int>);
static_assert(!HasPi<uni20::complex<double>>);
static_assert(!HasCatalan<double>);

template <class R> void check_constants()
{
  static_assert(std::same_as<decltype(uni20::pi<R>), R const>);
  static_assert(std::same_as<decltype(uni20::log_two<R>), R const>);
  static_assert(std::same_as<decltype(uni20::euler_gamma<R>), R const>);
  static_assert(uni20::pi<R> == std::numbers::pi_v<R>);
  static_assert(uni20::log_two<R> == std::numbers::ln2_v<R>);
  static_assert(uni20::euler_gamma<R> == std::numbers::egamma_v<R>);
  EXPECT_TRUE(uni20::pi<R> > R(3));
  EXPECT_TRUE(uni20::log_two<R> > R(0.69));
  EXPECT_TRUE(uni20::euler_gamma<R> < R(0.58));
  if constexpr (!std::same_as<R, float> && !std::same_as<R, double>)
  {
    if constexpr (uni20::numeric_limits<R>::digits > uni20::numeric_limits<double>::digits)
    {
      EXPECT_TRUE(uni20::pi<R> != R(uni20::pi<double>));
      EXPECT_TRUE(uni20::log_two<R> != R(uni20::log_two<double>));
      EXPECT_TRUE(uni20::euler_gamma<R> != R(uni20::euler_gamma<double>));
    }
  }
}
TEST(MathConstants, Float) { check_constants<float>(); }
TEST(MathConstants, Double) { check_constants<double>(); }
TEST(MathConstants, LongDouble) { check_constants<long double>(); }
#if UNI20_HAS_FLOAT128
TEST(MathConstants, Float128) { check_constants<uni20::float128>(); }
#endif
} // namespace
