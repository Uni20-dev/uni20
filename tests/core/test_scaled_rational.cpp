#include <gtest/gtest.h>
#include <uni20/core/math.hpp>

#include <cstdio>
#include <cstdlib>

using namespace uni20;

namespace
{
mpreal power_of_two(mpfr_exp_t exponent, Precision p)
{
  detail::mpfr_value native(p);
  mpfr_set_ui_2exp(native.value, 1, exponent, MPFR_RNDN);
  return mpreal(native.value, p);
}

// Install only in a freshly executed death-test child. A compact scalar operation
// must not request megabytes merely to expand a power of two into an integer.
struct GmpAllocationBound
{
    static inline void* (*allocate)(std::size_t) = nullptr;
    static inline void* (*reallocate)(void*, std::size_t, std::size_t) = nullptr;
    static inline void (*release)(void*, std::size_t) = nullptr;
    static void check(std::size_t size)
    {
      if (size > 1024 * 1024) std::_Exit(2);
    }
    static void install()
    {
      mp_get_memory_functions(&allocate, &reallocate, &release);
      mp_set_memory_functions(
          [](std::size_t size) -> void* {
            check(size);
            return allocate(size);
          },
          [](void* data, std::size_t old_size, std::size_t size) -> void* {
            check(size);
            return reallocate(data, old_size, size);
          },
          release);
    }
};
bool compact_operations()
{
  auto p = Precision::bits(80);
  auto huge = power_of_two(1'000'000'000, p), tiny = power_of_two(-1'000'000'000, p);
  bool valid = mpreal{1} / huge == tiny && mpreal{1} / tiny == huge;
#if UNI20_ENABLE_MPC
  using C = complex<mpreal>;
  C a(huge, huge), b(mpreal{1}, mpreal{-1});
  valid = valid && a * b == C(2 * huge, mpreal(0, p));
  valid = valid && a / b == C(mpreal(0, p), huge);
  valid = valid && b / a == C(mpreal(0, p), -tiny);
#endif
  return valid;
}
} // namespace

TEST(ScaledRational, RealQuotientMatchesExactOracle)
{
  for (auto bits : {3, 80})
    for (int exponent : {-5000, 5000})
    {
      auto p = Precision::bits(bits);
      auto x = 3 * power_of_two(exponent, p);
      auto exact_x = mpreal{3} * pow(mpreal{2}, mpreal{exponent});
      for (auto text : {"1/3", "9/8", "-11/10"})
      {
        mpreal q{exact_constant(text)};
        EXPECT_EQ(q / x, (q / exact_x).at(p));
      }
    }
}

TEST(ScaledRational, CorrectRoundingAndRangeFlagsMatchMpfr)
{
  // Exercise the comparison-based rounding directly against MPFR's independent
  // rational conversion, including ties and values on either side of zero.
  for (auto bits : {1, 2, 3, 80})
    for (auto text : {"0", "1/3", "-1/3", "9/8", "11/8", "15/8", "31/32"})
    {
      auto p = Precision::bits(bits);
      detail::mpfr_value actual(p);
      exact_constant q(text);
      detail::scaled_rational_ratio({detail::scaled_rational{q, 0}, {}, {exact_constant{1}, 0}, {}})
          .round_to(actual.value);
      EXPECT_EQ(mpreal(actual.value, p), q.at(p));
    }
  for (auto bits : {1, 2, 3, 80})
    for (auto exponent : {mpfr_get_emin() - 2, mpfr_get_emin() - 1, mpfr_get_emax() - 1})
      for (auto text : {"1/2", "1", "3/2", "7/4", "15/8", "31/16", "2", "-1", "-3/2"})
      {
        auto p = Precision::bits(bits);
        detail::mpfr_value actual(p), expected(p), input(Precision::bits(8));
        exact_constant q(text);
        mpfr_set_q(input.value, q.native_handle(), MPFR_RNDN);
        mpfr_clear_flags();
        mpfr_mul_2si(expected.value, input.value, exponent, MPFR_RNDN);
        auto flags = mpfr_flags_save();
        mpfr_clear_flags();
        detail::scaled_rational_ratio({detail::scaled_rational{q, exponent}, {}, {exact_constant{1}, 0}, {}})
            .round_to(actual.value);
        SCOPED_TRACE(testing::Message() << bits << " bits: " << text << " * 2^" << exponent);
        EXPECT_EQ(mpfr_flags_save(), flags);
        EXPECT_EQ(mpreal(actual.value, p), mpreal(expected.value, p));
        EXPECT_EQ(mpfr_signbit(actual.value), mpfr_signbit(expected.value));
      }
}

#if UNI20_ENABLE_MPC
TEST(ScaledRational, ComplexProductsAndQuotientsMatchExactOracle)
{
  using C = complex<mpreal>;
  for (auto bits : {3, 80})
    for (auto exponents : {std::pair{-5000, -5000}, std::pair{5000, 5000}, std::pair{-5000, 5000}})
      for (auto re : {"9/8", "1/7", "-11/10"})
        for (auto im : {"1/8", "-19/13", "1/10000000000000000000000000000000000000000"})
        {
          auto p = Precision::bits(bits);
          C x(pow(mpreal{2}, mpreal{exponents.first}), -pow(mpreal{2}, mpreal{exponents.second}));
          C a = x.at(p), b(mpreal{exact_constant(re)}, mpreal{exact_constant(im)});
          EXPECT_EQ(a * b, (x * b).at(p));
          EXPECT_EQ(b * a, (b * x).at(p));
          EXPECT_EQ(a / b, (x / b).at(p));
          EXPECT_EQ(b / a, (b / x).at(p));
        }
  auto p = Precision::bits(3);
  C a(power_of_two(5000, p), power_of_two(-5000, p));
  C below(mpreal{exact_constant("9/8")}, mpreal{1});
  C above(mpreal{exact_constant("9/8")}, mpreal{-1});
  EXPECT_EQ((a * below).real(), power_of_two(5000, p));
  EXPECT_EQ((a * above).real(), mpreal("1.25", p) * power_of_two(5000, p));
  // Cancellation of huge terms must still recover a much smaller exact result.
  C x(power_of_two(5000, p), -power_of_two(5000, p));
  mpreal tiny{exact_constant("1/10000000000000000000000000000000000000000")};
  C b(mpreal{1} + tiny, mpreal{-1});
  EXPECT_EQ((x * b).real(), (pow(mpreal{2}, mpreal{5000}) * tiny).at(p));
}

TEST(ScaledRational, LargeExponentComplexSignedZerosMatchMpc)
{
  using C = complex<mpreal>;
  auto p = Precision::bits(80);
  auto check = [](C const& actual, C const& expected) {
    EXPECT_EQ(actual, expected);
    if (expected.real() == 0)
    {
      EXPECT_EQ(signbit(actual.real()), signbit(expected.real()));
    }
    if (expected.imag() == 0)
    {
      EXPECT_EQ(signbit(actual.imag()), signbit(expected.imag()));
    }
  };
  for (auto exponent : {-5000, 5000})
    for (auto re : {"0", "-0", "1", "-1"})
      for (auto im : {"0", "-0", "1", "-1"})
        for (auto qr : {-1, 0, 1})
          for (auto qi : {-1, 0, 1})
          {
            auto scale = power_of_two(exponent, p);
            C a(mpreal(re, p) * scale, mpreal(im, p) * scale);
            C b(mpreal{qr}, mpreal{qi}), finite_b = b.at(p);
            check(a * b, a * finite_b);
            check(b * a, finite_b * a);
            if (b != 0) check(a / b, a / finite_b);
            if (a != 0) check(b / a, finite_b / a);
          }
}

#endif

TEST(ScaledRationalDeathTest, LargeExponentsDoNotExpandIntoLargeIntegers)
{
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT(
      {
        GmpAllocationBound::install();
        std::_Exit(compact_operations() ? 0 : 1);
      },
      ::testing::ExitedWithCode(0), "");
}
