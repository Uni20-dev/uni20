#include <gtest/gtest.h>
#include <limits>
#include <sstream>
#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/async/tbb_scheduler.hpp>
#include <uni20/core/math.hpp>
#include <uni20/core/mpreal.hpp>
#include <uni20/core/scalar_concepts.hpp>
#include <vector>

using namespace uni20;
using namespace uni20::literals;
using namespace uni20::async;

namespace
{
template <typename T>
concept AddsNativeFloat = requires(T x) {
  x + 0.1;
  0.1 + x;
};
template <typename T>
concept HasPrecisionlessSqrt = requires(T x) { uni20::sqrt(x); };
template <typename T>
concept HasComplexSpelling = requires { typename uni20::complex<T>; };
template <typename T>
concept HasComplexCounterpart = requires { typename uni20::make_complex_t<T>; };
static_assert(std::default_initializable<mpreal>);
static_assert(std::is_nothrow_move_constructible_v<mpreal>);
static_assert(!std::convertible_to<double, mpreal>);
static_assert(!std::convertible_to<mpreal, double>);
static_assert(!AddsNativeFloat<mpreal>);
static_assert(HasPrecisionlessSqrt<decimal_literal>);
static_assert(Real<mpreal> && Scalar<mpreal>);
static_assert(!BlasReal<mpreal> && !LapackReal<mpreal>);
#if UNI20_ENABLE_MPC
static_assert(HasComplexSpelling<mpreal> && HasComplexCounterpart<mpreal>);
#else
static_assert(!HasComplexSpelling<mpreal> && !HasComplexCounterpart<mpreal>);
#endif
static_assert(HasComplexSpelling<double> && HasComplexCounterpart<double>);
static_assert(!has_numeric_limits_v<mpreal>);
static_assert(std::same_as<decltype(std::declval<mpreal>() + std::declval<mpreal>()), mpreal>);
constexpr auto literal = 0.123456789012345678901234567890123456789_mp;
static_assert(literal.spelling() == "0.123456789012345678901234567890123456789");
} // namespace

TEST(MpReal, PrecisionIsExplicitAndValidated)
{
  EXPECT_THROW(Precision::bits(0), std::invalid_argument);
  EXPECT_THROW(Precision::bits(-1), std::invalid_argument);
  EXPECT_THROW(Precision::decimal_digits(0), std::invalid_argument);
  EXPECT_THROW(Precision::decimal_digits(-1), std::invalid_argument);
  EXPECT_THROW(Precision::decimal_digits(std::numeric_limits<std::int64_t>::max()), std::invalid_argument);
  EXPECT_EQ(Precision::decimal_digits(80).bit_count(), 266);
  auto p = Precision::bits(256);
  mpreal zero(p);
  EXPECT_EQ(zero, 0);
  EXPECT_EQ(zero.precision(), p);
  EXPECT_FALSE(signbit(zero));
}

TEST(MpReal, UnsetValuesHaveSafeOwnershipButNoNumericalMeaning)
{
  mpreal unset(uninitialized);
  EXPECT_FALSE(unset.initialized());
  EXPECT_THROW(unset.precision(), std::logic_error);
  EXPECT_THROW(unset.native_handle(), std::logic_error);
  EXPECT_THROW(unset.to_string(), std::logic_error);
  EXPECT_THROW((void)static_cast<double>(unset), std::logic_error);
  EXPECT_THROW(isfinite(unset), std::logic_error);
  EXPECT_THROW(isnan(unset), std::logic_error);
  EXPECT_THROW(isinf(unset), std::logic_error);
  EXPECT_THROW(signbit(unset), std::logic_error);
  EXPECT_THROW(+unset, std::logic_error);
  EXPECT_THROW(-unset, std::logic_error);
  EXPECT_THROW(sqrt(unset), std::logic_error);
  EXPECT_THROW((void)(unset == 0), std::logic_error);
  EXPECT_THROW((void)(unset < 0), std::logic_error);
  auto p = Precision::bits(256);
  mpreal valid(1, p);
  EXPECT_THROW(unset + valid, std::logic_error);
  EXPECT_THROW(valid + unset, std::logic_error);
  EXPECT_THROW((void)(valid == unset), std::logic_error);
  EXPECT_THROW(unset += valid, std::logic_error);
  EXPECT_THROW(unset.at(p), std::logic_error);
  EXPECT_THROW((void)format_real(unset), std::logic_error);
  EXPECT_THROW((void)mpreal(unset, p), std::logic_error);

  auto copy = unset;
  auto moved = std::move(unset);
  EXPECT_FALSE(copy.initialized());
  EXPECT_FALSE(moved.initialized());
  unset = valid;
  EXPECT_EQ(unset, 1);
  EXPECT_EQ(unset.precision(), p);
  moved = std::move(unset);
  EXPECT_EQ(moved, 1);
  EXPECT_FALSE(unset.initialized());
  moved.swap(copy);
  EXPECT_FALSE(moved.initialized());
  EXPECT_EQ(copy, 1);
  copy = moved;
  EXPECT_FALSE(copy.initialized());
  std::vector<mpreal> buffer(4);
  buffer[2] = valid;
  auto duplicate = buffer;
  EXPECT_TRUE(duplicate[0].is_exact());
  EXPECT_EQ(duplicate[0], 0);
  EXPECT_EQ(duplicate[2], 1);
  buffer.resize(40);
  EXPECT_EQ(buffer[2], 1);
  EXPECT_TRUE(buffer[39].is_exact());
  EXPECT_EQ(buffer[39], 0);
}

TEST(MpReal, ArithmeticIsEagerAndOwnsItsResult)
{
  auto p = Precision::bits(256);
  mpreal x(1, p), y(2, p);
  auto result = x + y;
  x = mpreal(10, p);
  y = mpreal(20, p);
  EXPECT_EQ(result, 3);
  auto temporary_result = mpreal(1, p) / mpreal(8, p);
  EXPECT_EQ(temporary_result, exact_constant("0.125"));
  auto high = mpreal(1, p) + 1e-70_mp;
  EXPECT_GT(high, 1);
  EXPECT_EQ(static_cast<double>(high), 1.0);
}

TEST(MpReal, MixedPrecisionArithmeticRejectsBeforeMutation)
{
  mpreal x(2, Precision::bits(80)), y(3, Precision::bits(256));
  EXPECT_THROW(x + y, std::invalid_argument);
  EXPECT_THROW(x - y, std::invalid_argument);
  EXPECT_THROW(x * y, std::invalid_argument);
  EXPECT_THROW(x / y, std::invalid_argument);
  EXPECT_THROW(x += y, std::invalid_argument);
  EXPECT_THROW(x -= y, std::invalid_argument);
  EXPECT_THROW(x *= y, std::invalid_argument);
  EXPECT_THROW(x /= y, std::invalid_argument);
  EXPECT_EQ(x, 2);
  EXPECT_EQ(x.precision(), Precision::bits(80));
  EXPECT_EQ(x.at(y.precision()) + y, 5);
  EXPECT_LT(x, y); // Comparisons compare stored values without conversion.
}

TEST(MpReal, AssignmentCopiesPrecisionAndMoveLeavesValidSource)
{
  auto p = Precision::bits(400);
  mpreal source = mpreal(1, p) + 1e-100_mp;
  mpreal destination(0, Precision::bits(80));
  destination = source;
  EXPECT_EQ(destination.precision(), p);
  EXPECT_EQ(destination, source);
  EXPECT_GT(destination, 1);
  auto copy = destination;
  auto& alias = destination;
  destination = alias;
  EXPECT_EQ(destination, copy);
  auto moved = std::move(source);
  EXPECT_EQ(moved, copy);
  EXPECT_FALSE(source.initialized());
  source = mpreal(7, Precision::bits(64));
  destination = std::move(source);
  EXPECT_EQ(destination, 7);
  EXPECT_EQ(destination.precision(), Precision::bits(64));
  destination = std::move(alias);
  EXPECT_EQ(destination, 7);
}

TEST(MpReal, CompoundArithmeticSupportsAliasingAndPreservesPrecision)
{
  auto p = Precision::bits(256);
  mpreal x("1.5", p);
  EXPECT_EQ(&(x += x), &x);
  EXPECT_EQ(x, 3);
  EXPECT_EQ(&(x *= x), &x);
  EXPECT_EQ(x, 9);
  EXPECT_EQ(&(x /= x), &x);
  EXPECT_EQ(x, 1);
  EXPECT_EQ(&(x -= x), &x);
  EXPECT_EQ(x, 0);
  EXPECT_EQ(x.precision(), p);

  auto low = Precision::bits(3);
  x = mpreal(1, low);
  x += 0.14_mp; // Round 1.14 once, without first rounding the literal.
  EXPECT_EQ(x, 1.25_mp);
  x -= 0.07_mp;
  EXPECT_EQ(x, 1.25_mp);
  x = mpreal("1.25", low);
  x *= 1.1_mp; // 1.375 is a tie, rounded to the even significand.
  EXPECT_EQ(x, 1.5_mp);
  x = mpreal(1, low);
  x /= 1.1_mp;
  EXPECT_EQ(x, 0.875_mp);
  EXPECT_EQ(x.precision(), low);
}

TEST(MpReal, MixedBasicArithmeticRoundsTheResultOnce)
{
  for (auto bits : {3, 128})
  {
    auto p = Precision::bits(bits);
    for (auto text : {"1", "-1", "5/4", "-3/2"})
    {
      exact_constant x(text);
      mpreal a(x, p);
      for (auto fraction : {"1/7", "9/8", "-19/13", "11/10", "1/10000000000000000000000000000000000000000"})
      {
        exact_constant q(fraction);
        mpreal b(q);
        EXPECT_EQ(a + b, mpreal(x + q, p));
        EXPECT_EQ(b + a, mpreal(q + x, p));
        EXPECT_EQ(a - b, mpreal(x - q, p));
        EXPECT_EQ(b - a, mpreal(q - x, p));
        EXPECT_EQ(a * b, mpreal(x * q, p));
        EXPECT_EQ(b * a, mpreal(q * x, p));
        EXPECT_EQ(a / b, mpreal(x / q, p));
        EXPECT_EQ(b / a, mpreal(q / x, p));
        auto c = b;
        c -= a;
        EXPECT_EQ(c, b - a);
        EXPECT_EQ(c.precision(), p);
        c = b;
        c /= a;
        EXPECT_EQ(c, b / a);
        EXPECT_EQ(c.precision(), p);
      }
    }
    auto epsilon = exact_constant("1/10000000000000000000000000000000000000000");
    mpreal just_above_one(exact_constant{1} + epsilon);
    EXPECT_EQ(just_above_one - mpreal(1, p), epsilon.at(p));
    EXPECT_EQ(mpreal(1, p) - just_above_one, (-epsilon).at(p));
  }
}

TEST(MpReal, MixedArithmeticRetainsExceptionalValuesAndSignedZeros)
{
  auto p = Precision::bits(80);
  mpreal zero{}, one{1}, negative_zero("-0", p), negative_one(-1, p);
  EXPECT_FALSE(signbit(one - mpreal(1, p)));
  EXPECT_FALSE(signbit(zero - negative_zero));
  EXPECT_TRUE(signbit(negative_zero - zero));
  EXPECT_TRUE(signbit(zero / negative_one));
  EXPECT_TRUE(signbit(zero * negative_one));
  EXPECT_TRUE(isinf(one / negative_zero));
  EXPECT_TRUE(signbit(one / negative_zero));
  EXPECT_TRUE(isnan(zero / negative_zero));
  mpreal inf("inf", p), nan("nan", p);
  EXPECT_EQ(mpreal(exact_constant("1e10000")) / inf, 0);
  EXPECT_TRUE(isnan(one / nan));
  EXPECT_TRUE(isinf(inf + one));
  EXPECT_TRUE(isnan(inf * zero));
  EXPECT_TRUE(isinf(one / zero.at(p)));
}

TEST(MpReal, RoundNearestTiesToEven)
{
  auto p = Precision::bits(3);
  EXPECT_EQ(mpreal("1.125", p), 1);
  EXPECT_EQ(mpreal("1.375", p), exact_constant("1.5"));
  EXPECT_EQ(mpreal("-1.125", p), -1);
  EXPECT_EQ((1.125_mp).at(p), 1);
}

TEST(MpReal, ExactRationalArithmeticAndLiteralDigits)
{
  auto sum = 0.1_mp + 0.2_mp;
  EXPECT_EQ(sum, 0.3_mp);
  EXPECT_EQ(sum.to_string(), "3/10");
  EXPECT_EQ((1_mp / 3_mp * 3_mp).to_string(), "1");
  EXPECT_EQ((-(1_mp / 3_mp) + 1_mp).to_string(), "2/3");
  EXPECT_EQ((1.2e3_mp - 200_mp).to_string(), "1000");
  EXPECT_EQ((1'000_mp + 1_mp).to_string(), "1001");
  EXPECT_EQ((1e1'0_mp + 1_mp).to_string(), "10000000001");
  EXPECT_LT(0.1_mp, 0.2_mp);
  EXPECT_THROW(1_mp / 0_mp, std::domain_error);
  auto p = Precision::bits(256);
  EXPECT_EQ(sum.at(p), mpreal("0.3", p));
  EXPECT_EQ(literal.at(p), mpreal(literal.spelling(), p));
  EXPECT_NE(literal.at(p), mpreal(0.123456789012345678901234567890123456789L, p));
  auto rational = exact_constant("0.1");
  auto owned_sum = rational + rational;
  rational = exact_constant("9");
  EXPECT_EQ(owned_sum, 0.2_mp);
  auto moved = std::move(owned_sum);
  EXPECT_EQ(moved, 0.2_mp);
  EXPECT_EQ(owned_sum, 0_mp);
}

TEST(MpReal, ParenthesesDetermineTheRoundingBoundary)
{
  auto p = Precision::bits(3);
  mpreal x(1, p);
  EXPECT_EQ(x + (0.1_mp + 0.2_mp), mpreal("1.25", p));
  EXPECT_EQ((x + 0.1_mp) + 0.2_mp, mpreal("1.25", p));
  // Two exact 1/8 terms sum to 1/4, while each separate tie rounds back to 1.
  EXPECT_EQ(x + (0.125_mp + 0.125_mp), mpreal("1.25", p));
  EXPECT_EQ((x + 0.125_mp) + 0.125_mp, 1);
  EXPECT_NE(mpreal("0.1", p), 0.1_mp); // Comparison does not round the exact rational.
}

TEST(MpReal, DecimalExpansionRejectsExcessiveScales)
{
  for (auto text : {"1e1000001", "1e-1000001", "0.1e-1000000", "0e1000001", "1e999999999"})
    EXPECT_THROW((void)exact_constant(text), std::out_of_range) << text;
  // The limit applies to the combined scale, not the exponent in isolation.
  auto positive = exact_constant("1.0e1000001");
  auto negative = exact_constant("1e-1000000");
  EXPECT_EQ(positive * negative, 10_mp);
  EXPECT_EQ(exact_constant("0e1000000"), 0_mp);
}

TEST(MpReal, IntegralOperandsRemainExactUntilConversion)
{
  auto p = Precision::bits(256);
  auto minimum = std::numeric_limits<long long>::min();
  auto maximum = std::numeric_limits<unsigned long long>::max();
  EXPECT_EQ(mpreal(minimum, p), exact_constant("-9223372036854775808"));
  EXPECT_EQ(mpreal(maximum, p), exact_constant("18446744073709551615"));
  mpreal x(3, p);
  EXPECT_EQ(x + 1, 4);
  EXPECT_EQ(1 + x, 4);
  EXPECT_EQ(9 - x, 6);
  EXPECT_EQ(9 / x, 3);
  x += 2;
  x *= 3;
  x -= 6;
  x /= 3;
  EXPECT_EQ(x, 3);
}

TEST(MpReal, IntegerComparisonsPreserveExactValues)
{
  auto check_type = []<std::integral I>() {
    auto p = Precision::bits(256);
    for (I value : {I{0}, I{1}, std::numeric_limits<I>::lowest(), std::numeric_limits<I>::max()})
    {
      mpreal x(value, p);
      EXPECT_TRUE(x == value);
      EXPECT_TRUE(value == x);
      EXPECT_FALSE(x != value);
      EXPECT_FALSE(value != x);
      EXPECT_EQ(x <=> value, std::partial_ordering::equivalent);
      EXPECT_EQ(value <=> x, std::partial_ordering::equivalent);
      EXPECT_LT(x - 1, value);
      EXPECT_GT(value, x - 1);
      EXPECT_GT(x + 1, value);
      EXPECT_LT(value, x + 1);
    }
  };
  check_type.operator()<signed char>();
  check_type.operator()<unsigned char>();
  check_type.operator()<short>();
  check_type.operator()<unsigned short>();
  check_type.operator()<int>();
  check_type.operator()<unsigned int>();
  check_type.operator()<long>();
  check_type.operator()<unsigned long>();
  check_type.operator()<long long>();
  check_type.operator()<unsigned long long>();

  // Rounding the integer to the real's precision would incorrectly give equality.
  mpreal low(8, Precision::bits(3));
  EXPECT_LT(low, 9);
  EXPECT_NE(low, 9);
  EXPECT_GT(-low, -9);
  mpreal beyond_double("9007199254740992", Precision::bits(53));
  EXPECT_LT(beyond_double, 9007199254740993ULL);
  EXPECT_NE(beyond_double, 9007199254740993ULL);
  EXPECT_LT(mpreal(-1, Precision::bits(80)), std::numeric_limits<unsigned long long>::max());
}

TEST(MpReal, IntegerComparisonsHandleZeroAndExceptionalValues)
{
  auto p = Precision::bits(80);
  for (auto text : {"0", "-0"})
  {
    mpreal zero(text, p);
    EXPECT_TRUE(zero == 0);
    EXPECT_TRUE(0U == zero);
    EXPECT_EQ(zero <=> 0, std::partial_ordering::equivalent);
    EXPECT_EQ(0U <=> zero, std::partial_ordering::equivalent);
  }
  mpreal nan("nan", p), inf("inf", p);
  for (int value : {0, 1, -1})
  {
    EXPECT_FALSE(nan == value);
    EXPECT_FALSE(value == nan);
    EXPECT_TRUE(nan != value);
    EXPECT_FALSE(nan < value);
    EXPECT_FALSE(nan >= value);
    EXPECT_EQ(nan <=> value, std::partial_ordering::unordered);
    EXPECT_EQ(value <=> nan, std::partial_ordering::unordered);
    EXPECT_NE(inf, value);
    EXPECT_GT(inf, value);
    EXPECT_LT(-inf, value);
  }
}

TEST(MpReal, ParsingAndRoundTripFormatting)
{
  for (auto bits : {2, 53, 80, 256, 400})
  {
    auto p = Precision::bits(bits);
    for (auto text : {"0.1", "-12345.6789", "1e-1000", "1e1000", "0", "-0"})
    {
      mpreal value(text, p);
      auto output = value.to_string();
      mpreal round_trip(output, p);
      EXPECT_EQ(round_trip, value) << output;
      EXPECT_EQ(signbit(round_trip), signbit(value));
    }
  }
  auto p = Precision::bits(256);
  EXPECT_EQ(mpreal("1.23456", p).to_string(4), "1.235e0");
  for (auto text : {"", "abc", "1.2junk", "1.2.3", " 1", "1 ", ".", "1e"})
    EXPECT_THROW(mpreal(text, p), std::invalid_argument) << text;
  EXPECT_THROW(mpreal(std::string_view("1\0x", 3), p), std::invalid_argument);
  for (auto text : {"", "+", "nan", "1/3/4", "1.2.3", "1e", "0x10", "1''2"})
    EXPECT_THROW((void)exact_constant(text), std::invalid_argument) << text;
  EXPECT_THROW(exact_constant("1e184467440737095516160"), std::out_of_range);
  EXPECT_THROW(exact_constant("1.0e-18446744073709551615"), std::out_of_range);
  std::ostringstream os;
  os << mpreal("0.1", p);
  EXPECT_EQ(mpreal(os.str(), p), mpreal("0.1", p));
}

TEST(MpReal, SingleDigitFormattingRoundsAndCarries)
{
  // MPFR >= 4.1 supports one output digit; older MPFR documentation required two.
  auto p = Precision::bits(80);
  EXPECT_EQ(mpreal("1.234", p).to_string(1), "1e0");
  EXPECT_EQ(mpreal("1.5", p).to_string(1), "2e0");
  EXPECT_EQ(mpreal("2.5", p).to_string(1), "2e0");
  EXPECT_EQ(mpreal("9.5", p).to_string(1), "1e1");
  EXPECT_EQ(mpreal("-9.5", p).to_string(1), "-1e1");
  EXPECT_EQ(mpreal("0.125", p).to_string(1), "1e-1");
  EXPECT_EQ(mpreal("0", p).to_string(1), "0");
  EXPECT_EQ(mpreal("-0", p).to_string(1), "-0");
  EXPECT_EQ(mpreal("inf", p).to_string(1), "inf");
  EXPECT_EQ(mpreal("nan", p).to_string(1), "nan");
}

TEST(MpReal, ConstantsMathAndExceptionalValues)
{
  auto p = Precision::bits(256);
  mpreal one(1, p);
  auto pi_value = pi<mpreal>.at(p);
  EXPECT_EQ(pi_value,
            mpreal("3.141592653589793238462643383279502884197169399375105820974944592307816406286208998628034825", p));
  EXPECT_EQ(one * pi<mpreal>, pi_value);
  EXPECT_EQ(pi<mpreal> / one, pi_value);
  EXPECT_EQ(sqrt(mpreal(4, p)), 2);
  EXPECT_EQ(exp(mpreal(0, p)), 1);
  EXPECT_EQ(log(one), 0);
  EXPECT_LT(abs(sin(pi_value)), epsilon(p));
  EXPECT_EQ(hypot(mpreal(3, p), mpreal(4, p)), 5);
  EXPECT_EQ(pow(mpreal(2, p), mpreal(8, p)), 256);
  EXPECT_EQ(one + epsilon(p) / 2, one);
  EXPECT_GT(one + epsilon(p), one);
  auto inf = one / 0;
  auto nan = sqrt(-one);
  EXPECT_TRUE(isinf(inf));
  EXPECT_FALSE(isfinite(inf));
  EXPECT_FALSE(isfinite<mpreal>(inf));
  EXPECT_TRUE(isnan(nan));
  EXPECT_FALSE(isfinite(nan));
  EXPECT_FALSE(nan == nan);
  EXPECT_EQ(nan <=> one, std::partial_ordering::unordered);
  EXPECT_GT(inf, 0);
  EXPECT_TRUE(isinf(mpreal(inf.to_string(), p)));
  EXPECT_TRUE(isnan(mpreal(nan.to_string(), p)));
  EXPECT_EQ(conj(one), one);
}

TEST(MpReal, DoesNotUseOrModifyDefaultPrecision)
{
  struct Restore
  {
      mpfr_prec_t saved = mpfr_get_default_prec();
      mpfr_rnd_t rounding = mpfr_get_default_rounding_mode();
      ~Restore()
      {
        mpfr_set_default_prec(saved);
        mpfr_set_default_rounding_mode(rounding);
      }
  } restore;
  mpfr_set_default_prec(17);
  mpfr_set_default_rounding_mode(MPFR_RNDD);
  auto p = Precision::bits(256);
  auto x = (1_mp / 3_mp).at(p);
  mpfr_set_default_prec(31);
  auto y = x + 1;
  EXPECT_EQ(y.precision(), p);
  EXPECT_EQ(pi<mpreal>.at(p).precision(), p);
  EXPECT_EQ(mpfr_get_default_prec(), 31);
  EXPECT_EQ(mpfr_get_default_rounding_mode(), MPFR_RNDD);
  EXPECT_EQ(mpreal("1.375", Precision::bits(3)), 1.5_mp);
}

TEST(MpReal, AsyncValuesCarryPrecisionAcrossSuspension)
{
  DebugScheduler scheduler;
  ScopedScheduler scope(&scheduler);
  Async<int> gate;
  auto p = Precision::bits(256);
  Async<mpreal> output{mpreal(p)};
  bool finished = false;
  scheduler.schedule(
      [](ReadBuffer<int> ready, WriteBuffer<mpreal> out, Precision p, bool& finished) static -> AsyncTask {
        auto x = (1_mp / 3_mp).at(p);
        auto const count = co_await ready;
        co_await out = x * count;
        finished = true;
      }(gate.read(), output.write(), p, finished));
  scheduler.run_all();
  EXPECT_FALSE(finished);
  scheduler.schedule([](WriteBuffer<int> ready) static -> AsyncTask { co_await ready = 3; }(gate.write()));
  scheduler.run_all();
  ASSERT_TRUE(finished);
  EXPECT_EQ(output.get_wait().precision(), p);
  EXPECT_EQ(output.get_wait(), 1);
}

TEST(MpReal, ConcurrentTasksUseIndependentWorkingPrecisions)
{
  TbbScheduler scheduler{4};
  ScopedScheduler scope(&scheduler);
  std::vector<Async<mpreal>> outputs;
  for (int bits : {64, 128, 256, 400})
  {
    auto p = Precision::bits(bits);
    outputs.emplace_back(mpreal(p));
    scheduler.schedule([](WriteBuffer<mpreal> out, Precision p) static -> AsyncTask {
      auto const before = mpfr_get_default_prec();
      co_await out = sqrt(mpreal{2}, p);
      EXPECT_EQ(mpfr_get_default_prec(), before);
    }(outputs.back().write(), p));
  }
  int i = 0;
  for (int bits : {64, 128, 256, 400})
  {
    auto const& value = outputs[i++].get_wait();
    auto p = Precision::bits(bits);
    EXPECT_EQ(value.precision(), p);
    EXPECT_EQ(value, sqrt(mpreal(2, p)));
  }
}

TEST(MpReal, ScalarIoUsesValuePrecisionAndFormattingOptions)
{
  auto p = Precision::bits(256);
  mpreal value("0.123456789012345678901234567890123456789", p);
  EXPECT_EQ(parse_real<mpreal>(format_scalar(value), p), value);
  EXPECT_EQ(format_real(value, {.precision = 4}), "0.1235");
  EXPECT_EQ(format_real(value, {.precision = 4, .notation = real_format_notation::fixed}), "0.1235");
  EXPECT_EQ(format_real(value, {.precision = 3, .notation = real_format_notation::scientific}), "1.235e-01");
  EXPECT_EQ(format_real(mpreal("-0", p)), "0");
  EXPECT_EQ(format_real(mpreal("-0", p), {.normalize_negative_zero = false}), "-0");
  EXPECT_EQ(format_scalar(mpreal("inf", p)), "inf");
  std::istringstream input("1.23456789012345678901234567890123456789 bad");
  read_real(input, value);
  EXPECT_EQ(value.precision(), p);
  EXPECT_EQ(value, mpreal("1.23456789012345678901234567890123456789", p));
  auto saved = value;
  read_real(input, value);
  EXPECT_TRUE(input.fail());
  EXPECT_EQ(value, saved);
}

TEST(MpReal, ArithmeticMatchesExactDyadicResults)
{
  // At four bits these inputs and expected results are exact binary fractions.
  // 13/8 * 9/8 = 117/64 rounds to 15/8; 13/9 rounds to 3/2.
  auto low = Precision::bits(4);
  mpreal a("1.625", low), b("1.125", low);
  EXPECT_EQ(a + b, 2.75_mp);
  EXPECT_EQ(a - b, 0.5_mp);
  EXPECT_EQ(a * b, 1.875_mp);
  EXPECT_EQ(a / b, 1.5_mp);

  // (1 + 2^-100)(1 - 2^-100) - 1 = -2^-200, exactly at 256 bits.
  // This detects hidden double/long-double arithmetic and lost small terms.
  auto high = Precision::bits(256);
  auto delta = (1_mp / 1267650600228229401496703205376_mp).at(high);
  auto expected = -1_mp / 1606938044258990275541962092341162602522202993782792835301376_mp;
  EXPECT_EQ((1 + delta) * (1 - delta) - 1, expected);
}

TEST(MpReal, ElementaryFunctionsMatchIndependentDecimalReferences)
{
  // Independently generated with Python's decimal module (libmpdec, not MPFR):
  // with localcontext() as c:
  //   c.prec = 125
  //   print(Decimal(2).sqrt(), Decimal(1).exp(), Decimal(2).ln())
  // More than 45 guard decimal digits beyond the largest tested precision.
  char const* root_two = "1."
                         "414213562373095048801688724209698078569671875376948073176679737990732478462107038850387534327"
                         "6415727350138462309122970249248";
  char const* e = "2."
                  "7182818284590452353602874713526624977572470936999595749669676277240766303535475945713821785251664274"
                  "274663919320030599218174";
  char const* log_two = "0."
                        "6931471805599453094172321214581765680755001343602552541206800094933936219696947156058633269964"
                        "1868754200148102057068573368552";
  for (int bits : {80, 256})
  {
    auto p = Precision::bits(bits);
    EXPECT_EQ(sqrt(mpreal(2, p)), mpreal(root_two, p));
    EXPECT_EQ(sqrt(mpreal{2}, p), mpreal(root_two, p));
    EXPECT_EQ(exp(mpreal(1, p)), mpreal(e, p));
    EXPECT_EQ(exp(mpreal{1}, p), mpreal(e, p));
    EXPECT_EQ(log(mpreal(2, p)), mpreal(log_two, p));
    EXPECT_EQ(log(mpreal{2}, p), mpreal(log_two, p));
    auto angle = pi<mpreal>.at(p);
    EXPECT_LT(abs(sin(angle / 6) - 0.5_mp), 2 * epsilon(p));
    EXPECT_LT(abs(cos(angle / 3) - 0.5_mp), 2 * epsilon(p));
  }
}

namespace
{
template <class T> T generic_sum(std::vector<T> const& input)
{
  T result{};
  for (auto const& x : input)
    result += x;
  return result;
}
template <class T> T generic_product(std::vector<T> const& input)
{
  T result{1};
  for (auto const& x : input)
    result *= x;
  return result;
}
} // namespace

TEST(MpReal, ExactStateSupportsGenericConstructionAndArithmetic)
{
  auto exact = Precision::exact();
  EXPECT_TRUE(exact.is_exact());
  EXPECT_THROW(exact.bit_count(), std::logic_error);
  mpreal zero{}, one{1}, two{2};
  EXPECT_TRUE(zero.initialized());
  EXPECT_EQ(zero.precision(), exact);
  EXPECT_EQ(zero, 0);
  auto third = one / mpreal{3};
  EXPECT_TRUE(third.is_exact());
  EXPECT_EQ(third.to_string(), "1/3");
  EXPECT_EQ(third * 3, one);
  EXPECT_EQ(mpreal(0.1_mp) + mpreal(0.2_mp), 0.3_mp);
  EXPECT_EQ((third - one).to_string(), "-2/3");
  EXPECT_EQ(abs(third - one).to_string(), "2/3");
  EXPECT_EQ(conj(third), third);
  EXPECT_TRUE(isfinite(third));
  EXPECT_FALSE(isnan(third));
  EXPECT_FALSE(isinf(third));
  EXPECT_TRUE(signbit(-third));
  EXPECT_EQ(generic_sum(std::vector<mpreal>{third, third, third}), one);
  EXPECT_EQ(generic_product(std::vector<mpreal>{third, two, mpreal{3}}), two);
  EXPECT_TRUE(generic_sum(std::vector<mpreal>{}).is_exact());
  EXPECT_EQ(generic_sum(std::vector<double>{0.25, 0.75}), 1.0);
  EXPECT_EQ(generic_product(std::vector<double>{2, 3}), 6.0);
  auto copy = third;
  third += third;
  EXPECT_EQ(copy.to_string(), "1/3");
  third *= third;
  EXPECT_EQ(third.to_string(), "4/9");
  third /= third;
  EXPECT_EQ(third, 1);
  third -= third;
  EXPECT_EQ(third, 0);
  EXPECT_THROW(copy / zero, std::domain_error);
  EXPECT_THROW(copy /= zero, std::domain_error);
  EXPECT_EQ(copy.to_string(), "1/3");
}

TEST(MpReal, ExactValuesAcquirePrecisionOnlyAtAnApproximationBoundary)
{
  auto p = Precision::bits(80), q = Precision::bits(256);
  mpreal third = mpreal{1} / mpreal{3};
  mpreal x(2, p);
  EXPECT_EQ((third + x).precision(), p);
  EXPECT_EQ((x + third).precision(), p);
  EXPECT_EQ(third.at(p), mpreal(1, p) / 3);
  EXPECT_NE(third, third.at(p)); // Exact comparison must not round the rational.
  EXPECT_EQ(third <=> third.at(p), 1_mp / 3_mp <=> third.at(p));
  auto sum = generic_sum(std::vector<mpreal>{mpreal{}, x, mpreal{3}});
  EXPECT_EQ(sum, 5);
  EXPECT_EQ(sum.precision(), p);
  EXPECT_EQ((x - x).precision(), p); // An integer result remains approximate.
  EXPECT_THROW(generic_sum(std::vector<mpreal>{x, mpreal(1, q)}), std::invalid_argument);
  EXPECT_THROW(third.native_handle(), std::logic_error);
  EXPECT_THROW(x.at(Precision::exact()), std::invalid_argument);
  EXPECT_THROW(mpreal(0.1, Precision::exact()), std::logic_error);
  EXPECT_THROW(sqrt(mpreal{2}), std::logic_error);
  EXPECT_EQ(exp(mpreal{}), 1);
  EXPECT_EQ(pow(mpreal{2}, mpreal{3}), 8);
  EXPECT_THROW(pi<mpreal>.at(Precision::exact()), std::logic_error);
  EXPECT_THROW(epsilon(Precision::exact()), std::logic_error);
  EXPECT_EQ(sqrt(scalar_like(x, 2)), sqrt(mpreal(2, p)));
  EXPECT_EQ(scalar_like(x, 0).precision(), p);
  EXPECT_EQ(scalar_like(1.0, 2), 2.0);
  EXPECT_EQ(scalar_like(1.0f, 2), 2.0f);
  EXPECT_TRUE(scalar_like(third, 2).is_exact());
  x = mpreal{1};
  EXPECT_TRUE(x.is_exact());
  EXPECT_EQ(format_real(third), "1/3");
  std::vector<mpreal> states{mpreal{}, mpreal(1, p), mpreal(uninitialized)};
  for (auto const& a : states)
    for (auto const& b : states)
    {
      auto lhs = a, rhs = b;
      lhs.swap(rhs);
      EXPECT_EQ(lhs.initialized(), b.initialized());
      EXPECT_EQ(rhs.initialized(), a.initialized());
      if (lhs.initialized())
      {
        EXPECT_EQ(lhs.precision(), b.precision());
      }
      if (rhs.initialized())
      {
        EXPECT_EQ(rhs.precision(), a.precision());
      }
    }
}

TEST(MpReal, ExactRootsPowersAndIdentities)
{
  auto exact = Precision::exact();
  EXPECT_EQ(sqrt(mpreal("4/9", exact)), mpreal("2/3", exact));
  EXPECT_EQ(sqrt(mpreal{}), 0);
  EXPECT_EQ(pow(mpreal{2}, mpreal{-3}), mpreal("1/8", exact));
  EXPECT_EQ(pow(mpreal("-8/27", exact), mpreal("1/3", exact)), mpreal("-2/3", exact));
  EXPECT_EQ(pow(mpreal{16}, mpreal("3/4", exact)), 8);
  EXPECT_EQ(pow(mpreal{}, mpreal{}), 1);
  EXPECT_THROW(pow(mpreal{}, mpreal{-1}), std::domain_error);
  EXPECT_THROW(pow(mpreal{2}, mpreal("1/3", exact)), std::logic_error);
  EXPECT_THROW(sqrt(mpreal{-1}), std::logic_error);
  EXPECT_EQ(hypot(mpreal{3}, mpreal{4}), 5);
  EXPECT_EQ(log(mpreal{1}), 0);
  EXPECT_EQ(sin(mpreal{}), 0);
  EXPECT_EQ(cos(mpreal{}), 1);
  EXPECT_EQ(tan(mpreal{}), 0);
  EXPECT_EQ(atan(mpreal{}), 0);
  EXPECT_EQ(atan2(mpreal{}, mpreal{2}), 0);
  EXPECT_TRUE(is_exact(sqrt(mpreal{4})));
  EXPECT_FALSE(is_exact(sqrt(mpreal(4, Precision::bits(80)))));
}

TEST(MpReal, FractionsParseAndRoundTrip)
{
  auto exact = Precision::exact();
  auto p = Precision::bits(100);
  for (auto text : {"1/3", "-10/-15", "+4/-6", "0/9", "12345678901234567890/19"})
  {
    mpreal x(text, exact);
    EXPECT_EQ(mpreal(x.to_string(), exact), x);
    EXPECT_EQ(mpreal(text, p), x.at(p));
    EXPECT_EQ(parse_real<mpreal>(x.to_string(), exact), x);
  }
  for (auto text : {"/3", "1/", "1//2", "1/2/3", "1.5/2", "1/2e3", "1 /2", "1/ 2", "1/+"})
  {
    EXPECT_THROW(mpreal(text, exact), std::invalid_argument) << text;
    EXPECT_THROW(mpreal(text, p), std::invalid_argument) << text;
  }
  EXPECT_THROW(mpreal("1/0", exact), std::domain_error);
  EXPECT_THROW(mpreal("0/0", p), std::domain_error);
}

TEST(MpReal, NativeConversionRoundsExactValuesExplicitly)
{
  static_assert(!std::convertible_to<mpreal, float>);
  static_assert(!std::convertible_to<mpreal, double>);
  static_assert(!std::convertible_to<mpreal, long double>);
  auto exact = Precision::exact();
  EXPECT_EQ(static_cast<double>(mpreal("1/3", exact)), 1.0 / 3.0);
  EXPECT_EQ(static_cast<float>(mpreal("1/3", exact)), 1.0f / 3.0f);
  EXPECT_EQ(static_cast<long double>(mpreal("1/3", exact)), 1.0L / 3.0L);
  // Values immediately around a double midpoint force refinement beyond 69 bits.
  auto half_ulp = pow(mpreal{2}, mpreal{-53});
  auto tiny = pow(mpreal{2}, mpreal{-200});
  EXPECT_EQ(static_cast<double>(mpreal{1} + half_ulp), 1.0);
  EXPECT_EQ(static_cast<double>(mpreal{1} + half_ulp - tiny), 1.0);
  EXPECT_EQ(static_cast<double>(mpreal{1} + half_ulp + tiny), std::nextafter(1.0, 2.0));
  auto half_subnormal = pow(mpreal{2}, mpreal{-1075});
  EXPECT_EQ(static_cast<double>(half_subnormal), 0.0);
  EXPECT_EQ(static_cast<double>(half_subnormal * mpreal{3}), 2 * std::numeric_limits<double>::denorm_min());
  auto epsilon = pow(mpreal{2}, mpreal{-1200});
  EXPECT_EQ(static_cast<double>(half_subnormal + epsilon), std::numeric_limits<double>::denorm_min());
  EXPECT_TRUE(std::signbit(static_cast<double>(-half_subnormal)));
  EXPECT_EQ(static_cast<double>(pow(mpreal{2}, mpreal{1024})), std::numeric_limits<double>::infinity());
  EXPECT_THROW((void)static_cast<double>(mpreal(uninitialized)), std::logic_error);
}

TEST(MpReal, GenericExactnessQueries)
{
  static_assert(is_exact(1));
  static_assert(!is_exact(1.0));
  static_assert(!numeric_limits<mpreal>::is_exact);
  EXPECT_TRUE(is_exact(mpreal{}));
  EXPECT_TRUE(is_exact(exact_constant("1/3")));
  EXPECT_TRUE(is_exact(0.1_mp));
  EXPECT_FALSE(is_exact(mpreal(1, Precision::bits(80))));
  EXPECT_FALSE(is_exact(mpreal(uninitialized)));
  EXPECT_FALSE(is_exact(complex<double>{1, 0}));
}

TEST(MpReal, ExplicitMathPrecisionEvaluatesExactInputs)
{
  auto p = Precision::bits(80);
  using Unary = mpreal (*)(mpreal const&, Precision);
  for (Unary function : {static_cast<Unary>(&uni20::abs), &uni20::sqrt, &uni20::exp, &uni20::log, &uni20::sin,
                         &uni20::cos, &uni20::tan, &uni20::atan})
  {
    auto result = function(mpreal{1}, p);
    EXPECT_EQ(result.precision(), p);
    EXPECT_FALSE(result.is_exact());
    EXPECT_THROW(function(mpreal{1}, Precision::exact()), std::logic_error);
    EXPECT_THROW(function(mpreal(uninitialized), p), std::logic_error);
  }
  EXPECT_EQ(sqrt(mpreal{4}, p), 2);
  EXPECT_FALSE(sqrt(mpreal{4}, p).is_exact());
  EXPECT_EQ(exp(mpreal{}, p), 1);
  EXPECT_EQ(log(mpreal{1}, p), 0);
  EXPECT_EQ(sin(mpreal{}, p), 0);
  EXPECT_EQ(cos(mpreal{}, p), 1);
  EXPECT_EQ(tan(mpreal{}, p), 0);
  EXPECT_EQ(atan(mpreal{}, p), 0);
  EXPECT_EQ(abs(mpreal{-2}, p), 2);
  EXPECT_TRUE(isnan(sqrt(mpreal{-1}, p)));
}

TEST(MpReal, ExplicitMathPrecisionConvertsInputsBeforeEvaluation)
{
  auto p = Precision::bits(2), high = Precision::bits(80);
  mpreal x("32/5", Precision::exact());
  // At two bits 6.4 first rounds to 6; sqrt(6) then rounds to 2.
  // Rounding sqrt(6.4) directly would instead produce 3.
  EXPECT_EQ(sqrt(x, p), 2);
  EXPECT_EQ(sqrt(x.at(high)).at(p), 3);
  EXPECT_TRUE(x.is_exact());
  EXPECT_EQ(sqrt(x.at(high), p), 2);

  mpreal a(3, Precision::bits(80)), b(4, Precision::bits(256));
  EXPECT_THROW(hypot(a, b), std::invalid_argument);
  EXPECT_EQ(hypot(a, b, high), 5);
  EXPECT_EQ(pow(a, b, high), 81);
  EXPECT_EQ(atan2(a, b, high), atan2(a, b.at(high)));
  EXPECT_EQ(a.precision(), Precision::bits(80));
  EXPECT_EQ(b.precision(), Precision::bits(256));
  using Binary = mpreal (*)(mpreal const&, mpreal const&, Precision);
  for (Binary function : {static_cast<Binary>(&uni20::hypot), &uni20::pow, &uni20::atan2})
  {
    auto result = function(mpreal{3}, mpreal{4}, high);
    EXPECT_EQ(result.precision(), high);
    EXPECT_FALSE(result.is_exact());
    EXPECT_THROW(function(a, b, Precision::exact()), std::logic_error);
    EXPECT_THROW(function(mpreal(uninitialized), b, high), std::logic_error);
    EXPECT_THROW(function(a, mpreal(uninitialized), high), std::logic_error);
  }
}
