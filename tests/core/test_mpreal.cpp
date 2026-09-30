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
static_assert(!std::default_initializable<mpreal>);
static_assert(!std::convertible_to<double, mpreal>);
static_assert(!std::convertible_to<mpreal, double>);
static_assert(!AddsNativeFloat<mpreal>);
static_assert(!HasPrecisionlessSqrt<decimal_literal>);
static_assert(Real<mpreal> && Scalar<mpreal>);
static_assert(!BlasReal<mpreal> && !LapackReal<mpreal>);
static_assert(!HasComplexSpelling<mpreal> && !HasComplexCounterpart<mpreal>);
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
  EXPECT_EQ(source, 0);
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
  x += 0.14_mp; // 0.14 first rounds to 0.125; the sum is a tie that rounds to 1.
  EXPECT_EQ(x, 1);
  x -= 0.07_mp; // 0.07 first rounds to 0.0625; the difference rounds to 1.
  EXPECT_EQ(x, 1);
  x = mpreal("1.25", low);
  x *= 1.1_mp; // 1.1 rounds to 1 before multiplication.
  EXPECT_EQ(x, 1.25_mp);
  x = mpreal(1, low);
  x /= 1.1_mp;
  EXPECT_EQ(x, 1);
  EXPECT_EQ(x.precision(), low);
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
  for (auto text : {"", "+", "nan", "1/3", "1.2.3", "1e", "0x10", "1''2"})
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
      co_await out = sqrt(mpreal(2, p));
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
    EXPECT_EQ(exp(mpreal(1, p)), mpreal(e, p));
    EXPECT_EQ(log(mpreal(2, p)), mpreal(log_two, p));
    auto angle = pi<mpreal>.at(p);
    EXPECT_LT(abs(sin(angle / 6) - 0.5_mp), 2 * epsilon(p));
    EXPECT_LT(abs(cos(angle / 3) - 0.5_mp), 2 * epsilon(p));
  }
}
