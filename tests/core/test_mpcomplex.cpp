#include <gtest/gtest.h>
#include <uni20/async/async.hpp>
#include <uni20/async/tbb_scheduler.hpp>
#include <uni20/core/math.hpp>
#include <uni20/core/scalar_io.hpp>
#include <vector>

using namespace uni20;
using namespace uni20::literals;
using C = uni20::complex<mpreal>;
static_assert(Complex<C> && Scalar<C>);
static_assert(std::same_as<make_real_t<C>, mpreal>);
static_assert(std::same_as<make_complex_t<mpreal>, C>);
static_assert(std::same_as<uni20::complex<double>, std::complex<double>>);
static_assert(!BlasComplex<C> && !LapackComplex<C>);
static_assert(std::is_nothrow_move_constructible_v<C>);

TEST(MpComplex, UnsetOwnershipAndExplicitPrecision)
{
  auto p = Precision::bits(256);
  C z(uninitialized);
  EXPECT_FALSE(z.initialized());
  EXPECT_THROW(z.native_handle(), std::logic_error);
  EXPECT_THROW(z.precision(), std::logic_error);
  EXPECT_THROW(z.real(), std::logic_error);
  EXPECT_THROW(abs(z), std::logic_error);
  EXPECT_THROW((void)(z == 0), std::logic_error);
  EXPECT_THROW(+z, std::logic_error);
  EXPECT_THROW(z.at(p), std::logic_error);
  auto copy = z;
  EXPECT_FALSE(copy.initialized());
  z = C("1.25", "-0.5", p);
  EXPECT_EQ(z.real(), 1.25_mp);
  EXPECT_EQ(z.imag(), -0.5_mp);
  EXPECT_EQ(z.precision(), p);
  auto moved = std::move(z);
  EXPECT_FALSE(z.initialized());
  EXPECT_EQ(moved.real(), 1.25_mp);
  copy = moved;
  moved.real(mpreal(3, p));
  EXPECT_EQ(copy.real(), 1.25_mp);
  auto& alias = copy;
  copy = std::move(alias);
  EXPECT_EQ(copy.real(), 1.25_mp);
  copy = z;
  EXPECT_FALSE(copy.initialized());
  std::vector<C> buffer(8);
  buffer[3] = moved;
  buffer.resize(24);
  EXPECT_EQ(buffer[3], moved);
  EXPECT_TRUE(buffer[23].is_exact());
  EXPECT_EQ(buffer[23], 0);
}

TEST(MpComplex, ComponentsConvertOnlyWhenExplicit)
{
  auto low = Precision::bits(3), high = Precision::bits(256);
  mpreal a("1.125", high), b(2, low);
  EXPECT_THROW((void)C(a, b), std::invalid_argument);
  C z(a, b, low);
  EXPECT_EQ(z.real(), 1);
  EXPECT_EQ(z.imag(), 2);
  z.real(a);
  z.imag(a);
  EXPECT_EQ(z.real(), 1);
  EXPECT_EQ(z.imag(), 1);
  EXPECT_EQ(z.precision(), low);
  z = C(a, high);
  EXPECT_EQ(z.precision(), high);
  EXPECT_EQ(z.real(), a);
  EXPECT_EQ(z.imag(), 0);
  C copy = z.at(low);
  EXPECT_EQ(copy.real(), 1);
  EXPECT_EQ(z.real(), 1.125_mp);
}

TEST(MpComplex, EagerArithmeticAndAliasing)
{
  auto p = Precision::bits(256);
  C a("1", "2", p), b("3", "-4", p);
  EXPECT_EQ(a + b, C("4", "-2", p));
  EXPECT_EQ(a - b, C("-2", "6", p));
  EXPECT_EQ(a * b, C("11", "2", p));
  EXPECT_EQ((a * b) / a, b);
  EXPECT_EQ(conj(a), C("1", "-2", p));
  EXPECT_EQ(norm(a), 5);
  EXPECT_EQ(abs(C("3", "4", p)), 5);
  auto result = a * b;
  a += a;
  EXPECT_EQ(a, C("2", "4", p));
  a *= a;
  EXPECT_EQ(a, C("-12", "16", p));
  a /= a;
  EXPECT_EQ(a, 1);
  a -= a;
  EXPECT_EQ(a, 0);
  EXPECT_EQ(result, C("11", "2", p));
  EXPECT_EQ(b + 2, C("5", "-4", p));
  EXPECT_EQ(2 - b, C("-1", "4", p));
  EXPECT_EQ(b * mpreal(2, p), C("6", "-8", p));
  EXPECT_EQ(mpreal(2, p) + b, C("5", "-4", p));
  EXPECT_EQ(b + 0.5_mp, C("3.5", "-4", p));
  b += 1;
  b *= 2;
  b -= 2;
  b /= 2;
  EXPECT_EQ(b, C("3", "-4", p));
  EXPECT_EQ(b, b.at(Precision::bits(400)));
  EXPECT_THROW(b += b.at(Precision::bits(80)), std::invalid_argument);
  EXPECT_EQ(b, C("3", "-4", p));
}

TEST(MpComplex, PrincipalBranchesAndRoundTripText)
{
  auto p = Precision::bits(256);
  EXPECT_EQ(sqrt(C("-4", "0", p)), C("0", "2", p));
  EXPECT_EQ(sqrt(C("-4", "-0", p)), C("0", "-2", p));
  EXPECT_EQ(exp(C(p)), 1);
  EXPECT_EQ(log(C(mpreal(1, p))), 0);
  EXPECT_EQ(log(C("-1", "0", p)).imag(), pi<mpreal>.at(p));
  EXPECT_EQ(log(C("-1", "-0", p)).imag(), -pi<mpreal>.at(p));
  C z("0.1", "-1.23456789012345678901234567890123456789", p);
  C round_trip(z.real().to_string(), z.imag().to_string(), p);
  EXPECT_EQ(z, round_trip);
  EXPECT_EQ(format_scalar(C("1.25", "-0.5", p), {.precision = 4}), "1.25-0.5i");
  EXPECT_EQ(format_scalar(C("0", "-0", p), {.precision = 4, .normalize_negative_zero = true}), "0+0i");
  EXPECT_TRUE(isfinite(z));
  EXPECT_TRUE(isnan(C("nan", "0", p)));
  EXPECT_TRUE(isinf(C("inf", "0", p)));
  EXPECT_FALSE(isfinite(C("inf", "0", p)));
  C nan("nan", "0", p);
  EXPECT_FALSE(nan == nan);
}

TEST(MpComplex, ElementaryFunctionsMatchIndependentReferences)
{
  // mpmath at 120 decimal digits, rounded here from 110-digit decimal references.
  for (auto bits : {80, 256})
  {
    auto p = Precision::bits(bits);
    C z("1", "2", p);
    EXPECT_EQ(exp(z), C("-1."
                        "1312043837568136384312552555107947106288679958265257502177219104165019166102261760470637653159"
                        "584169642662252",
                        "2."
                        "4717266720048189276169308935516645327361903692410081842007588352778396608113112040787193504355"
                        "692353270442651",
                        p));
    EXPECT_EQ(log(z), C("0."
                        "8047189562170501873003796666130938197628006771342588609563239457370894938538288823150669390465"
                        "8980539998315151",
                        "1."
                        "1071487177940905030170654601785370400700476454014326466765392074337103389773627940134171286861"
                        "70641434544191",
                        p));
    EXPECT_EQ(sqrt(z), C("1."
                         "272019649514068964252422461737491491715608041840096248616640382539297575536068011830384214988"
                         "4602585385141476",
                         "0."
                         "786151377757423286069558585842958929523122057837723237664901970101182047622310913711912889158"
                         "50813556487901224",
                         p));
  }
}

TEST(MpComplex, AsyncCalculationsCarryIndependentPrecisions)
{
  using namespace uni20::async;
  TbbScheduler scheduler{4};
  ScopedScheduler scope(&scheduler);
  std::vector<Async<C>> outputs;
  for (int bits : {64, 128, 256, 400})
  {
    auto p = Precision::bits(bits);
    outputs.emplace_back(C(p));
    scheduler.schedule([](WriteBuffer<C> output, Precision p) static -> AsyncTask {
      auto const before = mpfr_get_default_prec();
      auto root = sqrt(C("3", "4", p));
      EXPECT_EQ(mpfr_get_default_prec(), before);
      co_await output = std::move(root);
    }(outputs.back().write(), p));
  }
  int i = 0;
  for (int bits : {64, 128, 256, 400})
  {
    auto const& value = outputs[i++].get_wait();
    EXPECT_EQ(value.precision(), Precision::bits(bits));
    EXPECT_EQ(value.real(), 2);
    EXPECT_EQ(value.imag(), 1);
  }
}

TEST(MpComplex, ExactComplexArithmeticAndComponentTransitions)
{
  C zero{}, one{1};
  EXPECT_TRUE(zero.is_exact());
  EXPECT_EQ(zero, 0);
  C a(mpreal{1}, mpreal{2}), b(mpreal{3}, mpreal{-4});
  EXPECT_TRUE(a.is_exact());
  auto c = a * b;
  EXPECT_TRUE(c.is_exact());
  EXPECT_EQ(c.real(), 11);
  EXPECT_EQ(c.imag(), 2);
  EXPECT_EQ(c / b, a);
  EXPECT_EQ(norm(a), 5);
  EXPECT_TRUE(norm(a).is_exact());
  EXPECT_EQ(conj(a), C(mpreal{1}, mpreal{-2}));
  auto fraction = a / C{3};
  EXPECT_EQ(fraction.real().to_string(), "1/3");
  EXPECT_EQ(fraction.imag().to_string(), "2/3");
  EXPECT_THROW(a / zero, std::domain_error);
  EXPECT_THROW(a /= zero, std::domain_error);
  EXPECT_EQ(a.real(), 1);
  EXPECT_THROW(sqrt(a), std::logic_error);
  EXPECT_THROW(abs(a), std::logic_error);
  EXPECT_THROW(a.native_handle(), std::logic_error);
  EXPECT_TRUE(isfinite(a));
  EXPECT_EQ(format_scalar(fraction), "1/3+2/3i");
  auto p = Precision::bits(80), q = Precision::bits(256);
  auto approximate = a.at(p);
  EXPECT_EQ(approximate, a);
  EXPECT_FALSE(approximate.is_exact());
  EXPECT_EQ((a + approximate).precision(), p);
  EXPECT_EQ((approximate * a).precision(), p);
  EXPECT_THROW(approximate + a.at(q), std::invalid_argument);
  EXPECT_THROW(approximate.at(Precision::exact()), std::invalid_argument);
  C sum{};
  sum += a;
  EXPECT_TRUE(sum.is_exact());
  sum += approximate;
  EXPECT_EQ(sum.precision(), p);
  EXPECT_EQ(sum, C(mpreal{2}, mpreal{4}));
  C mixed(mpreal{1}, mpreal(2, p));
  EXPECT_EQ(mixed.precision(), p);
  EXPECT_EQ(mixed.real().precision(), p);
  a.real(mpreal{3});
  EXPECT_TRUE(a.is_exact());
  a.imag(mpreal(7, p));
  EXPECT_EQ(a.precision(), p);
  EXPECT_EQ(a.real(), 3);
  a.real(mpreal(8, q));
  EXPECT_EQ(a.precision(), p); // Existing finite component-setter policy.
  EXPECT_EQ(scalar_like(a, 1).precision(), p);
  EXPECT_EQ(scalar_like(a, 1), one);
  auto copied = fraction;
  auto moved = std::move(fraction);
  EXPECT_FALSE(fraction.initialized());
  EXPECT_EQ(copied, moved);
  copied.swap(approximate);
  EXPECT_FALSE(copied.is_exact());
  EXPECT_TRUE(approximate.is_exact());
}

TEST(MpComplex, ExactRootsPowersAndIdentities)
{
  using C = complex<mpreal>;
  C z(mpreal{3}, mpreal{4});
  EXPECT_EQ(sqrt(z), C(mpreal{2}, mpreal{1}));
  EXPECT_EQ(sqrt(conj(z)), C(mpreal{2}, mpreal{-1}));
  EXPECT_EQ(sqrt(C{-4}), C(mpreal{}, mpreal{2}));
  EXPECT_EQ(abs(z), 5);
  EXPECT_TRUE(is_exact(abs(z)));
  EXPECT_EQ(pow(z, C{3}), z * z * z);
  EXPECT_EQ(pow(z, C{-2}), C{1} / (z * z));
  EXPECT_EQ(pow(z, C(mpreal("1/2", Precision::exact()))), sqrt(z));
  EXPECT_EQ(exp(C{}), C{1});
  EXPECT_EQ(log(C{1}), C{});
  EXPECT_EQ(cos(C{}), C{1});
  EXPECT_EQ(sin(C{}), C{});
  EXPECT_EQ(arg(C{1}), 0);
  EXPECT_TRUE(is_exact(sqrt(z)));
  EXPECT_FALSE(is_exact(z.at(Precision::bits(100))));
  EXPECT_FALSE(is_exact(C(uninitialized)));
  EXPECT_THROW(sqrt(C{2}), std::logic_error);
}

TEST(MpComplex, ExplicitMathPrecisionEvaluatesExactInputs)
{
  auto p = Precision::bits(80);
  C z(mpreal{3}, mpreal{4});
  using Unary = C (*)(C const&, Precision);
  for (Unary function : {static_cast<Unary>(&uni20::sqrt), &uni20::exp, &uni20::log, &uni20::sin, &uni20::cos})
  {
    auto result = function(z, p);
    EXPECT_EQ(result.precision(), p);
    EXPECT_FALSE(result.is_exact());
    EXPECT_THROW(function(z, Precision::exact()), std::logic_error);
    EXPECT_THROW(function(C(uninitialized), p), std::logic_error);
  }
  using RealResult = mpreal (*)(C const&, Precision);
  for (RealResult function : {static_cast<RealResult>(&uni20::abs), &uni20::norm, &uni20::arg})
  {
    auto result = function(z, p);
    EXPECT_EQ(result.precision(), p);
    EXPECT_FALSE(result.is_exact());
    EXPECT_THROW(function(z, Precision::exact()), std::logic_error);
    EXPECT_THROW(function(C(uninitialized), p), std::logic_error);
  }
  EXPECT_EQ(sqrt(z, p), C(mpreal{2}, mpreal{1}));
  EXPECT_EQ(abs(z, p), 5);
  EXPECT_EQ(norm(z, p), 25);
  EXPECT_EQ(arg(C{1}, p), 0);
  EXPECT_EQ(exp(C{}, p), C{1});
  EXPECT_EQ(log(C{1}, p), C{});
  EXPECT_EQ(sin(C{}, p), C{});
  EXPECT_EQ(cos(C{}, p), C{1});
  EXPECT_EQ(abs(C(mpreal{1}, mpreal{1}), p), sqrt(mpreal{2}, p));
  EXPECT_TRUE(z.is_exact());
}

TEST(MpComplex, ExplicitMathPrecisionOverridesInputsAndPreservesCutSide)
{
  auto low = Precision::bits(2), p = Precision::bits(80), high = Precision::bits(256);
  C x(mpreal("32/5", Precision::exact()));
  EXPECT_EQ(sqrt(x, low), C{2});
  EXPECT_EQ(sqrt(x.at(high)).at(low), C{3});
  C a(mpreal{1}, mpreal{1});
  auto b = C{2}.at(high);
  EXPECT_THROW(pow(a.at(p), b), std::invalid_argument);
  auto square = pow(a.at(p), b, p);
  EXPECT_EQ(square, C(mpreal{}, mpreal{2}));
  EXPECT_EQ(square.precision(), p);
  EXPECT_EQ(b.precision(), high);
  EXPECT_FALSE(pow(C{2}, C{3}, p).is_exact());
  EXPECT_THROW(pow(a, b, Precision::exact()), std::logic_error);
  EXPECT_THROW(pow(C(uninitialized), b, p), std::logic_error);
  EXPECT_THROW(pow(a, C(uninitialized), p), std::logic_error);
  EXPECT_EQ(sqrt(C("-4", "-0", high), p), C("0", "-2", p));
  EXPECT_LT(log(C("-1", "-0", high), p).imag(), 0);
  EXPECT_GT(log(C{-1}, p).imag(), 0); // Exact zero has no negative sign.
}
