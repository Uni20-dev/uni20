#include <uni20/core/math.hpp>

#include <gtest/gtest.h>
#include <string_view>

namespace
{
using namespace uni20;

struct UnaryFunction
{
    char const* name;
    mpreal (*scalar)(mpreal const&);
    mpreal (*with_precision)(mpreal const&, Precision);
    mpreal (*generic)(mpreal const&);
    mpreal (*generic_with_precision)(mpreal const&, Precision);
    int (*provider)(mpfr_ptr, mpfr_srcptr, mpfr_rnd_t);
    char const* input;
};

UnaryFunction const functions[] = {
    {"exp2", uni20::exp2, uni20::exp2, [](mpreal const& x) { return math::exp2(x); },
     [](mpreal const& x, Precision p) { return math::exp2(x, p); }, mpfr_exp2, "1/4"},
    {"expm1", uni20::expm1, uni20::expm1, [](mpreal const& x) { return math::expm1(x); },
     [](mpreal const& x, Precision p) { return math::expm1(x, p); }, mpfr_expm1, "1/4"},
    {"log2", uni20::log2, uni20::log2, [](mpreal const& x) { return math::log2(x); },
     [](mpreal const& x, Precision p) { return math::log2(x, p); }, mpfr_log2, "1/4"},
    {"log10", uni20::log10, uni20::log10, [](mpreal const& x) { return math::log10(x); },
     [](mpreal const& x, Precision p) { return math::log10(x, p); }, mpfr_log10, "1/4"},
    {"log1p", uni20::log1p, uni20::log1p, [](mpreal const& x) { return math::log1p(x); },
     [](mpreal const& x, Precision p) { return math::log1p(x, p); }, mpfr_log1p, "1/4"},
    {"cbrt", uni20::cbrt, uni20::cbrt, [](mpreal const& x) { return math::cbrt(x); },
     [](mpreal const& x, Precision p) { return math::cbrt(x, p); }, mpfr_cbrt, "-1/4"},
    {"asin", uni20::asin, uni20::asin, [](mpreal const& x) { return math::asin(x); },
     [](mpreal const& x, Precision p) { return math::asin(x, p); }, mpfr_asin, "1/4"},
    {"acos", uni20::acos, uni20::acos, [](mpreal const& x) { return math::acos(x); },
     [](mpreal const& x, Precision p) { return math::acos(x, p); }, mpfr_acos, "1/4"},
    {"sinh", uni20::sinh, uni20::sinh, [](mpreal const& x) { return math::sinh(x); },
     [](mpreal const& x, Precision p) { return math::sinh(x, p); }, mpfr_sinh, "1/4"},
    {"cosh", uni20::cosh, uni20::cosh, [](mpreal const& x) { return math::cosh(x); },
     [](mpreal const& x, Precision p) { return math::cosh(x, p); }, mpfr_cosh, "1/4"},
    {"tanh", uni20::tanh, uni20::tanh, [](mpreal const& x) { return math::tanh(x); },
     [](mpreal const& x, Precision p) { return math::tanh(x, p); }, mpfr_tanh, "1/4"},
    {"asinh", uni20::asinh, uni20::asinh, [](mpreal const& x) { return math::asinh(x); },
     [](mpreal const& x, Precision p) { return math::asinh(x, p); }, mpfr_asinh, "1/4"},
    {"acosh", uni20::acosh, uni20::acosh, [](mpreal const& x) { return math::acosh(x); },
     [](mpreal const& x, Precision p) { return math::acosh(x, p); }, mpfr_acosh, "5/4"},
    {"atanh", uni20::atanh, uni20::atanh, [](mpreal const& x) { return math::atanh(x); },
     [](mpreal const& x, Precision p) { return math::atanh(x, p); }, mpfr_atanh, "1/4"},
};

TEST(MprealElementary, MatchesDedicatedProviderAndPrecisionOverrides)
{
  for (auto p : {Precision::bits(80), Precision::bits(256)})
    for (auto const& f : functions)
    {
      SCOPED_TRACE(f.name);
      SCOPED_TRACE(p.bit_count());
      auto x = mpreal(f.input, p);
      mpfr_t expected;
      mpfr_init2(expected, p.bit_count());
      f.provider(expected, x.native_handle(), MPFR_RNDN);
      auto result = f.scalar(x);
      EXPECT_EQ(result.precision(), p);
      EXPECT_FALSE(result.is_exact());
      EXPECT_EQ(mpfr_cmp(result.native_handle(), expected), 0);
      EXPECT_EQ(f.generic(x), result);
      EXPECT_EQ(f.generic(x).precision(), p);

      // Explicit precision converts the input before evaluation, including
      // when the original value was exact or had a different finite precision.
      for (auto input_precision : {Precision::exact(), Precision::bits(320)})
      {
        auto input = mpreal(f.input, input_precision);
        auto explicit_result = f.with_precision(input, p);
        EXPECT_EQ(explicit_result, result);
        EXPECT_EQ(explicit_result.precision(), p);
        EXPECT_EQ(f.generic_with_precision(input, p), result);
        EXPECT_EQ(input.precision(), input_precision);
      }
      EXPECT_THROW(f.with_precision(x, Precision::exact()), std::logic_error);
      EXPECT_THROW(f.scalar(mpreal(uninitialized)), std::logic_error);
      EXPECT_THROW(f.with_precision(mpreal(uninitialized), p), std::logic_error);
      mpfr_clear(expected);
    }
}

TEST(MprealElementary, ExactRationalResults)
{
  auto exact = [](char const* text) { return mpreal(text, Precision::exact()); };
  auto check = [&](mpreal const& result, char const* expected) {
    EXPECT_TRUE(result.is_exact());
    EXPECT_EQ(result, exact(expected));
  };
  check(cbrt(exact("-8/27")), "-2/3");
  check(cbrt(mpreal{}), "0");
  check(cbrt(exact("1/1000000")), "1/100");
  check(exp2(mpreal{-3}), "1/8");
  check(exp2(mpreal{}), "1");
  check(log2(exact("1/8")), "-3");
  check(log2(mpreal{8}), "3");
  check(log2(exp2(mpreal{300})), "300");
  check(log10(exact("1/1000")), "-3");
  check(log10(mpreal{100}), "2");
  check(log10(pow(mpreal{10}, mpreal{100})), "100");
  check(log2(mpreal{1}), "0");
  check(log10(mpreal{1}), "0");
  check(expm1(mpreal{}), "0");
  check(log1p(mpreal{}), "0");
  check(asin(mpreal{}), "0");
  check(acos(mpreal{1}), "0");
  check(sinh(mpreal{}), "0");
  check(cosh(mpreal{}), "1");
  check(tanh(mpreal{}), "0");
  check(asinh(mpreal{}), "0");
  check(acosh(mpreal{1}), "0");
  check(atanh(mpreal{}), "0");
  EXPECT_THROW(cbrt(mpreal{2}), std::logic_error);
  EXPECT_THROW(exp2(exact("1/2")), std::logic_error);
  for (auto x : {exact("0"), exact("-1"), exact("3"), exact("2/3")})
  {
    EXPECT_THROW(log2(x), std::logic_error);
    EXPECT_THROW(log10(x), std::logic_error);
  }
  EXPECT_THROW(log10(mpreal{2}), std::logic_error);
  EXPECT_THROW(log10(mpreal{5}), std::logic_error);
  EXPECT_THROW(log10(mpreal{20}), std::logic_error);
  // Every added function requires finite precision for these nontrivial inputs.
  for (auto const& f : functions)
  {
    if (std::string_view(f.name) == "log2") continue; // log2(1/4) is exactly -2.
    SCOPED_TRACE(f.name);
    EXPECT_THROW(f.scalar(exact(f.input)), std::logic_error);
  }
}

TEST(MprealElementary, ApproximateIdentitiesStayApproximate)
{
  auto p = Precision::bits(80);
  for (auto const& f : functions)
  {
    SCOPED_TRACE(f.name);
    auto name = std::string_view(f.name);
    int n = ((name.starts_with("log") && name != "log1p") || name == "acos" || name == "acosh") ? 1 : 0;
    auto input = mpreal(n, p);
    EXPECT_FALSE(f.scalar(input).is_exact());
    EXPECT_EQ(f.scalar(input).precision(), p);
    EXPECT_EQ(f.with_precision(mpreal{n}, p).precision(), p);
  }
}

TEST(MprealElementary, SignedZeroAndExceptionalValues)
{
  auto p = Precision::bits(128);
  auto minus_zero = mpreal("-0", p);
  mpreal (*preserves_zero_sign[])(mpreal const&) = {uni20::expm1, uni20::log1p, uni20::cbrt,  uni20::asin,
                                                    uni20::sinh,  uni20::tanh,  uni20::asinh, uni20::atanh};
  for (auto f : preserves_zero_sign)
  {
    EXPECT_EQ(f(minus_zero), 0);
    EXPECT_TRUE(signbit(f(minus_zero)));
  }
  EXPECT_TRUE(isnan(log1p(mpreal(-2, p))));
  EXPECT_TRUE(isnan(asin(mpreal(2, p))));
  EXPECT_TRUE(isnan(acos(mpreal(2, p))));
  EXPECT_TRUE(isnan(acosh(mpreal(0, p))));
  EXPECT_TRUE(isnan(atanh(mpreal(2, p))));
  auto negative_infinity = mpreal("-inf", p);
  EXPECT_EQ(log1p(mpreal(-1, p)), negative_infinity);
  EXPECT_EQ(log2(mpreal(0, p)), negative_infinity);
  EXPECT_EQ(log10(mpreal(0, p)), negative_infinity);
  EXPECT_EQ(atanh(mpreal(-1, p)), negative_infinity);
  EXPECT_EQ(atanh(mpreal(1, p)), -negative_infinity);
  EXPECT_EQ(cbrt(negative_infinity), negative_infinity);
  EXPECT_EQ(expm1(negative_infinity), -1);
  EXPECT_EQ(tanh(mpreal("1e1000", p)), 1);
  EXPECT_EQ(tanh(mpreal("-1e1000", p)), -1);
  for (auto const& f : functions)
    EXPECT_TRUE(isnan(f.scalar(mpreal("nan", p))));
}
} // namespace
