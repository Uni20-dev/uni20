#include <uni20/core/math.hpp>

#include <cstdint>
#include <gtest/gtest.h>

namespace
{
using namespace uni20;
namespace m = uni20::math;
mpreal q(char const* text) { return mpreal(text, Precision::exact()); }

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
    {"tgamma", uni20::tgamma, uni20::tgamma, [](mpreal const& x) { return m::tgamma(x); },
     [](mpreal const& x, Precision p) { return m::tgamma(x, p); }, mpfr_gamma, "1/4"},
    {"digamma", uni20::digamma, uni20::digamma, [](mpreal const& x) { return m::digamma(x); },
     [](mpreal const& x, Precision p) { return m::digamma(x, p); }, mpfr_digamma, "1/4"},
    {"erf", uni20::erf, uni20::erf, [](mpreal const& x) { return m::erf(x); },
     [](mpreal const& x, Precision p) { return m::erf(x, p); }, mpfr_erf, "1/4"},
    {"erfc", uni20::erfc, uni20::erfc, [](mpreal const& x) { return m::erfc(x); },
     [](mpreal const& x, Precision p) { return m::erfc(x, p); }, mpfr_erfc, "1/4"},
    {"zeta", uni20::zeta, uni20::zeta, [](mpreal const& x) { return m::zeta(x); },
     [](mpreal const& x, Precision p) { return m::zeta(x, p); }, mpfr_zeta, "5/4"},
    {"expint", uni20::expint, uni20::expint, [](mpreal const& x) { return m::expint(x); },
     [](mpreal const& x, Precision p) { return m::expint(x, p); }, mpfr_eint, "1/4"},
    {"dilog_real", uni20::dilog_real, uni20::dilog_real, [](mpreal const& x) { return m::dilog_real(x); },
     [](mpreal const& x, Precision p) { return m::dilog_real(x, p); }, mpfr_li2, "1/4"},
    {"airy_ai", uni20::airy_ai, uni20::airy_ai, [](mpreal const& x) { return m::airy_ai(x); },
     [](mpreal const& x, Precision p) { return m::airy_ai(x, p); }, mpfr_ai, "1/4"},
    {"exp10", uni20::exp10, uni20::exp10, [](mpreal const& x) { return m::exp10(x); },
     [](mpreal const& x, Precision p) { return m::exp10(x, p); }, mpfr_exp10, "1/4"},
    {"exp2m1", uni20::exp2m1, uni20::exp2m1, [](mpreal const& x) { return m::exp2m1(x); },
     [](mpreal const& x, Precision p) { return m::exp2m1(x, p); }, mpfr_exp2m1, "1/4"},
    {"exp10m1", uni20::exp10m1, uni20::exp10m1, [](mpreal const& x) { return m::exp10m1(x); },
     [](mpreal const& x, Precision p) { return m::exp10m1(x, p); }, mpfr_exp10m1, "1/4"},
    {"log2p1", uni20::log2p1, uni20::log2p1, [](mpreal const& x) { return m::log2p1(x); },
     [](mpreal const& x, Precision p) { return m::log2p1(x, p); }, mpfr_log2p1, "1/4"},
    {"log10p1", uni20::log10p1, uni20::log10p1, [](mpreal const& x) { return m::log10p1(x); },
     [](mpreal const& x, Precision p) { return m::log10p1(x, p); }, mpfr_log10p1, "1/4"},
    {"sinpi", uni20::sinpi, uni20::sinpi, [](mpreal const& x) { return m::sinpi(x); },
     [](mpreal const& x, Precision p) { return m::sinpi(x, p); }, mpfr_sinpi, "1/4"},
    {"cospi", uni20::cospi, uni20::cospi, [](mpreal const& x) { return m::cospi(x); },
     [](mpreal const& x, Precision p) { return m::cospi(x, p); }, mpfr_cospi, "1/4"},
    {"tanpi", uni20::tanpi, uni20::tanpi, [](mpreal const& x) { return m::tanpi(x); },
     [](mpreal const& x, Precision p) { return m::tanpi(x, p); }, mpfr_tanpi, "1/8"},
    {"asinpi", uni20::asinpi, uni20::asinpi, [](mpreal const& x) { return m::asinpi(x); },
     [](mpreal const& x, Precision p) { return m::asinpi(x, p); }, mpfr_asinpi, "1/4"},
    {"acospi", uni20::acospi, uni20::acospi, [](mpreal const& x) { return m::acospi(x); },
     [](mpreal const& x, Precision p) { return m::acospi(x, p); }, mpfr_acospi, "1/4"},
    {"atanpi", uni20::atanpi, uni20::atanpi, [](mpreal const& x) { return m::atanpi(x); },
     [](mpreal const& x, Precision p) { return m::atanpi(x, p); }, mpfr_atanpi, "1/4"},
    {"sec", uni20::sec, uni20::sec, [](mpreal const& x) { return m::sec(x); },
     [](mpreal const& x, Precision p) { return m::sec(x, p); }, mpfr_sec, "1/4"},
    {"csc", uni20::csc, uni20::csc, [](mpreal const& x) { return m::csc(x); },
     [](mpreal const& x, Precision p) { return m::csc(x, p); }, mpfr_csc, "1/4"},
    {"cot", uni20::cot, uni20::cot, [](mpreal const& x) { return m::cot(x); },
     [](mpreal const& x, Precision p) { return m::cot(x, p); }, mpfr_cot, "1/4"},
    {"sech", uni20::sech, uni20::sech, [](mpreal const& x) { return m::sech(x); },
     [](mpreal const& x, Precision p) { return m::sech(x, p); }, mpfr_sech, "1/4"},
    {"csch", uni20::csch, uni20::csch, [](mpreal const& x) { return m::csch(x); },
     [](mpreal const& x, Precision p) { return m::csch(x, p); }, mpfr_csch, "1/4"},
    {"coth", uni20::coth, uni20::coth, [](mpreal const& x) { return m::coth(x); },
     [](mpreal const& x, Precision p) { return m::coth(x, p); }, mpfr_coth, "1/4"},
};

TEST(MprealSpecial, UnaryProviderWiringPrecisionAndUnset)
{
  for (auto p : {Precision::bits(80), Precision::bits(128), Precision::bits(256)})
    for (auto const& f : functions)
    {
      SCOPED_TRACE(f.name);
      SCOPED_TRACE(p.bit_count());
      auto x = mpreal(f.input, p);
      auto expected =
          detail::mpreal_access::finite_result(p, [&](mpfr_ptr out) { f.provider(out, x.native_handle(), MPFR_RNDN); });
      EXPECT_EQ(f.scalar(x), expected);
      EXPECT_EQ(f.scalar(x).precision(), p);
      EXPECT_EQ(f.generic(x), expected);
      EXPECT_EQ(f.generic(x).precision(), p);
      for (auto source : {Precision::exact(), Precision::bits(320)})
      {
        auto input = mpreal(f.input, source);
        EXPECT_EQ(f.with_precision(input, p), expected);
        EXPECT_EQ(f.generic_with_precision(input, p), expected);
        EXPECT_EQ(f.with_precision(input, p).precision(), p);
        EXPECT_EQ(input.precision(), source);
      }
      EXPECT_THROW(f.with_precision(x, Precision::exact()), std::logic_error);
      EXPECT_THROW(f.scalar(mpreal(uninitialized)), std::logic_error);
      EXPECT_THROW(f.with_precision(mpreal(uninitialized), p), std::logic_error);
      EXPECT_TRUE(isnan(f.generic(mpreal("nan", p))));
    }
}

struct BinaryFunction
{
    char const* name;
    mpreal (*scalar)(mpreal const&, mpreal const&);
    mpreal (*with_precision)(mpreal const&, mpreal const&, Precision);
    mpreal (*generic)(mpreal const&, mpreal const&);
    mpreal (*generic_with_precision)(mpreal const&, mpreal const&, Precision);
    int (*provider)(mpfr_ptr, mpfr_srcptr, mpfr_srcptr, mpfr_rnd_t);
};
BinaryFunction const binary_functions[] = {
    {"beta", uni20::beta, uni20::beta, [](mpreal const& x, mpreal const& y) { return m::beta(x, y); },
     [](mpreal const& x, mpreal const& y, Precision p) { return m::beta(x, y, p); }, mpfr_beta},
    {"upper_gamma", uni20::upper_gamma, uni20::upper_gamma,
     [](mpreal const& x, mpreal const& y) { return m::upper_gamma(x, y); },
     [](mpreal const& x, mpreal const& y, Precision p) { return m::upper_gamma(x, y, p); }, mpfr_gamma_inc},
    {"agm", uni20::agm, uni20::agm, [](mpreal const& x, mpreal const& y) { return m::agm(x, y); },
     [](mpreal const& x, mpreal const& y, Precision p) { return m::agm(x, y, p); }, mpfr_agm},
    {"atan2pi", uni20::atan2pi, uni20::atan2pi, [](mpreal const& x, mpreal const& y) { return m::atan2pi(x, y); },
     [](mpreal const& x, mpreal const& y, Precision p) { return m::atan2pi(x, y, p); }, mpfr_atan2pi},
};

TEST(MprealSpecial, BinaryProviderWiringAndMixedPrecision)
{
  for (auto p : {Precision::bits(128), Precision::bits(256)})
    for (auto const& f : binary_functions)
    {
      SCOPED_TRACE(f.name);
      auto a = mpreal("5/4", p), b = mpreal("1/4", p);
      auto expected = detail::mpreal_access::finite_result(
          p, [&](mpfr_ptr out) { f.provider(out, a.native_handle(), b.native_handle(), MPFR_RNDN); });
      EXPECT_EQ(f.scalar(a, b), expected);
      EXPECT_EQ(f.generic(a, b), expected);
      EXPECT_EQ(f.scalar(a, b).precision(), p);
      EXPECT_EQ(f.generic(q("5/4"), b), expected);
      EXPECT_EQ(f.generic(a, q("1/4")), expected);
      auto high = a.at(Precision::bits(320));
      EXPECT_THROW(f.scalar(high, b), std::invalid_argument);
      EXPECT_EQ(f.with_precision(high, b, p), expected);
      EXPECT_EQ(f.generic_with_precision(high, b, p), expected);
      EXPECT_EQ(f.with_precision(q("5/4"), q("1/4"), p).precision(), p);
      EXPECT_THROW(f.with_precision(a, b, Precision::exact()), std::logic_error);
      EXPECT_THROW(f.scalar(mpreal(uninitialized), b), std::logic_error);
      EXPECT_THROW(f.scalar(a, mpreal(uninitialized)), std::logic_error);
      EXPECT_THROW(f.with_precision(a, mpreal(uninitialized), p), std::logic_error);
    }
}

TEST(MprealSpecial, ExactIdentitiesAndRationalValues)
{
  auto check = [](mpreal const& value, char const* expected) {
    EXPECT_TRUE(value.is_exact());
    EXPECT_EQ(value, q(expected));
  };
  check(m::tgamma(mpreal{7}), "720");
  check(m::factorial(10, Precision::exact()), "3628800");
  check(m::beta(mpreal{3}, mpreal{4}), "1/60");
  check(m::beta(q("1/2"), mpreal{3}), "16/15");
  auto large = m::pown(mpreal{10}, 100);
  EXPECT_EQ(m::beta(large, mpreal{1}), 1 / large);
  EXPECT_EQ(m::beta(mpreal{1}, large), 1 / large);
  check(m::upper_gamma(mpreal{4}, mpreal{}), "6");
  check(m::zeta(mpreal{}), "-1/2");
  check(m::zeta(mpreal{-1}), "-1/12");
  check(m::zeta(mpreal{-3}), "1/120");
  check(m::zeta(mpreal{-11}), "691/32760");
  check(m::zeta(mpreal{-24}), "0");
  check(m::erf(mpreal{}), "0");
  check(m::erfc(mpreal{}), "1");
  check(m::dilog_real(mpreal{}), "0");
  check(m::agm(q("2/3"), q("2/3")), "2/3");
  check(m::agm(mpreal{}, mpreal{2}), "0");
  check(m::bessel_j(0, mpreal{}), "1");
  check(m::bessel_j(-3, mpreal{}), "0");
  check(m::lgamma(mpreal{2}), "0");
  EXPECT_EQ(m::lgamma_sign(mpreal{1}).sign, 1);
  EXPECT_THROW(m::tgamma(q("1/2")), std::logic_error);
  EXPECT_THROW(m::digamma(mpreal{1}), std::logic_error);
  EXPECT_THROW(m::zeta(mpreal{2}), std::logic_error);
  EXPECT_THROW(m::beta(q("1/2"), q("1/2")), std::logic_error);
  EXPECT_THROW(m::upper_gamma(mpreal{3}, mpreal{1}), std::logic_error);
  EXPECT_THROW(m::expint(mpreal{1}), std::logic_error);
  EXPECT_THROW(m::dilog_real(mpreal{1}), std::logic_error);
  EXPECT_THROW(m::airy_ai(mpreal{}), std::logic_error);
  EXPECT_THROW(m::agm(mpreal{1}, mpreal{2}), std::logic_error);
  EXPECT_THROW(m::bessel_j(0, mpreal{1}), std::logic_error);
  EXPECT_THROW(m::bessel_y(0, mpreal{1}), std::logic_error);
}

TEST(MprealSpecial, LogGammaSignsAndDomains)
{
  auto p = Precision::bits(128);
  for (auto text : {"-1/2", "-3/2", "1/4", "1", "2", "-0", "0", "inf"})
  {
    auto x = mpreal(text, p);
    int sign = 0;
    auto expected = detail::mpreal_access::finite_result(
        p, [&](mpfr_ptr out) { mpfr_lgamma(out, &sign, x.native_handle(), MPFR_RNDN); });
    auto result = m::lgamma_sign(x);
    EXPECT_EQ(result.value, expected);
    EXPECT_EQ(result.value.precision(), p);
    EXPECT_EQ(result.sign, sign);
    EXPECT_EQ(m::lgamma(x), result.value);
    EXPECT_EQ(m::lgamma_sign(x.at(Precision::bits(256)), p).value, result.value);
    EXPECT_EQ(m::lgamma_sign(x.at(Precision::bits(256)), p).sign, sign);
    EXPECT_EQ(m::lgamma(x.at(Precision::bits(256)), p), result.value);
  }
  EXPECT_EQ(m::lgamma_sign(mpreal("-1/2", p)).sign, -1);
  EXPECT_EQ(m::lgamma_sign(mpreal("-3/2", p)).sign, 1);
  for (auto text : {"-1", "-2", "-inf", "nan"})
    EXPECT_EQ(m::lgamma_sign(mpreal(text, p)).sign, 0);
  EXPECT_THROW(m::lgamma_sign(mpreal(uninitialized)), std::logic_error);
  EXPECT_THROW(m::lgamma_sign(mpreal{3}), std::logic_error);
  EXPECT_THROW(m::lgamma_sign(mpreal{1}, Precision::exact()), std::logic_error);
  EXPECT_TRUE(isnan(m::tgamma(mpreal(-2, p))));
  EXPECT_TRUE(isnan(m::digamma(mpreal(-2, p))));
  EXPECT_TRUE(isnan(m::bessel_y(1, mpreal(-1, p))));
  EXPECT_TRUE(isinf(m::zeta(mpreal(1, p))));
  EXPECT_TRUE(isnan(m::agm(mpreal(-1, p), mpreal(2, p))));
  EXPECT_TRUE(isnan(m::asinpi(mpreal(2, p))));
  EXPECT_TRUE(isnan(m::log2p1(mpreal(-2, p))));
  EXPECT_TRUE(isnan(m::log10p1(mpreal(-2, p))));
  EXPECT_EQ(m::erf(mpreal("inf", p)), 1);
  EXPECT_EQ(m::erfc(mpreal("inf", p)), 0);
  EXPECT_TRUE(signbit(m::erf(mpreal("-0", p))));
  mpreal (*preserves_zero_sign[])(mpreal const&) = {uni20::exp2m1, uni20::exp10m1, uni20::log2p1, uni20::log10p1,
                                                    uni20::sinpi,  uni20::tanpi,   uni20::asinpi, uni20::atanpi};
  for (auto f : preserves_zero_sign)
    EXPECT_TRUE(signbit(f(mpreal("-0", p))));
}

TEST(MprealSpecial, IntegerOrdersAndConstants)
{
  for (auto p : {Precision::bits(128), Precision::bits(256)})
  {
    auto x = mpreal("1/4", p);
    for (int n : {-3, 0, 1, 4})
    {
      auto j =
          detail::mpreal_access::finite_result(p, [&](mpfr_ptr out) { mpfr_jn(out, n, x.native_handle(), MPFR_RNDN); });
      auto y =
          detail::mpreal_access::finite_result(p, [&](mpfr_ptr out) { mpfr_yn(out, n, x.native_handle(), MPFR_RNDN); });
      EXPECT_EQ(m::bessel_j(n, x), j);
      EXPECT_EQ(m::bessel_y(n, x), y);
      EXPECT_EQ(m::bessel_j(n, q("1/4"), p), j);
      EXPECT_EQ(m::bessel_y(n, q("1/4"), p), y);
      EXPECT_EQ(m::bessel_j(n, x).precision(), p);
      EXPECT_EQ(m::bessel_y(n, x).precision(), p);
    }
    auto fac = detail::mpreal_access::finite_result(p, [&](mpfr_ptr out) { mpfr_fac_ui(out, 100, MPFR_RNDN); });
    EXPECT_EQ(m::factorial(100, p), fac);
    EXPECT_EQ(m::factorial(100, p).precision(), p);
    auto check = [&](auto descriptor, auto provider) {
      auto expected = detail::mpreal_access::finite_result(p, [&](mpfr_ptr out) { provider(out, MPFR_RNDN); });
      EXPECT_EQ(descriptor.at(p), expected);
      EXPECT_EQ(descriptor.at(p).precision(), p);
      EXPECT_EQ(mpreal(1, p) * descriptor, expected);
      EXPECT_EQ(descriptor / mpreal(1, p), expected);
      EXPECT_THROW(descriptor.at(Precision::exact()), std::logic_error);
      EXPECT_THROW(mpreal{1} * descriptor, std::logic_error);
    };
    check(pi<mpreal>, mpfr_const_pi);
    check(log_two<mpreal>, mpfr_const_log2);
    check(euler_gamma<mpreal>, mpfr_const_euler);
    check(catalan<mpreal>, mpfr_const_catalan);
  }
  auto p = Precision::bits(128);
  auto huge = numeric_limits<std::uint64_t>::max();
  EXPECT_THROW(m::bessel_j(huge, mpreal(1, p)), std::out_of_range);
  EXPECT_THROW(m::bessel_y(huge, mpreal(1, p)), std::out_of_range);
  EXPECT_THROW(m::factorial(huge, p), std::out_of_range);
  EXPECT_THROW(m::factorial(-1, p), std::domain_error);
  EXPECT_THROW(m::bessel_j(1, mpreal(uninitialized)), std::logic_error);
  EXPECT_THROW(m::bessel_y(1, mpreal(uninitialized)), std::logic_error);
  EXPECT_THROW(m::bessel_j(1, mpreal{0}, Precision::exact()), std::logic_error);
  EXPECT_THROW(m::bessel_y(1, mpreal{0}, Precision::exact()), std::logic_error);
}
} // namespace
