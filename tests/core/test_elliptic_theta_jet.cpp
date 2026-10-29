#include "theta_jet_references.hpp"
#include <gtest/gtest.h>
#include <uni20/core/elliptic_theta_jet.hpp>
#include <uni20/core/scalar_io.hpp>

namespace
{
using namespace uni20;
template <Complex C> C actual(theta_jet_result<C> const& jet, unsigned d = 0)
{
  return math::exp(jet.log_scale) * jet.derivative[d];
}

template <class R> class ThetaJetNative : public testing::Test {};
using NativeReals = testing::Types<float, double, long double
#if UNI20_HAS_FLOAT128
                                   ,
                                   float128
#endif
                                   >;
TYPED_TEST_SUITE(ThetaJetNative, NativeReals);

TYPED_TEST(ThetaJetNative, IndependentValuesAndDerivatives)
{
  using R = TypeParam;
  using C = complex<R>;
  std::array<std::array<char const*, 2>, 4> const reference{
      {{"0.69219558972113418875657598555125487595689802745775963095538321277550698026515883",
        "0.30295842568440231704359734352559357616656419512939285695290802635041584770007338"},
       {"0.97370036030675749908185309462479722684202005621228971607055473603044726260643021",
        "-0.23134011872194558048036441888787309500055703385516093235047558690923750031527775"},
       {"1.0820563314307416382141902966895917672243922526193374482791519769533824502881634",
        "-0.14172351980229852150061527154356932129353427560991712023159400835264615038631135"},
       {"0.91701403643718451700049740426185963471839253409158838269591152285563434881747893",
        "0.14114932439016119087397800722902770968704101496998396921409793366468937888358458"}}};
  R const eps = numeric_limits<R>::epsilon(), t = parse_real<R>("0.7");
  C const u(parse_real<R>("0.2"), parse_real<R>("0.1"));
  for (unsigned k = 1; k <= 4; ++k)
  {
    auto jet = elliptic_theta_jet(k, u, t);
    C expected(parse_real<R>(reference[k - 1][0]), parse_real<R>(reference[k - 1][1]));
    EXPECT_LE(math::abs(actual(jet) - expected), R(512) * eps);
  }
  auto jet = elliptic_theta_jet(1, u, t);
  C first(parse_real<R>("3.140373175101863767109126990443280790919216481608393060449146373266355474"),
          parse_real<R>("-0.542140463819793229105053265208634636705604224096074518836382433938263608"));
  C second(parse_real<R>("-5.256238827195634225526959228714510493158642457037190918060930185253611353"),
           parse_real<R>("-3.365849286153387228448025426576455363601810484091901718883225391186697882"));
  EXPECT_LE(math::abs(actual(jet, 1) - first), R(1024) * eps);
  EXPECT_LE(math::abs(actual(jet, 2) - second), R(4096) * eps);
}

TYPED_TEST(ThetaJetNative, PeriodsParityAndCommonScale)
{
  using R = TypeParam;
  using C = complex<R>;
  R const eps = numeric_limits<R>::epsilon(), pi_value = R(4) * math::atan(R(1));
  for (R t : {R(1) / 2, R(1), R(2)})
    for (unsigned k = 1; k <= 4; ++k)
    {
      C const u(R(1) / 8, t / 8);
      auto base = elliptic_theta_jet(k, u, t);
      for (int n : {-2, -1, 0, 1, 2})
        for (int m : {-2, -1, 0, 1, 2})
        {
          auto shifted = elliptic_theta_jet(k, u + C(R(n), R(m) * t), t);
          int sign = ((k == 1 ? n + m : k == 2 ? n : k == 4 ? m : 0) % 2) ? -1 : 1;
          C slope(R(0), -R(2) * pi_value * R(m));
          C multiplier = R(sign) * math::exp(pi_value * R(m * m) * t + slope * u);
          std::array<C, 3> expected{actual(base), actual(base, 1) + slope * actual(base),
                                    actual(base, 2) + R(2) * slope * actual(base, 1) + slope * slope * actual(base)};
          for (unsigned d = 0; d < 3; ++d)
            EXPECT_LE(math::abs(actual(shifted, d) / multiplier - expected[d]),
                      R(32768) * eps * (R(1) + math::abs(expected[d])));
        }
      auto zero = elliptic_theta_jet(k, C{}, t);
      EXPECT_TRUE(zero.derivative[k == 1 ? 0 : 1] == C{});
      if (k == 1)
      {
        EXPECT_TRUE(zero.derivative[2] == C{});
      }
    }
  // Both theta1 and theta2 are too small to reconstruct, but their quotient is finite.
  auto a = elliptic_theta_jet(1, C(R(1) / 8, 0), R(40000));
  auto b = elliptic_theta_jet(2, C(R(1) / 8, 0), R(40000));
  EXPECT_EQ(math::exp(a.log_scale), R(0));
  EXPECT_LE(math::abs(elliptic_theta_ratio(a, 0, b, 0) - math::tan(pi_value / 8)), R(64) * eps);
}

TYPED_TEST(ThetaJetNative, LargeLatticeRetainsFiniteScale)
{
  using R = TypeParam;
  using C = complex<R>;
  R const t = numeric_limits<R>::max() / 2, one = R(1);
  R const pi_value = R(4) * math::atan(one), eps = numeric_limits<R>::epsilon();
  C const u(one / 8, 0);
  for (unsigned kind : {1u, 2u})
  {
    auto jet = elliptic_theta_jet(kind, u, t);
    EXPECT_TRUE(uni20::isfinite(jet.log_scale));
    EXPECT_EQ(jet.log_scale, -pi_value * (t / 4));
    C const expected = R(2) * (kind == 1 ? math::sin(pi_value * u) : math::cos(pi_value * u));
    EXPECT_LE(math::abs(jet.derivative[0] - expected), R(32) * eps);
  }
}

TYPED_TEST(ThetaJetNative, LargeImaginaryHalfPeriodHasFiniteScaledDerivatives)
{
  using R = TypeParam;
  using C = complex<R>;
  R const t = numeric_limits<R>::max() / 16;
  R const pi_value = R(4) * math::atan(R(1)), eps = numeric_limits<R>::epsilon();
  for (R sign : {R(-1), R(1)})
  {
    // In this limit all other Fourier terms are smaller than the scalar range.
    // The leading term (two terms for theta3/4) gives each derivative directly.
    std::array<std::array<C, 3>, 4> const expected{
        {{C(0, sign), C(pi_value, 0), C(0, -sign * pi_value * pi_value)},
         {C(1, 0), C(0, -sign * pi_value), C(-pi_value * pi_value, 0)},
         {C(2, 0), C(0, -sign * R(2) * pi_value), C(-R(4) * pi_value * pi_value, 0)},
         {C(0, 0), C(0, sign * R(2) * pi_value), C(R(4) * pi_value * pi_value, 0)}}};
    for (unsigned kind = 1; kind <= 4; ++kind)
    {
      SCOPED_TRACE(kind);
      auto const jet = elliptic_theta_jet(kind, C(0, sign * (t / 2)), t);
      EXPECT_EQ(jet.log_scale, kind < 3 ? pi_value * (t / 4) : R(0));
      for (unsigned d = 0; d < 3; ++d)
        EXPECT_LE(math::abs(jet.derivative[d] - expected[kind - 1][d]),
                  R(256) * eps * (R(1) + math::abs(expected[kind - 1][d])));
    }
  }
}

TYPED_TEST(ThetaJetNative, RatioAvoidsOverflowOfFiniteComponentNorms)
{
  using R = TypeParam;
  using C = complex<R>;
  R const large = (numeric_limits<R>::max() / 4) * 3;
  R const eps = numeric_limits<R>::epsilon();
  C const zero(0, 0), diagonal(large, large), conjugate(large, -large);
  for (R shift : {R(-2), R(2)})
  {
    theta_jet_result<C> const a{{diagonal, zero, zero}, shift};
    theta_jet_result<C> const b{{diagonal, conjugate, zero}, R(0)};
    C const real_expected(math::exp(shift), 0), imag_expected(0, math::exp(shift));
    EXPECT_LE(math::abs(elliptic_theta_ratio(a, 0, b, 0) - real_expected), R(256) * eps * math::abs(real_expected));
    EXPECT_LE(math::abs(elliptic_theta_ratio(a, 0, b, 1) - imag_expected), R(256) * eps * math::abs(imag_expected));
  }
}

TEST(ThetaJet, DomainBudgetAndRatio)
{
  using C = complex<double>;
  EXPECT_THROW(elliptic_theta_jet(0, C{}, 1.0), std::invalid_argument);
  EXPECT_THROW(elliptic_theta_jet(5, C{}, 1.0), std::invalid_argument);
  EXPECT_THROW(elliptic_theta_jet(1, C{}, 0.0), std::invalid_argument);
  EXPECT_THROW(elliptic_theta_jet(1, C{}, 1.0, 0), std::invalid_argument);
  EXPECT_THROW(elliptic_theta_jet(1, C{}, 1.0, 1), std::runtime_error);
  EXPECT_THROW(elliptic_theta_jet(1, C(1e20, 0), 1.0), std::overflow_error);
  EXPECT_THROW(elliptic_theta_jet(1, C{}, 1.0, Precision::bits(64)), std::invalid_argument);
  auto a = elliptic_theta_jet(1, C{}, 1.0), b = elliptic_theta_jet(2, C{}, 1.0);
  EXPECT_THROW(elliptic_theta_ratio(a, 3, b, 0), std::invalid_argument);
  EXPECT_THROW(elliptic_theta_ratio(b, 0, a, 0), std::domain_error);
  EXPECT_EQ(elliptic_theta_ratio(a, 0, b, 0), C{});
}

#if UNI20_HAS_MPREAL
using R = mpreal;
using C = complex<R>;
using uni20::test::theta_jet_reference::product_reference;

TEST(ThetaJetMpfr, IndependentProductsAndPrecisionGrowth)
{
  auto hp = Precision::bits(1280);
  auto const ambient = mpfr_get_default_prec(), emin = mpfr_get_emin(), emax = mpfr_get_emax();
  for (char const* ts : {"0.09", "0.7", "1", "3"})
    for (unsigned k = 1; k <= 4; ++k)
    {
      R previous(1, hp);
      for (int bits : {80, 160, 256, 333})
      {
        auto p = Precision::bits(bits);
        R t(ts, p);
        C u(R("0.2", p), R("0.1", p));
        auto jet = elliptic_theta_jet(k, u, t);
        auto reference = product_reference(k, u.at(hp), t.at(hp));
        R worst(0, hp);
        for (unsigned d = 0; d < 3; ++d)
        {
          EXPECT_EQ(precision_of(jet.derivative[d]), p);
          C const value = math::exp(jet.log_scale.at(hp)) * jet.derivative[d].at(hp);
          R const error = math::abs(value - reference[d]) / (R(1, hp) + math::abs(reference[d]));
          worst = std::max(worst, error);
          EXPECT_LT(error, 32 * numeric_limits<R>::epsilon(t).at(hp) * (1 + math::abs(jet.log_scale.at(hp))));
        }
        EXPECT_EQ(precision_of(jet.log_scale), p);
        EXPECT_LT(worst, previous / R("1e15", hp));
        if (bits >= 256)
        {
          EXPECT_LT(worst, R("1e-70", hp));
        }
        previous = worst;
      }
    }
  EXPECT_EQ(mpfr_get_default_prec(), ambient);
  EXPECT_EQ(mpfr_get_emin(), emin);
  EXPECT_EQ(mpfr_get_emax(), emax);
}

TEST(ThetaJetMpfr, ExactInputsTinyArgumentsAndZeros)
{
  auto p = Precision::bits(256), hp = Precision::bits(1280);
  R zero(0, p), one(1, p);
  C cz(zero, zero);
  EXPECT_THROW(elliptic_theta_jet(1, C(R(0), R(0)), R(1)), std::invalid_argument);
  EXPECT_THROW(elliptic_theta_jet(1, cz, R(1, Precision::bits(80))), std::invalid_argument);
  auto converted = elliptic_theta_jet(1, C(R(0), R(0)), R(1), p);
  EXPECT_EQ(precision_of(converted.log_scale), p);
  EXPECT_EQ(precision_of(converted.derivative[0]), p);
  EXPECT_EQ(converted.derivative[0], cz);
  EXPECT_EQ(converted.derivative[2], cz);
  for (unsigned k = 2; k <= 4; ++k)
  {
    R t("0.1", p);
    C u(R("1e-100", p), R("-1e-100", p));
    auto jet = elliptic_theta_jet(k, u, t);
    auto expected = product_reference(k, u.at(hp), t.at(hp));
    EXPECT_GT(math::abs(jet.derivative[1]), zero);
    C value = math::exp(jet.log_scale.at(hp)) * jet.derivative[1].at(hp);
    EXPECT_LT(math::abs(value / expected[1] - R(1, hp)), R("1e-70", hp));
    auto at_zero = elliptic_theta_jet(k, cz, t);
    EXPECT_EQ(at_zero.derivative[1], cz);
    auto ratio = elliptic_theta_ratio(converted, 0, at_zero, 0);
    EXPECT_EQ(ratio, cz);
    EXPECT_EQ(precision_of(ratio), p);
  }
}
#endif
} // namespace
