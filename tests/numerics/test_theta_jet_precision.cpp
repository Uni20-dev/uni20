#include "../core/theta_jet_references.hpp"
#include "precision_registry.hpp"
#include <uni20/core/elliptic_theta_jet.hpp>

namespace uni20::test
{
UNI20_PRECISION_TEST(NumericalScalar, ThetaJetIdentities)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  auto const zero = C::real(0), one = C::real(1);
  auto const pi_value = pi<typename C::real_type> * one;
  auto const u = C::scalar(one / 8, one / 16);
  auto actual = [](auto const& jet, unsigned d) { return math::exp(jet.log_scale) * jet.derivative[d]; };
  for (int index : {1, 4, 8, 16})
    for (unsigned kind = 1; kind <= 4; ++kind)
    {
      auto const t = C::real(index) / 8;
      auto a = elliptic_theta_jet(kind, u, t);
      auto b = elliptic_theta_jet(kind, uni20::conj(u), t);
      auto c = elliptic_theta_jet(kind, u + one, t);
      auto v = elliptic_theta_jet(kind, u + C::scalar(zero, t), t);
      auto const sign = kind < 3 ? -one : one;
      auto const vertical_sign = kind == 1 || kind == 4 ? -one : one;
      auto const slope = C::scalar(zero, -2 * pi_value);
      auto const multiplier = vertical_sign * math::exp(pi_value * t + slope * u);
      std::array expected{actual(a, 0), actual(a, 1) + slope * actual(a, 0),
                          actual(a, 2) + C::real(2) * slope * actual(a, 1) + slope * slope * actual(a, 0)};
      for (unsigned d = 0; d < 3; ++d)
      {
        C::expect_precision(a.derivative[d]);
        auto const tolerance = C::epsilon() * 2048 * (one + math::abs(actual(a, d)));
        expect_error_at_most(actual(b, d), uni20::conj(actual(a, d)), tolerance);
        expect_error_at_most(actual(c, d), sign * actual(a, d), tolerance);
        expect_error_at_most(actual(v, d) / multiplier, expected[d],
                             C::epsilon() * 2048 * (one + math::abs(expected[d])));
      }
      C::expect_precision(a.log_scale);
      auto at_zero = elliptic_theta_jet(kind, C::scalar(zero, zero), t);
      expect_equal(at_zero.derivative[kind == 1 ? 0 : 1], C::scalar(zero, zero));
      auto ratio = elliptic_theta_ratio(a, 1, a, 0);
      C::expect_precision(ratio);
      expect_error_at_most(ratio, actual(a, 1) / actual(a, 0), C::epsilon() * 128 * (one + math::abs(ratio)));
    }
}

#if UNI20_HAS_MPREAL
UNI20_PRECISION_TEST(NumericalScalar, ThetaJetAccuracy)
{
  using C = TypeParam;
  this->RecordProperty("backend", "scalar_math");
  auto const hp = Precision::bits(1280);
  auto const one = C::real(1);
  auto const eps = mpreal(C::epsilon(), hp);
  auto const u = C::scalar(one / 5, one / 10);
  constexpr int bits = C::digits();
  constexpr int previous = bits <= 24    ? 12
                           : bits <= 53  ? 24
                           : bits <= 64  ? 53
                           : bits <= 113 ? 64
                           : bits <= 128 ? 113
                                         : 128;
  auto error_sum = mpreal(0, hp), previous_sum = mpreal(0, hp);
  for (int index : {1, 4, 8, 24})
    for (unsigned kind = 1; kind <= 4; ++kind)
    {
      auto t = C::real(index) / 8;
      auto jet = elliptic_theta_jet(kind, u, t);
      // Widen the original inputs, including their final low bits, instead of
      // comparing different rounded mathematical problems across precisions.
      complex<mpreal> wide_u(mpreal(u.real(), hp), mpreal(u.imag(), hp));
      auto reference = theta_jet_reference::product_reference(kind, wide_u, mpreal(t, hp));
      for (unsigned d = 0; d < 3; ++d)
      {
        complex<mpreal> scaled(mpreal(jet.derivative[d].real(), hp), mpreal(jet.derivative[d].imag(), hp));
        auto value = math::exp(mpreal(jet.log_scale, hp)) * scaled;
        auto const scale = 1 + math::abs(reference[d]);
        auto const error = math::abs(value - reference[d]) / scale;
        EXPECT_LE(error, eps * 256 * (1 + math::abs(mpreal(jet.log_scale, hp))));
        error_sum += error;
        auto narrowed = reference[d].at(Precision::bits(previous)).at(hp);
        previous_sum += math::abs(narrowed - reference[d]) / scale;
      }
    }
  // Even best rounding to the preceding format cannot attain this accuracy.
  EXPECT_LT(error_sum * 8, previous_sum);
  this->RecordProperty("theta_jet_error_sum", format_scalar(error_sum));
  this->RecordProperty("theta_jet_preceding_format_error_sum", format_scalar(previous_sum));
}
#endif
} // namespace uni20::test
