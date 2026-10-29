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

  // Half-period translations of an even function give f'/f=-pi*i;
  // translations of an odd function at its zero give f''/f'=-2*pi*i.
  // These ratios remove the (potentially enormous) common logarithmic scale.
  auto const tiny_t = C::epsilon();
  for (unsigned kind = 1; kind <= 4; ++kind)
    for (int edge : {0, 1})
      for (int sign : {-1, 1})
      {
        SCOPED_TRACE(::testing::Message() << "half-period kind=" << kind << " edge=" << edge << " sign=" << sign);
        bool const at_zero = (kind == 3 && edge == 1) || (kind == 4 && edge == 0);
        auto const jet = elliptic_theta_jet(kind, C::scalar(C::real(edge) / 2, sign * (tiny_t / 2)), tiny_t);
        auto const ratio = jet.derivative[at_zero ? 2 : 1] / jet.derivative[at_zero ? 1 : 0];
        auto const expected = C::scalar(zero, -sign * (at_zero ? 2 : 1) * pi_value);
        expect_error_at_most(ratio, expected, C::epsilon() * 256 * math::abs(expected));
        if (at_zero) expect_equal(jet.derivative[0], C::scalar(zero, zero));
      }

  // A tiny imaginary displacement from a real stationary point must not be
  // swamped by the phase error from sin(pi) or cos(pi/2).
  auto const delta = C::epsilon() * C::epsilon();
  for (int lattice : {1, 2, 4})
    for (unsigned kind : {1u, 3u, 4u})
    {
      SCOPED_TRACE(::testing::Message() << "stationary kind=" << kind << " lattice=" << lattice);
      auto const t = C::real(lattice) / 2;
      auto const center = elliptic_theta_jet(kind, C::scalar(one / 2, zero), t);
      auto const near = elliptic_theta_jet(kind, C::scalar(one / 2, delta), t);
      auto const ratio = near.derivative[1] / (C::scalar(zero, delta) * center.derivative[2]);
      expect_error_at_most(ratio, C::scalar(one, zero), C::epsilon() * 512);
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
  constexpr int bits = C::digits();
  constexpr int previous = bits <= 24    ? 12
                           : bits <= 53  ? 24
                           : bits <= 64  ? 53
                           : bits <= 113 ? 64
                           : bits <= 128 ? 113
                                         : 128;
  auto error_sum = mpreal(0, hp), previous_sum = mpreal(0, hp);
  for (auto const& x : {one / 5, math::next_down(one / 4), math::next_up(one / 4), 3 * one / 8, -3 * one / 8})
    for (int index : {1, 4, 8, 24})
      for (unsigned kind = 1; kind <= 4; ++kind)
      {
        auto const u = C::scalar(x, one / 10);
        auto t = C::real(index) / 8;
        auto jet = elliptic_theta_jet(kind, u, t);
        // Widen the original inputs, including their final low bits, instead of
        // comparing different rounded mathematical problems across precisions.
        complex<mpreal> wide_u(mpreal(u.real(), hp), mpreal(u.imag(), hp));
        auto reference = theta_jet_reference::product_reference(kind, wide_u, mpreal(t, hp));
        if (index == 1)
        {
          auto const gaussian = theta_jet_reference::gaussian_reference(kind, wide_u, mpreal(t, hp));
          for (unsigned d : {0u, 2u})
          {
            auto const expected = reference[d] / reference[1];
            expect_error_at_most(gaussian[d] / gaussian[1], expected,
                                mpreal("1e-300", hp) * (1 + math::abs(expected)));
          }
        }
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
  // A direct wide Gaussian sum, without production period reduction or
  // phase helpers, checks adjacent inputs near the simple theta3/theta4 zeros.
  for (auto const& t : {C::epsilon(), one / 8})
    for (unsigned kind : {3u, 4u})
    {
      auto const x = kind == 3 ? one / 2 : C::real(0), y = t / 2;
      std::array xs{kind == 3 ? math::next_down(x) : -C::epsilon() * t, x,
                    kind == 3 ? math::next_up(x) : C::epsilon() * t};
      std::array ys{math::next_down(y), y, math::next_up(y)};
      for (auto const& xx : xs)
        for (auto const& yy : ys)
        {
          SCOPED_TRACE(::testing::Message() << "zero kind=" << kind << " t=" << format_scalar(t)
                                           << " x=" << format_scalar(xx) << " y=" << format_scalar(yy));
          auto const u_near = C::scalar(xx, yy);
          auto const jet = elliptic_theta_jet(kind, u_near, t);
          auto const reference = theta_jet_reference::gaussian_reference(
              kind, complex<mpreal>(mpreal(xx, hp), mpreal(yy, hp)), mpreal(t, hp));
          for (unsigned d : {0u, 2u})
          {
            if (d == 0 && xx == x && yy == y) continue; // Exact zero is tested above.
            auto const ratio = jet.derivative[d] / jet.derivative[1];
            complex<mpreal> const wide_ratio(mpreal(ratio.real(), hp), mpreal(ratio.imag(), hp));
            auto const expected = reference[d] / reference[1];
            expect_error_at_most(wide_ratio, expected, eps * 2048 * math::abs(expected));
          }
        }
    }

  // Even best rounding to the preceding format cannot attain this accuracy.
  EXPECT_LT(error_sum * 8, previous_sum);
  this->RecordProperty("theta_jet_error_sum", format_scalar(error_sum));
  this->RecordProperty("theta_jet_preceding_format_error_sum", format_scalar(previous_sum));
}
#endif
} // namespace uni20::test
