#include "precision_registry.hpp"
#include <uni20/core/elliptic_theta.hpp>

namespace uni20::test
{
namespace
{
template <class R> R theta_value(theta_constant_result<R> const& result)
{
  return result.scaled_value * math::exp(result.log_scale);
}
} // namespace

UNI20_PRECISION_TEST(NumericalScalar, ThetaConstantIdentities)
{
  using C = TypeParam;
  using R = typename C::real_type;
  this->RecordProperty("backend", "scalar_math");
  for (int index = 1; index <= 32; ++index)
  {
    R t = C::real(index) / 8;
    auto a = elliptic_theta_constant(2, t), b = elliptic_theta_constant(3, t), c = elliptic_theta_constant(4, t);
    for (auto const& result : {a, b, c})
    {
      C::expect_precision(result.scaled_value);
      C::expect_precision(result.log_scale);
      EXPECT_GT(result.scaled_value, 0);
    }
    R x = theta_value(a), y = theta_value(b), z = theta_value(c);
    R lhs = y * y * y * y, rhs = x * x * x * x + z * z * z * z;
    expect_error_at_most(lhs, rhs, C::epsilon() * 32 * lhs);
    for (unsigned kind : {2u, 3u, 4u})
    {
      R direct = theta_value(elliptic_theta_constant(kind, t));
      R dual = theta_value(elliptic_theta_constant(kind == 3 ? 3 : 6 - kind, C::real(1) / t)) / math::sqrt(t);
      expect_error_at_most(direct, dual, C::epsilon() * 64 * direct);
    }
  }
}

#if UNI20_HAS_MPREAL
namespace
{
// Jacobi products are independent of the implementation's modular Fourier
// sums. For t>=1/16, 4096 factors leave <2^-2000 relative truncation error.
mpreal product_reference(unsigned kind, mpreal const& t)
{
  auto const p = t.precision();
  auto q = math::exp(-pi<mpreal>.at(p) * t), q2 = q * q;
  auto even = q2, odd = q, product = mpreal(1, p);
  for (int n = 1; n <= 4096; ++n)
  {
    auto factor = kind == 2 ? 1 + even : kind == 3 ? 1 + odd : 1 - odd;
    product *= (1 - even) * factor * factor;
    even *= q2;
    odd *= q2;
  }
  if (kind == 2) product *= 2 * math::exp(-pi<mpreal>.at(p) * t / 4);
  return product;
}

template <class C> void theta_accuracy()
{
  using R = typename C::real_type;
  auto const p = Precision::bits(512);
  auto maximum_error = mpreal(0, p);
  for (int index = 1; index <= 32; ++index)
    for (unsigned kind : {2u, 3u, 4u})
    {
      R t = C::real(index) / 16;
      auto result = elliptic_theta_constant(kind, t);
      auto reference = product_reference(kind, widen(t));
      // Reconstruct at oracle precision to separate the two returned fields'
      // error from an additional exp/multiply rounding at the caller.
      auto actual = widen(result.scaled_value) * math::exp(widen(result.log_scale));
      auto error = math::abs(actual - reference) / reference;
      auto bound = widen(C::epsilon()) * 8 * (1 + math::abs(widen(result.log_scale)));
      EXPECT_LE(error, bound) << "kind=" << kind << " t=" << format_scalar(t);
      auto normalized = error / (widen(C::epsilon()) * (1 + math::abs(widen(result.log_scale))));
      if (normalized > maximum_error) maximum_error = normalized;
    }
  testing::Test::RecordProperty("theta_max_scaled_error", format_scalar(maximum_error));

  // Lemniscatic constant: theta3(0|i)=Gamma(1/4)/(math::sqrt(2)*pi^(3/4)).
  // Unlike the product oracle, this does not use a theta expansion at all.
  auto one = mpreal(1, p);
  auto reference = math::tgamma(one / 4) / (math::sqrt(2 * one) * math::pow(pi<mpreal>.at(p), 3 * one / 4));
  auto result = elliptic_theta_constant(3, C::real(1));
  auto error = math::abs(widen(result.scaled_value) - reference);
  EXPECT_LE(error, widen(C::epsilon()) * 4);
  constexpr int bits = C::digits();
  constexpr int previous = bits <= 24    ? 12
                           : bits <= 53  ? 24
                           : bits <= 64  ? 53
                           : bits <= 113 ? 64
                           : bits <= 128 ? 113
                                         : 128;
  auto previous_error = math::abs(reference.at(Precision::bits(previous)).at(p) - reference);
  EXPECT_LT(error * 8, previous_error);
  testing::Test::RecordProperty("theta_gamma_reference_error", format_scalar(error));

  // Check points on either side of the modular branch, without rounding them
  // through binary64 before evaluating the independent reference.
  for (R t : {math::next_down(C::real(1)), C::real(1), math::next_up(C::real(1))})
    for (unsigned kind : {2u, 3u, 4u})
    {
      auto actual = theta_value(elliptic_theta_constant(kind, t));
      auto reference_value = product_reference(kind, widen(t));
      EXPECT_LE(math::abs(widen(actual) - reference_value), widen(C::epsilon()) * 16);
    }
}
} // namespace
UNI20_PRECISION_TEST(NumericalScalar, ThetaConstantAccuracy)
{
  this->RecordProperty("backend", "scalar_math");
  theta_accuracy<TypeParam>();
}
#endif
} // namespace uni20::test
