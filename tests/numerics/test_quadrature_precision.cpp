#include "precision_registry.hpp"
#include <uni20/core/finite_quadrature.hpp>

namespace uni20::test
{
UNI20_PRECISION_TEST(NumericalScalar, FiniteQuadratureIdentities)
{
  using C = TypeParam;
  using R = typename C::real_type;
  this->RecordProperty("backend", "scalar_math");
  tanh_sinh_options<R, 6> options(C::precision());
  auto one = C::real(1), two = C::real(2);
  auto f = [&](quadrature_point<R> const& x) {
    C::expect_precision(x.value);
    C::expect_precision(x.lower_distance);
    C::expect_precision(x.upper_distance);
    return std::array<R, 6>{one,
                            x.value,
                            x.value * x.value,
                            one / math::sqrt(x.lower_distance),
                            one / math::sqrt(x.upper_distance),
                            math::log(x.upper_distance)};
  };
  auto result = tanh_sinh_integrate(f, one, two, options);
  ASSERT_EQ(result.status, quadrature_status::converged);
  std::array<R, 6> const expected{one, C::real(3) / 2, C::real(7) / 3, two, two, -one};
  for (std::size_t j = 0; j < 6; ++j)
  {
    C::expect_precision(result.value[j]);
    C::expect_precision(result.error[j]);
    C::expect_precision(result.absolute_integral[j]);
    C::expect_precision(result.endpoint_indicator[j]);
    expect_error_at_most(result.value[j], expected[j], C::epsilon() * 64 * math::abs(expected[j]));
    EXPECT_LE(result.error[j], options.relative_tolerance * math::abs(result.value[j]));
  }
  // A zero integral with nonzero absolute integral needs an absolute target.
  tanh_sinh_options<R> cancellation(C::precision());
  cancellation.absolute_tolerance[0] = C::epsilon() * 64;
  auto odd = [&](auto const& x) { return std::array<R, 1>{x.value - one * 3 / 2}; };
  auto cancelled = tanh_sinh_integrate(odd, one, two, cancellation);
  ASSERT_EQ(cancelled.status, quadrature_status::converged);
  expect_error_at_most(cancelled.value[0], C::real(0), cancellation.absolute_tolerance[0]);
  EXPECT_GT(cancelled.absolute_integral[0], 0);
  auto zero = [&](auto const&) { return std::array<R, 1>{C::real(0)}; };
  auto zero_result = tanh_sinh_integrate(zero, one, two, cancellation);
  EXPECT_EQ(zero_result.status, quadrature_status::converged);
  EXPECT_EQ(zero_result.value[0], 0);
  C::expect_precision(zero_result.value[0]);
}

#if UNI20_HAS_MPREAL
UNI20_PRECISION_TEST(NumericalScalar, FiniteQuadratureAccuracy)
{
  using C = TypeParam;
  using R = typename C::real_type;
  this->RecordProperty("backend", "scalar_math");
  auto one = C::real(1);
  tanh_sinh_options<R> options(C::precision());
  auto f = [&](auto const& x) { return std::array<R, 1>{one / (one + x.value)}; };
  auto result = tanh_sinh_integrate(f, C::real(0), one, options);
  ASSERT_EQ(result.status, quadrature_status::converged);
  auto p = Precision::bits(512);
  auto reference = math::log(mpreal(2, p));
  auto error = math::abs(widen(result.value[0]) - reference);
  EXPECT_LE(error, widen(C::epsilon()) * 16);
  constexpr int bits = C::digits();
  constexpr int previous = bits <= 24    ? 12
                           : bits <= 53  ? 24
                           : bits <= 64  ? 53
                           : bits <= 113 ? 64
                           : bits <= 128 ? 113
                                         : 128;
  auto previous_error = math::abs(reference.at(Precision::bits(previous)).at(p) - reference);
  EXPECT_LT(error * 8, previous_error);
  this->RecordProperty("quadrature_log_two_error", format_scalar(error));
}
#endif
} // namespace uni20::test
