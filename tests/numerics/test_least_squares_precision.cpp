#include "precision_registry.hpp"
#include <uni20/linalg/ops/least_squares.hpp>

namespace uni20::test
{
template <class C, class Backend> void check_least_squares(Backend backend)
{
  using S = typename C::scalar_type;
  using R = typename C::real_type;
  auto a = C::matrix(3, 2), b = C::matrix(3, 2);
  auto phase = C::scalar(1, 1);
  a[0, 0] = phase;
  a[1, 1] = C::scalar(1);
  a[2, 0] = phase;
  a[2, 1] = C::scalar(1);
  b[0, 0] = C::scalar(1);
  b[1, 0] = C::scalar(2);
  b[2, 0] = C::scalar(4);
  for (index_type i = 0; i < 3; ++i)
    b[i, 1] = b[i, 0] * phase;
  auto result = linalg::least_squares_with_info(backend, a, b, C::precision());
  ASSERT_TRUE(result.info.succeeded());
  ASSERT_TRUE(result.solution);
  auto const& x = *result.solution;
  auto tol = C::real(64) * C::epsilon();
  expect_error_at_most(S(x[0, 0] * phase), S(C::scalar(4) / C::real(3)), tol);
  expect_error_at_most(x[1, 0], S(C::scalar(7) / C::real(3)), tol);
  expect_error_at_most(x[0, 1], S(C::scalar(4) / C::real(3)), tol);
  expect_error_at_most(result.residual_norms[0], R(C::real(1) / math::sqrt(C::real(3))), tol);
  expect_error_at_most(result.residual_norms[1], R(result.residual_norms[0] * math::abs(phase)), tol);
  // The residual must be orthogonal to every coefficient column, including complex phases.
  for (index_type k = 0; k < 2; ++k)
  {
    S normal = C::scalar(0);
    for (index_type i = 0; i < 3; ++i)
      normal += uni20::conj(a[i, k]) * (a[i, 0] * x[0, 0] + a[i, 1] * x[1, 0] - b[i, 0]);
    expect_error_at_most(normal, C::scalar(0), tol);
  }
  EXPECT_EQ(precision_of(x), C::precision());
  C::expect_precision(x[0, 0]);
  C::expect_precision(result.residual_norms[0]);
  expect_error_at_most(a[0, 0], phase, C::real(0));
  expect_error_at_most(b[2, 0], C::scalar(4), C::real(0));
#if UNI20_HAS_MPREAL
  // Independent exact-rational oracle measured at 512 bits. Each case must
  // improve on the preceding significand tier, not just meet a double tolerance.
  auto p = Precision::bits(512);
  auto expected = mpreal(7, p) / 3;
  auto actual = widen(uni20::real(x[1, 0]));
  auto lower = Precision::bits(C::digits() <= 24    ? 12
                               : C::digits() <= 53  ? 24
                               : C::digits() <= 64  ? 53
                               : C::digits() <= 113 ? 64
                                                    : 113);
  EXPECT_LT(abs(actual - expected), abs(expected.at(lower).at(p) - expected) / 8);
#endif
  auto d = C::matrix(3, 2), rhs = C::matrix(3, 1);
  d[0, 0] = C::scalar(1);
  d[1, 1] = C::scalar(1);
  rhs[0, 0] = C::scalar(C::real(1) + C::gap(), C::real(0));
  rhs[1, 0] = C::scalar(2);
  auto retained = linalg::least_squares_with_info(backend, d, rhs);
  ASSERT_TRUE(retained.info.succeeded());
  expect_error_at_most((*retained.solution)[0, 0], rhs[0, 0], C::real(0));
  // Normal equations lose delta^2 at this precision. QR must still resolve
  // the independent columns, with an error scaled by the problem's condition.
  R delta = C::power_of_two(-C::digits() / 2 - 3);
  for (index_type i = 0; i < 3; ++i)
  {
    d[i, 0] = C::scalar(1);
    R offset = i == 0 ? C::real(0) : i == 1 ? delta : -delta;
    d[i, 1] = C::scalar(C::real(1) + offset, C::real(0)) * phase;
    rhs[i, 0] = C::scalar(-offset, C::real(0));
  }
  auto sensitive = linalg::least_squares_with_info(backend, d, rhs);
  ASSERT_TRUE(sensitive.info.succeeded());
  expect_error_at_most((*sensitive.solution)[0, 0], C::scalar(1), tol / delta);
  expect_error_at_most(S((*sensitive.solution)[1, 0] * phase), C::scalar(-1), tol / delta);
  // Exercise a nontrivial rotation outside double's exponent range.
  if constexpr ([] {
                  if constexpr (C::runtime)
                    return true;
                  else
                    return numeric_limits<R>::max_exponent > 1024;
                }())
  {
    R large = C::power_of_two(2000);
    for (index_type i = 0; i < 3; ++i)
    {
      for (index_type j = 0; j < 2; ++j)
        a[i, j] *= large;
      b[i, 0] *= large;
    }
    auto wide = linalg::least_squares_with_info(backend, a, b, C::precision());
    ASSERT_TRUE(wide.info.succeeded());
    expect_error_at_most((*wide.solution)[1, 0], S(C::scalar(7) / C::real(3)), tol);
    expect_error_at_most(R(wide.residual_norms[0] / large), result.residual_norms[0], tol);
  }
}

UNI20_PRECISION_TEST(NumericalLinalg, CpuLeastSquares)
{
  check_least_squares<TypeParam>(linalg::CpuReferenceBackend{});
}
UNI20_PRECISION_TEST(NumericalLinalg, ProviderLeastSquares) { check_least_squares<TypeParam>(linalg::LapackBackend{}); }
} // namespace uni20::test
