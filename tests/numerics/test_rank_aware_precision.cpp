#include "precision_registry.hpp"
#include <uni20/linalg/ops/lu_rcond.hpp>
#include <uni20/linalg/ops/svd_least_squares.hpp>

namespace uni20::test
{
UNI20_PRECISION_TEST(NumericalLinalg, ProviderSvdLeastSquares)
{
  using C = TypeParam;
  using S = typename C::scalar_type;
  using R = typename C::real_type;
  auto a = C::matrix(1, 2), b = C::matrix(1, 1);
  auto phase = C::scalar(1, 1);
  a[0, 0] = phase;
  a[0, 1] = C::scalar(2) * phase;
  b[0, 0] = C::scalar(3) * phase;
  auto r = linalg::svd_least_squares_with_info(linalg::LapackBackend{}, a, b, C::precision(),
                                               {.return_singular_values = true});
  ASSERT_TRUE(r.info.succeeded());
  ASSERT_TRUE(r.solution);
  ASSERT_TRUE(r.diagnostics);
  EXPECT_EQ(r.diagnostics->rank, 1);
  EXPECT_EQ(r.diagnostics->relative_cutoff, C::epsilon());
  auto tolerance = C::real(64) * C::epsilon();
  expect_error_at_most((*r.solution)[0, 0], S(C::scalar(3) / C::real(5)), tolerance);
  expect_error_at_most((*r.solution)[1, 0], S(C::scalar(6) / C::real(5)), tolerance);
  expect_error_at_most(r.residual_norms[0], C::real(0), tolerance);
  C::expect_precision((*r.solution)[0, 0]);
  C::expect_precision(r.residual_norms[0]);
  C::expect_precision(r.diagnostics->relative_cutoff);
  C::expect_precision(r.diagnostics->singular_values[0]);
#if UNI20_HAS_MPREAL
  auto p = Precision::bits(512);
  auto expected = mpreal(3, p) / 5;
  auto actual = widen(uni20::real((*r.solution)[0, 0]));
  auto lower = Precision::bits(C::digits() <= 24    ? 12
                               : C::digits() <= 53  ? 24
                               : C::digits() <= 64  ? 53
                               : C::digits() <= 113 ? 64
                                                    : 113);
  EXPECT_LT(abs(actual - expected), abs(expected.at(lower).at(p) - expected) / 8);
#endif
  // The default must retain this direction at every precision. A dimension
  // multiplier in the cutoff would incorrectly drop it.
  auto d = C::matrix(8, 2), rhs = C::matrix(8, 1);
  d[0, 0] = C::scalar(1);
  d[1, 1] = C::scalar(C::epsilon() * C::real(2), C::real(0));
  rhs[0, 0] = C::scalar(1);
  rhs[1, 0] = d[1, 1];
  auto full = linalg::svd_least_squares_with_info(linalg::LapackBackend{}, d, rhs);
  ASSERT_TRUE(full.info.succeeded());
  EXPECT_EQ(full.diagnostics->rank, 2);
  expect_error_at_most((*full.solution)[1, 0], C::scalar(1), tolerance);
  auto no_rhs = C::matrix(8, 0);
  auto rank_only = linalg::svd_least_squares_with_info(linalg::LapackBackend{}, d, no_rhs);
  ASSERT_TRUE(rank_only.info.succeeded());
  EXPECT_EQ(rank_only.diagnostics->rank, 2);
  auto cut = linalg::svd_least_squares_with_info(linalg::LapackBackend{}, d, rhs,
                                                 {.relative_cutoff = C::epsilon() * C::real(4)});
  ASSERT_TRUE(cut.info.succeeded());
  EXPECT_EQ(cut.diagnostics->rank, 1);
  expect_error_at_most((*cut.solution)[1, 0], C::scalar(0), C::real(0));
  expect_error_at_most(cut.residual_norms[0], R(C::epsilon() * C::real(2)), C::real(0));
  if constexpr ([] {
                  if constexpr (C::runtime)
                    return true;
                  else
                    return numeric_limits<R>::max_exponent > 1024;
                }())
  {
    auto scale = C::power_of_two(2000);
    a[0, 0] *= scale;
    a[0, 1] *= scale;
    b[0, 0] *= scale;
    auto scaled = linalg::svd_least_squares_with_info(linalg::LapackBackend{}, a, b);
    ASSERT_TRUE(scaled.info.succeeded());
    expect_error_at_most((*scaled.solution)[0, 0], S(C::scalar(3) / C::real(5)), tolerance);
  }
}

UNI20_PRECISION_TEST(NumericalLinalg, ProviderLuCondition)
{
  using C = TypeParam;
  using R = typename C::real_type;
  auto a = C::matrix(2, 2);
  auto phase = C::scalar(1, 1);
  a[0, 0] = C::scalar(3) * phase;
  a[1, 1] = C::scalar(7) * phase;
  auto f = linalg::lu_factor(linalg::LapackBackend{}, a, C::precision());
  auto one = linalg::lu_rcond_with_info(linalg::LapackBackend{}, f);
  auto inf = linalg::lu_rcond_with_info(linalg::LapackBackend{}, f, linalg::MatrixNorm::Infinity);
  ASSERT_TRUE(one.info.succeeded());
  ASSERT_TRUE(inf.info.succeeded());
  ASSERT_TRUE(one.estimate);
  ASSERT_TRUE(inf.estimate);
  auto tolerance = C::epsilon() * C::real(32);
  expect_error_at_most(*one.estimate, R(C::real(3) / C::real(7)), tolerance);
  expect_error_at_most(*inf.estimate, R(C::real(3) / C::real(7)), tolerance);
  expect_error_at_most(f.norms()->one, R(C::real(7) * math::abs(phase)), tolerance * C::real(10));
  C::expect_precision(*one.estimate);
  C::expect_precision(f.norms()->one);
  C::expect_precision(f.norms()->infinity);
#if UNI20_HAS_MPREAL
  auto p = Precision::bits(512);
  auto expected = mpreal(3, p) / 7;
  auto lower = Precision::bits(C::digits() <= 24    ? 12
                               : C::digits() <= 53  ? 24
                               : C::digits() <= 64  ? 53
                               : C::digits() <= 113 ? 64
                                                    : 113);
  EXPECT_LT(abs(widen(*one.estimate) - expected), abs(expected.at(lower).at(p) - expected) / 8);
#endif
  // The factorization and retained norms must both keep a sub-double increment.
  a[0, 0] = C::scalar(1);
  a[1, 1] = C::scalar(C::real(1) + C::gap(), C::real(0));
  auto gap = linalg::lu_factor(linalg::LapackBackend{}, a);
  EXPECT_EQ(gap.norms()->one, C::real(1) + C::gap());
  auto gap_estimate = linalg::lu_rcond(linalg::LapackBackend{}, gap);
  expect_error_at_most(gap_estimate, R(C::real(1) / (C::real(1) + C::gap())), C::epsilon());
  EXPECT_LT(gap_estimate, C::real(1));
  if constexpr ([] {
                  if constexpr (C::runtime)
                    return true;
                  else
                    return numeric_limits<R>::max_exponent > 1024;
                }())
  {
    auto scale = C::power_of_two(2000);
    a[0, 0] = C::scalar(3) * phase * scale;
    a[1, 1] = C::scalar(7) * phase * scale;
    auto scaled = linalg::lu_factor(linalg::LapackBackend{}, a);
    expect_error_at_most(linalg::lu_rcond(linalg::LapackBackend{}, scaled), *one.estimate, tolerance);
  }
}
} // namespace uni20::test
