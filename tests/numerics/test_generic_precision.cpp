#include "precision_registry.hpp"
#include <uni20/linalg/ops/linear_solve.hpp>
#include <uni20/linalg/ops/matrix_product.hpp>
#include <uni20/linalg/ops/slogdet.hpp>

namespace uni20::test
{
UNI20_PRECISION_TEST(NumericalLinalg, GenericPrecisionOperations)
{
  using C = TypeParam;
  using S = typename C::scalar_type;
  using R = typename C::real_type;
  auto p = C::precision();
  auto a = C::matrix(2, 2), b = C::matrix(2, 1), out = C::matrix(1, 1);
  // Dyadic inputs with bits beyond the previous precision in the ladder.
  a[0, 0] = C::scalar(2);
  a[1, 1] = C::scalar(4);
  b[0, 0] = C::scalar(C::real(1) + C::gap(), C::real(1));
  b[1, 0] = C::scalar(3, -1);
  if constexpr (!C::runtime)
  {
    auto wrong = Precision::bits(C::digits() + 1);
    out[0, 0] = C::scalar(7);
    EXPECT_THROW(linalg::assign_product(out, a, b, wrong), std::invalid_argument);
    EXPECT_EQ(out.extent(0), 1);
    expect_equal(S(out[0, 0]), C::scalar(7));
    EXPECT_THROW(linalg::solve_inplace(a, b, wrong), std::invalid_argument);
    expect_equal(S(a[0, 0]), C::scalar(2));
    expect_equal(S(b[1, 0]), C::scalar(3, -1));
    EXPECT_THROW((void)linalg::lu_factor(a, wrong), std::invalid_argument);
    EXPECT_THROW((void)linalg::slogdet(a, wrong), std::invalid_argument);
  }
  linalg::assign_product(out, a, b, p);
  expect_equal(S(out[0, 0]), S(C::scalar(2) * b[0, 0]));
  expect_equal(S(out[1, 0]), C::scalar(12, -4));
  EXPECT_EQ(precision_of(out), p);
  linalg::add_product(out, a, b, p);
  expect_equal(S(out[1, 0]), C::scalar(24, -8));
  linalg::gemm(out, 1, a, b, 0, p);
  auto x = linalg::solve(a, out, p);
  expect_equal(S(x[0, 0]), S(b[0, 0]));
  EXPECT_EQ(precision_of(x), p);
  auto factor = linalg::lu_factor(a, p);
  EXPECT_EQ(factor.precision(), p);
  auto reused = linalg::lu_solve(factor, out);
  expect_equal(S(reused[1, 0]), S(b[1, 0]));
  auto determinant = linalg::slogdet(a, p);
  expect_equal(determinant.phase, C::scalar(1));
  expect_error_at_most(determinant.log_absolute, R(math::log(C::real(8))), C::real(4) * C::epsilon());
  auto report = linalg::solve_inplace_with_info(a, out, p);
  EXPECT_TRUE(report.succeeded());
  expect_equal(S(out[0, 0]), S(b[0, 0]));
}
} // namespace uni20::test
