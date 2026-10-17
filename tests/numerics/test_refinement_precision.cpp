#include "precision_registry.hpp"
#include <uni20/linalg/ops/lu_refine.hpp>

namespace uni20::test
{
template <class C, class Backend> void check_refinement(Backend backend)
{
  using namespace linalg;
  using S = typename C::scalar_type;
  using F = typename TestScalar<float, C::is_complex>::type;
  using R = typename C::real_type;
  auto a = C::matrix(2, 2), b = C::matrix(2, 1);
  auto phase = C::scalar(1, 1);
  a[0, 0] = C::scalar(3) * phase;
  a[0, 1] = a[1, 0] = phase;
  a[1, 1] = C::scalar(2) * phase;
  b[0, 0] = phase;
  auto residual_precision = C::runtime ? Precision::bits(C::digits() * 2) : C::precision();
  LuRefinementPrecision<F, S, S> p{native_precision<F>(), C::precision(), residual_precision};
  auto initial = lu_refine_with_info(backend, a, b, p, {.max_iterations = 0});
  ASSERT_TRUE(initial.solution);
  auto result = lu_refine_with_info(backend, a, b, p);
  ASSERT_TRUE(result.info.succeeded());
  ASSERT_TRUE(result.solution);
  auto expected = S(C::scalar(2) / C::real(5));
  expect_error_at_most((*result.solution)[0, 0], expected, C::epsilon() * C::real(4));
  expect_error_at_most((*result.solution)[1, 0], S(-C::scalar(1) / C::real(5)), C::epsilon() * C::real(4));
  C::expect_precision((*result.solution)[0, 0]);
  EXPECT_EQ(precision_of(*result.solution), C::precision());
  EXPECT_LE(result.columns[0].error.componentwise_backward_error,
            scalar_at_precision<R>(C::epsilon(), residual_precision));
  if constexpr (C::digits() > 24)
  {
    EXPECT_EQ(initial.info.status, RefinementStatus::iteration_limit);
    EXPECT_GT(result.columns[0].corrections, 0);
    EXPECT_LT(math::abs((*result.solution)[0, 0] - expected),
              math::abs((*initial.solution)[0, 0] - expected) / C::real(100));
  }
#if UNI20_HAS_MPREAL
  auto reference = mpreal("2/5", Precision::bits(512));
  auto lower = Precision::bits(C::digits() <= 24 ? 12 : C::digits() / 2);
  EXPECT_LT(abs(widen(uni20::real((*result.solution)[0, 0])) - reference),
            abs(reference.at(lower).at(reference.precision()) - reference) / 16);
#endif
}

// Case scalars describe working/diagnostic arithmetic. Factors deliberately
// use float32 for both backends, including when MPLAPACK is unavailable.
UNI20_PRECISION_TEST(NumericalLinalg, CpuLuRefinement) { check_refinement<TypeParam>(linalg::CpuReferenceBackend{}); }
UNI20_PRECISION_TEST(NumericalLinalg, ProviderLuRefinement) { check_refinement<TypeParam>(linalg::LapackBackend{}); }

// Precision ordering is intentionally unrestricted. In particular, converting
// binary128 epsilon to a native residual scalar must not depend on whether the
// standard library recognizes the extension type in constructible_from.
UNI20_PRECISION_TEST(NumericalLinalg, CpuLuRefinementMixedResidual)
{
  using namespace linalg;
  using C = TypeParam;
  using S = typename C::scalar_type;
  using F = typename TestScalar<float, C::is_complex>::type;
  using R = typename TestScalar<double, C::is_complex>::type;
  auto a = C::matrix(1, 1), b = C::matrix(1, 1);
  a[0, 0] = C::scalar(2);
  b[0, 0] = C::scalar(1, 1);
  LuRefinementPrecision<F, S, R> p{native_precision<F>(), C::precision(), native_precision<R>()};
  auto result = lu_refine_with_info(CpuReferenceBackend{}, a, b, p);
  ASSERT_TRUE(result.info.succeeded());
  ASSERT_TRUE(result.solution);
  expect_error_at_most((*result.solution)[0, 0], S(C::scalar(1, 1) / C::real(2)), C::real(0));
  C::expect_precision((*result.solution)[0, 0]);
  EXPECT_EQ(result.columns[0].error.componentwise_backward_error, 0.0);
  EXPECT_EQ(precision_of(result.columns[0].error.componentwise_backward_error), native_precision<double>());
}
} // namespace uni20::test
