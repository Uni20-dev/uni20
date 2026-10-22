#include "precision_registry.hpp"
#include <uni20/krylov/dense_host_vector.hpp>
#include <uni20/krylov/gmres.hpp>
#include <uni20/krylov/gmres_refinement.hpp>

namespace uni20::test
{
UNI20_PRECISION_TEST(NumericalKrylov, GmresRotationAndSolve)
{
  using C = TypeParam;
  using S = typename C::scalar_type;
  using V = krylov::DenseHostVector<S>;
  auto phase = C::scalar(1, 1);
  krylov::DenseHostVectorOps<S> ops(2, {C::scalar(0), -phase, phase, C::scalar(0)});
  V x{{C::scalar(0), C::scalar(0)}}, b{{C::scalar(1), C::scalar(0)}};
  auto result = krylov::gmres_with_info<S>(
      ops, ops, x, b, {.relative_tolerance = C::epsilon() * C::real(32), .restart = 2, .max_iterations = 4},
      C::precision());
  ASSERT_TRUE(result.succeeded());
  EXPECT_EQ(result.iterations, 2);
  expect_error_at_most(x.values[0], C::scalar(0), C::epsilon() * C::real(16));
  expect_error_at_most(x.values[1], S(-C::scalar(1) / phase), C::epsilon() * C::real(16));
  C::expect_precision(x.values[0]);
  C::expect_precision(x.values[1]);
  C::expect_precision(result.residual->norm);

  krylov::DenseHostVectorOps<S> gap_ops(
      2, {C::scalar(1), C::scalar(0), C::scalar(0), C::scalar(C::real(1) + C::gap(), C::real(0))});
  x.values = {C::scalar(0), C::scalar(0)};
  b.values = {C::scalar(1), C::scalar(1)};
  result = krylov::gmres_with_info<S>(gap_ops, gap_ops, x, b, {.relative_tolerance = C::epsilon() * C::real(16)},
                                      C::precision());
  ASSERT_TRUE(result.succeeded());
  EXPECT_TRUE(x.values[0] != x.values[1]);
  expect_error_at_most(x.values[1], S(C::scalar(1) / (C::real(1) + C::gap())), C::epsilon() * C::real(16));
}

UNI20_PRECISION_TEST(NumericalKrylov, GmresAccuracyImprovesWithPrecision)
{
#if UNI20_HAS_MPREAL
  using C = TypeParam;
  using S = typename C::scalar_type;
  using V = krylov::DenseHostVector<S>;
  auto phase = C::scalar(1, 1);
  krylov::DenseHostVectorOps<S> ops(2, {C::real(3) * phase, phase, phase, C::real(2) * phase});
  V x{{C::scalar(0), C::scalar(0)}}, b{{phase, C::scalar(0)}};
  auto result =
      krylov::gmres_with_info<S>(ops, ops, x, b, {.relative_tolerance = C::epsilon() * C::real(32)}, C::precision());
  ASSERT_TRUE(result.succeeded());
  mpreal reference("2/5", Precision::bits(512));
  auto error = abs(widen(uni20::real(x.values[0])) - reference);
  auto low = Precision::bits(C::digits() <= 24 ? 12 : C::digits() / 2);
  EXPECT_LT(error, abs(reference.at(low).at(reference.precision()) - reference) / 16);
  expect_error_at_most(x.values[1], S(-C::scalar(1) / C::real(5)), C::epsilon() * C::real(16));
#endif
}
UNI20_PRECISION_TEST(NumericalKrylov, GmresMixedRefinement)
{
  using namespace krylov;
  using C = TypeParam;
  using S = typename C::scalar_type;
  using F = typename TestScalar<float, C::is_complex>::type;
  using V = DenseHostVector<S>;
  auto pw = C::precision();
  auto pr = C::runtime ? Precision::bits(C::digits() * 2) : pw;
  auto phase = C::scalar(1, 1);
  std::vector<S> matrix{C::real(3) * phase, phase, phase, C::real(2) * phase};
  std::vector<F> inner_matrix;
  std::vector<S> residual_matrix;
  for (auto const& value : matrix)
  {
    inner_matrix.push_back(gmres_detail::convert_scalar<F>(value, native_precision<F>()));
    residual_matrix.push_back(gmres_detail::convert_scalar<S>(value, pr));
  }
  DenseHostVectorOps<S> wops(2, matrix), rops(2, residual_matrix);
  DenseHostVectorOps<F> iops(2, inner_matrix);
  V x{{C::scalar(0), C::scalar(0)}};
  V b{{gmres_detail::convert_scalar<S>(phase, pr), scalar_at_precision<S>(0, pr)}};
  DenseHostVector<F> inner{{F{}, F{}}};
  IdentityPreconditioner preconditioner{iops};
  DenseHostVectorTransfer transfer;
  GmresRefinementContext context{.working = krylov_space<S>(wops, x, pw),
                                 .inner = krylov_space<F>(iops, inner),
                                 .residual = krylov_space<S>(rops, b, pr),
                                 .inner_operator = iops,
                                 .residual_operator = rops,
                                 .preconditioner = preconditioner,
                                 .transfer = transfer};
  auto info = gmres_refine_with_info(context, x, b, {.max_corrections = 20});
  ASSERT_TRUE(info.succeeded());
  if constexpr (C::digits() > 24)
  {
    EXPECT_GT(info.corrections, 1);
  }
  expect_error_at_most(x.values[0], S(C::scalar(2) / C::real(5)), C::epsilon() * C::real(4));
  expect_error_at_most(x.values[1], S(-C::scalar(1) / C::real(5)), C::epsilon() * C::real(4));
  C::expect_precision(x.values[0]);
  auto tolerance = gmres_detail::convert_scalar<typename C::real_type>(C::epsilon(), pr);
  EXPECT_TRUE(info.residual->norm / info.residual->rhs_norm <= tolerance);
}
UNI20_PRECISION_TEST(NumericalKrylov, GmresErrorDiagnostics)
{
  using namespace krylov;
  using C = TypeParam;
  using S = typename C::scalar_type;
  using R = typename C::real_type;
  using E = GmresOperatorEstimate<R>;
  auto phase = C::scalar(1, 1);
  R phase_norm = math::abs(phase);
  DenseHostVectorOps<S> ops(2, {C::real(4) * phase, C::scalar(0), C::scalar(0), C::real(2) * phase});
  DenseHostVector<S> x{{C::scalar(0), C::scalar(C::real(1) / C::real(4), C::real(0))}};
  DenseHostVector<S> b{{C::scalar(0), phase}};
  R anorm = (C::real(4) + C::gap()) * phase_norm;
  R inverse = C::real(1) / (C::real(2) * phase_norm);
  R condition = C::real(2) + C::gap();
  auto info = gmres_with_info<S>(
      ops, ops, x, b,
      {.max_iterations = 0,
       .operator_estimates = {.norm = E{anorm}, .inverse_norm = E{inverse}, .condition_number = E{condition}}},
      C::precision());
  ASSERT_TRUE(info.errors);
  auto const& errors = *info.errors;
  R expected_backward = (phase_norm / C::real(2)) / (anorm / C::real(4) + phase_norm);
  expect_error_at_most(*errors.normwise_backward_error, expected_backward, C::epsilon() * C::real(4));
  expect_error_at_most(*errors.absolute_solution_error, R(C::real(1) / C::real(4)), C::epsilon() * C::real(4));
  expect_error_at_most(*errors.relative_solution_error, R(condition / C::real(2)), C::epsilon() * C::real(4));
  EXPECT_TRUE(*errors.relative_solution_error > C::real(1));
  C::expect_precision(*errors.normwise_backward_error);
  C::expect_precision(*errors.absolute_solution_error);
  C::expect_precision(*errors.relative_solution_error);
  C::expect_precision(info.residual->solution_norm);
}
} // namespace uni20::test
