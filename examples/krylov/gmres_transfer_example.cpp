#include <uni20/krylov/dense_host_vector.hpp>
#include <uni20/krylov/gmres.hpp>

#include <fmt/format.h>

int main()
{
  using namespace uni20::krylov;
  using Vector = DenseHostVector<double>;
  // T has one fixed point. P projects onto it, so I-T+P is invertible
  // without changing the equation on the complementary subspace.
  struct ProjectedTransfer
  {
      void matvec(Vector& out, Vector const& in) const
      {
        out.values[0] = in.values[0];
        out.values[1] = 0.001 * in.values[1];
        out.values[2] = 0.3 * in.values[2];
      }
  } op;
  DenseHostVectorAlgebra<double> ops(3);
  Vector x{{0, 900, 2}}, b{{0, 1, 1}};
  fmt::print("Matrix-free projected transfer equation: (I - T + P)x = b.\n"
             "T has eigenvalues 1, 0.999, 0.7; b has no fixed-point component.\n"
             "The supplied guess is (0, 900, 2); the expected solution is (0, 1000, 10/3).\n"
             "Known 2-norm bounds: ||A|| <= 1 and ||A^-1|| <= 1000.\n");
  using Estimate = GmresOperatorEstimate<double>;
  auto info =
      gmres_with_info<double>(ops, op, x, b,
                              {.relative_tolerance = 1e-12,
                               .operator_estimates = {.norm = Estimate{1, GmresEstimateKind::upper_bound},
                                                      .inverse_norm = Estimate{1000, GmresEstimateKind::upper_bound}}});
  fmt::print("Converged: {}; Arnoldi steps: {}; recomputed residual norm: {:.3e}\n", info.succeeded(), info.iterations,
             info.residual ? info.residual->norm : -1.0);
  fmt::print("Solution: ({:.12g}, {:.12g}, {:.12g})\n", x.values[0], x.values[1], x.values[2]);
  if (info.errors)
    fmt::print("Estimated backward error: {:.3e}; absolute solution error: {:.3e}; relative solution error: {:.3e}\n"
               "These estimates do not include rounding error in residual evaluation.\n",
               *info.errors->normwise_backward_error, *info.errors->absolute_solution_error,
               *info.errors->relative_solution_error);
  return info.succeeded() ? 0 : 1;
}
