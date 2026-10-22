#include <uni20/krylov/dense_host_vector.hpp>
#include <uni20/krylov/gmres_refinement.hpp>

#include <fmt/format.h>

template <class S> struct Stencil
{
    void matvec(uni20::krylov::DenseHostVector<S>& out, uni20::krylov::DenseHostVector<S> const& in) const
    {
      for (std::size_t i = 0; i < in.values.size(); ++i)
      {
        out.values[i] = S(3) * in.values[i];
        if (i > 0) out.values[i] -= in.values[i - 1];
        if (i + 1 < in.values.size()) out.values[i] -= in.values[i + 1];
      }
    }
};

int main()
{
  using namespace uni20::krylov;
  constexpr std::size_t n = 12;
  DenseHostVectorAlgebra<double> working_ops(n);
  DenseHostVectorAlgebra<float> inner_ops(n);
  DenseHostVectorAlgebra<long double> residual_ops(n);
  DenseHostVector<double> x{std::vector<double>(n, 0)};
  DenseHostVector<float> inner{std::vector<float>(n, 0)};
  DenseHostVector<long double> b{std::vector<long double>(n, 1)};
  b.values.front() = b.values.back() = 2;
  Stencil<float> inner_operator;
  Stencil<long double> residual_operator;
  IdentityPreconditioner preconditioner{inner_ops};
  DenseHostVectorTransfer transfer;
  GmresRefinementContext context{.working = krylov_space<double>(working_ops, x),
                                 .inner = krylov_space<float>(inner_ops, inner),
                                 .residual = krylov_space<long double>(residual_ops, b),
                                 .inner_operator = inner_operator,
                                 .residual_operator = residual_operator,
                                 .preconditioner = preconditioner,
                                 .transfer = transfer};
  fmt::print("Matrix-free refinement for a 12-point tridiagonal stencil.\n"
             "Inner GMRES uses float; solution updates use double; true residuals use long double.\n"
             "The RHS is chosen so every exact solution component equals one.\n"
             "The stencil has eigenvalues in [1,5], giving 2-norm bounds ||A|| <= 5, ||A^-1|| <= 1.\n");
  using Estimate = GmresOperatorEstimate<long double>;
  auto info =
      gmres_refine_with_info(context, x, b,
                             {.operator_estimates = {.norm = Estimate{5, GmresEstimateKind::upper_bound},
                                                     .inverse_norm = Estimate{1, GmresEstimateKind::upper_bound}}});
  double max_error = 0;
  for (double value : x.values)
    max_error = std::max(max_error, std::abs(value - 1));
  fmt::print("Converged: {}; corrections: {}; total inner Arnoldi steps: {}\n", info.succeeded(), info.corrections,
             info.inner_iterations);
  if (info.residual)
    fmt::print("Recomputed relative residual: {:.3e}\n", info.residual->norm / info.residual->rhs_norm);
  fmt::print("Maximum component error against the known solution: {:.3e}\n", max_error);
  if (info.errors)
    fmt::print("Estimated absolute solution 2-norm error: {:.3e}; relative error: {:.3e}\n"
               "A zero computed residual gives zero estimates; residual rounding is not bounded here.\n",
               *info.errors->absolute_solution_error, *info.errors->relative_solution_error);
  return info.succeeded() ? 0 : 1;
}
