#include <iostream>
#include <uni20/core/scalar_io.hpp>
#include <uni20/linalg/ops/lu_refine.hpp>

template <class F, class W, class R> void demonstrate(char const* name, uni20::linalg::LuRefinementPrecision<F, W, R> p)
{
  using namespace uni20;
  using namespace uni20::linalg;
  auto value = [&](int n) { return scalar_at_precision<W>(n, p.working); };
  DenseMatrix<W> a(2, 2, p.working), b(2, 1, p.working);
  a[0, 0] = value(3);
  a[0, 1] = a[1, 0] = value(1);
  a[1, 1] = value(2);
  b[0, 0] = value(1);
  auto initial = lu_refine_with_info(a, b, p, {.max_iterations = 0});
  auto refined = lu_refine_with_info(a, b, p);
  if (!initial.solution || !refined.info.succeeded()) throw std::runtime_error("refinement example failed");
  std::cout << "\n"
            << name << ": factor/working/residual bits = " << p.factor.bit_count() << "/" << p.working.bit_count()
            << "/" << p.residual.bit_count() << "\n"
            << "Initial solution: (" << format_scalar((*initial.solution)[0, 0]) << ", "
            << format_scalar((*initial.solution)[1, 0]) << ")\n"
            << "Refined solution: (" << format_scalar((*refined.solution)[0, 0]) << ", "
            << format_scalar((*refined.solution)[1, 0]) << ")\n"
            << "Componentwise backward error: " << format_scalar(initial.columns[0].error.componentwise_backward_error)
            << " -> " << format_scalar(refined.columns[0].error.componentwise_backward_error) << "\n"
            << "Correction steps: " << refined.columns[0].corrections << "\n";
}

int main()
{
  std::cout << "Solve 3*x + y = 1 and x + 2*y = 0. The exact solution is (2/5, -1/5).\n"
               "LU factors use cheaper arithmetic; residuals from the original equations drive corrections.\n"
               "The solution accumulates at working precision. A small backward error describes satisfaction\n"
               "of these equations, and is not a forward-error bound for an arbitrary ill-conditioned system.\n";
  demonstrate("Native arithmetic", uni20::linalg::LuRefinementPrecision<float, double, long double>::native());
#if UNI20_HAS_MPLAPACK_MPREAL
  using namespace uni20;
  demonstrate("MPFR arithmetic",
              linalg::LuRefinementPrecision<mpreal>{Precision::bits(64), Precision::bits(128), Precision::bits(256)});
#endif
}
