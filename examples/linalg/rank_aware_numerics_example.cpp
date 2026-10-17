#include <iostream>
#include <uni20/core/scalar_io.hpp>
#include <uni20/linalg/ops/lu_rcond.hpp>
#include <uni20/linalg/ops/svd_least_squares.hpp>

template <class R> void demonstrate(uni20::Precision p)
{
  using namespace uni20;
  using namespace uni20::linalg;
  auto value = [p](int n) { return scalar_at_precision<R>(n, p); };
  DenseMatrix<R> a(1, 2, p), b(1, 1, p);
  a[0, 0] = value(1);
  a[0, 1] = value(2);
  b[0, 0] = value(3);
  auto fit = svd_least_squares_with_info(a, b, p, {.return_singular_values = true});
  if (!fit.info.succeeded() || fit.diagnostics->rank != 1) throw std::runtime_error("SVD example failed");
  std::cout << "\n"
            << p.bit_count() << " bits: minimum-norm solution of x0 + 2*x1 = 3 is ("
            << format_scalar((*fit.solution)[0, 0]) << ", " << format_scalar((*fit.solution)[1, 0])
            << "). Expected: (0.6, 1.2).\n"
            << "Rank = " << fit.diagnostics->rank
            << ", relative cutoff = " << format_scalar(fit.diagnostics->relative_cutoff)
            << ", residual = " << format_scalar(fit.residual_norms[0]) << "\n";

  DenseMatrix<R> diagonal(2, 2, p), rhs(2, 1, p);
  diagonal[0, 0] = rhs[0, 0] = value(1);
  diagonal[1, 1] = rhs[1, 0] = numeric_limits<R>::epsilon(p) * value(2);
  auto factors = lu_factor(diagonal, p);
  auto condition = lu_rcond_with_info(factors);
  if (!condition.info.succeeded()) throw std::runtime_error("condition example failed");
  auto full = svd_least_squares_with_info(diagonal, rhs, p);
  auto truncated =
      svd_least_squares_with_info(diagonal, rhs, p, {.relative_cutoff = numeric_limits<R>::epsilon(p) * value(4)});
  if (!full.info.succeeded() || !truncated.info.succeeded()) throw std::runtime_error("cutoff example failed");
  std::cout << "For diag(1, 2*epsilon), the one-norm reciprocal condition estimate is "
            << format_scalar(*condition.estimate) << ".\n"
            << "The default epsilon cutoff retains rank " << full.diagnostics->rank
            << "; choosing 4*epsilon gives rank " << truncated.diagnostics->rank
            << " and discards the second solution component.\n"
            << "That truncation is application policy, not a failure to converge.\n";

  auto without_norms = lu_factor(diagonal, p, LuOptions<R>{.retain_norms = false});
  auto supplied = lu_rcond(without_norms, MatrixNorm::One, value(1));
  std::cout << "With norm capture disabled, supplying the original norm gives rcond = " << format_scalar(supplied)
            << ". The LU factors remain reusable for solves.\n";
}

int main()
{
  std::cout << "This example contrasts numerical rank policy with conditioning.\n"
               "SVD selects a minimum-norm solution when there are fewer equations than unknowns.\n"
               "LU condition estimates describe sensitivity; they do not decide whether a calculation is acceptable.\n";
  demonstrate<double>(uni20::native_precision<double>());
#if UNI20_HAS_MPLAPACK_FLOAT80
  demonstrate<uni20::float80>(uni20::native_precision<uni20::float80>());
#endif
#if UNI20_HAS_MPLAPACK_MPREAL
  demonstrate<uni20::mpreal>(uni20::Precision::bits(128));
  demonstrate<uni20::mpreal>(uni20::Precision::bits(256));
#endif
}
