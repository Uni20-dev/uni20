#include <fmt/format.h>
#include <uni20/common/presentation.hpp>
#include <uni20/linalg/ops/slogdet.hpp>

int main()
{
  using namespace uni20;
  using namespace uni20::linalg;
  fmt::print("Factor A once, solve two right-hand sides, then read its signed log determinant.\n"
             "A = [[0, 2], [1, 3]]; its determinant is -2 and its LU needs a row swap.\n");
  DenseMatrix<double> a(2, 2), b(2, 1);
  a[0, 0] = 0;
  a[0, 1] = 2;
  a[1, 0] = 1;
  a[1, 1] = 3;
  auto factor = lu_factor(a);
  for (double scale : {1.0, 2.0})
  {
    b[0, 0] = 4 * scale;
    b[1, 0] = 7 * scale;
    auto x = lu_solve(factor, b);
    fmt::print("b = [{}, {}] -> x = [{}, {}] (expected [{}, {}])\n", b[0, 0], b[1, 0], x[0, 0], x[1, 0], scale,
               2 * scale);
    if (std::abs(x[0, 0] - scale) > 1e-14 || std::abs(x[1, 0] - 2 * scale) > 1e-14) return 1;
  }
  auto determinant = slogdet(factor);
  fmt::print("det(A): sign = {}, log(abs(det)) = {} (log(2))\n", determinant.phase, determinant.log_absolute);
  if (determinant.phase != -1 || std::abs(determinant.log_absolute - std::log(2.0)) > 1e-14) return 1;

  a[1, 0] = 0;
  auto singular = slogdet_with_info(a);
  fmt::print("Making the first column zero gives a singular diagnostic and the conventional (0, -infinity).\n");
  if (singular.info.status != SolveStatus::singular || !singular.value || singular.value->phase != 0) return 1;

#if UNI20_ENABLE_MPLAPACK_MPFR
  auto p = Precision::bits(256);
  DenseMatrix<mpreal> tiny(2, 2, Precision::exact());
  tiny[0, 0] = mpreal(exact_constant("1e-1000"));
  tiny[1, 1] = mpreal(exact_constant("1e-1000"));
  auto high_precision = slogdet(tiny, p);
  fmt::print("MPFR: diag(1e-1000, 1e-1000), factored at 256 bits, has log(abs(det)) = {}.\n"
             "The exact source stays exact; the logarithm is approximate at the requested precision.\n",
             uni20::presentation::format_real(high_precision.log_absolute, {}));
  if (high_precision.phase != 1 || high_precision.log_absolute.precision() != p) return 1;
#endif
}
