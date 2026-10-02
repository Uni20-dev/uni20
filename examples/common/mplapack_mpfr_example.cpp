#include <iostream>
#include <uni20/core/math.hpp>
#include <uni20/core/scalar_io.hpp>
#include <uni20/linalg/ops/linear_solve.hpp>
#include <uni20/linalg/ops/matrix_product.hpp>

int main()
{
  using namespace uni20;
  using C = complex<mpreal>;
  auto p = Precision::bits(400), check = Precision::bits(512);
  auto epsilon = pow(mpreal(2, p), mpreal(-200, p));
  DenseMatrix<C> a(2, 2, p), b(2, 1, p);
  a[0, 0] = C("1", "0", p);
  a[1, 0] = C("1", "0", p);
  a[0, 1] = C("1", "0", p);
  a[1, 1] = C(1 + epsilon);
  b[0, 0] = C("3", "1", p);
  b[1, 0] = C(3 + 2 * epsilon, 1 - epsilon);
  std::cout << "Solving a nearly singular complex system at 400 bits.\n"
            << "A = [[1,1],[1,1+2^-200]], exact x = [1+2i, 2-i].\n"
            << "At binary128's 113-bit precision, the distinction in A rounds away.\n";
  auto x = linalg::solve(a, b);
  bool correct = true;
  for (int i = 0; i < 2; ++i)
  {
    auto residual = a[i, 0].at(check) * x[0, 0].at(check) + a[i, 1].at(check) * x[1, 0].at(check) - b[i, 0].at(check);
    std::cout << "x[" << i << "] = " << format_scalar(x[i, 0])
              << "; independent 512-bit residual magnitude = " << abs(residual) << '\n';
    correct = correct && residual == 0;
  }
  std::cout << "A successful solve reports finite output, not an accuracy guarantee; the residual above is a separate "
               "check.\n";
  auto low_a = a, low_b = b;
  auto info = linalg::solve_inplace_with_info(low_a, low_b, Precision::bits(113));
  std::cout << "Repeating at 113 bits reports singular: " << std::boolalpha
            << (info.status == linalg::SolveStatus::singular) << '\n';
  return correct && info.status == linalg::SolveStatus::singular ? 0 : 1;
}
