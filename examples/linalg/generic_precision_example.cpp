#include <uni20/core/scalar_io.hpp>
#include <uni20/linalg/ops/linear_solve.hpp>

#include <iostream>

// The same function works with fixed or runtime precision, without scalar-specific branches.
template <uni20::Real R> void demonstrate(char const* label, uni20::Precision p)
{
  using namespace uni20;
  DenseMatrix<R> a(2, 2, p), b(2, 1, p);
  a[0, 0] = scalar_at_precision<R>(2, p);
  a[0, 1] = a[1, 0] = scalar_at_precision<R>(1, p);
  a[1, 1] = scalar_at_precision<R>(3, p);
  b[0, 0] = scalar_at_precision<R>(1, p);
  auto x = linalg::solve(a, b, precision_of(a));
  std::cout << label << ": " << numeric_limits<R>::digits(p)
            << " significand bits, epsilon=" << format_scalar(numeric_limits<R>::epsilon(p)) << "\n"
            << "  x = (" << format_scalar(R(x[0, 0])) << ", " << format_scalar(R(x[1, 0])) << ")\n";
}

int main()
{
  std::cout << "Solve [[2,1],[1,3]] x = [1,0] with one precision-generic function.\n"
            << "The exact solution is (3/5, -1/5); each run rounds at its chosen precision.\n";
  demonstrate<double>("double", uni20::native_precision<double>());
#if UNI20_ENABLE_MPLAPACK_MPFR
  demonstrate<uni20::mpreal>("MPFR", uni20::Precision::bits(128));
  demonstrate<uni20::mpreal>("MPFR", uni20::Precision::bits(256));
#endif
  try
  {
    demonstrate<double>("incompatible request", uni20::Precision::bits(128));
    return 1;
  }
  catch (std::invalid_argument const& error)
  {
    std::cout << "A 128-bit request for a double result is rejected: " << error.what() << '\n';
  }
}
