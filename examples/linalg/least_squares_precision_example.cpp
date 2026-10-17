#include <iostream>
#include <uni20/core/scalar_io.hpp>
#include <uni20/linalg/ops/least_squares.hpp>

template <class R> void demonstrate(uni20::Precision p)
{
  using namespace uni20;
  DenseMatrix<R> a(3, 2, p), b(3, 1, p);
  a[0, 0] = a[1, 1] = a[2, 0] = a[2, 1] = scalar_at_precision<R>(1, p);
  b[0, 0] = scalar_at_precision<R>(1, p);
  b[1, 0] = scalar_at_precision<R>(2, p);
  b[2, 0] = scalar_at_precision<R>(4, p);
  auto result = linalg::least_squares_with_info(a, b, p);
  if (!result.info.succeeded()) throw std::runtime_error("example least-squares fit failed");
  std::cout << "\n"
            << p.bit_count() << " bits: x = (" << format_scalar((*result.solution)[0, 0]) << ", "
            << format_scalar((*result.solution)[1, 0])
            << "), residual norm = " << format_scalar(result.residual_norms[0]) << "\n";
  for (index_type i = 0; i < 3; ++i)
    a[i, 1] = scalar_at_precision<R>(0, p);
  auto singular = linalg::least_squares_with_info(a, b, p);
  if (singular.info.succeeded()) throw std::runtime_error("example failed to reject a zero column");
  std::cout << "Zeroing the second column produces recoverable diagonal rejection; no solution is published.\n";
}
int main()
{
  std::cout << "Least squares fits x0=1, x1=2, x0+x1=4 simultaneously.\n"
               "These equations are inconsistent. The best fit is (4/3, 7/3),\n"
               "with residual norm 1/sqrt(3); a nonzero residual is expected.\n"
               "The same API selects native or explicit arbitrary precision.\n";
  demonstrate<double>(uni20::native_precision<double>());
#if UNI20_HAS_FLOAT80
  demonstrate<uni20::float80>(uni20::native_precision<uni20::float80>());
#endif
#if UNI20_HAS_MPREAL
  demonstrate<uni20::mpreal>(uni20::Precision::bits(128));
  demonstrate<uni20::mpreal>(uni20::Precision::bits(256));
#endif
}
