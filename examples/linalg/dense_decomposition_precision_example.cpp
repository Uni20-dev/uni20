#include <iostream>
#include <span>
#include <vector>
#include <uni20/core/scalar_io.hpp>
#include <uni20/linalg/ops/lq.hpp>
#include <uni20/linalg/ops/nonsymmetric_eigen.hpp>
#include <uni20/linalg/ops/qr.hpp>
#include <uni20/linalg/ops/self_adjoint_eigh.hpp>
#include <uni20/linalg/ops/svd.hpp>

// One algorithm for native and runtime precision. The explicit request is
// optional here: each matrix already carries (or encodes) its working precision.
template <class R> void demonstrate(uni20::Precision p)
{
  using namespace uni20;
  DenseMatrix<R> a(2, 2, p);
  a[0, 0] = scalar_at_precision<R>(2, p);
  a[0, 1] = a[1, 0] = a[1, 1] = scalar_at_precision<R>(1, p);
  auto [q, r] = linalg::qr(a, p);
  auto [l, q_lq] = linalg::lq(a, p);
  auto [u, s, vh] = linalg::svd(a, p);
  auto [w, v] = linalg::eigh(a, p);
  auto work = a;
  std::vector<complex<R>> eigenvalues(2);
  DenseMatrix<complex<R>> eigenvectors(2, 2, p);
  linalg::nonsymmetric_eigen(work, std::span(eigenvalues), eigenvectors, true, p);
  R qr_error = scalar_at_precision<R>(0, p), lq_error = qr_error, svd_error = qr_error;
  for (index_type i = 0; i < 2; ++i)
    for (index_type j = 0; j < 2; ++j)
    {
      qr_error += math::abs(q[i, 0] * r[0, j] + q[i, 1] * r[1, j] - a[i, j]);
      lq_error += math::abs(l[i, 0] * q_lq[0, j] + l[i, 1] * q_lq[1, j] - a[i, j]);
      svd_error += math::abs(u[i, 0] * s[0] * vh[0, j] + u[i, 1] * s[1] * vh[1, j] - a[i, j]);
    }
  std::cout << "\nWorking precision: " << p.bit_count() << " bits\n"
            << "Sum of absolute reconstruction errors (smaller is better):\n"
            << "  QR:  " << format_scalar(qr_error) << "\n"
            << "  LQ:  " << format_scalar(lq_error) << "\n"
            << "  SVD: " << format_scalar(svd_error) << "\n"
            << "Self-adjoint eigenvalues, ascending: " << format_scalar(w[0]) << ", " << format_scalar(w[1]) << "\n"
            << "Singular values, descending:        " << format_scalar(s[0]) << ", " << format_scalar(s[1]) << "\n"
            << "General eigenvalues, unsorted:      " << format_scalar(eigenvalues[0]) << ", "
            << format_scalar(eigenvalues[1]) << "\n";
}
int main()
{
  std::cout << "Dense decompositions of A = [[2,1],[1,1]].\n"
            << "A is positive definite: its singular values equal its eigenvalues, (3 +/- sqrt(5))/2.\n"
            << "The same C++ calls below use native double and, when enabled, MPFR.\n";
  demonstrate<double>(uni20::native_precision<double>());
#if UNI20_HAS_MPLAPACK_MPREAL
  demonstrate<uni20::mpreal>(uni20::Precision::bits(128));
  demonstrate<uni20::mpreal>(uni20::Precision::bits(256));
#endif
}
