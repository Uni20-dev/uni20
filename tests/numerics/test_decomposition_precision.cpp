#include "precision_registry.hpp"
#include <uni20/linalg/ops/lq.hpp>
#include <uni20/linalg/ops/nonsymmetric_eigen.hpp>
#include <uni20/linalg/ops/qr.hpp>
#include <uni20/linalg/ops/self_adjoint_eigh.hpp>
#include <uni20/linalg/ops/svd.hpp>

namespace uni20::test
{
template <class C, class A, class L, class R> void check_factor_product(A const& a, L const& l, R const& r)
{
  for (index_type i = 0; i < a.extent(0); ++i)
    for (index_type j = 0; j < a.extent(1); ++j)
    {
      auto sum = C::scalar(0);
      for (index_type k = 0; k < l.extent(1); ++k)
        sum += l[i, k] * r[k, j];
      expect_error_at_most(sum, typename C::scalar_type(a[i, j]), C::real(128) * C::epsilon());
    }
}
template <class C, class Q> void check_orthogonal(Q const& q, bool rows)
{
  index_type n = q.extent(rows ? 0 : 1), k = q.extent(rows ? 1 : 0);
  for (index_type i = 0; i < n; ++i)
    for (index_type j = 0; j < n; ++j)
    {
      auto sum = C::scalar(0);
      for (index_type t = 0; t < k; ++t)
        sum += rows ? q[i, t] * uni20::conj(q[j, t]) : uni20::conj(q[t, i]) * q[t, j];
      expect_error_at_most(sum, C::scalar(i == j), C::real(64) * C::epsilon());
    }
}
UNI20_PRECISION_TEST(NumericalLinalg, DecompositionFactors)
{
  using C = TypeParam;
  for (auto shape : {std::pair{4, 2}, std::pair{2, 4}, std::pair{3, 3}})
  {
    auto a = C::matrix(shape.first, shape.second);
    for (index_type j = 0; j < a.extent(1); ++j)
      for (index_type i = 0; i < a.extent(0); ++i)
        a[i, j] = C::scalar(static_cast<int>(i == j ? 3 : i + j + 1), static_cast<int>(i - j));
    a[0, 0] += C::scalar(C::gap(), C::real(0));
    auto [q, r] = linalg::qr(linalg::LapackBackend{}, a, C::precision());
    check_factor_product<C>(a, q, r);
    check_orthogonal<C>(q, false);
    auto [l, q2] = linalg::lq(linalg::LapackBackend{}, a);
    check_factor_product<C>(a, l, q2);
    check_orthogonal<C>(q2, true);
    auto [u, s, vh] = linalg::svd(linalg::LapackBackend{}, a, C::precision());
    check_orthogonal<C>(u, false);
    check_orthogonal<C>(vh, true);
    auto us = u;
    for (index_type j = 0; j < us.extent(1); ++j)
      for (index_type i = 0; i < us.extent(0); ++i)
        us[i, j] *= s[j];
    check_factor_product<C>(a, us, vh);
    for (index_type j = 1; j < s.extent(0); ++j)
      EXPECT_GE(s[j - 1], s[j]);
    C::expect_precision(q[0, 0]);
    C::expect_precision(r[0, 0]);
    C::expect_precision(l[0, 0]);
    C::expect_precision(q2[0, 0]);
    C::expect_precision(u[0, 0]);
    C::expect_precision(s[0]);
    C::expect_precision(vh[0, 0]);
    EXPECT_EQ(precision_of(q), C::precision());
    EXPECT_EQ(precision_of(s), C::precision());
  }
}
UNI20_PRECISION_TEST(NumericalLinalg, DecompositionSpectrum)
{
  using C = TypeParam;
  using S = typename C::scalar_type;
  using Z = uni20::complex<typename C::real_type>;
  auto a = C::matrix(2, 2);
  a[0, 0] = C::scalar(2);
  a[1, 1] = C::scalar(2);
  a[0, 1] = C::scalar(1, 1);
  a[1, 0] = uni20::conj(a[0, 1]);
  auto [w, v] = linalg::eigh(linalg::LapackBackend{}, a, C::precision());
  check_orthogonal<C>(v, false);
  for (index_type j = 0; j < 2; ++j)
    for (index_type i = 0; i < 2; ++i)
      expect_error_at_most(S(a[i, 0] * v[0, j] + a[i, 1] * v[1, j]), S(v[i, j] * w[j]), C::real(64) * C::epsilon());
  auto work = a;
  std::vector<Z> ev(2);
  DenseMatrix<Z> vectors(2, 2, C::precision());
  linalg::nonsymmetric_eigen(linalg::LapackBackend{}, work, std::span(ev), vectors, true, C::precision());
  for (index_type j = 0; j < 2; ++j)
    for (index_type i = 0; i < 2; ++i)
      expect_error_at_most(Z(a[i, 0]) * vectors[0, j] + Z(a[i, 1]) * vectors[1, j], vectors[i, j] * ev[j],
                           C::real(64) * C::epsilon());
  C::expect_precision(w[0]);
  C::expect_precision(v[0, 0]);
  C::expect_precision(ev[0]);
  EXPECT_EQ(precision_of(vectors), C::precision());
  auto diagonal = C::matrix(2, 2);
  diagonal[0, 0] = C::scalar(1);
  diagonal[1, 1] = C::scalar(C::real(1) + C::gap(), C::real(0));
  auto spectrum = linalg::eigh(diagonal);
  EXPECT_EQ(spectrum.eigenvalues[1], C::real(1) + C::gap());
  auto singular = linalg::singular_values(diagonal);
  EXPECT_EQ(singular[0], C::real(1) + C::gap());
}
UNI20_PRECISION_TEST(NumericalLinalg, DecompositionAccuracyImprovesWithPrecision)
{
#if UNI20_ENABLE_MPFR
  using C = TypeParam;
  auto a = C::matrix(2, 2);
  a[0, 0] = C::scalar(2);
  a[0, 1] = a[1, 0] = a[1, 1] = C::scalar(1);
  auto [w, v] = linalg::eigh(a);
  auto s = linalg::singular_values(a);
  auto [q, r] = linalg::qr(a);
  auto oracle = Precision::bits(512);
  auto root = uni20::sqrt(mpreal(5, oracle));
  auto expected = (3 + root) / 2;
  auto check = [&](auto actual, mpreal const& reference) {
    auto error = abs(widen(actual) - reference);
    auto lower = Precision::bits(C::digits() <= 24 ? 12 : C::digits() <= 53 ? 24 : C::digits() <= 113 ? 53 : 113);
    auto previous = abs(reference.at(lower).at(oracle) - reference);
    EXPECT_LT(error, previous / 8);
    EXPECT_LE(error, widen(C::epsilon()) * 8);
  };
  check(w[1], expected);
  check(s[0], expected);
  check(math::abs(r[0, 0]), root);
#endif
}
} // namespace uni20::test
