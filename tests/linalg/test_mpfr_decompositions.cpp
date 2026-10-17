#include <gtest/gtest.h>
#include <uni20/config.hpp>
#include <uni20/linalg/async/lq.hpp>
#include <uni20/linalg/async/qr.hpp>
#include <uni20/linalg/async/self_adjoint_eigh.hpp>
#include <uni20/linalg/async/svd.hpp>
#include <uni20/linalg/async/truncated_svd.hpp>
#include <uni20/linalg/ops/nonsymmetric_eigen.hpp>

using namespace uni20;
using namespace uni20::linalg;

TEST(DecompositionPrecision, FixedPrecisionRejectedBeforeConsumptionOrResize)
{
  DenseMatrix<double> a(2, 2), q(1, 1), r(1, 1);
  a[0, 0] = 2;
  a[1, 1] = 3;
  auto pointer = a.data();
  auto wrong = Precision::bits(100);
  EXPECT_THROW((void)qr(std::move(a), wrong), std::invalid_argument);
  EXPECT_THROW((void)lq(std::move(a), wrong), std::invalid_argument);
  EXPECT_THROW((void)svd(std::move(a), wrong), std::invalid_argument);
  EXPECT_THROW((void)eigh(std::move(a), wrong), std::invalid_argument);
  EXPECT_THROW(qr_factorization(q, r, a, wrong), std::invalid_argument);
  EXPECT_EQ(a.data(), pointer);
  EXPECT_EQ(q.extent(0), 1);
  EXPECT_EQ(r.extent(0), 1);
}

#if UNI20_ENABLE_MPLAPACK_MPFR
#include <uni20/async/tbb_scheduler.hpp>
#include <uni20/tensor/conjugate.hpp>
#include <uni20/tensor/reshape.hpp>
namespace
{
template <class S> class MpfrDecomposition : public testing::Test {};
using Scalars = testing::Types<mpreal, complex<mpreal>>;
TYPED_TEST_SUITE(MpfrDecomposition, Scalars);

template <class S> auto sample(Precision p)
{
  DenseMatrix<S> a(3, 2, p);
  a[0, 0] = S(2, p);
  a[1, 0] = S(1, p);
  a[1, 1] = S(2, p);
  a[2, 1] = S(3, p);
  if constexpr (Complex<S>) a[0, 1] = S(mpreal(0, p), mpreal(1, p));
  return a;
}
template <class T> void check_precision(T const& tensor, Precision p)
{
  EXPECT_EQ(precision_of(tensor), p);
  for (auto const& v : tensor.storage())
    EXPECT_EQ(precision_of(v), p);
}

TYPED_TEST(MpfrDecomposition, ExactAndMissingDefaultsNeedExplicitPrecision)
{
  using S = TypeParam;
  auto exact = Precision::exact(), p = Precision::bits(192);
  auto a = sample<S>(exact);
  EXPECT_THROW((void)qr(a), std::logic_error);
  EXPECT_THROW((void)lq(a), std::logic_error);
  EXPECT_THROW((void)svd(a), std::logic_error);
  EXPECT_THROW((void)singular_values(a), std::logic_error);
  auto q = qr(a, p);
  check_precision(q.q, p);
  check_precision(q.r, p);
  auto l = lq(a, p);
  check_precision(l.l, p);
  check_precision(l.q, p);
  auto s = svd(a, p);
  check_precision(s.singular_values, p);
  EXPECT_EQ(precision_of(a), exact);
  DenseMatrix<S> missing(uninitialized, 2, 2);
  for (auto& value : missing.storage())
    value = S{0};
  missing[0, 0] = S{1};
  missing[1, 1] = S{2};
  EXPECT_THROW((void)eigh(missing), std::logic_error);
  auto eig = eigh(missing, p);
  check_precision(eig.eigenvalues, p);
  check_precision(eig.eigenvectors, p);
}

TYPED_TEST(MpfrDecomposition, ConsumingSvdAndEighPreserveStorageAndPrecision)
{
  using S = TypeParam;
  auto p = Precision::bits(192), target = Precision::bits(256);
  auto a = sample<S>(p);
  auto pointer = a.data();
  auto factors = svd(std::move(a), target);
  EXPECT_EQ(factors.left_singular_vectors.storage().data(), pointer);
  check_precision(factors.singular_values, target);
  EXPECT_EQ(precision_of(factors.left_singular_vectors), target);
  auto left = svd_left(sample<S>(p), target);
  check_precision(left.singular_values, target);
  auto right = svd_right(sample<S>(p), target);
  check_precision(right.singular_values, target);
  auto values = singular_values(sample<S>(p), target);
  check_precision(values, target);
  RowMajorTensor<S, 2> hermitian(2, 2, p);
  hermitian[0, 0] = S(2, p);
  hermitian[1, 1] = S(3, p);
  if constexpr (Complex<S>)
    hermitian[0, 1] = S(mpreal(1, p), mpreal(1, p));
  else
    hermitian[0, 1] = S(1, p);
  hermitian[1, 0] = uni20::conj(hermitian[0, 1]);
  auto original = hermitian;
  auto hptr = hermitian.data();
  auto [w, v] = eigh(std::move(hermitian), target);
  EXPECT_EQ(v.storage().data(), hptr);
  EXPECT_EQ(precision_of(v), target);
  check_precision(w, target);
  for (index_type j = 0; j < 2; ++j)
    for (index_type i = 0; i < 2; ++i)
    {
      auto residual = original[i, 0].at(target) * v[0, j] + original[i, 1].at(target) * v[1, j] - v[i, j] * w[j];
      EXPECT_LT(math::abs(residual), math::ldexp(mpreal(1, target), -240));
    }
}

TYPED_TEST(MpfrDecomposition, ConsumedSvdReconstructsTallWideAndFullFactors)
{
  using S = TypeParam;
  auto p = Precision::bits(192);
  for (auto shape : {std::pair{3, 2}, std::pair{2, 3}})
    for (auto left : {SvdVectorExtent::Reduced, SvdVectorExtent::Full})
      for (auto right : {SvdVectorExtent::Reduced, SvdVectorExtent::Full})
      {
        DenseMatrix<S> a(shape.first, shape.second, p);
        for (index_type j = 0; j < a.extent(1); ++j)
          for (index_type i = 0; i < a.extent(0); ++i)
          {
            a[i, j] = S(i == j ? 4 : i + j + 1, p);
            if constexpr (Complex<S>) a[i, j].imag(mpreal(i - j, p));
          }
        auto original = a;
        auto [u, w, vh] = svd(std::move(a), p, SvdOptions{.left = left, .right = right});
        for (index_type j = 0; j < original.extent(1); ++j)
          for (index_type i = 0; i < original.extent(0); ++i)
          {
            auto sum = S(0, p);
            for (index_type k = 0; k < w.extent(0); ++k)
              sum += u[i, k] * w[k] * vh[k, j];
            EXPECT_LT(math::abs(sum - original[i, j]), math::ldexp(mpreal(1, p), -175));
          }
      }
}

TYPED_TEST(MpfrDecomposition, SelectedTriangleAndEmptyFullFactors)
{
  using S = TypeParam;
  auto p = Precision::bits(192);
  DenseMatrix<S> a(uninitialized, 2, 2, p);
  a[0, 0] = S(2, p);
  a[0, 1] = S(1, p);
  a[1, 1] = S(2, p);
  // The lower triangle is unset and must never be numerically read.
  auto [w, v] = eigh(a);
  EXPECT_LT(abs(w[0] - 1), math::ldexp(mpreal(1, p), -180));
  EXPECT_LT(abs(w[1] - 3), math::ldexp(mpreal(1, p), -180));
  for (auto shape : {std::pair{0, 3}, std::pair{3, 0}})
  {
    DenseMatrix<S> empty(shape.first, shape.second, p);
    auto [u, s, vh] = svd(empty, p, SvdOptions{.left = SvdVectorExtent::Full, .right = SvdVectorExtent::Full});
    check_precision(u, p);
    check_precision(s, p);
    check_precision(vh, p);
    for (index_type j = 0; j < u.extent(1); ++j)
      for (index_type i = 0; i < u.extent(0); ++i)
        EXPECT_EQ((u[i, j]), S(i == j ? 1 : 0));
    for (index_type j = 0; j < vh.extent(1); ++j)
      for (index_type i = 0; i < vh.extent(0); ++i)
        EXPECT_EQ((vh[i, j]), S(i == j ? 1 : 0));
    auto qr_result = qr(empty);
    auto lq_result = lq(empty);
    EXPECT_EQ(precision_of(qr_result.q), p);
    EXPECT_EQ(precision_of(lq_result.q), p);
  }
}

TYPED_TEST(MpfrDecomposition, AsyncSelectsPrecisionAfterPublication)
{
  using S = TypeParam;
  using Matrix = DenseMatrix<S>;
  async::TbbScheduler scheduler{4};
  async::ScopedScheduler scope(&scheduler);
  struct Results
  {
      decltype(qr(std::declval<async::Async<Matrix> const&>())) qr_result;
      decltype(lq(std::declval<async::Async<Matrix> const&>())) lq_result;
      decltype(svd(std::declval<async::Async<Matrix> const&>())) svd_result;
      decltype(eigh(std::declval<async::Async<Matrix> const&>())) eig_result;
  };
  std::vector<Results> results;
  for (int bits : {128, 192, 256, 320})
  {
    auto p = Precision::bits(bits);
    async::Async<Matrix> a;
    results.push_back({qr(a), lq(a, p), svd(a, p), eigh(a)});
    scheduler.schedule([](async::WriteBuffer<Matrix> out, Precision p) static -> async::AsyncTask {
      Matrix x(2, 2, p);
      x[0, 0] = S(2, p);
      x[1, 1] = S(3, p);
      x[0, 1] = x[1, 0] = S(1, p);
      co_await out = std::move(x);
    }(a.write(), p));
  }
  int bits = 128;
  for (auto& result : results)
  {
    auto p = Precision::bits(bits);
    bits += 64;
    check_precision(result.qr_result.q.get_wait(), p);
    check_precision(result.qr_result.r.get_wait(), p);
    check_precision(result.lq_result.l.get_wait(), p);
    check_precision(result.lq_result.q.get_wait(), p);
    check_precision(result.svd_result.singular_values.get_wait(), p);
    check_precision(result.eig_result.eigenvalues.get_wait(), p);
  }
  async::Async<Matrix> a(sample<S>(Precision::bits(128)));
  auto consuming = svd(std::move(a), Precision::bits(256));
  check_precision(consuming.singular_values.get_wait(), Precision::bits(256));
  auto p = Precision::bits(192);
  auto cq = qr(async::Async<Matrix>(sample<S>(p)), p);
  auto cl = lq(async::Async<Matrix>(sample<S>(p)), p);
  auto sl = svd_left(async::Async<Matrix>(sample<S>(p)), p);
  auto sr = svd_right(async::Async<Matrix>(sample<S>(p)), p);
  auto sv = singular_values(async::Async<Matrix>(sample<S>(p)), p);
  Matrix h(2, 2, p);
  h[0, 0] = S(2, p);
  h[1, 1] = S(1, p);
  auto ce = eigh(async::Async<Matrix>(std::move(h)), p);
  check_precision(cq.q.get_wait(), p);
  check_precision(cl.q.get_wait(), p);
  check_precision(sl.singular_values.get_wait(), p);
  check_precision(sr.singular_values.get_wait(), p);
  check_precision(sv.get_wait(), p);
  check_precision(ce.eigenvalues.get_wait(), p);
}

TYPED_TEST(MpfrDecomposition, ViewsKeepParentDefaultsAndProviderContextIsRestored)
{
  using S = TypeParam;
  auto p = Precision::bits(192), old = Precision::bits(80);
  auto a = sample<S>(p);
  auto conjugated = uni20::conj(a);
  auto [q, r] = qr(conjugated, p);
  for (index_type j = 0; j < a.extent(1); ++j)
    for (index_type i = 0; i < a.extent(0); ++i)
    {
      auto sum = S(0, p);
      for (index_type k = 0; k < r.extent(0); ++k)
        sum += q[i, k] * r[k, j];
      EXPECT_LT(math::abs(sum - uni20::conj(a[i, j])), math::ldexp(mpreal(1, p), -180));
    }
  DenseMatrix<S> q_owner(3, 2, old), r_owner(2, 2, old);
  auto q_view = reshape_view(q_owner, 3, 2), r_view = reshape_view(r_owner, 2, 2);
  auto before_precision = mpfr_get_default_prec();
  auto before_rounding = mpfr_get_default_rounding_mode();
  struct Restore
  {
      mpfr_prec_t precision;
      mpfr_rnd_t rounding;
      ~Restore()
      {
        mpfr_set_default_prec(precision);
        mpfr_set_default_rounding_mode(rounding);
      }
  } restore{before_precision, before_rounding};
  mpfr_set_default_prec(73);
  mpfr_set_default_rounding_mode(MPFR_RNDD);
  qr_factorization(q_view, r_view, a, p);
  EXPECT_EQ(precision_of(q_owner), old);
  EXPECT_EQ(precision_of(r_owner), old);
  EXPECT_EQ((q_owner[0, 0].precision()), p);
  (void)lq(a, p);
  (void)svd(a, p);
  DenseMatrix<S> h(2, 2, p);
  h[0, 0] = S(2, p);
  h[0, 1] = h[1, 0] = h[1, 1] = S(1, p);
  Tensor<mpreal, 1> w;
  self_adjoint_eigh(w, h, p);
  EXPECT_EQ(precision_of(w), p);
  EXPECT_EQ(mpfr_get_default_prec(), 73);
  EXPECT_EQ(mpfr_get_default_rounding_mode(), MPFR_RNDD);
}

TYPED_TEST(MpfrDecomposition, TruncationRetainsPrecisionAndReportsDiscardedWeight)
{
  using S = TypeParam;
  auto p = Precision::bits(192);
  DenseMatrix<S> a(2, 2, Precision::exact());
  a[0, 0] = S{3};
  a[1, 1] = S{1};
  SvdTruncationPolicy<mpreal> policy{.maximum_retained_extent = 1};
  auto [u, s, vh, info] = truncated_svd(a, p, policy);
  check_precision(u, p);
  check_precision(s, p);
  check_precision(vh, p);
  EXPECT_EQ(info.retained_rank, 1);
  EXPECT_LT(abs(info.original_squared_norm - 10), math::ldexp(mpreal(1, p), -180));
  EXPECT_LT(abs(info.discarded_weight - mpreal("0.1", p)), math::ldexp(mpreal(1, p), -180));
  async::DebugScheduler scheduler;
  async::ScopedScheduler scope(&scheduler);
  auto result = truncated_svd(async::Async<DenseMatrix<S>>(std::move(a)), p, policy);
  check_precision(result.singular_values.get_wait(scheduler), p);
  EXPECT_EQ(result.truncation.get_wait(scheduler).retained_rank, 1);
}

TEST(MpfrDecompositions, GeneralRealEigenvectorsUnpackConjugatePairs)
{
  auto p = Precision::bits(192);
  DenseMatrix<mpreal> a(2, 2, p);
  a[0, 1] = mpreal(-1, p);
  a[1, 0] = mpreal(1, p);
  auto original = a;
  std::vector<complex<mpreal>> w(2);
  DenseMatrix<complex<mpreal>> v(2, 2, p);
  nonsymmetric_eigen(a, std::span(w), v, true);
  EXPECT_EQ(w[0].real(), 0);
  EXPECT_EQ(w[0].imag(), 1);
  EXPECT_EQ(w[1], uni20::conj(w[0]));
  for (index_type j = 0; j < 2; ++j)
    for (index_type i = 0; i < 2; ++i)
      EXPECT_LT(math::abs(original[i, 0] * v[0, j] + original[i, 1] * v[1, j] - v[i, j] * w[j]),
                math::ldexp(mpreal(1, p), -180));
  auto dummy = sample<complex<mpreal>>(p);
  auto saved = dummy;
  a = original;
  nonsymmetric_eigen(a, std::span(w), dummy, false);
  EXPECT_EQ((dummy[0, 0]), (saved[0, 0]));
}
} // namespace
#endif
