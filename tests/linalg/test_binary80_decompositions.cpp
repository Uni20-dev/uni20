#include <gtest/gtest.h>
#include <uni20/core/math.hpp>
#include <uni20/linalg/async/lq.hpp>
#include <uni20/linalg/async/qr.hpp>
#include <uni20/linalg/async/self_adjoint_eigh.hpp>
#include <uni20/linalg/async/svd.hpp>
#include <uni20/linalg/ops/nonsymmetric_eigen.hpp>
#include <uni20/linalg/ops/schur.hpp>
#include <uni20/linalg/ops/tridiagonal_eigen.hpp>
#include <uni20/tensor/conjugate.hpp>

using namespace uni20;
using namespace uni20::linalg;

namespace
{
using A = decltype(mdspec_of(std::declval<DenseMatrix<float80>&>()));
using Z = decltype(mdspec_of(std::declval<DenseMatrix<complex160>&>()));
using V = decltype(mdspec_of(std::declval<Tensor<float80, 1>&>()));
static_assert(!BlasScalar<float80> && !LapackScalar<float80>);
static_assert(probe_dispatch_kernel_types<LapackBackend, qr_op, A&, A&, A&>() == KernelTypeAcceptance::yes);
static_assert(probe_dispatch_kernel_types<LapackBackend, lq_op, A&, A&, A&>() == KernelTypeAcceptance::yes);
static_assert(probe_dispatch_kernel_types<LapackBackend, self_adjoint_eigh_op, V&, A&>() == KernelTypeAcceptance::yes);
static_assert(probe_dispatch_kernel_types<LapackBackend, singular_values_op, V&, A&>() == KernelTypeAcceptance::yes);
static_assert(probe_dispatch_kernel_types<LapackBackend, nonsymmetric_eigen_op, A&, std::span<complex160>, Z&>() ==
              KernelTypeAcceptance::yes);
// Adding decomposition leaves must not accidentally admit unrelated raw LAPACK routines.
static_assert(probe_dispatch_kernel_types<LapackBackend, schur_op, A&, std::span<complex160>, A&>() ==
              KernelTypeAcceptance::no);
static_assert(probe_dispatch_kernel_types<LapackBackend, symmetric_tridiagonal_eigen_op, std::span<float80>,
                                          std::span<float80>, A&>() == KernelTypeAcceptance::no);
constexpr auto complex_acceptance =
    UNI20_HAS_MPLAPACK_FLOAT80_COMPLEX ? KernelTypeAcceptance::yes : KernelTypeAcceptance::no;
static_assert(probe_dispatch_kernel_types<LapackBackend, qr_op, Z&, Z&, Z&>() == complex_acceptance);
static_assert(probe_dispatch_kernel_types<LapackBackend, lq_op, Z&, Z&, Z&>() == complex_acceptance);
static_assert(probe_dispatch_kernel_types<LapackBackend, self_adjoint_eigh_op, V&, Z&>() == complex_acceptance);
static_assert(probe_dispatch_kernel_types<LapackBackend, singular_values_op, V&, Z&>() == complex_acceptance);
static_assert(probe_dispatch_kernel_types<LapackBackend, nonsymmetric_eigen_op, Z&, std::span<complex160>, Z&>() ==
              complex_acceptance);

template <class S> class Binary80Decomposition : public testing::Test {};
#if UNI20_HAS_MPLAPACK_FLOAT80_COMPLEX
using Scalars = testing::Types<float80, complex160>;
#else
using Scalars = testing::Types<float80>;
#endif
TYPED_TEST_SUITE(Binary80Decomposition, Scalars);
constexpr float80 tolerance = 256 * numeric_limits<float80>::epsilon();

template <class S> auto sample(index_type rows = 3, index_type cols = 2)
{
  RowMajorTensor<S, 2> a(rows, cols);
  for (index_type i = 0; i < rows; ++i)
    for (index_type j = 0; j < cols; ++j)
    {
      a[i, j] = S(i == j ? 4 : i + j + 1);
      if constexpr (Complex<S>) a[i, j].imag(float80(i - j));
    }
  a[0, 0] += S(0x1p-60L);
  return a;
}

template <class Matrix, class Left, class Right> void reconstruct(Matrix const& a, Left const& l, Right const& r)
{
  using S = tensor_element_t<Matrix>;
  auto as = a.mdspan();
  for (index_type i = 0; i < a.extent(0); ++i)
    for (index_type j = 0; j < a.extent(1); ++j)
    {
      S sum{};
      for (index_type k = 0; k < l.extent(1); ++k)
        sum += l[i, k] * r[k, j];
      EXPECT_LE(math::abs(sum - as[i, j]), tolerance);
    }
}

TYPED_TEST(Binary80Decomposition, LogicalViewsAndRowMajorFactors)
{
  using S = TypeParam;
  auto a = sample<S>();
  auto view = uni20::conj(a);
  auto [q, r] = qr(LapackBackend{}, view);
  reconstruct(view, q, r);
  auto [l, q2] = lq(LapackBackend{}, a);
  reconstruct(a, l, q2);
  auto work = a;
  RowMajorTensor<S, 2> qout, rout;
  qr_factorization(LapackBackend{}, qout, rout, work, precision_of(a));
  reconstruct(a, qout, rout);
}

TYPED_TEST(Binary80Decomposition, ConsumingAndOneSidedSvd)
{
  using S = TypeParam;
  for (auto shape : {std::pair{3, 2}, std::pair{2, 3}})
    for (auto left : {SvdVectorExtent::Reduced, SvdVectorExtent::Full})
      for (auto right : {SvdVectorExtent::Reduced, SvdVectorExtent::Full})
      {
        auto a = sample<S>(shape.first, shape.second);
        auto original = a;
        auto [u, s, vh] = svd(LapackBackend{}, std::move(a), SvdOptions{.left = left, .right = right});
        for (index_type i = 0; i < original.extent(0); ++i)
          for (index_type j = 0; j < original.extent(1); ++j)
          {
            S sum{};
            for (index_type k = 0; k < s.extent(0); ++k)
              sum += u[i, k] * s[k] * vh[k, j];
            EXPECT_LE(math::abs(sum - original[i, j]), tolerance);
          }
        auto sl = svd_left(original);
        auto sr = svd_right(original);
        auto values = singular_values(original);
        for (index_type k = 0; k < s.extent(0); ++k)
        {
          EXPECT_LE(math::abs(sl.singular_values[k] - s[k]), tolerance);
          EXPECT_LE(math::abs(sr.singular_values[k] - s[k]), tolerance);
          EXPECT_LE(math::abs(values[k] - s[k]), tolerance);
        }
      }
}

TYPED_TEST(Binary80Decomposition, SelectedTriangleAndEmptyFactors)
{
  using S = TypeParam;
  for (auto triangle : {MatrixTriangle::Upper, MatrixTriangle::Lower})
  {
    RowMajorTensor<S, 2> a(2, 2);
    a[0, 0] = a[1, 1] = S(2);
    a[0, 1] = triangle == MatrixTriangle::Upper ? S(1) : S(numeric_limits<float80>::quiet_NaN());
    a[1, 0] = triangle == MatrixTriangle::Lower ? S(1) : S(numeric_limits<float80>::quiet_NaN());
    auto [w, v] = eigh(a, triangle);
    EXPECT_LE(math::abs(w[0] - 1), tolerance);
    EXPECT_LE(math::abs(w[1] - 3), tolerance);
  }
  for (auto shape : {std::pair{0, 3}, std::pair{3, 0}, std::pair{0, 0}})
  {
    DenseMatrix<S> a(shape.first, shape.second);
    auto [u, s, vh] = svd(a, SvdOptions{.left = SvdVectorExtent::Full, .right = SvdVectorExtent::Full});
    EXPECT_EQ(s.extent(0), 0);
    for (index_type i = 0; i < u.extent(0); ++i)
      for (index_type j = 0; j < u.extent(1); ++j)
        EXPECT_EQ((u[i, j]), S(i == j));
    for (index_type i = 0; i < vh.extent(0); ++i)
      for (index_type j = 0; j < vh.extent(1); ++j)
        EXPECT_EQ((vh[i, j]), S(i == j));
    EXPECT_EQ(qr(a).q.extent(1), 0);
    EXPECT_EQ(lq(a).q.extent(0), 0);
  }
}

TEST(Binary80Decomposition, GeneralRealConjugatePairsAndValuesOnly)
{
  DenseMatrix<float80> a(2, 2);
  a[0, 1] = -1;
  a[1, 0] = 1;
  auto original = a;
  std::vector<complex160> values(2);
  RowMajorTensor<complex160, 2> vectors(2, 2);
  nonsymmetric_eigen(LapackBackend{}, a, std::span(values), vectors, true);
  EXPECT_LE(math::abs(values[0] - complex160(0, 1)), tolerance);
  EXPECT_LE(math::abs(values[1] - complex160(0, -1)), tolerance);
  for (index_type j = 0; j < 2; ++j)
  {
    EXPECT_LE(math::abs(-vectors[1, j] - values[j] * vectors[0, j]), tolerance);
    EXPECT_LE(math::abs(vectors[0, j] - values[j] * vectors[1, j]), tolerance);
  }
  vectors[0, 0] = complex160(42);
  nonsymmetric_eigen(original, std::span(values), vectors, false);
  EXPECT_EQ((vectors[0, 0]), complex160(42));
}

TYPED_TEST(Binary80Decomposition, DecompositionsPreserveExtendedExponentRange)
{
  using S = TypeParam;
  for (float80 scale : {1e3000L, 1e-3000L})
  {
    DenseMatrix<S> a(1, 1);
    a[0, 0] = S(scale);
    auto [q, r] = qr(LapackBackend{}, a);
    EXPECT_LE(math::abs((q[0, 0] * r[0, 0]) / scale - S(1)), tolerance);
    auto s = singular_values(LapackBackend{}, a);
    EXPECT_LE(math::abs(s[0] / scale - 1), tolerance);
    auto [w, v] = eigh(LapackBackend{}, a);
    EXPECT_LE(math::abs(w[0] / scale - 1), tolerance);
    std::vector<complex160> values(1);
    DenseMatrix<complex160> vectors(1, 1);
    nonsymmetric_eigen(LapackBackend{}, a, std::span(values), vectors, true);
    EXPECT_LE(math::abs(values[0] / scale - complex160(1)), tolerance);
  }
}

TYPED_TEST(Binary80Decomposition, PrecisionRejectionPrecedesMutation)
{
  auto a = sample<TypeParam>();
  auto pointer = a.data();
  auto wrong = Precision::bits(53);
  EXPECT_THROW((void)qr(std::move(a), wrong), std::invalid_argument);
  EXPECT_THROW((void)lq(std::move(a), wrong), std::invalid_argument);
  EXPECT_THROW((void)svd(std::move(a), wrong), std::invalid_argument);
  EXPECT_EQ(a.data(), pointer);
  EXPECT_EQ((a[0, 0]), TypeParam(4 + 0x1p-60L));
}

TYPED_TEST(Binary80Decomposition, AsyncPreservingAndConsuming)
{
  using S = TypeParam;
  using Matrix = DenseMatrix<S>;
  async::DebugScheduler scheduler;
  async::ScopedScheduler scope(&scheduler);
  Matrix a(2, 2);
  a[0, 0] = S(2 + 0x1p-60L);
  a[1, 1] = S(1);
  async::Async<Matrix> input(a);
  auto q = qr(input);
  auto l = lq(input, precision_of(a));
  auto eig = eigh(input);
  auto factors = svd(std::move(input));
  auto const& w = eig.eigenvalues.get_wait();
  auto const& s = factors.singular_values.get_wait();
  EXPECT_EQ(w[1], 2 + 0x1p-60L);
  EXPECT_EQ(s[0], 2 + 0x1p-60L);
  reconstruct(a, q.q.get_wait(), q.r.get_wait());
  reconstruct(a, l.l.get_wait(), l.q.get_wait());
}
} // namespace
