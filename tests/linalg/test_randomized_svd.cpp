#include <cmath>
#include <gtest/gtest.h>
#include <uni20/linalg/ops/randomized_svd.hpp>

using namespace uni20;
using namespace uni20::linalg;
namespace
{
template <class R> DenseMatrix<R> spectrum_matrix(int m, int n)
{
  // Orthogonal left and right rotations preserve the prescribed singular values.
  DenseMatrix<R> a(m, n);
  for (int j = 0; j < std::min(m, n); ++j)
    a[j, j] = std::ldexp(R{1}, -j);
  for (int i = 1; i < m; i += 2)
    for (int j = 0; j < n; ++j)
    {
      R x = a[i - 1, j], y = a[i, j];
      a[i - 1, j] = R(0.6) * x - R(0.8) * y;
      a[i, j] = R(0.8) * x + R(0.6) * y;
    }
  for (int j = 2; j < n; j += 2)
    for (int i = 0; i < m; ++i)
    {
      R x = a[i, j - 1], y = a[i, j];
      a[i, j - 1] = R(0.8) * x - R(0.6) * y;
      a[i, j] = R(0.6) * x + R(0.8) * y;
    }
  return a;
}

template <class T> auto host_factors(T const& t)
{
#if UNI20_BACKEND_CUDA
  if constexpr (std::same_as<tensor_storage_policy_t<T>, CudaStorage>)
    return to_host(t);
  else
#endif
    return t;
}

template <class A, class B> double projector_error(A const& a, B const& b, int k)
{
  double error = 0;
  for (int i = 0; i < a.extent(0); ++i)
    for (int j = 0; j < a.extent(0); ++j)
    {
      double d = 0;
      for (int l = 0; l < k; ++l)
        d += double(a[i, l]) * a[j, l] - double(b[i, l]) * b[j, l];
      error += d * d;
    }
  return std::sqrt(error);
}
template <class Matrix> void check_svd(Matrix const& a)
{
  using R = tensor_element_t<Matrix>;
  constexpr int k = 4, p = 4, q = 2;
  auto original = host_factors(a);
  auto ref = svd(original);
  auto id = random::root(17);
  // Exercise the default power count against the explicit two-step replay below.
  auto [ud, sd, vhd, info] = randomized_truncated_svd(a, {.rank = k, .oversampling = p}, id);
  auto u = host_factors(ud);
  auto s = host_factors(sd);
  auto vh = host_factors(vhd);
  EXPECT_EQ(info.retained_rank, k);
  EXPECT_EQ(info.sketch_width, k + p);
  // Power iteration suppresses the unresolved tail by (sigma[width]/sigma[k-1])^(2q+1).
  double bound = std::max(100. * numeric_limits<R>::epsilon(), 32. * std::pow(std::ldexp(1., -1 - p), 2 * q + 1));
  for (int i = 0; i < k; ++i)
  {
    EXPECT_NEAR(s[i], std::ldexp(1., -i), bound);
    EXPECT_NEAR(s[i], ref.singular_values[i], bound);
    double squared = 0;
    for (int row = 0; row < a.extent(0); ++row)
    {
      double av = 0;
      for (int col = 0; col < a.extent(1); ++col)
        av += double(original[row, col]) * vh[i, col];
      squared += std::pow(av - double(u[row, i]) * s[i], 2);
    }
    EXPECT_LT(std::sqrt(squared), bound);
  }
  EXPECT_LT(projector_error(u, ref.left_singular_vectors, k), bound);
  EXPECT_LT(projector_error(permute<1, 0>(vh), permute<1, 0>(ref.right_singular_vectors_adjoint), k), bound);
  auto replay = randomized_truncated_svd(a, {k, p, q}, id);
  auto u2 = host_factors(replay.left_singular_vectors);
  auto s2 = host_factors(replay.singular_values);
  auto vh2 = host_factors(replay.right_singular_vectors_adjoint);
  for (int j = 0; j < k; ++j)
  {
    EXPECT_NEAR(s[j], s2[j], bound);
    for (int i = 0; i < u.extent(0); ++i)
      EXPECT_NEAR((u[i, j]), (u2[i, j]), bound);
    for (int i = 0; i < vh.extent(1); ++i)
      EXPECT_NEAR((vh[j, i]), (vh2[j, i]), bound);
  }
  auto weak = randomized_truncated_svd(a, {k, 0, 0}, id);
  auto other = randomized_truncated_svd(a, {k, 0, 0}, random::root(18));
  EXPECT_GT(projector_error(host_factors(weak.left_singular_vectors), host_factors(other.left_singular_vectors), k),
            0.01);
}
template <class R>
concept AcceptsSketch = requires(DenseMatrix<R> const& a) { randomized_truncated_svd(a, {1}, random::root(0)); };
static_assert(!AcceptsSketch<long double> && !AcceptsSketch<uni20::complex<double>>);
TEST(RandomizedSvd, TallAndWide)
{
  check_svd(spectrum_matrix<float>(23, 16));
  check_svd(spectrum_matrix<double>(16, 23));
}
TEST(RandomizedSvd, ThinBasisMatchesFullQr)
{
  auto a = spectrum_matrix<double>(13, 5);
  auto q = orthonormal_basis(LapackBackend{}, a);
  auto reference = qr(a);
  EXPECT_LT(projector_error(q, reference.q, 5), 1.e-12);
  EXPECT_THROW((void)orthonormal_basis(permute<1, 0>(a)), std::invalid_argument);
}

TEST(RandomizedSvd, RankAndEmptyShapes)
{
  DenseMatrix<double> a(4, 3);
  a[0, 0] = 2;
  auto [u, s, vh, info] = randomized_truncated_svd(a, {2, 16, 1}, random::root(1));
  EXPECT_EQ(info.sketch_width, 3);
  EXPECT_NEAR(s[0], 2, 1e-13);
  EXPECT_NEAR(s[1], 0, 1e-13);
  DenseMatrix<double> empty(0, 7);
  auto zero = randomized_truncated_svd(empty, {0}, random::root(1));
  EXPECT_EQ(zero.truncation.sketch_width, 0);
  EXPECT_EQ(zero.right_singular_vectors_adjoint.extent(1), 7);
  for (auto options : {RandomizedRangeOptions{-1}, RandomizedRangeOptions{4}, RandomizedRangeOptions{1, -1},
                       RandomizedRangeOptions{1, 0, -1}})
    EXPECT_THROW((void)randomized_truncated_svd(a, options, random::root(2)), std::invalid_argument);
}
#if UNI20_BACKEND_CUSOLVER && UNI20_BACKEND_CUBLAS
TEST(RandomizedSvd, CudaTallAndWide)
{
  int devices = 0;
  auto status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && !devices))
    GTEST_SKIP();
  ASSERT_EQ(status, cudaSuccess);
  auto const ordinals = devices > 1 ? std::vector<int>{0, 1} : std::vector<int>{0};
  auto runtime = uni20::cuda::initialize({.device_ordinals = ordinals, .streams_per_device = 1});
  int const ordinal = ordinals.back();
  for (auto [m, n] : {std::pair{9, 4}, std::pair{4, 9}, std::pair{1, 4}, std::pair{4, 1}, std::pair{1, 1},
                      std::pair{4, 0}, std::pair{0, 4}})
  {
    auto a = spectrum_matrix<double>(m, n);
    auto device = to_device(a, ordinal);
    auto [qd, rd] = qr(device);
    static_assert(std::same_as<decltype(qd), CudaMatrix<double>>);
    if (std::min(m, n) != 0)
    {
      EXPECT_EQ(qd.mdspec().data_descriptor().buffer().device().ordinal(), ordinal);
    }
    auto q = to_host(qd);
    auto r = to_host(rd);
    int const k = std::min(m, n);
    ASSERT_EQ(q.extent(1), k);
    ASSERT_EQ(r.extent(0), k);
    for (int j = 0; j < n; ++j)
      for (int i = 0; i < m; ++i)
      {
        double value = 0;
        for (int l = 0; l < k; ++l)
          value += q[i, l] * r[l, j];
        EXPECT_NEAR(value, (a[i, j]), 1.e-12);
      }
    if (m >= n)
    {
      auto only_q = to_host(orthonormal_basis(std::move(device)));
      for (int j = 0; j < k; ++j)
        for (int i = 0; i < m; ++i)
          EXPECT_NEAR((only_q[i, j]), (q[i, j]), 1.e-13);
    }
  }
  auto empty = randomized_truncated_svd(to_device(DenseMatrix<float>(0, 4), 0), {0}, random::root(0));
  EXPECT_EQ(empty.right_singular_vectors_adjoint.extent(1), 4);
  auto one = randomized_truncated_svd(to_device(spectrum_matrix<float>(3, 1), 0), {1}, random::root(0));
  EXPECT_NEAR(to_host(one.singular_values)[0], 1.f, 1e-5f);
  check_svd(to_device(spectrum_matrix<float>(23, 16), ordinal));
  check_svd(to_device(spectrum_matrix<double>(16, 23), 0));
}
#endif
} // namespace
