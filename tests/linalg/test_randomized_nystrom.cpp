#include <array>
#include <chrono>
#include <gtest/gtest.h>
#include <iostream>
#include <uni20/linalg/ops/randomized_nystrom.hpp>
#include <uni20/linalg/ops/randomized_svd.hpp>

namespace
{
using namespace uni20;
using namespace uni20::linalg;

// A Gram matrix with known spectrum. Pair rotations also mix the last kept and first discarded vectors for odd k.
template <class R> DenseMatrix<R> rotated_gram(int n)
{
  DenseMatrix<R> a(n, n);
  for (int i = 0; i < n; i += 2)
  {
    R x = static_cast<R>(std::pow(0.88, i)), y = static_cast<R>(std::pow(0.88, i + 1));
    a[i, i] = R(0.36) * x + R(0.64) * y;
    a[i + 1, i + 1] = R(0.64) * x + R(0.36) * y;
    a[i, i + 1] = a[i + 1, i] = R(0.48) * (x - y);
  }
  return a;
}
template <class T> auto on_host(T const& t)
{
#if UNI20_BACKEND_CUDA
  if constexpr (std::same_as<tensor_storage_policy_t<T>, CudaStorage>)
    return to_host(t);
  else
#endif
    return t;
}
template <class Matrix> void compare(Matrix const& matrix)
{
  using R = tensor_element_t<Matrix>;
  constexpr int k = 7;
  // Four applications of the large matrix in either algorithm.
  constexpr RandomizedRangeOptions ny{k, 12, 3}, sv{k, 12, 1};
  auto host = on_host(matrix);
  auto id = random::root(31);
  auto measure = [&](char const* name, auto solve, auto factors) {
    auto sync = [] {
#if UNI20_BACKEND_CUDA
      if constexpr (std::same_as<tensor_storage_policy_t<Matrix>, CudaStorage>)
        uni20::cuda::check(cudaDeviceSynchronize(), "synchronize PSD comparison", 0);
#endif
    };
    std::array<double, 3> ms{};
    double error = 0, residual = 0, orthogonality = 0;
    for (int iter = -1; iter < 3; ++iter)
    {
      sync();
      auto start = std::chrono::steady_clock::now();
      auto result = solve();
      sync();
      if (iter >= 0)
        ms[iter] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - start).count();
      if (iter != 2) continue;
      auto [values, vectors] = factors(result);
      auto e = on_host(values);
      auto u = on_host(vectors);
      for (int j = 0; j < k; ++j)
      {
        error = std::max(error, std::abs(double(e[j]) / std::pow(0.88, j) - 1));
        double sq = 0;
        for (int i = 0; i < matrix.extent(0); ++i)
        {
          double au = double(host[i, i]) * u[i, j] + double(host[i, i ^ 1]) * u[i ^ 1, j];
          sq += std::pow(au - double(e[j]) * u[i, j], 2);
        }
        residual = std::max(residual, std::sqrt(sq) / e[j]);
        for (int l = 0; l <= j; ++l)
        {
          double dot = 0;
          for (int i = 0; i < matrix.extent(0); ++i)
            dot += double(u[i, j]) * u[i, l];
          orthogonality = std::max(orthogonality, std::abs(dot - (l == j)));
        }
      }
    }
    std::sort(ms.begin(), ms.end());
    std::cout << name << " " << sizeof(R) * 8 << "-bit median_ms=" << ms[1] << " max_rel_eigenvalue_error=" << error
              << " max_rel_eigen_residual=" << residual << '\n';
    EXPECT_LT(error, 0.001);
    EXPECT_LT(residual, 0.03);
    EXPECT_LT(orthogonality, 100 * numeric_limits<R>::epsilon());
  };
  measure(
      "Nystrom", [&] { return randomized_nystrom(matrix, ny, id); },
      [](auto const& r) { return std::tie(r.eigenvalues, r.eigenvectors); });
  measure(
      "rSVD", [&] { return randomized_truncated_svd(matrix, sv, id); },
      [](auto const& r) { return std::tie(r.singular_values, r.left_singular_vectors); });
}
template <class Matrix> void check_deficient(Matrix const& a, double scale)
{
  auto [ed, vd, info] = randomized_nystrom(a, {3, 1, 0}, random::root(29));
  auto e = on_host(ed);
  auto v = on_host(vd);
  EXPECT_EQ(info.retained_rank, 3);
  EXPECT_EQ(info.sketch_width, 4);
  EXPECT_NEAR(e[0] / scale, 1., 2e-12);
  EXPECT_NEAR(e[1] / scale, 0., 1e-13);
  EXPECT_NEAR(e[2] / scale, 0., 1e-13);
  EXPECT_NEAR(std::abs(v[0, 0]), 1., 1e-12);
}
TEST(RandomizedNystrom, GramComparison)
{
  compare(rotated_gram<float>(128));
  compare(rotated_gram<double>(128));
}
TEST(RandomizedNystrom, RankDeficientScaledAndEmpty)
{
  for (double scale : {1e-200, 1., 1e200})
  {
    DenseMatrix<double> a(9, 9);
    a[0, 0] = scale;
    check_deficient(a, scale);
  }
  DenseMatrix<double> zero(4, 4);
  auto z = randomized_nystrom(zero, {3, 4, 1}, random::root(9));
  for (int j = 0; j < 3; ++j)
    EXPECT_EQ(z.eigenvalues[j], 0);
  auto empty = randomized_nystrom(DenseMatrix<double>(0, 0), {0}, random::root(1));
  EXPECT_EQ(empty.eigenvalues.size(), 0);
  EXPECT_EQ(empty.eigenvectors.extent(1), 0);
  EXPECT_THROW((void)randomized_nystrom(DenseMatrix<double>(3, 4), {2}, random::root(2)), std::invalid_argument);
  EXPECT_THROW((void)randomized_nystrom(zero, {5}, random::root(2)), std::invalid_argument);
}
TEST(RandomizedNystrom, OneColumnAgreesWithExplicitNystrom)
{
  // For one column, A omega (omega^T A omega)^-1 omega^T A has one nonzero eigenvalue.
  DenseMatrix<double> a(3, 3);
  a[0, 0] = 4;
  a[1, 1] = 2;
  a[2, 2] = 1;
  auto id = random::root(59);
  DenseMatrix<double> omega(3, 1);
  random::fill(omega, random::child(id, 0), random::Normal{});
  double denominator = 0, numerator = 0;
  for (int i = 0; i < 3; ++i)
  {
    double y = a[i, i] * omega[i, 0];
    denominator += omega[i, 0] * y;
    numerator += y * y;
  }
  auto [values, vectors, info] = randomized_nystrom(a, {1, 0, 0}, id);
  EXPECT_NEAR(values[0], numerator / denominator, 2e-13);
  double alignment = 0;
  for (int i = 0; i < 3; ++i)
    alignment += vectors[i, 0] * a[i, i] * omega[i, 0];
  EXPECT_NEAR(std::abs(alignment) / std::sqrt(numerator), 1., 1e-13);
}
TEST(RandomizedNystrom, UnpoweredApproximationUsesSketch)
{
  auto a = rotated_gram<double>(16);
  auto run = [&](int seed) { return randomized_nystrom(a, {2, 0, 0}, random::root(seed)); };
  auto x = run(1), same = run(1), other = run(2);
  EXPECT_NEAR(x.eigenvalues[0], same.eigenvalues[0], 1e-12);
  EXPECT_GT(std::abs(x.eigenvalues[0] - other.eigenvalues[0]), 1e-4);
  // With a full-rank PSD input the Nyström estimate is below the exact leading eigenvalue.
  EXPECT_GT(x.eigenvalues[0], 0.);
  EXPECT_LT(x.eigenvalues[0], 0.99);
}
#if UNI20_BACKEND_CUSOLVER && UNI20_BACKEND_CUBLAS
TEST(RandomizedNystrom, CudaGramComparisonAndRankDeficiency)
{
  int devices = 0;
  auto status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && !devices))
    GTEST_SKIP();
  ASSERT_EQ(status, cudaSuccess);
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  compare(to_device(rotated_gram<float>(128), 0));
  compare(to_device(rotated_gram<double>(128), 0));
  DenseMatrix<double> a(9, 9);
  a[0, 0] = 1e-180;
  check_deficient(to_device(a, 0), 1e-180);
}
#endif
} // namespace
