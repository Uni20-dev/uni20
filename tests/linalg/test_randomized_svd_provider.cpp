#include <uni20/config.hpp>
#if UNI20_BACKEND_CUSOLVER && UNI20_BACKEND_CUBLAS
#include <chrono>
#include <gtest/gtest.h>
#include <iostream>
#include <uni20/linalg/ops/cuda_gesvdr.hpp>
#include <uni20/linalg/ops/randomized_svd.hpp>

namespace
{
using namespace uni20;
using namespace uni20::linalg;
template <class R> void compare_providers()
{
  constexpr int m = 384, n = 256, k = 8;
  constexpr RandomizedRangeOptions opts{k, 16, 2};
  DenseMatrix<R> a(m, n);
  for (int i = 0; i < n; ++i)
    a[i, i] = static_cast<R>(std::pow(0.90, i));
  auto d = to_device(a, 0);
  auto measure = [&](char const* method, auto solve) {
    std::array<double, 3> ms{};
    double value_error = 0, triplet_error = 0;
    for (int repeat = -1; repeat < 3; ++repeat)
    {
      uni20::cuda::check(cudaDeviceSynchronize(), "synchronize before timing", 0);
      auto begin = std::chrono::steady_clock::now();
      auto factors = solve();
      uni20::cuda::check(cudaDeviceSynchronize(), "synchronize after timing", 0);
      if (repeat >= 0)
        ms[repeat] = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - begin).count();
      if (repeat == 2)
      {
        auto s = to_host(factors.singular_values);
        auto u = to_host(factors.left_singular_vectors);
        auto vh = to_host(factors.right_singular_vectors_adjoint);
        for (int j = 0; j < k; ++j)
        {
          value_error = std::max(value_error, std::abs(double(s[j]) / a[j, j] - 1));
          double squared = 0;
          for (int i = 0; i < m; ++i)
          {
            double av = i < n ? double(a[i, i]) * vh[j, i] : 0.;
            squared += std::pow(av - double(u[i, j]) * s[j], 2);
          }
          triplet_error = std::max(triplet_error, std::sqrt(squared) / s[j]);
        }
      }
    }
    std::sort(ms.begin(), ms.end());
    std::cout << method << " " << sizeof(R) * 8 << "-bit median_ms=" << ms[1] << " max_rel_value_error=" << value_error
              << " max_rel_triplet_residual=" << triplet_error << '\n';
    EXPECT_LT(value_error, 0.001);
    EXPECT_LT(triplet_error, 0.02);
  };
  measure("Uni20", [&] { return randomized_truncated_svd(d, opts, random::root(17)); });
  measure("Xgesvdr", [&] {
    return svd_gesvdr(d, {.rank = k, .oversampling = opts.oversampling, .power_iterations = opts.power_iterations});
  });
}
TEST(RandomizedSvd, CudaProviderVectorSelection)
{
  int devices = 0;
  auto status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && !devices))
    GTEST_SKIP();
  ASSERT_EQ(status, cudaSuccess);
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  DenseMatrix<double> a(12, 17);
  a[1, 3] = 8.;
  a[4, 2] = -4.;
  auto d = to_device(a, 0);
  for (bool left : {false, true})
    for (bool right : {false, true})
    {
      auto [u, s, vh] = svd_gesvdr(d, {.rank = 2, .oversampling = 3, .left_vectors = left, .right_vectors = right});
      EXPECT_EQ(u.extent(1), left ? 2 : 0);
      EXPECT_EQ(vh.extent(0), right ? 2 : 0);
      auto values = to_host(s);
      EXPECT_NEAR(values[0], 8., 1.e-11);
      EXPECT_NEAR(values[1], 4., 1.e-11);
      if (left)
      {
        auto v = to_host(u);
        EXPECT_NEAR(std::abs((v[1, 0])), 1., 1.e-10);
      }
      if (right)
      {
        auto v = to_host(vh);
        EXPECT_NEAR(std::abs((v[1, 2])), 1., 1.e-10);
      }
    }
  EXPECT_EQ(svd_gesvdr(d, {.rank = 0}).singular_values.extent(0), 0);
  EXPECT_THROW((void)svd_gesvdr(d, {.rank = 5, .oversampling = 7}), std::invalid_argument);
  EXPECT_EQ((to_host(d)[4, 2]), -4.);
}
TEST(RandomizedSvd, CudaVendorComparison)
{
  int devices = 0;
  auto status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && !devices))
    GTEST_SKIP();
  ASSERT_EQ(status, cudaSuccess);
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  compare_providers<float>();
  compare_providers<double>();
}
} // namespace
#endif
