#include <uni20/config.hpp>
#if UNI20_BACKEND_CUSOLVER && UNI20_BACKEND_CUBLAS
#include <chrono>
#include <gtest/gtest.h>
#include <iostream>
#include <uni20/linalg/backends/cusolver/workspace.hpp>
#include <uni20/linalg/ops/randomized_svd.hpp>

namespace
{
using namespace uni20;
using namespace uni20::linalg;
namespace rd = uni20::linalg::detail::randomized;
namespace cs = uni20::linalg::detail::cusolver_backend;
template <class R> auto vendor_svd(CudaTensor<R, 2> const& input, RandomizedRangeOptions opts)
{
  auto& resources = input.mdspec().data_descriptor().buffer().resources();
  auto device = resources.device().ordinal();
  uni20::cuda::ScopedDevice guard(device);
  auto m = input.extent(0), n = input.extent(1), k = opts.rank;
  auto a = to_device(input, resources);
  // Use the full buffer dimensions specified by NVIDIA for Xgesvdr.
  auto values = make_tensor_like(uninitialized, a, std::min(m, n));
  auto left = make_tensor_like(uninitialized, a, m, m);
  auto right = make_tensor_like(uninitialized, a, n, n);
  {
    auto lease = cusolver::execution_pool(resources).acquire();
    auto const& stream = lease.stream();
    auto handle = lease.handle().native_handle();
    cusolverDnParams_t raw = nullptr;
    cusolver::check(cusolverDnCreateParams(&raw), "create Xgesvdr parameters", device);
    std::unique_ptr<std::remove_pointer_t<cusolverDnParams_t>, decltype(&cusolverDnDestroyParams)> params(
        raw, cusolverDnDestroyParams);
    auto write = [&](auto& matrix) {
      return matrix.mdspec().data_descriptor().buffer().write_synchronized_with(stream);
    };
    auto aw = write(a), sw = write(values), uw = write(left), vw = write(right);
    constexpr auto dtype = std::same_as<R, float> ? CUDA_R_32F : CUDA_R_64F;
    auto invoke = [&](auto fn, auto... workspace) {
      return fn(handle, params.get(), 'S', 'S', m, n, k, opts.oversampling, opts.power_iterations, dtype, aw.data(), m,
                dtype, sw.data(), dtype, uw.data(), m, dtype, vw.data(), n, dtype, workspace...);
    };
    std::size_t device_bytes = 0, host_bytes = 0;
    cusolver::check(invoke(cusolverDnXgesvdr_bufferSize, &device_bytes, &host_bytes), "Xgesvdr_bufferSize", device);
    bool pools = resources.device().capabilities().memory_pools_supported;
    cs::StreamAllocation<std::byte> scratch(stream, device_bytes, pools);
    cs::StreamAllocation<int> info(stream, 1, pools);
    std::vector<std::byte> host(host_bytes);
    cusolver::check(invoke(cusolverDnXgesvdr, scratch.data(), device_bytes, host.data(), host_bytes, info.data()),
                    "Xgesvdr", device);
    int status = 0;
    uni20::cuda::check(
        cudaMemcpyAsync(&status, info.data(), sizeof(status), cudaMemcpyDeviceToHost, stream.native_handle()),
        "read Xgesvdr status", device);
    stream.synchronize();
    aw.release_after_synchronization();
    sw.release_after_synchronization();
    uw.release_after_synchronization();
    vw.release_after_synchronization();
    EXPECT_EQ(status, 0);
  }
  auto u = make_tensor_like(uninitialized, a, m, k);
  auto s = make_tensor_like(uninitialized, a, k);
  auto vh = make_tensor_like(uninitialized, a, k, n);
  rd::copy_prefix(u, left);
  rd::copy_prefix(s, values);
  rd::copy_prefix(vh, permute<1, 0>(right));
  return SvdResult<decltype(u), decltype(s), decltype(vh)>{std::move(u), std::move(s), std::move(vh)};
}
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
  measure("Xgesvdr", [&] { return vendor_svd(d, opts); });
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
