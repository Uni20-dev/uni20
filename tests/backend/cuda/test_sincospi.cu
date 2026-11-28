#include <uni20/core/math.hpp>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <memory>

namespace
{
template <class R> __global__ void evaluate(R x, uni20::sincos_result<R>* out) { *out = uni20::math::sincospi(x); }

template <class R> class CudaSincospi : public testing::Test {};
using DeviceReals = testing::Types<float, double>;
TYPED_TEST_SUITE(CudaSincospi, DeviceReals);

TYPED_TEST(CudaSincospi, EvaluatesAxesAndNearbyPhases)
{
  using R = TypeParam;
  using Pair = uni20::sincos_result<R>;
  int count = 0;
  auto status = cudaGetDeviceCount(&count);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && count == 0))
    GTEST_SKIP() << "No CUDA device available";
  ASSERT_EQ(status, cudaSuccess);
  Pair* raw = nullptr;
  ASSERT_EQ(cudaMalloc(&raw, sizeof(Pair)), cudaSuccess);
  std::unique_ptr<Pair, decltype(&cudaFree)> device(raw, cudaFree);
  R const eps = uni20::numeric_limits<R>::epsilon();
  for (R x : {R(-1), R(0), R(0.5), R(1.5), R(1) - eps, R(1) + eps, R(-0.25)})
  {
    SCOPED_TRACE(x);
    evaluate<<<1, 1>>>(x, device.get());
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    Pair result;
    ASSERT_EQ(cudaMemcpy(&result, raw, sizeof(result), cudaMemcpyDeviceToHost), cudaSuccess);
    auto const s = uni20::math::sinpi(x), c = uni20::math::cospi(x);
    EXPECT_NEAR(result.sin, s, 4 * eps * std::abs(s));
    EXPECT_NEAR(result.cos, c, 4 * eps * std::abs(c));
  }
}
} // namespace
