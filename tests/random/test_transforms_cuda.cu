#include <uni20/random/transforms.hpp>

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <bit>
#include <limits>
#include <memory>

namespace uni20::random::test
{
namespace
{
struct Results
{
    Philox4x32Block words;
    ValueBlock<float, 4> floats;
    ValueBlock<double, 2> doubles;
    ValueBlock<float, 4> scaled_float;
    ValueBlock<double, 2> scaled_double;
    ValueBlock<double, 2> normal;
    ValueBlock<complex<double>, 1> z;
};

__global__ void sample(Philox4x32Block counter, Results* out)
{
  auto const w = philox4x32_10(counter, {});
  *out = {w,
          Uniform<float>{}(w),
          Uniform<double>{}(w),
          Uniform{-1.7f, 4.3f}(w),
          Uniform{-1.7, 4.3}(w),
          Normal{-0.75, 1.25}(w),
          ComplexNormal{{1.25, -0.25}, 0.75}(w)};
}

TEST(RandomTransformCudaTest, MatchesHostWithinTransformBounds)
{
  int n = 0;
  auto const status = cudaGetDeviceCount(&n);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && n == 0))
    GTEST_SKIP() << "No CUDA device available";
  ASSERT_EQ(status, cudaSuccess);
  Results* raw = nullptr;
  ASSERT_EQ(cudaMalloc(&raw, sizeof(Results)), cudaSuccess);
  std::unique_ptr<Results, decltype(&cudaFree)> const device(raw, cudaFree);
  for (unsigned i = 0; i < 96; ++i)
  {
    Philox4x32Block const counter{{i, 0, 17, 31}};
    sample<<<1, 1>>>(counter, device.get());
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    Results gpu{};
    ASSERT_EQ(cudaMemcpy(&gpu, device.get(), sizeof(gpu), cudaMemcpyDeviceToHost), cudaSuccess);
    auto w = philox4x32_10(counter, {});
    auto unit_f = Uniform<float>{}(w);
    auto unit_d = Uniform<double>{}(w);
    auto pair = Normal{}(w);
    constexpr double eps = std::numeric_limits<double>::epsilon();
    for (unsigned j = 0; j < 4; ++j)
    {
      EXPECT_EQ(gpu.words.words[j], w.words[j]);
      EXPECT_EQ(std::bit_cast<std::uint32_t>(gpu.floats.values[j]), std::bit_cast<std::uint32_t>(unit_f.values[j]));
      EXPECT_NEAR(gpu.scaled_float.values[j], (Uniform{-1.7f, 4.3f}(w).values[j]),
                  4 * std::numeric_limits<float>::epsilon() * (1.7f + 6 * unit_f.values[j]));
    }
    for (unsigned j = 0; j < 2; ++j)
    {
      EXPECT_EQ(std::bit_cast<std::uint64_t>(gpu.doubles.values[j]), std::bit_cast<std::uint64_t>(unit_d.values[j]));
      EXPECT_NEAR(gpu.scaled_double.values[j], (Uniform{-1.7, 4.3}(w).values[j]),
                  4 * eps * (1.7 + 6 * unit_d.values[j]));
      EXPECT_NEAR(gpu.normal.values[j], (Normal{-0.75, 1.25}(w).values[j]),
                  8 * eps * (0.75 + 1.25 * std::abs(pair.values[j])));
    }
    auto const z = ComplexNormal{{1.25, -0.25}, 0.75}(w).values[0];
    EXPECT_NEAR(gpu.z.values[0].real(), z.real(), 8 * eps * (1.25 + 0.75 * std::abs(pair.values[0])));
    EXPECT_NEAR(gpu.z.values[0].imag(), z.imag(), 8 * eps * (0.25 + 0.75 * std::abs(pair.values[1])));
  }
}
} // namespace
} // namespace uni20::random::test
