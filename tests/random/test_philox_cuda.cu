#include "philox_vectors.hpp"

#include <cuda_runtime.h>
#include <gtest/gtest.h>

#include <memory>

namespace uni20::random::test
{
namespace
{
__global__ void evaluate_philox(Philox4x32Block counter, Philox4x32Key key, Philox4x32Block* output)
{
  *output = philox4x32_10(counter, key);
}
} // namespace

TEST(Philox4x32CudaTest, MatchesRandom123KnownAnswers)
{
  int devices = 0;
  auto const status = cudaGetDeviceCount(&devices);
  if (status == cudaErrorInsufficientDriver || status == cudaErrorNoDevice || (status == cudaSuccess && devices == 0))
    GTEST_SKIP() << "No CUDA device available";
  ASSERT_EQ(status, cudaSuccess) << cudaGetErrorString(status);

  Philox4x32Block* device_block = nullptr;
  ASSERT_EQ(cudaMalloc(&device_block, sizeof(*device_block)), cudaSuccess);
  std::unique_ptr<Philox4x32Block, decltype(&cudaFree)> const output(device_block, cudaFree);
  for (auto const& v : known_answers)
  {
    evaluate_philox<<<1, 1>>>(v.counter, v.key, output.get());
    ASSERT_EQ(cudaGetLastError(), cudaSuccess);
    Philox4x32Block block;
    ASSERT_EQ(cudaMemcpy(&block, output.get(), sizeof(block), cudaMemcpyDeviceToHost), cudaSuccess);
    for (int lane = 0; lane < 4; ++lane)
      EXPECT_EQ(block.words[lane], v.expected.words[lane]) << "lane " << lane;
  }
}

} // namespace uni20::random::test
