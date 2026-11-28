#include <cuda_runtime.h>
#include <gtest/gtest.h>
#include <memory>
#include <uni20/random/stream.hpp>

namespace uni20::random::test
{
namespace
{
struct Walker
{
    PhiloxStream stream;
    std::uint32_t words[6];
};
__global__ void walk(Walker* walker)
{
  walker->stream.discard(2);
  for (auto& word : walker->words)
    word = walker->stream();
}
TEST(RandomStreamCudaTest, MigratesPartiallyConsumedStateBothWays)
{
  int n = 0;
  auto status = cudaGetDeviceCount(&n);
  if (status == cudaErrorNoDevice || status == cudaErrorInsufficientDriver || (status == cudaSuccess && n == 0))
    GTEST_SKIP() << "No CUDA device available";
  ASSERT_EQ(status, cudaSuccess);
  Walker host{PhiloxStream(child(root(27), 5)), {}};
  (void)host.stream();
  auto reference = host.stream;
  Walker* raw = nullptr;
  ASSERT_EQ(cudaMalloc(&raw, sizeof(host)), cudaSuccess);
  std::unique_ptr<Walker, decltype(&cudaFree)> device(raw, cudaFree);
  ASSERT_EQ(cudaMemcpy(raw, &host, sizeof(host), cudaMemcpyHostToDevice), cudaSuccess);
  walk<<<1, 1>>>(raw);
  ASSERT_EQ(cudaGetLastError(), cudaSuccess);
  ASSERT_EQ(cudaMemcpy(&host, raw, sizeof(host), cudaMemcpyDeviceToHost), cudaSuccess);
  reference.discard(2);
  for (auto word : host.words)
    EXPECT_EQ(word, reference());
  for (int i = 0; i < 7; ++i)
    EXPECT_EQ(host.stream(), reference());
}
} // namespace
} // namespace uni20::random::test
