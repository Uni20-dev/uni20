// Batch items in a coroutine that update one value through one write proxy.
#include <gtest/gtest.h>
#include <uni20/async/async.hpp>
#include <uni20/async/tbb_scheduler.hpp>

#include <cstddef>
#include <vector>

using namespace uni20::async;

// Batch items may update disjoint parts of one value through one write proxy. Their first in-place
// accesses then check for an inherited failure concurrently, which must not be a data race. Only
// ThreadSanitizer detects a regression here. With oneTBB uninstrumented, it also reports the batch
// hand-off itself, so look for reports at the check in epoch_context.hpp.
TEST(AsyncWriteProxyThreads, BatchItemsUpdateDisjointElementsThroughOneProxy)
{
  TbbScheduler scheduler{4};
  for (int round = 0; round < 20; ++round)
  {
    Async<std::vector<int>> value = std::vector<int>(64, 1);
    scheduler.schedule([](TbbScheduler* scheduler, WriteBuffer<std::vector<int>> w) static -> AsyncTask {
      auto proxy = co_await w;
      scheduler->execute_batch(64, [&](std::size_t i) { proxy.get()[i] *= 2; });
    }(&scheduler, value.write()));
    EXPECT_EQ(value.get_wait(scheduler), std::vector<int>(64, 2));
  }
}
