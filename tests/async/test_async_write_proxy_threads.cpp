// Several threads using one write proxy, as batch items in a coroutine do.
#include <gtest/gtest.h>
#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>

#include <thread>
#include <vector>

using namespace uni20::async;

// Batch items may update disjoint parts of one value through one write proxy. Their first in-place
// accesses then check for an inherited failure concurrently, which must not be a data race. Only a
// ThreadSanitizer build detects a regression here.
TEST(AsyncWriteProxyThreads, ConcurrentFirstInPlaceAccessesThroughOneProxy)
{
  DebugScheduler sched;
  Async<std::vector<int>> value = std::vector<int>(2, 0);
  sched.schedule([](WriteBuffer<std::vector<int>> w) static -> AsyncTask {
    auto proxy = co_await w;
    std::thread first([&] { proxy.get()[0] = 1; });
    std::thread second([&] { proxy.get()[1] = 2; });
    first.join();
    second.join();
  }(value.write()));
  sched.run_all();
  EXPECT_EQ(value.get_wait(sched), (std::vector<int>{1, 2}));
}
