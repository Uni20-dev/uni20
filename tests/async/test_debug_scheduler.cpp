#include <uni20/async/debug_scheduler.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <chrono>
#include <numeric>
#include <semaphore>
#include <stdexcept>
#include <thread>
#include <utility>
#include <vector>

namespace
{

using uni20::async::AsyncTask;
using uni20::async::DebugScheduler;
using uni20::async::DebugSchedulerOptions;
using uni20::async::DebugSchedulerOrder;

std::vector<int> execution_order(DebugSchedulerOptions options, int task_count = 8)
{
  DebugScheduler scheduler(options);
  std::vector<int> result;
  result.reserve(static_cast<std::size_t>(task_count));

  auto record = [](std::vector<int>* output, int value) static -> AsyncTask {
    output->push_back(value);
    co_return;
  };

  for (int value = 0; value < task_count; ++value)
  {
    scheduler.schedule(record(&result, value));
  }
  scheduler.run_all();
  return result;
}

TEST(DebugSchedulerTest, DefaultUsesReverseSubmissionOrder)
{
  EXPECT_EQ(execution_order({}), (std::vector{7, 6, 5, 4, 3, 2, 1, 0}));
}

TEST(DebugSchedulerTest, FifoUsesSubmissionOrder)
{
  EXPECT_EQ(execution_order({.order = DebugSchedulerOrder::fifo}), (std::vector{0, 1, 2, 3, 4, 5, 6, 7}));
}

TEST(DebugSchedulerTest, ReverseUsesReverseSubmissionOrder)
{
  EXPECT_EQ(execution_order({.order = DebugSchedulerOrder::reverse}), (std::vector{7, 6, 5, 4, 3, 2, 1, 0}));
}

TEST(DebugSchedulerTest, RandomOrderIsAReproduciblePermutation)
{
  DebugSchedulerOptions const options{.order = DebugSchedulerOrder::random, .random_seed = 0x5eed};
  auto const first = execution_order(options);
  auto const second = execution_order(options);
  EXPECT_EQ(first, second);

  auto sorted = first;
  std::ranges::sort(sorted);
  std::vector<int> expected(sorted.size());
  std::iota(expected.begin(), expected.end(), 0);
  EXPECT_EQ(sorted, expected);
  EXPECT_NE(first, expected);
  std::ranges::reverse(expected);
  EXPECT_NE(first, expected);
}

TEST(DebugSchedulerTest, LightweightBatchUsesConfiguredOrderExactlyOnce)
{
  DebugScheduler scheduler({.order = DebugSchedulerOrder::reverse});
  std::vector<std::size_t> visited;

  scheduler.execute_batch(5, [&](std::size_t index) { visited.push_back(index); });

  EXPECT_EQ(visited, (std::vector<std::size_t>{4, 3, 2, 1, 0}));
}

TEST(DebugSchedulerTest, EmptyLightweightBatchDoesNotInvokeCallable)
{
  DebugScheduler scheduler;
  bool invoked = false;

  scheduler.execute_batch(0, [&](std::size_t) { invoked = true; });

  EXPECT_FALSE(invoked);
}

TEST(DebugSchedulerTest, LightweightBatchPropagatesException)
{
  DebugScheduler scheduler({.order = DebugSchedulerOrder::fifo});

  EXPECT_THROW(scheduler.execute_batch(4,
                                       [](std::size_t index) {
                                         if (index == 2) throw std::runtime_error("batch failure");
                                       }),
               std::runtime_error);
}

TEST(DebugSchedulerTest, LightweightBatchItemMayScheduleAndWait)
{
  DebugScheduler scheduler({.order = DebugSchedulerOrder::fifo});
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<int> result;

  scheduler.execute_batch(1, [&](std::size_t) {
    scheduler.schedule([](uni20::async::WriteBuffer<int> output) static -> AsyncTask {
      co_await output = 42;
      co_return;
    }(result.write()));
    EXPECT_EQ(result.get_wait(), 42);
  });
}

// Hands its task to another thread, as a CUDA host callback does. The waker
// waits for `gate`, then delays briefly so the scheduler is already waiting.
struct ExternalWakeup
{
    std::thread* waker = nullptr;
    std::binary_semaphore* gate = nullptr;

    [[nodiscard]] bool await_ready() const noexcept { return false; }

    void await_suspend(uni20::async::BasicTask task) const noexcept
    {
      uni20::async::BasicTask::begin_external_wait(task);
      *waker = std::thread(
          [](uni20::async::BasicTask owned, std::binary_semaphore* opened) {
            opened->acquire();
            std::this_thread::sleep_for(std::chrono::milliseconds(20));
            uni20::async::BasicTask::reschedule_after_external_wait(std::move(owned));
          },
          std::move(task), gate);
    }

    void await_resume() const noexcept {}
};

AsyncTask resume_after(ExternalWakeup wakeup, bool* resumed)
{
  co_await wakeup;
  *resumed = true;
  co_return;
}

// run_all() only runs what is runnable; it is not a completion wait.
TEST(DebugSchedulerTest, RunAllReturnsWhileExternalWakeupIsPending)
{
  DebugScheduler scheduler;
  std::thread waker;
  std::binary_semaphore gate{0};
  bool resumed = false;

  scheduler.schedule(resume_after(ExternalWakeup{&waker, &gate}, &resumed));
  scheduler.run_all();
  EXPECT_FALSE(resumed);
  EXPECT_TRUE(scheduler.done());
  EXPECT_EQ(scheduler.external_waits(), 1U);

  gate.release();
  scheduler.wait_for([&resumed] { return resumed; });
  waker.join();
  EXPECT_TRUE(resumed);
  EXPECT_EQ(scheduler.external_waits(), 0U);
}

TEST(DebugSchedulerTest, WaitDoesNotReportDeadlockWhileExternalWakeupIsPending)
{
  DebugScheduler scheduler;
  std::thread waker;
  std::binary_semaphore gate{0};
  bool resumed = false;

  scheduler.schedule(resume_after(ExternalWakeup{&waker, &gate}, &resumed));
  scheduler.run();
  ASSERT_EQ(scheduler.external_waits(), 1U);

  gate.release();
  scheduler.wait_for([&resumed] { return resumed; });
  waker.join();
  EXPECT_TRUE(resumed);
  EXPECT_EQ(scheduler.external_waits(), 0U);
}

// Hands its task to a thread that wakes it at once, with no gate or delay.
struct ImmediateExternalWakeup
{
    std::thread* waker = nullptr;

    [[nodiscard]] bool await_ready() const noexcept { return false; }

    void await_suspend(uni20::async::BasicTask task) const noexcept
    {
      uni20::async::BasicTask::begin_external_wait(task);
      *waker = std::thread(
          [](uni20::async::BasicTask owned) {
            uni20::async::BasicTask::reschedule_after_external_wait(std::move(owned));
          },
          std::move(task));
    }

    void await_resume() const noexcept {}
};

AsyncTask resume_immediately(ImmediateExternalWakeup wakeup, bool* resumed)
{
  co_await wakeup;
  *resumed = true;
  co_return;
}

// Once the woken task has run, the waking thread must no longer use the scheduler: the program
// may conclude that all its tasks are done and destroy it. Destroy before joining the waker.
TEST(DebugSchedulerTest, SchedulerMayBeDestroyedOnceTheExternallyWokenTaskHasRun)
{
  for (int iteration = 0; iteration < 20000; ++iteration)
  {
    std::thread waker;
    bool resumed = false;
    {
      DebugScheduler scheduler;
      scheduler.schedule(resume_immediately(ImmediateExternalWakeup{&waker}, &resumed));
      scheduler.run();
      scheduler.wait_for([&resumed] { return resumed; });
      ASSERT_EQ(scheduler.external_waits(), 0U) << "iteration " << iteration;
    }
    waker.join();
    ASSERT_TRUE(resumed);
  }
}

} // namespace
