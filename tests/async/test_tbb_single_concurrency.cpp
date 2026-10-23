#include <uni20/async/async.hpp>
#include <uni20/async/tbb_scheduler.hpp>

#include <array>
#include <chrono>
#include <gtest/gtest.h>
#include <memory>
#include <oneapi/tbb/global_control.h>
#include <oneapi/tbb/task_arena.h>
#include <thread>
#include <tuple>

using namespace uni20::async;

namespace
{
// Exercise both constructor overloads, with and without a process-wide limit.
// The arena limit must hold independently of the global worker limit.
class TbbSingleConcurrency : public testing::TestWithParam<std::tuple<bool, bool>> {
  protected:
    void SetUp() override
    {
      if (std::get<1>(this->GetParam()))
        global_limit_ =
            std::make_unique<oneapi::tbb::global_control>(oneapi::tbb::global_control::max_allowed_parallelism, 1);
      if (std::get<0>(this->GetParam()))
      {
        oneapi::tbb::task_arena::constraints constraints;
        constraints.set_max_concurrency(1);
        scheduler_ = std::make_unique<TbbScheduler>(constraints);
      }
      else
      {
        scheduler_ = std::make_unique<TbbScheduler>(1);
      }
    }

    std::unique_ptr<oneapi::tbb::global_control> global_limit_;
    std::unique_ptr<TbbScheduler> scheduler_;
};

TEST_P(TbbSingleConcurrency, QueuedTasksObserveOneArenaParticipant)
{
  std::array<Async<int>, 32> results;
  scheduler_->pause();
  for (auto& result : results)
    scheduler_->schedule([](WriteBuffer<int> output) static -> AsyncTask {
      co_await output = oneapi::tbb::this_task_arena::max_concurrency();
    }(result.write()));
  scheduler_->resume();

  // Query from dispatched work, not an uninitialized arena. With (1, 1),
  // oneTBB's mandatory enqueue worker raises the reported concurrency to two,
  // even under global_control(max_allowed_parallelism, 1).
  for (auto const& result : results)
    EXPECT_EQ(result.get_wait(*scheduler_), 1);
}

TEST_P(TbbSingleConcurrency, BatchRunsInsideArenaAndCanWaitForChildren)
{
  scheduler_->execute_batch(3, [&](std::size_t index) {
    EXPECT_EQ(oneapi::tbb::this_task_arena::max_concurrency(), 1);
    Async<std::size_t> child;
    scheduler_->schedule([](WriteBuffer<std::size_t> output, std::size_t value) static -> AsyncTask {
      co_await output = value;
    }(child.write(), index));
    EXPECT_EQ(child.get_wait(*scheduler_), index);
  });
}

TEST_P(TbbSingleConcurrency, NestedWaitsProgressWithoutApplicationParticipation)
{
  Async<int> result;
  scheduler_->schedule([](TbbScheduler* scheduler, WriteBuffer<int> output) static -> AsyncTask {
    Async<int> first;
    scheduler->schedule([](WriteBuffer<int> child) static -> AsyncTask { co_await child = 3; }(first.write()));
    int const a = first.get_wait(*scheduler);

    Async<int> second;
    scheduler->schedule(
        [](WriteBuffer<int> child, int value) static -> AsyncTask { co_await child = value * 4; }(second.write(), a));
    co_await output = second.get_wait(*scheduler);
  }(scheduler_.get(), result.write()));

  // Readiness polling never joins the arena. The worker must execute the
  // parent, suspend each synchronous wait and run its children unaided.
  auto reader = result.read();
  auto const deadline = std::chrono::steady_clock::now() + std::chrono::seconds(5);
  while (!reader.await_ready() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::sleep_for(std::chrono::milliseconds(1));
  EXPECT_TRUE(reader.await_ready()) << "Queued work required application-thread participation";
  EXPECT_EQ(reader.get_wait(*scheduler_), 12);
}

INSTANTIATE_TEST_SUITE_P(ConstructorsAndGlobalLimits, TbbSingleConcurrency,
                         testing::Combine(testing::Bool(), testing::Bool()));
} // namespace
