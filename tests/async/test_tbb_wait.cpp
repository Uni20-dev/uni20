#include <gtest/gtest.h>
#include <oneapi/tbb/global_control.h>
#include <uni20/async/async.hpp>
#include <uni20/async/tbb_scheduler.hpp>

#include <array>
#include <atomic>
#include <memory>

using namespace uni20::async;

TEST(TbbWait, PendingPredicateRequiresNotification)
{
  TbbScheduler scheduler{1};
  EXPECT_NO_THROW(scheduler.wait_for([] { return true; }));
  EXPECT_THROW(scheduler.wait_for([] { return false; }), async_state_error);
}

TEST(TbbWait, InlineAndLateNotificationsAreSafe)
{
  TbbScheduler scheduler{1};
  bool ready = false;
  int registrations = 0;
  IAsyncScheduler::WaitWakeup retained;
  scheduler.wait_for(IAsyncScheduler::WaitRequest{
      .is_ready = [&] { return ready; },
      .notify_when_ready =
          [&](auto notify) {
            retained = notify;
            // The first inline notification precedes suspend-point installation
            // but does not satisfy the predicate, so the latch must retain it.
            ready = ++registrations == 2;
            notify();
            notify();
          },
  });
  ASSERT_TRUE(ready);
  EXPECT_EQ(registrations, 2);
  // Providers may retain callbacks after the requesting stack has returned.
  retained();
}

TEST(TbbWait, RechecksReadinessAndRegistersAgainAfterWakeup)
{
  oneapi::tbb::global_control limit(oneapi::tbb::global_control::max_allowed_parallelism, 1);
  TbbScheduler scheduler{1};
  std::atomic<bool> ready{false};
  int registrations = 0;
  scheduler.wait_for(IAsyncScheduler::WaitRequest{
      .is_ready = [&] { return ready.load(); },
      .notify_when_ready =
          [&](auto notify) {
            bool const publish = ++registrations == 2;
            scheduler.schedule(
                [](std::atomic<bool>* ready, bool publish, IAsyncScheduler::WaitWakeup notify) static -> AsyncTask {
                  if (publish) ready->store(true);
                  notify();
                  notify();
                  co_return;
                }(&ready, publish, std::move(notify)));
          },
  });
  EXPECT_TRUE(ready.load());
  EXPECT_EQ(registrations, 2);
  scheduler.run_all();
}

TEST(TbbWait, MoveWaitsForPrecedingReaderWithOneParticipant)
{
  oneapi::tbb::global_control limit(oneapi::tbb::global_control::max_allowed_parallelism, 1);
  TbbScheduler scheduler{1};
  ScopedScheduler scope(&scheduler);
  scheduler.execute_batch(1, [&](std::size_t) {
    Async<std::unique_ptr<int>> value = std::make_unique<int>(42);
    scheduler.schedule([](ReadBuffer<std::unique_ptr<int>> reader) static -> AsyncTask {
      EXPECT_EQ(*(co_await reader), 42);
    }(value.read()));
    // This batch occupies the only slot until move_from_wait() suspends it.
    auto writer = value.write();
    auto result = writer.move_from_wait();
    ASSERT_TRUE(result);
    EXPECT_EQ(*result, 42);
  });
}

TEST(TbbWait, MoveWaitsForActiveWriterWithOneParticipant)
{
  oneapi::tbb::global_control limit(oneapi::tbb::global_control::max_allowed_parallelism, 1);
  TbbScheduler scheduler{1};
  ScopedScheduler scope(&scheduler);
  scheduler.execute_batch(1, [&](std::size_t) {
    auto epoch = std::make_shared<EpochContext>();
    auto storage = make_shared_storage<int>(7);
    EpochContextWriter<int> first(storage, epoch);
    EpochContextWriter<int> second(storage, epoch);
    epoch->start();
    ASSERT_TRUE(first.ready());
    ASSERT_FALSE(second.ready());
    scheduler.schedule([](EpochContextWriter<int> first) static -> AsyncTask {
      first.data() = 23;
      first.release();
      co_return;
    }(std::move(first)));
    EXPECT_EQ(second.move_from_wait(), 23);
  });
}

TEST(TbbWait, ConcurrentNestedBatchesAndCoroutineWaits)
{
  // Batch bodies are ordinary TBB tasks and can run between resumable coroutine
  // stacks. Repeated nesting must not depend on a thread-local execution chain.
  oneapi::tbb::global_control limit(oneapi::tbb::global_control::max_allowed_parallelism, 2);
  TbbScheduler scheduler{4};
  std::array<Async<int>, 8> results;
  for (int job = 0; job < int(results.size()); ++job)
    scheduler.schedule([](TbbScheduler* scheduler, int job, WriteBuffer<int> output) static -> AsyncTask {
      std::array<int, 8> partials{};
      scheduler->execute_batch(partials.size(), [&](std::size_t index) {
        for (int iteration = 0; iteration < 8; ++iteration)
        {
          Async<int> result;
          scheduler->schedule([](TbbScheduler* scheduler, int value, WriteBuffer<int> output) static -> AsyncTask {
            Async<int> inner;
            scheduler->schedule([](int value, WriteBuffer<int> output) static -> AsyncTask {
              co_await output = value;
            }(value, inner.write()));
            co_await output = inner.get_wait(*scheduler);
          }(scheduler, job + int(index) + iteration, result.write()));
          partials[index] += result.get_wait(*scheduler);
        }
      });
      int sum = 0;
      for (int part : partials)
        sum += part;
      co_await output = sum;
    }(&scheduler, job, results[job].write()));
  for (int job = 0; job < int(results.size()); ++job)
    EXPECT_EQ(results[job].get_wait(scheduler), 64 * job + 448);
  scheduler.run_all();
}
