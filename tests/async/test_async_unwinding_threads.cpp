// A failure delivered while a reader on another thread holds the epoch (#163).
#include <gtest/gtest.h>
#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>

#include <atomic>
#include <future>
#include <thread>

using namespace uni20::async;

namespace
{

struct ProducerFailure
{};

// Destroyed after the proxy during unwinding, so it runs after the epoch is published and before
// the promise delivers the failure. The handshake is relaxed on purpose: nothing but the epoch
// lock orders the reader's lookup against the late store, which is what TSan checks here.
struct ReadBeforeDelivery
{
    std::atomic<bool>* published;
    std::atomic<bool>* read;

    ~ReadBeforeDelivery()
    {
      published->store(true, std::memory_order_relaxed);
      while (!read->load(std::memory_order_relaxed))
        std::this_thread::yield();
    }
};

AsyncTask fail_holding_proxy(WriteBuffer<int> writer, std::atomic<bool>* published, std::atomic<bool>* read)
{
  ReadBeforeDelivery before_delivery{published, read};
  auto proxy = co_await writer.transfer();
  static_cast<void>(proxy);
  throw ProducerFailure{};
}

} // namespace

TEST(AsyncUnwindingThreads, ReaderOnAnotherThreadAndTheLateFailureShareTheEpochLock)
{
  DebugScheduler scheduler;
  Async<int> value = 4;
  auto writer = value.write();
  auto reader = value.read();
  std::atomic<bool> published{false};
  std::atomic<bool> read{false};

  // This thread holds the read lease across the delivery of the failure.
  auto first = std::async(std::launch::async, [&] {
    while (!published.load(std::memory_order_relaxed))
      std::this_thread::yield();
    int const seen = reader.get_wait(scheduler);
    read.store(true, std::memory_order_relaxed);
    return seen;
  });
  scheduler.schedule(fail_holding_proxy(std::move(writer), &published, &read));
  scheduler.run_all();

  EXPECT_EQ(first.get(), 4);
  EXPECT_THROW((void)reader.get_wait(scheduler), ProducerFailure);
}
