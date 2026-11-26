#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <future>
#include <gtest/gtest.h>
#include <memory>
#include <optional>
#include <uni20/async/async.hpp>
#include <uni20/async/async_task.hpp>
#include <uni20/async/debug_scheduler.hpp>

using namespace uni20;
using namespace uni20::async;

// These tests check the coroutine cancel behavior, making sure that
// local variables of the coroutine are destroyed.

struct DestructionObserver
{
    bool* destroyed_;
    explicit DestructionObserver(bool* flag) : destroyed_(flag) {}
    ~DestructionObserver() { *destroyed_ = true; }
};

TEST(AsyncDestroyTest, DestroyWaitingReader)
{
  Async<int> a;
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> result;

  bool was_destroyed = false;

  auto Reader = [](ReadBuffer<int> in, bool* flag) static -> AsyncTask {
    DestructionObserver obs(flag);
    auto buf = co_await in.or_cancel();
    EXPECT_EQ(&buf, nullptr); // this should never be executed
    co_return;
  };

  {
    WriteBuffer<int> wb = result.write();
    // wb.writer_require();

    schedule(Reader(result.read(), &was_destroyed));
    sched.run(); // ensure that Reader blocks at co_await
  }
  // when wb goes out of scope, this should trigger the cancellation of Reader
  EXPECT_EQ(was_destroyed, false);

  sched.run(); // we need to run again since the .destroy() will happen on resume

  EXPECT_EQ(was_destroyed, true);
}

TEST(AsyncDestroyTest, DestroyNewReader)
{
  Async<int> a;
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> result;

  bool was_destroyed = false;

  auto Reader = [](ReadBuffer<int> in, bool* flag) static -> AsyncTask {
    DestructionObserver obs(flag);
    auto buf = co_await in.or_cancel();
    EXPECT_EQ(&buf, nullptr); // this should never be executed
    co_return;
  };

  {
    WriteBuffer<int> wb = result.write();
    // wb.writer_require();
    //  do not run the scheduler yet
  }
  // when wb goes out of scope, this marks the EpochContext as cancelled
  EXPECT_EQ(was_destroyed, false);

  // Now schedule a new reader; this should appear in the sme Epoch
  schedule(Reader(result.read(), &was_destroyed));
  // so when we finally run the scheduler, the coroutine will get destroyed when it suspends
  sched.run_all();
  EXPECT_EQ(was_destroyed, true);
}

// This test is currently disabled. Arguiably, writer_require() should be transitive: if a writer is required at epoch
// n, but there is no writer, and also no writer a a subsequent epoch n+1, then readers at epoch n+1 should also
// cancel.

TEST(AsyncDestroyTest, DestroySubsequentReader)
{
  Async<int> a;
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> result;

  bool was_destroyed1 = false;
  bool was_destroyed2 = false;

  auto Reader = [](ReadBuffer<int> in, bool* flag) static -> AsyncTask {
    DestructionObserver obs(flag);
    auto buf = co_await in.or_cancel();
    EXPECT_EQ(&buf, nullptr); // this should never be executed
    co_return;
  };

  // get a write buffer.  when wb goes out of scope, this marks the EpochContext as cancelled
  (void)result.write();

  EXPECT_EQ(was_destroyed1, false);

  // Now schedule a new reader; this should appear in the same Epoch
  schedule(Reader(result.read(), &was_destroyed1));

  // Now get another writer; this forces another epoch.
  (void)result.write();
  // Now schedule another reader; this should inherit the writer_require() flag from the previous epoch
  schedule(Reader(result.read(), &was_destroyed2));

  // so when we finally run the scheduler, both coroutines should get destroyed
  sched.run_all();
  EXPECT_EQ(was_destroyed1, true);
  EXPECT_EQ(was_destroyed2, true);
}

namespace
{
struct FirstFailure
{};

/// A task failure that owns a reader of the value its own task was writing.
struct FailureOwningAReader
{
    std::shared_ptr<std::optional<ReadBuffer<int>>> reader;
};

AsyncTask fail_first(WriteBuffer<int> writer)
{
  static_cast<void>(writer);
  throw FirstFailure{};
  co_return;
}

AsyncTask fail_owning_a_reader(WriteBuffer<int> writer, ReadBuffer<int> reader)
{
  static_cast<void>(writer);
  throw FailureOwningAReader{std::make_shared<std::optional<ReadBuffer<int>>>(std::move(reader))};
  co_return;
}
} // namespace

// An epoch that starts with an inherited failure replaces any failure stored while it was
// pending. Here the replaced failure owns a reader of that same epoch, so destroying it releases
// the reader, which locks the epoch. That used to happen while start() still held the epoch's
// mutex, so the releasing thread deadlocked on itself. A snapshot's diagnostic reader can keep
// the predecessor in Reading in the same way as `held` does here.
TEST(AsyncDestroyTest, ReplacedFailureIsDestroyedOutsideTheEpochLock)
{
  DebugScheduler sched;
  Async<int> value(1);
  sched.schedule(fail_first(value.write()));
  sched.run_all();

  // Keeps the failed epoch in Reading, so its successor is still pending when its own task fails.
  auto held = value.read();
  {
    // In this order, so that the reader belongs to the successor that the writer creates.
    auto writer = value.write();
    auto reader = value.read();
    sched.schedule(fail_owning_a_reader(std::move(writer), std::move(reader)));
  }
  sched.run_all();

  // Releasing `held` starts the successor with FirstFailure. Release on another thread, so a
  // deadlock stops the test loudly instead of hanging it.
  auto released = std::async(std::launch::async, [&held] { held.release(); });
  if (released.wait_for(std::chrono::seconds(10)) != std::future_status::ready)
  {
    std::fprintf(stderr, "Releasing the reader deadlocked while the next epoch replaced its failure\n");
    std::abort();
  }
  EXPECT_THROW(static_cast<void>(value.get_wait(sched)), FirstFailure);
}
