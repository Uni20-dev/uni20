#include <gtest/gtest.h>
#include <uni20/async/async.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/debug_scheduler.hpp>

using namespace uni20::async;

TEST(AsyncWriteTest, WriteValueCorrectly)
{
  DebugScheduler sched;
  Async<int> x;

  auto task = [](WriteBuffer<int> buffer) static -> AsyncTask {
    Async<int> x;
    co_await write_to(x.write(), 42);
    co_await write_to(buffer.transfer(), co_await x.read());
    co_return;
  }(x.write());

  sched.schedule(std::move(task));
  sched.run_all();

  EXPECT_EQ(x.get_wait(sched), 42);
}

// The reader holds the epoch, so the writer behind it suspends on its gate. When the reader goes,
// the epoch makes that writer active before resuming it, and write_to must accept that rather than
// asking for the gate again, which used to fail its "immediately writable" check.
TEST(AsyncWriteTest, WriteToAfterWaitingForTheGate)
{
  DebugScheduler sched;
  Async<int> x(1);
  Async<int> y(1);
  auto x_reader = x.read();
  auto y_reader = y.read();

  sched.schedule([](WriteBuffer<int> buffer) static -> AsyncTask {
    co_await write_to(std::move(buffer), 2);
    co_return;
  }(x.write()));
  sched.schedule([](WriteBuffer<int> buffer) static -> AsyncTask {
    co_await write_to(buffer.transfer(), 3);
    co_return;
  }(y.write()));
  sched.run();
  EXPECT_TRUE(sched.done());

  x_reader.release();
  y_reader.release();
  sched.run_all();
  EXPECT_EQ(x.get_wait(sched), 2);
  EXPECT_EQ(y.get_wait(sched), 3);
}
