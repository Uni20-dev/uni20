// Exceptions that escape a coroutine while an owning form holds its writer (#163, #169).
#include <gtest/gtest.h>
#include <uni20/async/async.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/debug_scheduler.hpp>

#include <stdexcept>

using namespace uni20::async;

namespace
{

struct ProducerFailure
{};

// Readers see whatever exception constructing a value raises.
struct Explodes
{
    Explodes(int) { throw ProducerFailure{}; } // NOLINT(google-explicit-constructor)
};

AsyncTask copy_value(ReadBuffer<int> in, WriteBuffer<int> out)
{
  int const value = co_await in;
  co_await out = value;
}

AsyncTask fail_holding_proxy(WriteBuffer<int> writer)
{
  auto proxy = co_await writer.transfer();
  static_cast<void>(proxy);
  throw ProducerFailure{};
}

AsyncTask fail_holding_storage(WriteBuffer<int> writer)
{
  auto storage = co_await writer.transfer().storage();
  static_cast<void>(storage);
  throw ProducerFailure{};
}

AsyncTask fail_with_storage(OwningStorageAccessProxy<int> storage)
{
  static_cast<void>(storage);
  throw ProducerFailure{};
  co_return;
}

AsyncTask fail_with_proxy(OwningWriteAccessProxy<int> proxy)
{
  proxy = 5;
  throw ProducerFailure{};
  co_return;
}

// Holds the writer until the frame is destroyed, so the failure is stored before readers resume.
AsyncTask fail_after_writing(WriteBuffer<int> writer)
{
  co_await writer = 1;
  throw ProducerFailure{};
}

// A holder that no coroutine-argument hook could know about.
struct HeldStorage
{
    OwningStorageAccessProxy<int> storage;
};

AsyncTask fail_with_held_storage(HeldStorage held)
{
  static_cast<void>(held);
  throw ProducerFailure{};
  co_return;
}

} // namespace

// The owning forms used to move the parameter's buffer, and its exception sink, into a temporary
// of the co_await expression. All four forms now deliver the constructor's exception (#169).
TEST(AsyncUnwindingTest, EveryWriteFormDeliversAConstructorFailure)
{
  DebugScheduler sched;
  auto check = [&](AsyncTask (*write)(WriteBuffer<Explodes>)) {
    Async<Explodes> value;
    sched.schedule(write(value.write()));
    sched.run_all();
    EXPECT_THROW((void)value.get_wait(sched), ProducerFailure);
  };
  check([](WriteBuffer<Explodes> b) static -> AsyncTask { co_await write_to(b.transfer(), 1); });
  check([](WriteBuffer<Explodes> b) static -> AsyncTask { (co_await b.transfer()).emplace(1); });
  check([](WriteBuffer<Explodes> b) static -> AsyncTask { co_await b = 1; });
  check([](WriteBuffer<Explodes> b) static -> AsyncTask { (co_await b).emplace(1); });
}

// A named owning proxy lost the failure on every schedule: readers saw the old value.
TEST(AsyncUnwindingTest, OwningProxyDeliversAnEscapingExceptionToReaders)
{
  {
    DebugScheduler sched;
    ScopedScheduler scoped(&sched);
    Async<int> value = 4;
    schedule(fail_holding_proxy(value.write()));
    EXPECT_THROW((void)value.get_wait(), ProducerFailure);
  }
  {
    DebugScheduler sched;
    ScopedScheduler scoped(&sched);
    Async<int> value = 4;
    Async<int> seen = 0;
    schedule(fail_holding_proxy(value.write()));
    schedule(copy_value(value.read(), seen.write()));
    EXPECT_THROW((void)seen.get_wait(), ProducerFailure);
  }
}

// A reader that still holds the epoch when the failure arrives receives it, and so does the
// successor that starts after that reader.
TEST(AsyncUnwindingTest, ReaderHoldingTheEpochSeesTheFailureBeforeASuccessorStarts)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);
  Async<int> value = 4;
  Async<int> seen = 0;
  schedule(fail_holding_storage(value.write()));
  schedule(copy_value(value.read(), seen.write()));
  async_assign(value, 9);
  EXPECT_THROW((void)seen.get_wait(), ProducerFailure);
  EXPECT_THROW((void)value.get_wait(), ProducerFailure);
}

// The owning holder publishes the epoch as the exception destroys it, before the promise reaches
// unhandled_exception(). If nothing holds the epoch, a queued successor starts first and keeps
// its own value. The late failure is then left to the task registry. It used to trip
// DEBUG_CHECK(phase_ <= Phase::Reading) in writer_set_exception() (#163).
TEST(AsyncUnwindingTest, FailureArrivingAfterTheEpochFinishedDoesNotAbort)
{
  for (auto* fail : {&fail_holding_storage, &fail_holding_proxy})
  {
    DebugScheduler sched;
    ScopedScheduler scoped(&sched);
    Async<int> value = 4;
    schedule(fail(value.write()));
    async_assign(value, 9);
    EXPECT_EQ(value.get_wait(), 9);
  }
}

// A caught exception does not fail the value, and the coroutine can go on to use the same queue.
TEST(AsyncUnwindingTest, CaughtExceptionPublishesTheOwningWriterUnchanged)
{
  {
    DebugScheduler sched;
    ScopedScheduler scoped(&sched);
    Async<int> value = 4;
    schedule([](WriteBuffer<int> writer) static -> AsyncTask {
      try
      {
        auto storage = co_await writer.transfer().storage();
        static_cast<void>(storage);
        throw ProducerFailure{};
      }
      catch (ProducerFailure const&)
      {}
    }(value.write()));
    async_assign(value, 9);
    EXPECT_EQ(value.get_wait(), 9);
  }
  {
    DebugScheduler sched;
    ScopedScheduler scoped(&sched);
    Async<int> value = 4;
    schedule([](WriteBuffer<int> writer, Async<int>* target) static -> AsyncTask {
      try
      {
        auto proxy = co_await writer.transfer();
        static_cast<void>(proxy);
        throw ProducerFailure{};
      }
      catch (ProducerFailure const&)
      {}
      co_await write_to(target->write(), 7);
    }(value.write(), &value));
    sched.run_all();
    EXPECT_EQ(value.get_wait(), 7);
  }
}

// After a caught exception, a later exception escaping the same coroutine reaches the parameter's
// sink after the epoch has finished. That used to abort in Debug.
TEST(AsyncUnwindingTest, LaterExceptionAfterACatchDoesNotAbort)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);
  Async<int> value = 4;
  Async<int> other = 0;
  schedule([](WriteBuffer<int> writer, Async<int>* other) static -> AsyncTask {
    try
    {
      auto storage = co_await writer.transfer().storage();
      static_cast<void>(storage);
      throw ProducerFailure{};
    }
    catch (ProducerFailure const&)
    {}
    co_await write_to(other->write(), 1);
    throw ProducerFailure{};
  }(value.write(), &other));
  async_assign(value, 9);
  sched.run_all();
  EXPECT_EQ(value.get_wait(), 9);
  EXPECT_EQ(other.get_wait(), 1);
}

// An owning holder that finished normally ends exception delivery, so a later exception escaping
// the coroutine does not fail the value it wrote. With the parameter's sink retained, it used to
// fail every later read of the epoch.
TEST(AsyncUnwindingTest, LaterExceptionLeavesANormallyPublishedOwningWriteAlone)
{
  {
    DebugScheduler sched;
    ScopedScheduler scoped(&sched);
    Async<int> value = 4;
    Async<int> seen = 0;
    schedule([](WriteBuffer<int> writer) static -> AsyncTask {
      {
        auto proxy = co_await writer.transfer();
        proxy = 5;
      }
      throw ProducerFailure{};
    }(value.write()));
    schedule(copy_value(value.read(), seen.write()));
    sched.run_all();
    EXPECT_EQ(seen.get_wait(), 5);
    EXPECT_EQ(value.get_wait(), 5);
  }
  {
    // write_to has finished writing before the coroutine reads a failed input.
    DebugScheduler sched;
    ScopedScheduler scoped(&sched);
    Async<int> input = 0;
    Async<int> output = 4;
    schedule(fail_holding_proxy(input.write()));
    schedule([](ReadBuffer<int> in, WriteBuffer<int> out) static -> AsyncTask {
      co_await write_to(out.transfer(), 5);
      int const x = co_await in;
      static_cast<void>(x);
    }(input.read(), output.write()));
    sched.run_all();
    EXPECT_EQ(output.get_wait(), 5);
    EXPECT_EQ(output.get_wait(), 5);
  }
}

// A coroutine parameter is destroyed after unhandled_exception() has returned. A failed coroutine's
// frame is now destroyed while its exception is rethrown, so an owning holder among the parameters
// keeps delivery open, as a local does, and the caller's sink delivers the failure. The holder used
// to count as a normal finish, and readers saw whatever the child left.
TEST(AsyncUnwindingTest, CoroutineHoldingAnOwningParameterDeliversItsFailure)
{
  auto check = [](auto parent) {
    DebugScheduler sched;
    ScopedScheduler scoped(&sched);
    Async<int> value = 4;
    Async<int> seen = 0;
    schedule(parent(value.write()));
    schedule(copy_value(value.read(), seen.write()));
    sched.run_all();
    EXPECT_THROW((void)seen.get_wait(), ProducerFailure);
    EXPECT_THROW((void)value.get_wait(), ProducerFailure);
  };
  check([](WriteBuffer<int> w) static -> AsyncTask { co_await fail_with_storage(co_await w.transfer().storage()); });
  check([](WriteBuffer<int> w) static -> AsyncTask { co_await fail_with_proxy(co_await w.transfer()); });
  check([](WriteBuffer<int> w) static -> AsyncTask {
    co_await fail_with_held_storage(HeldStorage{co_await w.transfer().storage()});
  });

  // A parameter that has already published keeps its value.
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);
  Async<int> value = 4;
  schedule([](WriteBuffer<int> w) static -> AsyncTask {
    co_await [](OwningWriteAccessProxy<int> proxy) static -> AsyncTask {
      proxy = 5;
      proxy.release();
      throw ProducerFailure{};
      co_return;
    }(co_await w.transfer());
  }(value.write()));
  sched.run_all();
  EXPECT_EQ(value.get_wait(), 5);
}

// An epoch that resumes a waiting coroutine with an upstream failure records it on the promise,
// and the parent still receives it after the child has handled it (#166). Only an exception that
// escapes the body destroys the frame as failed, so the value the child wrote stays.
TEST(AsyncUnwindingTest, ChildThatHandledAnUpstreamFailureKeepsItsOwningParameterValue)
{
  {
    DebugScheduler sched;
    ScopedScheduler scoped(&sched);
    Async<int> input = 0;
    Async<int> value = 4;
    auto producer = fail_after_writing(input.write());
    schedule([](ReadBuffer<int> in, WriteBuffer<int> w) static -> AsyncTask {
      co_await [](ReadBuffer<int> in, OwningWriteAccessProxy<int> out) static -> AsyncTask {
        try
        {
          out = co_await in;
        }
        catch (ProducerFailure const&)
        {
          out = -1;
        }
      }(std::move(in), co_await w.transfer());
    }(input.read(), value.write()));
    sched.run_all(); // the child waits for the input
    schedule(std::move(producer));
    sched.run_all();
    EXPECT_NO_THROW(EXPECT_EQ(value.get_wait(), -1));
  }
  {
    // A writer waiting for a failed predecessor is resumed with the failure, with nothing to catch.
    DebugScheduler sched;
    ScopedScheduler scoped(&sched);
    Async<int> other = 0;
    Async<int> value = 4;
    auto producer = fail_after_writing(other.write());
    schedule([](WriteBuffer<int> o, WriteBuffer<int> w) static -> AsyncTask {
      co_await [](WriteBuffer<int> o, OwningWriteAccessProxy<int> out) static -> AsyncTask {
        co_await o = 1;
        out = 5;
      }(std::move(o), co_await w.transfer());
    }(other.write(), value.write()));
    sched.run_all();
    schedule(std::move(producer));
    sched.run_all();
    EXPECT_NO_THROW(EXPECT_EQ(value.get_wait(), 5));
  }
}
