#include <gtest/gtest.h>
#include <memory>
#include <uni20/async/async.hpp>
#include <uni20/async/async_task.hpp>
#include <uni20/async/debug_scheduler.hpp>

using namespace uni20;
using namespace uni20::async;

TEST(AsyncMoveTest, MoveConstructKeepsStorage)
{
  Async<int> original = 7;
  Async<int> moved(std::move(original));

  DebugScheduler sched;
  sched.schedule([](ReadBuffer<int> reader) static -> AsyncTask {
    auto& value = co_await reader;
    EXPECT_EQ(value, 7);
    co_return;
  }(moved.read()));

  sched.run_all();
}

TEST(AsyncMoveTest, MoveAssignPreservesQueue)
{
  Async<int> lhs = 1;
  Async<int> rhs = 2;
  lhs = std::move(rhs);

  DebugScheduler sched;
  sched.schedule([](WriteBuffer<int> writer) static -> AsyncTask {
    auto value = co_await writer;
    value = 9;
    co_return;
  }(lhs.write()));

  sched.schedule([](ReadBuffer<int> reader) static -> AsyncTask {
    auto& value = co_await reader;
    EXPECT_EQ(value, 9);
    co_return;
  }(lhs.read()));

  sched.run_all();
}

TEST(AsyncMoveTest, AsyncMoveTransfersValue)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  using Ptr = std::unique_ptr<int>;
  Async<Ptr> src = std::make_unique<int>(42);
  Async<Ptr> dst;

  async_move(dst, src);

  sched.run_all();

  Ptr received = dst.move_from_wait();
  ASSERT_TRUE(received);
  EXPECT_EQ(*received, 42);

  EXPECT_THROW((void)src.move_from_wait(), buffer_write_uninitialized);
}

TEST(AsyncMoveTest, AsyncMoveFromRvalueAsync)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  using Ptr = std::unique_ptr<int>;
  Async<Ptr> dst;

  async_move(dst, Async<Ptr>(std::make_unique<int>(11)));

  sched.run_all();

  Ptr received = dst.move_from_wait();
  ASSERT_TRUE(received);
  EXPECT_EQ(*received, 11);
}

TEST(AsyncMoveTest, AsyncMoveFromValueBuffersOnStack)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  using Ptr = std::unique_ptr<int>;
  Ptr payload = std::make_unique<int>(7);
  Async<Ptr> dst;

  async_move(dst, std::move(payload));

  sched.run_all();

  Ptr received = dst.move_from_wait();
  ASSERT_TRUE(received);
  EXPECT_EQ(*received, 7);
  EXPECT_FALSE(payload);
}

TEST(AsyncMoveTest, SourceWriterOnDestinationQueueReleasesBeforeDestinationWrite)
{
  using Ptr = std::unique_ptr<int>;
  struct Writer
  {
      Async<Ptr>& source;
      auto write() { return source.write(); }
  };

  DebugScheduler sched;
  ScopedScheduler scoped(&sched);
  Async<Ptr> value;
  auto expected = std::make_unique<int>(17);
  auto const* address = expected.get();
  async_move(value, std::move(expected));
  Writer source{value};
  async_move(value, source);
  EXPECT_EQ(value.get_wait().get(), address);
}

TEST(AsyncMoveTest, ImmediateConversionCanReadPreviousDestinationValue)
{
  struct NextValue
  {
      Async<int> const& source;
      operator int() const { return source.get_wait() + 1; }
  };
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);
  Async<int> value = 4;
  async_move(value, NextValue{value});
  EXPECT_EQ(value.get_wait(), 5);
}

TEST(AsyncMoveTest, ThrowingImmediateConversionLeavesDestinationEpochAlone)
{
  struct Failure
  {};
  struct Operand
  {
      explicit operator int() const { throw Failure{}; }
  };
  Async<int> value = 4;
  auto previous = value.queue().latest();
  EXPECT_THROW(async_move(value, Operand{}), Failure);
  EXPECT_EQ(value.queue().latest(), previous);
  EXPECT_EQ(value.get_wait(), 4);
}

TEST(AsyncMoveTest, MovePropagatesProducerFailureWithOrWithoutAnOldValue)
{
  struct ProducerFailure
  {};
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);
  for (bool initialized : {false, true})
  {
    for (bool producer_finished : {false, true})
    {
      SCOPED_TRACE(::testing::Message() << "initialized=" << initialized
                                        << ", producer_finished=" << producer_finished);
      Async<int> source;
      if (initialized) async_move(source, 4);
      schedule([](WriteBuffer<int> writer) static -> AsyncTask {
        (void)(co_await writer);
        throw ProducerFailure{};
      }(source.write()));
      if (producer_finished) sched.run_all();
      Async<int> destination;
      async_move(destination, source);
      EXPECT_THROW((void)destination.get_wait(), ProducerFailure);
      EXPECT_THROW((void)source.move_from_wait(), ProducerFailure);
    }
  }
}

TEST(AsyncMoveTest, TakingAnUninitializedValueReportsTheAccessError)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);
  Async<int> source;
  Async<int> destination;
  async_move(destination, source);
  EXPECT_THROW((void)destination.get_wait(), buffer_write_uninitialized);
}

TEST(AsyncMoveTest, DestinationFailureDoesNotReachReleasedSourceWriter)
{
  struct DestinationFailure
  {};
  struct Value
  {
      explicit Value(int) {}
      Value(Value&&) = default;
      Value& operator=(Value&&) { throw DestinationFailure{}; }
  };
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);
  Async<Value> source(1);
  Async<Value> destination(2);
  async_move(destination, source);
  EXPECT_THROW((void)destination.get_wait(), DestinationFailure);
  EXPECT_THROW((void)source.get_wait(), buffer_read_uninitialized);
}

TEST(AsyncMoveTest, FailureAfterTakeReleaseOnlyReachesThePendingWriter)
{
  struct DestinationFailure
  {};
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);
  Async<int> value = 4;
  auto source = value.write();
  auto destination = value.write();
  schedule([](WriteBuffer<int> source, WriteBuffer<int> destination) static -> AsyncTask {
    EXPECT_EQ(co_await source.take_release(), 4);
    (void)(co_await destination);
    throw DestinationFailure{};
  }(std::move(source), std::move(destination)));
  EXPECT_THROW((void)value.get_wait(), DestinationFailure);
}
