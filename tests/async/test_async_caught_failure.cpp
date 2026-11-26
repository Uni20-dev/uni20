// A coroutine that catches a producer's failure finishes normally, whichever task ran first (#166).
#include <gtest/gtest.h>
#include <uni20/async/async.hpp>
#include <uni20/async/async_ops.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/debug_scheduler.hpp>

#include <array>

using namespace uni20::async;

namespace
{

struct ProducerFailure
{};

struct ChildFailure
{};

// Holds the writer until the frame is destroyed, so the failure is stored before readers resume.
AsyncTask fail_after_writing(WriteBuffer<int> writer)
{
  co_await writer = 1;
  throw ProducerFailure{};
}

enum class Order
{
  ProducerFirst,      // the child finds the failed value ready
  ChildSuspendsFirst, // the epoch resumes the waiting child with the failure
};

constexpr std::array orders{Order::ProducerFirst, Order::ChildSuspendsFirst};

char const* name(Order order) { return order == Order::ProducerFirst ? "producer first" : "child suspends first"; }

// Fails `input` in a producer whose writer is taken before `make_parent()` takes its buffers, and
// runs the two tasks in `order`.
template <typename MakeParent>
void run_in_order(Order order, DebugScheduler& sched, Async<int>& input, MakeParent make_parent)
{
  auto producer = fail_after_writing(input.write());
  if (order == Order::ProducerFirst)
  {
    schedule(std::move(producer));
    sched.run_all();
    schedule(make_parent());
    sched.run_all();
  }
  else
  {
    schedule(make_parent());
    sched.run_all(); // the child waits for the producer
    schedule(std::move(producer));
    sched.run_all();
  }
}

AsyncTask read_and_catch(ReadBuffer<int> in, bool& caught)
{
  try
  {
    static_cast<void>(co_await in);
  }
  catch (ProducerFailure const&)
  {
    caught = true;
  }
}

AsyncTask take_and_catch(WriteBuffer<int> w, bool& caught)
{
  try
  {
    static_cast<void>(co_await w.take());
  }
  catch (ProducerFailure const&)
  {
    caught = true;
  }
}

AsyncTask read_and_rethrow_own(ReadBuffer<int> in)
{
  try
  {
    static_cast<void>(co_await in);
  }
  catch (ProducerFailure const&)
  {
    throw ChildFailure{};
  }
}

AsyncTask read_uncaught(ReadBuffer<int> in) { static_cast<void>(co_await in); }

AsyncTask take_uncaught(WriteBuffer<int> w) { static_cast<void>(co_await w.take()); }

AsyncTask overwrite(WriteBuffer<int> w) { co_await w = 7; }

// Each in-place form reads the value the failed producer left, so each must rethrow its failure.
AsyncTask get_through_borrowed_proxy(WriteBuffer<int> w, int& seen, bool& caught)
{
  try
  {
    auto proxy = co_await w;
    seen = proxy.get();
  }
  catch (ProducerFailure const&)
  {
    caught = true;
  }
}

AsyncTask add_through_borrowed_proxy(WriteBuffer<int> w, bool& caught)
{
  try
  {
    co_await w += 10;
  }
  catch (ProducerFailure const&)
  {
    caught = true;
  }
}

AsyncTask get_through_owning_proxy(WriteBuffer<int> w, int& seen, bool& caught)
{
  try
  {
    auto proxy = co_await w.transfer();
    seen = proxy.get();
  }
  catch (ProducerFailure const&)
  {
    caught = true;
  }
}

AsyncTask add_through_owning_proxy(WriteBuffer<int> w, bool& caught)
{
  try
  {
    auto proxy = co_await w.transfer();
    proxy += 10;
  }
  catch (ProducerFailure const&)
  {
    caught = true;
  }
}

AsyncTask add_uncaught(WriteBuffer<int> w) { co_await w += 10; }

// A descriptor of the parent's value, as a view is of a tensor.
struct IntView
{
    using async_alias_tag = void;

    explicit IntView(int* value_in) : value(value_in) {}

    int* value;
};

AsyncTask read_borrowed_descriptor(WriteBuffer<IntView> w, bool& reached)
{
  auto proxy = co_await w;
  static_cast<void>(proxy.get().value);
  reached = true;
}

AsyncTask read_owning_descriptor(WriteBuffer<IntView> w, bool& reached)
{
  auto proxy = co_await w.transfer();
  static_cast<void>(proxy.get().value);
  reached = true;
}

// Counts the updates applied to it.
struct CountedValue
{
    int value = 0;
    static inline int updates = 0;

    CountedValue& operator+=(int x)
    {
      ++updates;
      value += x;
      return *this;
    }
};

AsyncTask fail_after_writing_counted(WriteBuffer<CountedValue> writer)
{
  co_await writer = CountedValue{1};
  throw ProducerFailure{};
}

// Writes 42 only after the child has returned.
AsyncTask parent(AsyncTask child, WriteBuffer<int> out)
{
  co_await std::move(child);
  co_await out = 42;
}

} // namespace

// The parent continues after a child that caught the failure of a value it read or took. The
// source keeps the failure for its other consumers.
TEST(AsyncCaughtFailureTest, ChildThatCaughtAnUpstreamFailureLetsItsParentContinue)
{
  for (auto order : orders)
  {
    SCOPED_TRACE(name(order));
    {
      DebugScheduler sched;
      ScopedScheduler scoped(&sched);
      Async<int> input = 0;
      Async<int> out = 0;
      bool caught = false;
      run_in_order(order, sched, input, [&] { return parent(read_and_catch(input.read(), caught), out.write()); });
      EXPECT_TRUE(caught);
      EXPECT_NO_THROW(EXPECT_EQ(out.get_wait(), 42));
      EXPECT_THROW((void)input.get_wait(), ProducerFailure);
    }
    {
      DebugScheduler sched;
      ScopedScheduler scoped(&sched);
      Async<int> input = 0;
      Async<int> out = 0;
      bool caught = false;
      run_in_order(order, sched, input, [&] { return parent(take_and_catch(input.write(), caught), out.write()); });
      EXPECT_TRUE(caught);
      EXPECT_NO_THROW(EXPECT_EQ(out.get_wait(), 42));
      EXPECT_THROW((void)input.get_wait(), ProducerFailure);
    }
  }
}

// A writer that overwrites a value whose predecessor failed never sees the failure, so its parent
// continues. Readers of the value still receive the inherited failure.
TEST(AsyncCaughtFailureTest, WriterAfterAFailedPredecessorLetsItsParentContinue)
{
  for (auto order : orders)
  {
    SCOPED_TRACE(name(order));
    DebugScheduler sched;
    ScopedScheduler scoped(&sched);
    Async<int> input = 0;
    Async<int> out = 0;
    run_in_order(order, sched, input, [&] { return parent(overwrite(input.write()), out.write()); });
    EXPECT_NO_THROW(EXPECT_EQ(out.get_wait(), 42));
    EXPECT_THROW((void)input.get_wait(), ProducerFailure);
  }
}

// In-place access to a value whose predecessor failed rethrows the failure, as `take()` does, so
// the child never reads the value the producer left (1). Readers still receive the failure.
TEST(AsyncCaughtFailureTest, InPlaceAccessAfterAFailedPredecessorRethrowsTheFailure)
{
  using MakeChild = AsyncTask (*)(WriteBuffer<int>, int&, bool&);
  MakeChild const children[] = {
      get_through_borrowed_proxy,
      [](WriteBuffer<int> w, int&, bool& caught) { return add_through_borrowed_proxy(std::move(w), caught); },
      get_through_owning_proxy,
      [](WriteBuffer<int> w, int&, bool& caught) { return add_through_owning_proxy(std::move(w), caught); },
  };
  for (auto order : orders)
  {
    SCOPED_TRACE(name(order));
    for (auto make_child : children)
    {
      DebugScheduler sched;
      ScopedScheduler scoped(&sched);
      Async<int> input = 0;
      Async<int> out = 0;
      int seen = -1;
      bool caught = false;
      run_in_order(order, sched, input, [&] { return parent(make_child(input.write(), seen, caught), out.write()); });
      EXPECT_TRUE(caught);
      EXPECT_EQ(seen, -1);
      EXPECT_NO_THROW(EXPECT_EQ(out.get_wait(), 42));
      EXPECT_THROW((void)input.get_wait(), ProducerFailure);
    }
  }
}

// Kernels overwrite as well as update through an async alias's descriptor, so reading the
// descriptor does not rethrow an inherited failure.
TEST(AsyncCaughtFailureTest, AliasDescriptorAccessAfterAFailedPredecessorDoesNotThrow)
{
  using MakeChild = AsyncTask (*)(WriteBuffer<IntView>, bool&);
  for (auto order : orders)
  {
    SCOPED_TRACE(name(order));
    for (MakeChild make_child : {read_borrowed_descriptor, read_owning_descriptor})
    {
      DebugScheduler sched;
      ScopedScheduler scoped(&sched);
      Async<int> input = 0;
      auto view = make_async_alias<IntView>(input, input.storage_address());
      Async<int> out = 0;
      bool reached = false;
      run_in_order(order, sched, input, [&] { return parent(make_child(view.write(), reached), out.write()); });
      EXPECT_TRUE(reached);
      EXPECT_NO_THROW(EXPECT_EQ(out.get_wait(), 42));
      EXPECT_THROW((void)input.get_wait(), ProducerFailure);
    }
  }
}

// `x += y` updates the old value, like `co_await w += y`, so it never applies the update to the
// value a failed producer left.
TEST(AsyncCaughtFailureTest, AsyncCompoundAssignmentAfterAFailedPredecessorDoesNotUpdate)
{
  for (auto order : orders)
  {
    SCOPED_TRACE(name(order));
    DebugScheduler sched;
    ScopedScheduler scoped(&sched);
    Async<CountedValue> x;
    CountedValue::updates = 0;
    auto producer = fail_after_writing_counted(x.write());
    if (order == Order::ProducerFirst)
    {
      schedule(std::move(producer));
      sched.run_all();
      x += 5;
      sched.run_all();
    }
    else
    {
      x += 5;
      sched.run_all(); // the update waits for the producer
      schedule(std::move(producer));
      sched.run_all();
    }
    EXPECT_EQ(CountedValue::updates, 0);
    EXPECT_THROW((void)x.get_wait(), ProducerFailure);
  }
}

// A failure that escapes the child reaches the parent at `co_await child`.
TEST(AsyncCaughtFailureTest, UncaughtUpstreamFailureReachesTheParent)
{
  for (auto order : orders)
  {
    SCOPED_TRACE(name(order));
    {
      DebugScheduler sched;
      ScopedScheduler scoped(&sched);
      Async<int> input = 0;
      Async<int> out = 0;
      run_in_order(order, sched, input, [&] { return parent(read_uncaught(input.read()), out.write()); });
      EXPECT_THROW((void)out.get_wait(), ProducerFailure);
    }
    {
      DebugScheduler sched;
      ScopedScheduler scoped(&sched);
      Async<int> input = 0;
      Async<int> out = 0;
      run_in_order(order, sched, input, [&] { return parent(take_uncaught(input.write()), out.write()); });
      EXPECT_THROW((void)out.get_wait(), ProducerFailure);
    }
    {
      DebugScheduler sched;
      ScopedScheduler scoped(&sched);
      Async<int> input = 0;
      Async<int> out = 0;
      run_in_order(order, sched, input, [&] { return parent(add_uncaught(input.write()), out.write()); });
      EXPECT_THROW((void)out.get_wait(), ProducerFailure);
    }
  }
}

// The parent receives the exception that escaped the child, not the upstream failure it caught.
TEST(AsyncCaughtFailureTest, ParentReceivesTheChildsOwnFailureAfterACaughtOne)
{
  for (auto order : orders)
  {
    SCOPED_TRACE(name(order));
    DebugScheduler sched;
    ScopedScheduler scoped(&sched);
    Async<int> input = 0;
    Async<int> out = 0;
    run_in_order(order, sched, input, [&] { return parent(read_and_rethrow_own(input.read()), out.write()); });
    EXPECT_THROW((void)out.get_wait(), ChildFailure);
  }
}
