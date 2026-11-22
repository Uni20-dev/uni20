#include "gtest/gtest.h"
#include <functional>
#include <uni20/async/async.hpp>
#include <uni20/async/async_ops.hpp>
#include <uni20/async/async_task.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <utility>

using namespace uni20::async;

namespace
{
class ErrorModeGuard {
  public:
    ErrorModeGuard() : previous_(trace::get_formatting_options().errors_abort())
    {
      trace::get_formatting_options().set_errors_abort(false);
    }

    ~ErrorModeGuard() { trace::get_formatting_options().set_errors_abort(previous_); }

  private:
    bool previous_;
};
} // namespace

namespace async_test_types
{
struct AssignmentAware
{
    explicit AssignmentAware(int value_in) : value(value_in) { ++constructions; }
    AssignmentAware(AssignmentAware const& other) : value(other.value) { ++constructions; }
    AssignmentAware(AssignmentAware&& other) noexcept : value(other.value) { ++constructions; }
    ~AssignmentAware() { ++destructions; }

    AssignmentAware& operator=(int new_value)
    {
      value = new_value;
      ++assignments;
      return *this;
    }

    static void reset()
    {
      constructions = 0;
      assignments = 0;
      destructions = 0;
    }

    int value;
    inline static int constructions = 0;
    inline static int assignments = 0;
    inline static int destructions = 0;
};

/// An immediate payload whose `queue()` is not an epoch queue.
struct QueueMember
{
    int items = 0;
    int const& queue() const { return items; }
};

/// An immediate payload that records an epoch queue as metadata.
struct Stamp
{
    EpochQueue const* origin;
    int sequence;
    EpochQueue const& queue() const { return *origin; }
};
} // namespace async_test_types

TEST(AsyncOpsTest, WriteProxyAssignmentConstructsThenUsesUnderlyingAssignment)
{
  async_test_types::AssignmentAware::reset();

  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  {
    Async<async_test_types::AssignmentAware> value;

    schedule([](WriteBuffer<async_test_types::AssignmentAware> out) static -> AsyncTask {
      co_await out = 42;
      co_return;
    }(value.write()));
    schedule([](WriteBuffer<async_test_types::AssignmentAware> out) static -> AsyncTask {
      co_await out = 77;
      co_return;
    }(value.write()));

    sched.run_all();

    EXPECT_EQ(value.get_wait().value, 77);
    EXPECT_EQ(async_test_types::AssignmentAware::constructions, 1);
    EXPECT_EQ(async_test_types::AssignmentAware::assignments, 1);
    EXPECT_EQ(async_test_types::AssignmentAware::destructions, 0);
  }

  EXPECT_EQ(async_test_types::AssignmentAware::destructions, 1);
}

TEST(AsyncOpsTest, WriteProxyEmplaceExplicitlyReconstructs)
{
  async_test_types::AssignmentAware::reset();

  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  {
    Async<async_test_types::AssignmentAware> value;

    schedule([](WriteBuffer<async_test_types::AssignmentAware> out) static -> AsyncTask {
      auto writer = co_await out;
      writer.emplace(3);
      writer.emplace(9);
      co_return;
    }(value.write()));

    sched.run_all();

    EXPECT_EQ(value.get_wait().value, 9);
    EXPECT_EQ(async_test_types::AssignmentAware::constructions, 2);
    EXPECT_EQ(async_test_types::AssignmentAware::assignments, 0);
    EXPECT_EQ(async_test_types::AssignmentAware::destructions, 1);
  }

  EXPECT_EQ(async_test_types::AssignmentAware::destructions, 2);
}

TEST(AsyncOpsTest, AddTwoAsyncInts)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched); // installs into global `schedule()` dispatch

  Async<int> a = 5;
  Async<int> b = 7;
  Async<int> c = a + b; // launches a coroutine via operator+

  EXPECT_EQ(c.get_wait(), 12);
}

TEST(AsyncOpsTest, UnaryNegation)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> value = 21;
  auto negated_value = -value;
  EXPECT_EQ(negated_value.get_wait(), -21);

  Async<int> lhs = 4;
  Async<int> rhs = 6;
  auto summed_async = lhs + rhs; // produces result through coroutine path
  auto negated_sum = -summed_async;
  EXPECT_EQ(negated_sum.get_wait(), -10);
}

TEST(AsyncOpsTest, AddMixedTypesIntDouble)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> a = 4;
  Async<double> b = 1.5;
  auto c = a + b; // Should deduce Async<double>

  EXPECT_DOUBLE_EQ(c.get_wait(), 5.5);
}

TEST(AsyncOpsTest, AddAsyncAndScalar)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> a = 10;
  auto c = a + 2.5; // should be Async<double>

  EXPECT_DOUBLE_EQ(c.get_wait(), 12.5);
}

TEST(AsyncOpsTest, AddScalarAndAsync)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<float> b = 3.5f;
  auto c = 1 + b; // should be Async<float>

  EXPECT_FLOAT_EQ(c.get_wait(), 4.5f);
}

TEST(AsyncOpsTest, BasicArithmeticOps)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> a = 6;
  Async<double> b = 2.0;

  auto sum = a + b;  // 8.0
  auto diff = a - b; // 4.0
  auto prod = a * b; // 12.0
  auto quot = a / b; // 3.0

  Async<double> x = 1.0;
  x += sum;  // 9.0
  x -= diff; // 5.0
  x *= prod; // 60.0
  x /= quot; // 20.0

  EXPECT_DOUBLE_EQ(x.get_wait(), 20.0);
}

TEST(AsyncOpsTest, BinaryOperationReadsBothInputsBeforeWritingTheirQueue)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);
  Async<int> value = 3;
  Async<int> other = 4;
  async_binary_op(value, value, other, std::plus<>{});
  async_binary_op(value, other, value, std::minus<>{});
  async_binary_op(value, value, value, std::plus<>{});
  EXPECT_EQ(value.get_wait(), -6);
}

TEST(AsyncOpsTest, BinaryOperationPropagatesFailureWhenTheOutputIsAnInput)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);
  Async<int> value = 3;
  async_compound_op(value, 1, [](int&, int) { throw std::runtime_error("upstream"); });
  bool called = false;
  async_binary_op(value, value, 1, [&](int a, int b) {
    called = true;
    return a + b;
  });
  EXPECT_THROW((void)value.get_wait(), std::runtime_error);
  EXPECT_FALSE(called);
}

TEST(AsyncOpsTest, BinaryCallbackFailureSurvivesSuccessfulInputOnOutputQueue)
{
  struct CallbackFailure
  {};
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);
  Async<int> value = 3;
  async_binary_op(value, value, 1, [](int, int) -> int { throw CallbackFailure{}; });
  EXPECT_THROW((void)value.get_wait(), CallbackFailure);
}

TEST(AsyncOpsTest, BinaryResultIsConvertedBeforeReleasingInputs)
{
  using Pair = std::pair<int, int>;
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);
  Async<Pair> value(Pair{1, 2});
  async_binary_op(value, value, 0,
                  [](Pair const& pair, int) { return std::pair{std::cref(pair.second), std::cref(pair.first)}; });
  EXPECT_EQ(value.get_wait(), (Pair{2, 1}));
}

TEST(AsyncOpsTest, BinaryResultCanBeConstructedWhenItHasNoMoveAssignment)
{
  using Result = async_test_types::AssignmentAware;
  static_assert(!std::is_move_assignable_v<Result>);
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);
  Async<int> input = 2;
  Async<Result> output;
  async_binary_op(output, input, 3, std::plus<>{});
  EXPECT_EQ(output.get_wait().value, 5);
}

TEST(AsyncOpsTest, CompoundOperationRejectsRhsQueueAliasAtSubmission)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> value = 3;
  ErrorModeGuard const error_mode;
  EXPECT_THROW(value += value, std::runtime_error);
  EXPECT_EQ(value.get_wait(), 3);
}

TEST(AsyncOpsTest, ImmediatePayloadQueueMemberIsNotAnEpochQueue)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<async_test_types::QueueMember> value;
  async_assign(value, async_test_types::QueueMember{3});
  EXPECT_EQ(value.get_wait().items, 3);

  async_move(value, async_test_types::QueueMember{5});
  EXPECT_EQ(value.get_wait().items, 5);

  async_compound_op(
      value, async_test_types::QueueMember{2},
      [](async_test_types::QueueMember& lhs, async_test_types::QueueMember const& rhs) { lhs.items += rhs.items; });
  EXPECT_EQ(value.get_wait().items, 7);
}

TEST(AsyncOpsTest, MoveUsesImmediateConversionEvenWhenTheOperandCanBeRead)
{
  struct Operand
  {
      Async<int>& destination;
      int& reads;
      EpochQueue const& queue() const { return destination.queue(); }
      auto read() const
      {
        ++reads;
        return destination.read();
      }
      operator int() const { return 17; }
  };
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);
  ErrorModeGuard const error_mode;
  Async<int> value = 4;
  int reads = 0;
  async_move(value, Operand{value, reads});
  EXPECT_EQ(value.get_wait(), 17);
  EXPECT_EQ(reads, 0);
}

TEST(AsyncOpsTest, OwningAssignmentReadsTheSourceBeforeEnrollingTheDestination)
{
  struct Reader
  {
      Async<int>& value;
      auto read() & { return value.read(); }
      operator int() const { return -1; }
  };
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);
  Async<int> value = 4;
  async_assign(value, 11);
  Reader reader{value};
  async_assign(value, reader);
  EXPECT_EQ(value.get_wait(), 11);
}

TEST(AsyncOpsTest, ImmediateValueRecordingTheDestinationQueueIsAssigned)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<async_test_types::Stamp> destination;
  async_test_types::Stamp stamp{&destination.queue(), 9};
  async_assign(destination, stamp);
  EXPECT_EQ(destination.get_wait().sequence, 9);

  async_move(destination, async_test_types::Stamp{&destination.queue(), 10});
  EXPECT_EQ(destination.get_wait().sequence, 10);
}

TEST(AsyncOpsTest, CompoundOperationExecutionFailurePropagatesThroughOutput)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> value = 3;
  Async<int> rhs = 4;

  async_compound_op(value, rhs, [](int&, int) { throw std::runtime_error("compound failure"); });
  sched.run_all();

  EXPECT_THROW((void)value.get_wait(), std::runtime_error);
}

TEST(AsyncOpsTest, MoveOnlyType)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  using Ptr = std::unique_ptr<std::string>;
  Async<Ptr> dst;

  Ptr src = std::make_unique<std::string>("test-move");
  async_move(dst, std::move(src));

  Ptr result = dst.move_from_wait(); // Must return by value
  ASSERT_TRUE(result);
  EXPECT_EQ(*result, "test-move");
}

TEST(AsyncOpsTest, AsyncAssignReadWriteSameAsyncDoesNotDeadlock)
{
  DebugScheduler sched;
  ScopedScheduler scoped(&sched);

  Async<int> value = 9;
  auto src = value.read();
  auto dst = value.write();
  async_assign(std::move(dst), std::move(src));

  sched.run_all();
  EXPECT_EQ(value.get_wait(), 9);
}
