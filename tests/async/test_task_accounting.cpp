#include <deque>
#include <gtest/gtest.h>
#include <stdexcept>
#include <uni20/async/accounted_task.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/tbb_numa_scheduler.hpp>
#include <uni20/async/tbb_scheduler.hpp>
#include <vector>

using namespace uni20::async;

namespace
{
template <class T>
concept Admitted = requires(IAsyncScheduler& scheduler, T&& task) { scheduler.schedule(std::forward<T>(task)); };
static_assert(Admitted<AccountedTask>);
static_assert(Admitted<AsyncTask>);
static_assert(!Admitted<BasicTask>);
static_assert(!Admitted<AccountedTask&>);
static_assert(!Admitted<AccountedTask const>);
static_assert(!Admitted<CudaTask>);
static_assert(std::same_as<decltype(std::declval<AccountedTask&>().debug_name("stage")), AccountedTask&>);

struct FakeState
{
    long wall = 0;
    long ticks[2]{};
    int worker = 0;
    int wall_reads = 0;
    int cpu_reads = 0;
    bool wall_available = true;
    bool cpu_available = true;
    void work(long n)
    {
      wall += n;
      ticks[worker] += n;
    }
};
struct FakeClock
{
    static constexpr bool has_cpu = true;
    FakeState* state;
    std::optional<AccountingDuration> wall() noexcept
    {
      ++state->wall_reads;
      return state->wall_available ? std::optional(AccountingDuration(state->wall)) : std::nullopt;
    }
    std::optional<AccountingDuration> cpu() noexcept
    {
      ++state->cpu_reads;
      return state->cpu_available ? std::optional(AccountingDuration(state->ticks[state->worker])) : std::nullopt;
    }
};
BasicAccountingOptions<FakeClock> options(FakeState& state, AccountingMode mode = AccountingMode::cpu_and_wall)
{
  return {.mode = mode, .clock = {&state}};
}
AccountingReport inspect(AccountingReceipt const& receipt)
{
  auto reader = receipt.read();
  EXPECT_TRUE(reader.await_ready());
  return reader.await_resume();
}
void expect_cost(AccountingReceipt const& receipt, long cpu, long elapsed, unsigned intervals,
                 AccountingOutcome outcome = AccountingOutcome::returned)
{
  auto report = inspect(receipt);
  EXPECT_EQ(report.own_cpu, AccountingDuration(cpu));
  EXPECT_EQ(report.active_wall, AccountingDuration(cpu));
  EXPECT_EQ(report.elapsed, AccountingDuration(elapsed));
  EXPECT_EQ(report.intervals, intervals);
  EXPECT_EQ(report.outcome, outcome);
}
// Deterministic scheduler for driving suspension and direct-transfer boundaries.
class Driver final : public IAsyncScheduler {
  public:
    explicit Driver(bool direct = true) : direct_(direct) {}
    void schedule_host(BasicTask&& t) override { this->submit(std::move(t)); }
    void submit(BasicTask t)
    {
      if (t.set_scheduler(this)) queue_.push_back(std::move(t));
    }
    void pause() override {}
    void resume() override {}
    void run()
    {
      while (!queue_.empty())
      {
        auto t = std::move(queue_.front());
        queue_.pop_front();
        t.resume();
      }
    }

  private:
    bool accepts_route(TaskRoute r) const noexcept override { return r.domain == TaskDomain::host; }
    bool can_direct_transfer(TaskHandle, TaskHandle) const noexcept override { return direct_; }
    void reschedule(BasicTask&& t) override { queue_.push_back(std::move(t)); }
    std::deque<BasicTask> queue_;
    bool direct_;
};
struct Hold
{
    BasicTask* held;
    bool await_ready() const noexcept { return false; }
    void await_suspend(BasicTask t) noexcept { *held = std::move(t); }
    void await_resume() const noexcept {}
};
struct ReturnSelf
{
    bool await_ready() const noexcept { return false; }
    BasicTask await_suspend(BasicTask t) noexcept { return t; }
    void await_resume() const noexcept {}
};
struct ResumeImmediately
{
    bool await_ready() const noexcept { return false; }
    void await_suspend(BasicTask t) noexcept { t.resume(); }
    void await_resume() const noexcept {}
};
struct ThrowBeforePublication
{
    bool await_ready() const noexcept { return false; }
    int num_awaiters() { throw std::runtime_error("registration failed"); }
    void await_suspend(TaskFactory) noexcept {}
    void await_resume() noexcept {}
};
struct Transfer
{
    BasicTask child;
    bool await_ready() const noexcept { return false; }
    BasicTask await_suspend(BasicTask t) noexcept
    {
      (void)t.release_handle();
      return std::move(child);
    }
    void await_resume() const noexcept {}
};
AsyncTask plain(FakeState& c, long ticks)
{
  c.work(ticks);
  co_return;
}
BasicAccountedTask<FakeClock> leaf(BasicAccountingOptions<FakeClock> options, long ticks)
{
  options.clock.state->work(ticks);
  co_return;
}
BasicAccountedTask<FakeClock> parent(BasicAccountingOptions<FakeClock> options, BasicTask child)
{
  options.clock.state->work(2);
  co_await child;
  options.clock.state->work(3);
}
AsyncTask ordinary_parent(BasicAccountedTask<FakeClock> child) { co_await child; }
BasicAccountedTask<FakeClock> held(BasicAccountingOptions<FakeClock> options, BasicTask& save)
{
  options.clock.state->work(2);
  co_await Hold{&save};
  options.clock.state->work(3);
}
BasicAccountedTask<FakeClock> ready(BasicAccountingOptions<FakeClock> options)
{
  options.clock.state->work(2);
  ValueAwaiter<int> value{9};
  int const& ref = co_await value;
  EXPECT_EQ(&ref, &value.value);
  auto owned = co_await ValueAwaiter<std::unique_ptr<int>>{std::make_unique<int>(4)};
  EXPECT_EQ(*owned, 4);
  co_await AsyncTask{};
  options.clock.state->work(3);
}
BasicAccountedTask<FakeClock> self_transfer(BasicAccountingOptions<FakeClock> options)
{
  options.clock.state->work(2);
  co_await ReturnSelf{};
  options.clock.state->work(3);
}
BasicAccountedTask<FakeClock> immediate(BasicAccountingOptions<FakeClock> options)
{
  options.clock.state->work(2);
  co_await ResumeImmediately{};
  options.clock.state->work(3);
}
BasicAccountedTask<FakeClock> forwarding(BasicAccountingOptions<FakeClock> options, BasicTask child)
{
  options.clock.state->work(2);
  co_await Transfer{std::move(child)};
  options.clock.state->work(3);
}
BasicAccountedTask<FakeClock> registration_failure(BasicAccountingOptions<FakeClock> options)
{
  options.clock.state->work(2);
  try
  {
    co_await ThrowBeforePublication{};
  }
  catch (std::runtime_error const&)
  {
    options.clock.state->work(3);
  }
}
BasicAccountedTask<FakeClock> failing(BasicAccountingOptions<FakeClock> options)
{
  options.clock.state->work(7);
  throw std::runtime_error("body failed");
  co_return;
}
BasicAccountedTask<FakeClock> catches_child(BasicAccountingOptions<FakeClock> options, BasicTask child)
{
  options.clock.state->work(2);
  try
  {
    co_await child;
  }
  catch (std::runtime_error const&)
  {
    options.clock.state->work(3);
  }
}
BasicAccountedTask<FakeClock> joins(BasicAccountingOptions<FakeClock> options, ReadBuffer<int> ar, ReadBuffer<int> br)
{
  options.clock.state->work(2);
  auto [x, y] = co_await all(ar, br);
  EXPECT_EQ(x + y, 7);
  options.clock.state->work(3);
}

TEST(TaskAccounting, ReadyAwaitersPreserveReferencesAndMoveOnlyValues)
{
  FakeState c;
  Driver d;
  auto t = ready(options(c));
  auto receipt = t.receipt();
  d.schedule(std::move(t));
  d.run();
  expect_cost(receipt, 5, 5, 1);
  EXPECT_EQ(c.cpu_reads, 2);
  EXPECT_EQ(c.wall_reads, 4);
}
TEST(TaskAccounting, AllNestingCombinationsAndQueuedTransfers)
{
  for (bool direct : {false, true})
  {
    FakeState c;
    Driver d(direct);
    auto child = leaf(options(c), 100);
    auto cr = child.receipt();
    auto outer = parent(options(c), std::move(child));
    auto pr = outer.receipt();
    d.schedule(std::move(outer));
    d.run();
    expect_cost(pr, 5, 105, 2);
    expect_cost(cr, 100, 102, 1);
    auto a = parent(options(c), plain(c, 80));
    auto ar = a.receipt();
    d.schedule(std::move(a));
    d.run();
    expect_cost(ar, 5, 85, 2);
    auto b = leaf(options(c), 17);
    auto br = b.receipt();
    d.schedule(ordinary_parent(std::move(b)));
    d.run();
    expect_cost(br, 17, 17, 1);
    auto plain_parent = [](FakeState& state) static -> AsyncTask { co_await plain(state, 11); };
    auto previous = c.wall;
    d.schedule(plain_parent(c));
    d.run();
    EXPECT_EQ(c.wall - previous, 11);
  }
}
TEST(TaskAccounting, SuspendedWorkExcludedAndCpuEpochChangesAtResume)
{
  FakeState c;
  Driver d;
  BasicTask save;
  auto t = held(options(c), save);
  auto r = t.receipt();
  d.schedule(std::move(t));
  d.run();
  EXPECT_FALSE(r.read().await_ready());
  d.schedule(plain(c, 100));
  d.run();
  c.worker = 1;
  c.ticks[1] = 700;
  BasicTask::reschedule(std::move(save));
  d.run();
  expect_cost(r, 5, 105, 2);
}
TEST(TaskAccounting, SelfForwardingAndImmediateTransfers)
{
  FakeState c;
  Driver d;
  auto a = self_transfer(options(c));
  auto ar = a.receipt();
  d.schedule(std::move(a));
  d.run();
  expect_cost(ar, 5, 5, 2);
  auto b = forwarding(options(c), plain(c, 100));
  auto br = b.receipt();
  d.schedule(std::move(b));
  d.run();
  expect_cost(br, 5, 105, 2);
  auto e = immediate(options(c));
  auto er = e.receipt();
  d.schedule(std::move(e));
  d.run();
  expect_cost(er, 5, 5, 2);
}
TEST(TaskAccounting, PrePublicationThrowRestartsBeforeHandler)
{
  FakeState c;
  Driver d;
  auto t = registration_failure(options(c));
  auto r = t.receipt();
  d.schedule(std::move(t));
  d.run();
  expect_cost(r, 5, 5, 2);
}
TEST(TaskAccounting, ChildExceptionRetainsBothReports)
{
  FakeState c;
  Driver d;
  auto child = failing(options(c));
  auto cr = child.receipt();
  auto t = catches_child(options(c), std::move(child));
  auto r = t.receipt();
  d.schedule(std::move(t));
  d.run();
  expect_cost(r, 5, 12, 2);
  expect_cost(cr, 7, 9, 1, AccountingOutcome::failed);
}
TEST(TaskAccounting, ExplicitCancellationBeforeBodyEntry)
{
  FakeState c;
  auto t = leaf(options(c), 20);
  auto r = t.receipt();
  {
    auto cancelled = std::move(t);
    cancelled.set_cancel_on_resume();
  }
  expect_cost(r, 0, 0, 0, AccountingOutcome::cancelled);
  EXPECT_FALSE(inspect(r).body_entered);
  EXPECT_EQ(c.cpu_reads, 0);
}
TEST(TaskAccounting, DestroyedSuspendedParentDoesNotInventCancellationFlag)
{
  FakeState c;
  Driver d;
  BasicTask save;
  auto child = held(options(c), save);
  auto cr = child.receipt();
  auto t = parent(options(c), std::move(child));
  auto r = t.receipt();
  d.schedule(std::move(t));
  d.run();
  save.set_cancel_on_resume();
  save = BasicTask{};
  expect_cost(cr, 2, 4, 1, AccountingOutcome::cancelled);
  expect_cost(r, 2, 4, 1, AccountingOutcome::interrupted);
}
TEST(TaskAccounting, FactoryJoinExcludesOtherWork)
{
  FakeState c;
  Driver d;
  Async<int> a, b;
  auto wa = a.write();
  auto wb = b.write();
  auto t = joins(options(c), a.read(), b.read());
  auto r = t.receipt();
  d.schedule(std::move(t));
  d.run();
  c.work(80);
  wa.emplace_assert(3);
  wa.release();
  d.run();
  EXPECT_FALSE(r.read().await_ready());
  c.work(90);
  wb.emplace_assert(4);
  wb.release();
  d.run();
  expect_cost(r, 5, 175, 2);
}
TEST(TaskAccounting, InitialAdmissionHonoursDebugPause)
{
  FakeState c;
  DebugScheduler d;
  d.pause();
  auto t = leaf(options(c), 8);
  auto r = t.receipt();
  d.schedule(std::move(t));
  d.run_all();
  EXPECT_EQ(c.cpu_reads, 0);
  EXPECT_FALSE(r.read().await_ready());
  c.work(100);
  d.resume();
  d.run_all();
  expect_cost(r, 8, 108, 1);
}
TEST(TaskAccounting, GlobalHelperAdmitsAccountedHostTask)
{
  FakeState c;
  DebugScheduler scheduler;
  ScopedScheduler scoped(&scheduler);
  auto task = leaf(options(c), 9);
  auto receipt = task.receipt();
  schedule(std::move(task));
  scheduler.run_all();
  expect_cost(receipt, 9, 9, 1);
}

TEST(TaskAccounting, NumaAdmissionPreservesPreferenceAndExplicitNode)
{
  TbbNumaScheduler scheduler;
  scheduler.pause();
  int node = scheduler.numa_nodes().front();
  auto body = [](AccountingOptions) static -> AccountedTask { co_return; };
  auto preferred = body({});
  preferred.set_preferred_numa_node(node);
  auto preferred_report = preferred.receipt();
  IAsyncScheduler& interface = scheduler;
  interface.schedule(std::move(preferred));
  EXPECT_EQ(scheduler.scheduled_count_for(node), 1U);
  EXPECT_FALSE(preferred_report.read().await_ready());

  auto explicit_node = body({});
  auto explicit_report = explicit_node.receipt();
  scheduler.schedule(std::move(explicit_node), node);
  EXPECT_EQ(scheduler.scheduled_count_for(node), 2U);
  EXPECT_FALSE(explicit_report.read().await_ready());
  scheduler.run_all();
  EXPECT_EQ(inspect(preferred_report).outcome, AccountingOutcome::returned);
  EXPECT_EQ(inspect(explicit_report).outcome, AccountingOutcome::returned);
}

TEST(TaskAccounting, CrossSchedulerContinuationReturnsToParentRoute)
{
  FakeState c;
  Driver outer, inner;
  auto child = leaf(options(c), 100);
  auto cr = child.receipt();
  ASSERT_TRUE(child.set_scheduler(&inner));
  auto t = parent(options(c), std::move(child));
  auto r = t.receipt();
  outer.schedule(std::move(t));
  outer.run();
  EXPECT_FALSE(cr.read().await_ready());
  inner.run();
  EXPECT_FALSE(r.read().await_ready());
  outer.run();
  expect_cost(r, 5, 105, 2);
}
struct EmptyFactory
{
    bool await_ready() const noexcept { return false; }
    int num_awaiters() const noexcept { return 0; }
    void await_suspend(TaskFactory) const noexcept {}
    void await_resume() const noexcept {}
};
TEST(TaskAccounting, ZeroOwnerFactoryDestroysDuringDelegation)
{
  FakeState c;
  Driver d;
  auto body = [](BasicAccountingOptions<FakeClock> options) static -> BasicAccountedTask<FakeClock> {
    options.clock.state->work(2);
    co_await EmptyFactory{};
    ADD_FAILURE();
  };
  auto t = body(options(c));
  auto r = t.receipt();
  d.schedule(std::move(t));
  d.run();
  expect_cost(r, 2, 2, 1, AccountingOutcome::interrupted);
}
TEST(TaskAccounting, WallOnlyModesNeverReadCpuClock)
{
  for (auto mode : {AccountingMode::elapsed, AccountingMode::active_wall})
  {
    FakeState c;
    Driver d;
    BasicTask save;
    auto t = held(options(c, mode), save);
    auto r = t.receipt();
    d.schedule(std::move(t));
    d.run();
    c.work(100);
    BasicTask::reschedule(std::move(save));
    d.run();
    auto report = inspect(r);
    EXPECT_EQ(c.cpu_reads, 0);
    EXPECT_FALSE(report.own_cpu);
    EXPECT_EQ(report.elapsed, AccountingDuration(105));
    if (mode == AccountingMode::elapsed)
    {
      EXPECT_FALSE(report.active_wall);
      EXPECT_EQ(c.wall_reads, 2);
    }
    else
    {
      EXPECT_EQ(report.active_wall, AccountingDuration(5));
      EXPECT_EQ(c.wall_reads, 6);
    }
  }
}
TEST(TaskAccounting, ClockFailureInvalidatesMetricsWithoutFailingWork)
{
  FakeState c;
  Driver d;
  BasicTask save;
  auto t = held(options(c), save);
  auto r = t.receipt();
  d.schedule(std::move(t));
  d.run();
  c.cpu_available = false;
  BasicTask::reschedule(std::move(save));
  d.run();
  auto report = inspect(r);
  EXPECT_FALSE(report.own_cpu);
  EXPECT_EQ(report.active_wall, AccountingDuration(5));
  EXPECT_EQ(report.outcome, AccountingOutcome::returned);
  auto bad = [](BasicAccountingOptions<FakeClock> opts) static -> BasicAccountedTask<FakeClock> {
    opts.clock.state->wall_available = false;
    co_return;
  };
  auto b = bad(options(c));
  auto br = b.receipt();
  d.schedule(std::move(b));
  d.run();
  EXPECT_FALSE(inspect(br).elapsed);
  EXPECT_FALSE(inspect(br).active_wall);
}
TEST(TaskAccounting, NumericalExceptionDoesNotPoisonRetainedReport)
{
  FakeState c;
  DebugScheduler d;
  Async<int> result;
  auto body = [](BasicAccountingOptions<FakeClock> opts, WriteBuffer<int> out) static -> BasicAccountedTask<FakeClock> {
    co_await out;
    opts.clock.state->work(7);
    throw std::runtime_error("numerical failure");
  };
  auto t = body(options(c), result.write());
  auto r = t.receipt();
  d.schedule(std::move(t));
  d.run_all();
  EXPECT_THROW((void)result.get_wait(d), std::runtime_error);
  expect_cost(r, 7, 7, 1, AccountingOutcome::failed);
}

TEST(TaskAccounting, CopiedReceiptCanBeAwaitedBeforeFailedTaskRuns)
{
  FakeState clock;
  DebugScheduler scheduler;
  Async<AccountingReport> observed;
  auto task = failing(options(clock));
  auto receipt = task.receipt();
  {
    auto copy = receipt;
    auto observe = [](ReadBuffer<AccountingReport> input, WriteBuffer<AccountingReport> output) static -> AsyncTask {
      auto const& report = co_await input;
      co_await output = report;
    };
    scheduler.schedule(observe(copy.read(), observed.write()));
  }
  scheduler.run_all();
  EXPECT_FALSE(observed.read().await_ready());
  scheduler.schedule(std::move(task.debug_name("failed measured stage")));
  scheduler.run_all();
  auto const& report = observed.get_wait(scheduler);
  EXPECT_EQ(report.outcome, AccountingOutcome::failed);
  EXPECT_EQ(report.own_cpu, AccountingDuration(7));
  EXPECT_EQ(receipt.wait(scheduler).own_cpu, report.own_cpu);
}
// A frame parameter outlives promise finalization. Keep this distinction tested:
// receipt readiness is suitable for statistics, not for resource teardown.
struct DestructionProbe
{
    std::optional<AccountingReceipt>* receipt;
    bool* destroyed;
    bool* ready_before_destruction;
    ~DestructionProbe()
    {
      *ready_before_destruction = receipt->value().read().await_ready();
      *destroyed = true;
    }
};
TEST(TaskAccounting, ReportPrecedesParameterDestructionButChildJoinFollowsIt)
{
  FakeState c;
  DebugScheduler d;
  std::optional<AccountingReceipt> receipt;
  bool destroyed = false, ready_before_destruction = false;
  auto child = [](BasicAccountingOptions<FakeClock>,
                  std::unique_ptr<DestructionProbe>) static -> BasicAccountedTask<FakeClock> { co_return; };
  auto join = [](BasicAccountedTask<FakeClock> task, bool& destroyed) static -> AsyncTask {
    co_await task;
    EXPECT_TRUE(destroyed);
  };
  auto probe = std::unique_ptr<DestructionProbe>(new DestructionProbe{&receipt, &destroyed, &ready_before_destruction});
  auto t = child(options(c), std::move(probe));
  receipt = t.receipt();
  d.schedule(join(std::move(t), destroyed));
  d.run_all();
  EXPECT_TRUE(destroyed);
  EXPECT_TRUE(ready_before_destruction);
  EXPECT_EQ(inspect(*receipt).outcome, AccountingOutcome::returned);
}

std::uint64_t burn(std::uint64_t value, unsigned count)
{
  for (unsigned i = 0; i < count; ++i)
    value = value * 6364136223846793005ULL + 1442695040888963407ULL;
  return value;
}
AccountedTask real_leaf(AccountingOptions opts, WriteBuffer<std::uint64_t> output, unsigned n)
{
  (void)opts;
  auto value = burn(9, n);
  co_await output = value;
}
TEST(TaskAccounting, RealTbbAdmissionReportsAndPartialFailures)
{
  for (int threads : {1, 4})
  {
    TbbScheduler d(threads);
    d.pause();
    std::vector<AccountingReceipt> receipts;
    std::vector<Async<std::uint64_t>> outputs(32);
    for (auto& output : outputs)
    {
      auto t = real_leaf({.mode = AccountingMode::cpu_and_wall}, output.write(), 20000);
      receipts.push_back(t.receipt());
      d.schedule(std::move(t));
    }
    for (auto const& r : receipts)
      EXPECT_FALSE(r.read().await_ready());
    d.resume();
    for (std::size_t i = 0; i < outputs.size(); ++i)
    {
      EXPECT_EQ(outputs[i].get_wait(d), burn(9, 20000));
      auto r = receipts[i].wait(d);
      EXPECT_TRUE(r.body_entered);
      EXPECT_EQ(r.intervals, 1);
      EXPECT_EQ(r.outcome, AccountingOutcome::returned);
      ASSERT_TRUE(r.elapsed);
      ASSERT_TRUE(r.active_wall);
      EXPECT_GE(*r.elapsed, *r.active_wall);
      EXPECT_GT(r.active_wall->count(), 0);
      if (SystemAccountingClock::has_cpu)
      {
        ASSERT_TRUE(r.own_cpu);
        EXPECT_GT(r.own_cpu->count(), 0);
      }
    }
    auto failure = [](AccountingOptions) static -> AccountedTask {
      throw std::runtime_error("failed attempt");
      co_return;
    };
    auto t = failure({});
    auto r = t.receipt();
    d.schedule(std::move(t));
    EXPECT_EQ(r.wait(d).outcome, AccountingOutcome::failed);
    // Quiesce activations before destroying scheduler-owned context.
    d.run_all();
  }
}
TEST(TaskAccounting, RealTbbSharedPreparationIsCountedOnce)
{
  TbbScheduler d(4);
  Async<int> prepared;
  auto writer = prepared.write();
  auto prepare = [](AccountingOptions, WriteBuffer<int> out) static -> AccountedTask { co_await out = 7; };
  auto consume = [](AccountingOptions, ReadBuffer<int> in, WriteBuffer<int> out) static -> AccountedTask {
    auto const& value = co_await in;
    co_await out = value + 1;
  };
  std::vector<Async<int>> outputs(16);
  std::vector<AccountingReceipt> reports;
  for (auto& out : outputs)
  {
    auto t = consume({.mode = AccountingMode::cpu_and_wall}, prepared.read(), out.write());
    reports.push_back(t.receipt());
    d.schedule(std::move(t));
  }
  d.run_all(); // All consumers are suspended, not completed.
  for (auto const& r : reports)
    EXPECT_FALSE(r.read().await_ready());
  auto t = prepare({.mode = AccountingMode::cpu_and_wall}, std::move(writer));
  auto preparation = t.receipt();
  d.schedule(std::move(t));
  EXPECT_EQ(preparation.wait(d).outcome, AccountingOutcome::returned);
  for (std::size_t i = 0; i < outputs.size(); ++i)
  {
    EXPECT_EQ(outputs[i].get_wait(d), 8);
    EXPECT_EQ(reports[i].wait(d).intervals, 2);
  }
  d.run_all();
}

struct PublishHeldTask
{
    BasicTask* held;
    WriteBuffer<int> signal;
    bool await_ready() const noexcept { return false; }
    void await_suspend(BasicTask task) noexcept
    {
      *held = std::move(task);
      signal.emplace_assert(1);
      signal.release();
    }
    void await_resume() const noexcept {}
};
TEST(TaskAccounting, RealWorkerToApplicationMigrationRestartsCpuClock)
{
  TbbScheduler worker(2);
  DebugScheduler application;
  Async<int> signal;
  BasicTask held;
  std::thread::id first, second;
  auto app_thread = std::this_thread::get_id();
  auto body = [](AccountingOptions opts, BasicTask& held, WriteBuffer<int> signal, std::thread::id& first,
                 std::thread::id& second) static -> AccountedTask {
    (void)opts;
    first = std::this_thread::get_id();
    co_await PublishHeldTask{&held, std::move(signal)};
    second = std::this_thread::get_id();
  };
  auto t = body({.mode = AccountingMode::cpu_and_wall}, held, signal.write(), first, second);
  auto r = t.receipt();
  auto ready = signal.read();
  worker.schedule(std::move(t));
  // Do not enter this arena: initial execution must be on a real TBB worker.
  // The Async publication supplies synchronization; the deadline diagnoses a
  // missing worker instead of hanging the test indefinitely.
  auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (!ready.await_ready() && std::chrono::steady_clock::now() < deadline)
    std::this_thread::yield();
  if (!ready.await_ready())
  {
    worker.run_all();
    held.set_cancel_on_resume();
    FAIL() << "No TBB worker reached the suspension point";
  }
  worker.run_all(); // await_suspend has returned; held still owns the continuation.
  EXPECT_NE(first, app_thread);
  EXPECT_FALSE(r.read().await_ready());
  ASSERT_TRUE(held.set_scheduler(&application));
  BasicTask::reschedule(std::move(held));
  application.run_all();
  EXPECT_EQ(second, app_thread);
  auto report = inspect(r);
  EXPECT_EQ(report.intervals, 2);
  EXPECT_TRUE(report.active_wall);
  EXPECT_TRUE(report.elapsed);
  if (SystemAccountingClock::has_cpu)
  {
    EXPECT_TRUE(report.own_cpu);
  }
  EXPECT_EQ(report.outcome, AccountingOutcome::returned);
}
} // namespace
