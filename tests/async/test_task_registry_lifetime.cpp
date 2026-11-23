// Snapshots of the task registry while the program releases what they describe (issue #164).

#include <algorithm>
#include <condition_variable>
#include <cstdint>
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <memory>
#include <mutex>
#include <optional>
#include <semaphore>
#include <string>
#include <thread>
#include <uni20/async/async.hpp>
#include <uni20/async/async_task.hpp>
#include <uni20/async/buffers.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/async/task_registry.hpp>
#include <uni20/config.hpp>
#include <vector>

using namespace uni20;
using namespace uni20::async;

#if UNI20_DEBUG_ASYNC_TASKS
namespace
{

/// Installs a snapshot unlock hook for one scope.
class SnapshotUnlockHookGuard {
  public:
    explicit SnapshotUnlockHookGuard(void (*hook)()) { TaskRegistry::set_snapshot_unlock_hook_for_testing(hook); }
    ~SnapshotUnlockHookGuard() { TaskRegistry::set_snapshot_unlock_hook_for_testing(nullptr); }

    SnapshotUnlockHookGuard(SnapshotUnlockHookGuard const&) = delete;
    SnapshotUnlockHookGuard& operator=(SnapshotUnlockHookGuard const&) = delete;
};

std::optional<Async<int>>* value_to_release = nullptr;

void release_value() { value_to_release->reset(); }

std::string address_text(void const* address) { return fmt::format("{}", address); }

TaskRegistry::GraphSnapshot take_snapshot(bool best_effort)
{
  return best_effort ? TaskRegistry::snapshot_best_effort() : TaskRegistry::snapshot();
}

#if UNI20_DEBUG_DAG
uni20::TaskRegistryGraphDataNode const* find_data_node(TaskRegistry::GraphSnapshot const& snapshot, std::uint64_t id)
{
  auto const node = std::ranges::find(snapshot.data_nodes, id, &uni20::TaskRegistryGraphDataNode::id);
  return node == snapshot.data_nodes.end() ? nullptr : &*node;
}

AsyncTask write_value(WriteBuffer<int> writer, int value)
{
  co_await writer = value;
  co_return;
}

// Holds the writer of `writer` active while it waits for `gate`.
AsyncTask write_then_wait(WriteBuffer<int> writer, ReadBuffer<int> gate)
{
  co_await writer = 2;
  static_cast<void>(co_await gate);
}

// Leaves an await record for `local` in the task's history, then suspends on `blocker`.
AsyncTask await_a_value_that_is_then_released(ReadBuffer<int> blocker)
{
  {
    Async<int> local(5);
    auto reader = local.read();
    static_cast<void>(co_await reader);
  }
  static_cast<void>(co_await blocker);
}
#endif

} // namespace

#if UNI20_DEBUG_DAG
/// A value whose debug hook releases the last owner of the very value it is formatting.
struct ReleasedByItsOwnHook
{
    inline static int destroyed = 0;
    inline static std::optional<Async<ReleasedByItsOwnHook>>* owner = nullptr;

    ~ReleasedByItsOwnHook() { ++destroyed; }
};

std::string uni20_async_debug_value(ReleasedByItsOwnHook const&)
{
  if (ReleasedByItsOwnHook::owner) ReleasedByItsOwnHook::owner->reset();
  // Still alive: the snapshot's observation holds the storage.
  return fmt::format("destroyed={}", ReleasedByItsOwnHook::destroyed);
}

/// A value whose debug hook asks an epoch it was given for that epoch's node.
struct QueriesItsEpoch
{
    // Captured by the test before the snapshot. A hook must not read the queue head itself: the
    // program serializes queue-head operations on its own threads, not with the snapshot.
    inline static std::shared_ptr<EpochContext> epoch;
};

std::string uni20_async_debug_value(QueriesItsEpoch const&)
{
  // Locks the epoch the snapshot's reader was admitted to, so formatting must not hold its mutex.
  if (!QueriesItsEpoch::epoch) return "no epoch";
  return fmt::format("node={}", QueriesItsEpoch::epoch->node()->global_index());
}

/// A value whose debug hook stops until the test lets it go, holding the snapshot's reader.
struct HeldByItsHook
{
    int value{0};

    struct Gate
    {
        std::mutex mutex;
        std::condition_variable changed;
        bool entered{false};
        bool released{false};
        // Only this thread's snapshot is held. Any other snapshot, such as the dump a scheduler
        // writes before reporting a deadlock, formats the value without stopping.
        std::thread::id holder{};
    };
    inline static Gate* gate = nullptr;
};

std::string uni20_async_debug_value(HeldByItsHook const& held)
{
  if (auto* gate = HeldByItsHook::gate; gate && gate->holder == std::this_thread::get_id())
  {
    std::unique_lock lock(gate->mutex);
    gate->entered = true;
    gate->changed.notify_all();
    gate->changed.wait(lock, [gate] { return gate->released; });
  }
  return std::to_string(held.value);
}

/// A value that owns a reader of another value. Its debug hook and its destructor both stop on the
/// snapshotting thread until the test lets them go.
struct OwnsAReader
{
    ReadBuffer<int> reader;

    struct Gates
    {
        std::mutex mutex;
        std::condition_variable changed;
        std::thread::id holder{};
        bool in_hook{false};
        bool hook_released{false};
        bool in_destructor{false};
        bool destructor_released{false};
    };
    inline static Gates* gates = nullptr;

    explicit OwnsAReader(ReadBuffer<int> owned) : reader(std::move(owned)) {}
    OwnsAReader(OwnsAReader&&) = default;

    ~OwnsAReader()
    {
      if (auto* g = gates; g && g->holder == std::this_thread::get_id())
      {
        std::unique_lock lock(g->mutex);
        g->in_destructor = true;
        g->changed.notify_all();
        g->changed.wait(lock, [g] { return g->destructor_released; });
      }
    }
};

std::string uni20_async_debug_value(OwnsAReader const&)
{
  if (auto* g = OwnsAReader::gates; g && g->holder == std::this_thread::get_id())
  {
    std::unique_lock lock(g->mutex);
    g->in_hook = true;
    g->changed.notify_all();
    g->changed.wait(lock, [g] { return g->hook_released; });
  }
  return "owns a reader";
}

namespace
{
AsyncTask write_held(WriteBuffer<HeldByItsHook> writer, int value)
{
  co_await writer = HeldByItsHook{value};
  co_return;
}

// Hands its task to another thread, which reschedules it once `gate` opens, as a CUDA host
// callback does.
struct ExternalWakeup
{
    std::thread* waker = nullptr;
    std::binary_semaphore* gate = nullptr;

    [[nodiscard]] bool await_ready() const noexcept { return false; }

    void await_suspend(BasicTask task) const noexcept
    {
      BasicTask::begin_external_wait(task);
      *waker = std::thread(
          [](BasicTask owned, std::binary_semaphore* opened) {
            opened->acquire();
            BasicTask::reschedule_after_external_wait(std::move(owned));
          },
          std::move(task), gate);
    }

    void await_resume() const noexcept {}
};

AsyncTask resume_after(ExternalWakeup wakeup, bool* resumed)
{
  co_await wakeup;
  *resumed = true;
  co_return;
}

void wait_until(std::mutex& mutex, std::condition_variable& changed, bool const& flag)
{
  std::unique_lock lock(mutex);
  changed.wait(lock, [&] { return flag; });
}

void set_and_notify(std::mutex& mutex, std::condition_variable& changed, bool& flag)
{
  std::lock_guard lock(mutex);
  flag = true;
  changed.notify_all();
}
} // namespace
#endif

// The snapshot lists the epoch under the registry lock, and the hook then drops the last owner of
// that epoch before the snapshot reads it. Without a pin taken under the lock, the read is a
// use-after-free (AddressSanitizer reports it in EpochContext::try_debug_snapshot or debug_snapshot).
TEST(TaskRegistryLifetimeTest, SnapshotReadsAnEpochReleasedAfterTheRegistryUnlocks)
{
  for (bool const best_effort : {false, true})
  {
    SCOPED_TRACE(best_effort ? "best effort" : "explicit");
    std::optional<Async<int>> value(std::in_place, 5);
    // Only the epoch is released here; keeping the storage isolates the epoch lifetime.
    auto const storage = value->storage();
    auto const epoch_address = address_text(value->queue().latest().get());

    value_to_release = &value;
    TaskRegistry::GraphSnapshot snapshot;
    {
      SnapshotUnlockHookGuard guard(&release_value);
      snapshot = take_snapshot(best_effort);
    }
    value_to_release = nullptr;

    EXPECT_FALSE(value.has_value());
    ASSERT_TRUE(snapshot.snapshot_available);
    auto const epoch = std::ranges::find(snapshot.epochs, epoch_address, &uni20::TaskRegistryGraphEpoch::address);
    ASSERT_NE(epoch, snapshot.epochs.end());
    EXPECT_TRUE(epoch->snapshot_available);
    EXPECT_EQ(epoch->generation, 0);
    EXPECT_EQ(epoch->phase, "Reading");
  }
}

#if UNI20_DEBUG_DAG
// A task's await history keeps the node of a value it awaited after that value is released. The
// snapshot keeps the dependency and reports the value as expired, instead of reading the freed
// storage (AddressSanitizer: heap-use-after-free in the construction-state query).
TEST(TaskRegistryLifetimeTest, SnapshotReportsAnAwaitedValueReleasedSinceAsExpired)
{
  DebugScheduler sched;
  Async<int> blocker;
  auto task = await_a_value_that_is_then_released(blocker.read());
  task.debug_name("awaited a released value");
  sched.schedule(std::move(task));
  sched.run();

  for (bool const best_effort : {false, true})
  {
    SCOPED_TRACE(best_effort ? "best effort" : "explicit");
    auto const snapshot = take_snapshot(best_effort);
    ASSERT_TRUE(snapshot.snapshot_available);
    auto const task_record =
        std::ranges::find(snapshot.tasks, "awaited a released value", &uni20::TaskRegistryGraphTask::label);
    ASSERT_NE(task_record, snapshot.tasks.end());

    std::vector<std::string> states;
    for (auto const& dependency : task_record->await_dependencies)
    {
      auto const* node = find_data_node(snapshot, dependency.node_id);
      ASSERT_NE(node, nullptr);
      states.push_back(node->state);
      if (node->state == "expired")
      {
        EXPECT_EQ(node->address, "(expired)");
        EXPECT_TRUE(node->value.empty());
        EXPECT_FALSE(node->value_constructed);
        EXPECT_FALSE(node->storage_address.empty());
      }
    }
    EXPECT_EQ(std::ranges::count(states, "expired"), 1);
    EXPECT_EQ(std::ranges::count(states, "unconstructed"), 1);
  }

  sched.schedule(write_value(blocker.write(), 1));
  sched.run_all();
}

// The hook releases the whole value, storage included, after the snapshot has listed its epoch.
// The pinned epoch still names the node, and the node reports the released storage as expired.
TEST(TaskRegistryLifetimeTest, SnapshotReportsStorageReleasedAfterTheRegistryUnlocksAsExpired)
{
  for (bool const best_effort : {false, true})
  {
    SCOPED_TRACE(best_effort ? "best effort" : "explicit");
    std::optional<Async<int>> value(std::in_place, 5);
    auto const epoch_address = address_text(value->queue().latest().get());

    value_to_release = &value;
    TaskRegistry::GraphSnapshot snapshot;
    {
      SnapshotUnlockHookGuard guard(&release_value);
      snapshot = take_snapshot(best_effort);
    }
    value_to_release = nullptr;

    ASSERT_FALSE(value.has_value());
    auto const epoch = std::ranges::find(snapshot.epochs, epoch_address, &uni20::TaskRegistryGraphEpoch::address);
    ASSERT_NE(epoch, snapshot.epochs.end());
    ASSERT_TRUE(epoch->has_node);
    auto const* node = find_data_node(snapshot, epoch->node_id);
    ASSERT_NE(node, nullptr);
    EXPECT_EQ(node->state, "expired");
    EXPECT_EQ(node->address, "(expired)");
    EXPECT_TRUE(node->value.empty());
  }
}

// The observation holds the storage, so a hook that drops the value's last owner does not destroy
// the value under itself, and the final release, which then runs on the snapshotting thread when
// the observation ends, does not wait for the observation that triggered it.
TEST(TaskRegistryLifetimeTest, ValueReleasedByItsOwnHookIsDestroyedAfterTheObservation)
{
  std::optional<Async<ReleasedByItsOwnHook>> value(std::in_place, ReleasedByItsOwnHook{});
  auto const destroyed_before = ReleasedByItsOwnHook::destroyed;
  auto const epoch_address = address_text(value->queue().latest().get());

  ReleasedByItsOwnHook::owner = &value;
  auto const snapshot = TaskRegistry::snapshot();
  ReleasedByItsOwnHook::owner = nullptr;

  EXPECT_FALSE(value.has_value());
  EXPECT_EQ(ReleasedByItsOwnHook::destroyed, destroyed_before + 1);
  auto const epoch = std::ranges::find(snapshot.epochs, epoch_address, &uni20::TaskRegistryGraphEpoch::address);
  ASSERT_NE(epoch, snapshot.epochs.end());
  auto const* node = find_data_node(snapshot, epoch->node_id);
  ASSERT_NE(node, nullptr);
  EXPECT_EQ(node->state, "constructed");
  EXPECT_EQ(node->value, fmt::format("destroyed={}", destroyed_before));
}

// A writer that is active, here suspended on another value, may change the payload at any time, so
// neither kind of snapshot formats it, and neither waits for the writer.
TEST(TaskRegistryLifetimeTest, ValueHeldByAnActiveWriterIsNotRead)
{
  DebugScheduler sched;
  Async<int> value(1);
  Async<int> gate;
  value.debug_name("held by a writer");
  sched.schedule(write_then_wait(value.write(), gate.read()));
  sched.run();

  for (bool const best_effort : {false, true})
  {
    SCOPED_TRACE(best_effort ? "best effort" : "explicit");
    auto const snapshot = take_snapshot(best_effort);
    auto const node =
        std::ranges::find(snapshot.data_nodes, "held by a writer", &uni20::TaskRegistryGraphDataNode::label);
    ASSERT_NE(node, snapshot.data_nodes.end());
    EXPECT_EQ(node->state, "constructed");
    EXPECT_TRUE(node->value.empty());
    EXPECT_EQ(node->value_unavailable, "writer active");
    EXPECT_NE(TaskRegistry::graphviz_dot(snapshot).find("value not read: writer active"), std::string::npos);
  }

  sched.schedule(write_value(gate.write(), 0));
  sched.run_all();
  EXPECT_EQ(value.read().get_wait(sched), 2);
}

// An epoch that has not started says nothing about the epoch before it, whose writer may be
// running. A pending successor therefore never admits a reader, whatever a snapshot saw of its
// predecessor.
TEST(TaskRegistryLifetimeTest, PendingSuccessorOfAnActiveWriterAdmitsNoReader)
{
  using Admission = EpochContext::DiagnosticReadAdmission;
  DebugScheduler sched;
  Async<int> value(1);
  Async<int> gate;
  auto const* node = value.queue().debug_node();
  sched.schedule(write_then_wait(value.write(), gate.read()));
  sched.run();
  auto const writing = value.queue().latest();

  auto pending_writer = value.write();
  auto const pending = value.queue().latest();
  ASSERT_NE(pending, writing);
  EXPECT_EQ(pending->diagnostic_reader_admit(node, false), Admission::NotReadable);
  EXPECT_EQ(pending->diagnostic_reader_admit(node, true), Admission::NotReadable);
  EXPECT_EQ(writing->diagnostic_reader_admit(node, true), Admission::WriterActive);
  EXPECT_EQ(writing->diagnostic_reader_admit(nullptr, true), Admission::OtherNode);

  sched.schedule(write_value(std::move(pending_writer), 3));
  sched.schedule(write_value(gate.write(), 0));
  sched.run_all();
  EXPECT_EQ(value.read().get_wait(sched), 3);
}

// The hook locks the epoch that the snapshot's reader was admitted to. Formatting runs with no
// registry or epoch mutex held, so this neither deadlocks nor loses the text.
TEST(TaskRegistryLifetimeTest, HookMayQueryTheEpochOfTheValueItFormats)
{
  Async<QueriesItsEpoch> value(QueriesItsEpoch{});
  value.debug_name("queries its epoch");
  QueriesItsEpoch::epoch = value.queue().latest();
  for (bool const best_effort : {false, true})
  {
    SCOPED_TRACE(best_effort ? "best effort" : "explicit");
    auto const snapshot = take_snapshot(best_effort);
    auto const node =
        std::ranges::find(snapshot.data_nodes, "queries its epoch", &uni20::TaskRegistryGraphDataNode::label);
    ASSERT_NE(node, snapshot.data_nodes.end());
    EXPECT_EQ(node->value, fmt::format("node={}", node->id));
    EXPECT_TRUE(node->value_unavailable.empty());
  }
  QueriesItsEpoch::epoch.reset();
}

// While a snapshot on another thread holds a diagnostic reader on `held`, the next writer of
// `held` waits for it without anything being queued on its scheduler. Waiting for the result
// must then look again instead of reporting a deadlock, and once the read ends it must find the
// writer that releasing the reader rescheduled. A scheduler with unrelated work is not held up at
// all: it services its queue, and its run_all() returns while the read is still in progress.
TEST(TaskRegistryLifetimeTest, SchedulersWaitForADiagnosticReadOnlyWhenItHoldsTheirWork)
{
  DebugScheduler sched;
  Async<HeldByItsHook> held(HeldByItsHook{1});
  held.debug_name("held by its hook");
  HeldByItsHook::Gate gate;
  HeldByItsHook::gate = &gate;

  TaskRegistry::GraphSnapshot snapshot;
  std::thread dumper([&] {
    {
      std::lock_guard lock(gate.mutex);
      gate.holder = std::this_thread::get_id();
    }
    snapshot = TaskRegistry::snapshot();
  });
  {
    std::unique_lock lock(gate.mutex);
    gate.changed.wait(lock, [&] { return gate.entered; });
  }
  EXPECT_TRUE(TaskRegistry::diagnostic_reads_in_progress());

  {
    DebugScheduler independent;
    Async<int> other;
    independent.schedule(write_value(other.write(), 3));
    independent.run_all();
    EXPECT_EQ(other.read().get_wait(independent), 3);
    EXPECT_TRUE(TaskRegistry::diagnostic_reads_in_progress());
  }

  // The writer suspends behind the snapshot's reader, leaving `sched` with nothing to run and no
  // external wait: the branch where help_while_waiting() used to report a deadlock.
  sched.schedule(write_held(held.write(), 2));
  sched.run();
  ASSERT_TRUE(sched.done());
  auto result = held.read();
  auto const is_ready = [&] { return result.await_ready(); };
  ASSERT_FALSE(is_ready());
  sched.help_while_waiting(is_ready);
  EXPECT_FALSE(is_ready());

  {
    std::lock_guard lock(gate.mutex);
    gate.released = true;
    gate.changed.notify_all();
  }
  EXPECT_EQ(result.get_wait(sched).value, 2);

  dumper.join();
  HeldByItsHook::gate = nullptr;
  EXPECT_FALSE(TaskRegistry::diagnostic_reads_in_progress());
  auto const node =
      std::ranges::find(snapshot.data_nodes, "held by its hook", &uni20::TaskRegistryGraphDataNode::label);
  ASSERT_NE(node, snapshot.data_nodes.end());
  EXPECT_EQ(node->value, "1");
}

// With an unrelated task in an external wait, an empty queue would normally park the waiting
// thread until that task's wakeup. Here the target becomes ready when the snapshot's reader is
// released, without any task being queued: the writer enrolled meanwhile was released without
// writing. So while the read is in progress the wait must return to its readiness loop instead.
TEST(TaskRegistryLifetimeTest, WaitDoesNotParkWhileADiagnosticReadHoldsItsTarget)
{
  DebugScheduler sched;
  std::thread waker;
  std::binary_semaphore unrelated_gate{0};
  bool resumed = false;
  sched.schedule(resume_after(ExternalWakeup{&waker, &unrelated_gate}, &resumed));
  sched.run();
  ASSERT_EQ(sched.external_waits(), 1U);

  // If the wait parks, only the unrelated wakeup ends it. Fire that after a long delay, so a
  // regression fails instead of hanging.
  std::binary_semaphore cancel_guard{0};
  bool guard_fired = false;
  std::thread guard([&] {
    if (cancel_guard.try_acquire_for(std::chrono::seconds(10))) return;
    guard_fired = true;
    unrelated_gate.release();
  });

  Async<HeldByItsHook> held(HeldByItsHook{1});
  HeldByItsHook::Gate gate;
  HeldByItsHook::gate = &gate;
  std::thread dumper([&] {
    {
      std::lock_guard lock(gate.mutex);
      gate.holder = std::this_thread::get_id();
    }
    static_cast<void>(TaskRegistry::snapshot());
  });
  wait_until(gate.mutex, gate.changed, gate.entered);

  {
    auto writer = held.write();
  }
  auto result = held.read();
  auto const is_ready = [&] { return result.await_ready(); };
  ASSERT_FALSE(is_ready());
  sched.help_while_waiting(is_ready);
  cancel_guard.release();
  guard.join();
  EXPECT_FALSE(guard_fired);

  set_and_notify(gate.mutex, gate.changed, gate.released);
  EXPECT_EQ(result.get_wait(sched).value, 1);
  dumper.join();
  HeldByItsHook::gate = nullptr;

  if (!guard_fired) unrelated_gate.release();
  sched.wait_for([&resumed] { return resumed; });
  waker.join();
}

// The read stays counted until the snapshot has released its pin on the value. Here the pin is the
// payload's last owner, so releasing it runs the payload's destructor on the snapshotting thread,
// and only that destructor releases the reader that the waiting thread's writer needs. A scheduler
// that saw no read in progress at that point would report a deadlock.
TEST(TaskRegistryLifetimeTest, ReadStaysCountedUntilTheSnapshotReleasesThePayload)
{
  DebugScheduler sched;
  Async<int> source(1);
  OwnsAReader::Gates gates;
  OwnsAReader::gates = &gates;
  std::optional<Async<OwnsAReader>> owner(std::in_place, OwnsAReader(source.read()));

  std::thread dumper([&] {
    {
      std::lock_guard lock(gates.mutex);
      gates.holder = std::this_thread::get_id();
    }
    static_cast<void>(TaskRegistry::snapshot());
  });
  wait_until(gates.mutex, gates.changed, gates.in_hook);

  owner.reset();
  sched.schedule(write_value(source.write(), 2));
  sched.run();
  ASSERT_TRUE(sched.done());

  set_and_notify(gates.mutex, gates.changed, gates.hook_released);
  wait_until(gates.mutex, gates.changed, gates.in_destructor);
  EXPECT_TRUE(TaskRegistry::diagnostic_reads_in_progress());
  auto result = source.read();
  auto const is_ready = [&] { return result.await_ready(); };
  ASSERT_FALSE(is_ready());
  sched.help_while_waiting(is_ready);

  set_and_notify(gates.mutex, gates.changed, gates.destructor_released);
  EXPECT_EQ(result.get_wait(sched), 2);
  dumper.join();
  OwnsAReader::gates = nullptr;
}
#endif
#endif
