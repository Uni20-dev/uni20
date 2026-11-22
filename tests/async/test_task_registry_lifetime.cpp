// Snapshots of the task registry while the program releases what they describe (issue #164).

#include <algorithm>
#include <cstdint>
#include <fmt/format.h>
#include <gtest/gtest.h>
#include <optional>
#include <string>
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
#endif
#endif
