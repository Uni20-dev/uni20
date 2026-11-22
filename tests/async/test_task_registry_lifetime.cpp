// Snapshots of the task registry while the program releases what they describe (issue #164).

#include <fmt/format.h>
#include <gtest/gtest.h>
#include <optional>
#include <string>
#include <uni20/async/async.hpp>
#include <uni20/async/task_registry.hpp>
#include <uni20/config.hpp>

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

} // namespace

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
      snapshot = best_effort ? TaskRegistry::snapshot_best_effort() : TaskRegistry::snapshot();
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
#endif
