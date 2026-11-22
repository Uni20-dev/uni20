#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <future>
#include <gtest/gtest.h>
#include <iterator>
#include <limits>
#include <mutex>
#include <stop_token>
#include <string>
#include <string_view>
#include <thread>
#include <uni20/async/async.hpp>
#include <uni20/async/async_task.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/buffers.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/async/task_registry.hpp>
#include <uni20/async/task_registry_presentation.hpp>
#include <uni20/common/presentation.hpp>
#include <uni20/config.hpp>
#include <uni20/core/scalar_precision.hpp>

#include "../common/env_var_guard.hpp"

#if defined(__linux__)
#include <signal.h>
#include <sys/wait.h>
#include <unistd.h>
#endif

using namespace uni20;
using namespace uni20::async;

#if UNI20_DEBUG_ASYNC_TASKS
namespace
{
using uni20::test::EnvVarGuard;

class StacktraceOptionsGuard {
  public:
    StacktraceOptionsGuard() : old_(TaskRegistry::stacktrace_options()) {}

    ~StacktraceOptionsGuard() { TaskRegistry::set_stacktrace_options(old_); }

  private:
    TaskRegistry::StacktraceOptions old_;
};

class CoroutineExceptionDiagnosticsOptionsGuard {
  public:
    CoroutineExceptionDiagnosticsOptionsGuard() : old_(TaskRegistry::coroutine_exception_diagnostics_options()) {}

    ~CoroutineExceptionDiagnosticsOptionsGuard() { TaskRegistry::set_coroutine_exception_diagnostics_options(old_); }

  private:
    TaskRegistry::CoroutineExceptionDiagnosticsOptions old_;
};

AsyncTask make_suspended_task() { co_return; }

AsyncTask wait_for_reader(ReadBuffer<int> reader)
{
  auto const& value = co_await reader;
  (void)value;
}

AsyncTask write_value(WriteBuffer<int> writer, int value)
{
  co_await writer = value;
  co_return;
}

AsyncTask fail_writer(WriteBuffer<int> writer)
{
  static_cast<void>(writer);
  throw std::runtime_error("deliberate task-registry failure");
  co_return;
}

AsyncTask copy_value(ReadBuffer<int> reader, WriteBuffer<int> writer)
{
  auto const& value = co_await reader;
  co_await writer = value;
  co_return;
}

AsyncTask sum_all_values(ReadBuffer<int> lhs, ReadBuffer<int> rhs, WriteBuffer<int> writer)
{
  auto [left, right] = co_await all(lhs, rhs);
  co_await writer = left + right;
  co_return;
}

struct ShapeLikeExtents
{
    static constexpr std::size_t rank() noexcept { return 3; }

    [[nodiscard]] std::size_t extent(std::size_t axis) const noexcept { return axis == 0 ? 2U : axis == 1 ? 3U : 4U; }
};

struct ShapeLikeValue
{
    [[nodiscard]] ShapeLikeExtents extents() const noexcept { return {}; }
};

struct CustomDebugValue
{
    int id{0};
};

struct OpaqueValue
{};

std::string uni20_async_debug_value(CustomDebugValue const& value) { return "custom id=" + std::to_string(value.id); }

std::filesystem::path make_temp_dir(std::string_view name)
{
  auto const stamp = std::chrono::steady_clock::now().time_since_epoch().count();
  auto path = std::filesystem::temp_directory_path() / (std::string(name) + "-" + std::to_string(stamp));
  std::filesystem::remove_all(path);
  std::filesystem::create_directories(path);
  return path;
}

std::string read_file(std::filesystem::path const& path)
{
  std::ifstream input(path);
  return std::string(std::istreambuf_iterator<char>(input), std::istreambuf_iterator<char>());
}

std::filesystem::path find_dot_file(std::filesystem::path const& dir)
{
  if (!std::filesystem::exists(dir)) return {};
  for (auto const& entry : std::filesystem::directory_iterator(dir))
  {
    if (entry.path().extension() == ".dot") return entry.path();
  }
  return {};
}

bool wait_for_dot_file(std::filesystem::path const& dir, std::chrono::milliseconds timeout)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline)
  {
    auto path = find_dot_file(dir);
    if (!path.empty() && read_file(path).find("digraph uni20_async_dag") != std::string::npos) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return false;
}

int dump_mode_probe(char const* value)
{
  if (value)
    ::setenv("UNI20_DEBUG_ASYNC_TASKS", value, 1);
  else
    ::unsetenv("UNI20_DEBUG_ASYNC_TASKS");

  switch (TaskRegistry::dump_mode())
  {
    case TaskRegistry::DumpMode::None:
      return 0;
    case TaskRegistry::DumpMode::Basic:
      return 1;
    case TaskRegistry::DumpMode::Full:
      return 2;
  }

  return 99;
}

#if UNI20_DEBUG_DAG
int dependency_cycle_probe()
{
  DebugScheduler sched;
  Async<int> a;
  Async<int> b;
  a.debug_name("cycle a");
  b.debug_name("cycle b");

  auto task1 = copy_value(b.read(), a.write());
  auto task2 = copy_value(a.read(), b.write());
  task1.debug_name("cycle task 1");
  task2.debug_name("cycle task 2");

  sched.schedule(std::move(task1));
  sched.schedule(std::move(task2));
  sched.run();

  auto const dot = TaskRegistry::graphviz_dot();
  bool const ok = dot.find("DAG diagnostics") != std::string::npos &&
                  dot.find("dependency cycle") != std::string::npos &&
                  dot.find("diagnostic: dependency cycle") != std::string::npos &&
                  dot.find("cycle task 1") != std::string::npos && dot.find("cycle task 2") != std::string::npos &&
                  dot.find("cycle a") != std::string::npos && dot.find("cycle b") != std::string::npos;
  return ok ? 0 : 1;
}
#endif

} // namespace
#endif

#if UNI20_DEBUG_ASYNC_TASKS
TEST(TaskRegistryDebugDeathTest, DumpModeDefaultsToBasicWhenUnset)
{
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT({ std::_Exit(dump_mode_probe(nullptr)); }, ::testing::ExitedWithCode(1), "");
}

TEST(TaskRegistryDebugDeathTest, DumpModeParsesNoneSynonyms)
{
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT({ std::_Exit(dump_mode_probe("0")); }, ::testing::ExitedWithCode(0), "");
  EXPECT_EXIT({ std::_Exit(dump_mode_probe("off")); }, ::testing::ExitedWithCode(0), "");
}

TEST(TaskRegistryDebugDeathTest, DumpModeParsesFullSynonyms)
{
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT({ std::_Exit(dump_mode_probe("2")); }, ::testing::ExitedWithCode(2), "");
  EXPECT_EXIT({ std::_Exit(dump_mode_probe("verbose")); }, ::testing::ExitedWithCode(2), "");
}

TEST(TaskRegistryDebugDeathTest, DumpModeTrimsAndNormalizesCase)
{
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT({ std::_Exit(dump_mode_probe("  YeS  ")); }, ::testing::ExitedWithCode(1), "");
}

TEST(TaskRegistryDebugDeathTest, DumpModeFallsBackToBasicForUnknownValue)
{
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT({ std::_Exit(dump_mode_probe("not-a-mode")); }, ::testing::ExitedWithCode(1), "");
}

#if UNI20_DEBUG_DAG
TEST(TaskRegistryDebugDeathTest, GraphvizDotHighlightsDependencyCycle)
{
  GTEST_FLAG_SET(death_test_style, "threadsafe");
  EXPECT_EXIT({ std::_Exit(dependency_cycle_probe()); }, ::testing::ExitedWithCode(0), "");
}
#endif

TEST(TaskRegistryDebugTest, DumpShowsTaskStateAndTransitions)
{
  auto task = make_suspended_task();
  auto const snapshot = TaskRegistry::snapshot();
  ASSERT_EQ(snapshot.tasks.size(), 1U);

  testing::internal::CaptureStderr();
  TaskRegistry::dump();
  auto const dump = testing::internal::GetCapturedStderr();

  EXPECT_NE(dump.find("Async task registry"), std::string::npos);
  EXPECT_NE(dump.find("tracked coroutine tasks"), std::string::npos);
  EXPECT_NE(dump.find("Coroutine tasks"), std::string::npos);
  EXPECT_NE(dump.find("task " + std::to_string(snapshot.tasks.front().id)), std::string::npos);
  EXPECT_NE(dump.find("suspended"), std::string::npos);
  EXPECT_NE(dump.find("transitions"), std::string::npos);
  EXPECT_NE(dump.find("source"), std::string::npos);
#if !UNI20_HAS_STACKTRACE
  EXPECT_NE(dump.find("std::stacktrace is unavailable"), std::string::npos);
#endif

  EXPECT_FALSE(snapshot.tasks.front().creation_timestamp.empty());
  EXPECT_FALSE(snapshot.tasks.front().last_transition_timestamp.empty());

  task.resume();

  testing::internal::CaptureStderr();
  TaskRegistry::dump();
  auto const after_resume_dump = testing::internal::GetCapturedStderr();
  EXPECT_NE(after_resume_dump.find("tracked coroutine tasks"), std::string::npos);
  EXPECT_EQ(after_resume_dump.find("Coroutine tasks"), std::string::npos);
}

TEST(TaskRegistryDebugTest, DumpShowsEpochContextBindingsForSuspendedTask)
{
  DebugScheduler sched;
  Async<int> value;

  sched.schedule(wait_for_reader(value.read()));
  sched.run();

  testing::internal::CaptureStderr();
  TaskRegistry::dump();
  auto const dump = testing::internal::GetCapturedStderr();

  EXPECT_NE(dump.find("tracked epoch contexts"), std::string::npos);
  EXPECT_NE(dump.find("Epoch contexts"), std::string::npos);
  EXPECT_NE(dump.find("Coroutine tasks"), std::string::npos);
  EXPECT_NE(dump.find("readers"), std::string::npos);

  sched.schedule(write_value(value.write(), 7));
  sched.run_all();

  testing::internal::CaptureStderr();
  TaskRegistry::dump();
  auto const after_completion_dump = testing::internal::GetCapturedStderr();
  EXPECT_NE(after_completion_dump.find("tracked coroutine tasks"), std::string::npos);
  EXPECT_EQ(after_completion_dump.find("Coroutine tasks"), std::string::npos);
}

TEST(TaskRegistryDebugTest, GraphvizDotShowsSuspendedTaskAndEpochReadEdges)
{
  DebugScheduler sched;
  Async<int> value;

  sched.schedule(wait_for_reader(value.read()));
  sched.run();

  auto const dot = TaskRegistry::graphviz_dot();
  EXPECT_NE(dot.find("digraph uni20_async_dag"), std::string::npos);
  EXPECT_NE(dot.find("rankdir=LR"), std::string::npos);
  EXPECT_NE(dot.find("task_"), std::string::npos);
  EXPECT_NE(dot.find("epoch_"), std::string::npos);
  EXPECT_NE(dot.find("suspended"), std::string::npos);
  EXPECT_NE(dot.find("await read"), std::string::npos);
#if UNI20_DEBUG_DAG
  EXPECT_NE(dot.find("data_"), std::string::npos);
  EXPECT_NE(dot.find("co_await read"), std::string::npos);
#endif

  sched.schedule(write_value(value.write(), 7));
  sched.run_all();
}

#if UNI20_DEBUG_DAG
TEST(TaskRegistryDebugTest, GraphvizDotShowsCoarseBufferArgumentDependencies)
{
  DebugScheduler sched;
  Async<int> input(3);
  Async<int> output;

  auto task = copy_value(input.read(), output.write());
  auto const dot = TaskRegistry::graphviz_dot();

  EXPECT_NE(dot.find("data_"), std::string::npos);
  EXPECT_NE(dot.find("arg read"), std::string::npos);
  EXPECT_NE(dot.find("arg write"), std::string::npos);

  sched.schedule(std::move(task));
  sched.run_all();

  EXPECT_EQ(output.read().get_wait(sched), 3);
}

TEST(TaskRegistryDebugTest, GraphvizDotShowsOptionalDebugLabels)
{
  DebugScheduler sched;
  Async<int> input(3);
  Async<int> output;
  input.debug_name("named input");
  output.debug_name("named output");

  auto task = copy_value(input.read(), output.write());
  task.debug_name("copy kernel");

  auto const dot = TaskRegistry::graphviz_dot();
  EXPECT_NE(dot.find("named input"), std::string::npos);
  EXPECT_NE(dot.find("named output"), std::string::npos);
  EXPECT_NE(dot.find("copy kernel"), std::string::npos);

  sched.schedule(std::move(task));
  sched.run_all();

  EXPECT_EQ(output.read().get_wait(sched), 3);
}

TEST(TaskRegistryDebugTest, GraphSnapshotCarriesOptionalStacktraceProvenance)
{
  DebugScheduler sched;
  Async<int> value;

#if UNI20_HAS_STACKTRACE
  auto const expected_creation_line = __LINE__ + 2;
#endif
  auto task = wait_for_reader(value.read());
  task.debug_name("provenance reader");
  sched.schedule(std::move(task));
  sched.run();

  auto const snapshot = TaskRegistry::snapshot();
  auto const task_it = std::find_if(snapshot.tasks.begin(), snapshot.tasks.end(),
                                    [](auto const& record) { return record.label == "provenance reader"; });

  ASSERT_NE(task_it, snapshot.tasks.end());
  EXPECT_EQ(task_it->state, "suspended");
  ASSERT_FALSE(task_it->await_dependencies.empty());

  auto const dot = TaskRegistry::graphviz_dot(snapshot);
  EXPECT_NE(dot.find("provenance reader"), std::string::npos);
  EXPECT_NE(dot.find("co_await read"), std::string::npos);

#if UNI20_HAS_STACKTRACE
  EXPECT_FALSE(task_it->creation_site.stacktrace.empty());
  EXPECT_FALSE(task_it->schedule_site.stacktrace.empty());
  EXPECT_FALSE(task_it->last_transition_site.stacktrace.empty());
  EXPECT_FALSE(task_it->last_await_site.stacktrace.empty());
  EXPECT_FALSE(task_it->await_dependencies.front().await_site.stacktrace.empty());
  if (!task_it->creation_site.location.empty())
  {
    auto const expected_suffix = ":" + std::to_string(expected_creation_line);
    EXPECT_TRUE(task_it->creation_site.location.ends_with(expected_suffix)) << task_it->creation_site.location;
  }
  EXPECT_NE(dot.find("tooltip=\""), std::string::npos);
  if (!task_it->creation_site.location.empty())
  {
    EXPECT_NE(dot.find("created_at="), std::string::npos);
  }
  if (!task_it->schedule_site.location.empty())
  {
    EXPECT_NE(dot.find("scheduled_at="), std::string::npos);
  }
  if (!task_it->last_await_site.location.empty())
  {
    EXPECT_NE(dot.find("awaiting_at="), std::string::npos);
  }
#else
  EXPECT_TRUE(task_it->creation_site.stacktrace.empty());
  EXPECT_TRUE(task_it->schedule_site.stacktrace.empty());
  EXPECT_TRUE(task_it->last_transition_site.stacktrace.empty());
  EXPECT_TRUE(task_it->last_await_site.stacktrace.empty());
  EXPECT_TRUE(task_it->await_dependencies.front().await_site.stacktrace.empty());
  EXPECT_EQ(dot.find("created_at="), std::string::npos);
  EXPECT_EQ(dot.find("scheduled_at="), std::string::npos);
  EXPECT_EQ(dot.find("awaiting_at="), std::string::npos);
#endif

  sched.schedule(write_value(value.write(), 7));
  sched.run_all();
}

TEST(TaskRegistryDebugTest, GraphvizTooltipsEscapeMarkupForXdot)
{
  TaskRegistry::GraphSnapshot snapshot;
  TaskRegistryGraphTask task;
  task.id = 7;
  task.state = "suspended";
  task.creation_site.function = "std::vector<int>& make_value<double>()";
  task.creation_site.stacktrace = "std::vector<int>& make_value<double>()\noperator&<int>()\n" + std::string(105, 'x');
  snapshot.tasks.push_back(std::move(task));

  auto const dot = TaskRegistry::graphviz_dot(snapshot);
  EXPECT_NE(dot.find("std::vector&lt;int&gt;&amp; make_value&lt;double&gt;()"), std::string::npos);
  EXPECT_NE(dot.find("operator&amp;&lt;int&gt;()"), std::string::npos);
  EXPECT_NE(dot.find("tooltip=\"task creation:&#10;"), std::string::npos);
  EXPECT_NE(dot.find(std::string(100, 'x') + "&#10;" + std::string(5, 'x')), std::string::npos);
  EXPECT_EQ(dot.find("std::vector<int>"), std::string::npos);
}

TEST(TaskRegistryDebugTest, SnapshotReportUsesPresentationDocument)
{
  TaskRegistry::GraphSnapshot snapshot;
  snapshot.data_nodes.push_back(TaskRegistryGraphDataNode{.id = 3,
                                                          .label = "input",
                                                          .type = "int",
                                                          .storage_address = "0x1234",
                                                          .state = "unconstructed",
                                                          .value_constructed = false});
  snapshot.tasks.push_back(TaskRegistryGraphTask{.id = 5,
                                                 .label = "blocked reader",
                                                 .state = "suspended",
                                                 .transition_count = 4,
                                                 .creation_timestamp = "2026-07-15 18:00:00",
                                                 .read_dependencies = {3}});

  TaskRegistry::GraphDiagnostics diagnostics;
  diagnostics.notes.push_back("missing writers: 1");
  diagnostics.blocked_read_task_ids.push_back(5);
  diagnostics.missing_writer_node_ids.push_back(3);

  auto const report = task_registry_report(snapshot, diagnostics,
                                           {.title = "Test async snapshot", .reason = "presentation regression"});
  auto policy = presentation::strict_ascii_policy();
  policy.wrap_width = 132;
  auto const text = presentation::render_plain(report, policy);

  EXPECT_NE(text.find("Test async snapshot"), std::string::npos);
  EXPECT_NE(text.find("presentation regression"), std::string::npos);
  EXPECT_NE(text.find("missing writer detected"), std::string::npos);
  EXPECT_NE(text.find("data 3: input"), std::string::npos);
  EXPECT_NE(text.find("task 5: blocked reader"), std::string::npos);
  EXPECT_NE(text.find("blocked read"), std::string::npos);
}

#if UNI20_HAS_STACKTRACE
TEST(TaskRegistryDebugTest, StacktraceOptionsSuppressSnapshotStacktraceText)
{
  StacktraceOptionsGuard guard;
  auto options = TaskRegistry::stacktrace_options();
  options.max_frames = 0;
  TaskRegistry::set_stacktrace_options(options);

  DebugScheduler sched;
  Async<int> value;

  auto task = wait_for_reader(value.read());
  task.debug_name("hidden stacktrace reader");
  sched.schedule(std::move(task));
  sched.run();

  auto const snapshot = TaskRegistry::snapshot();
  auto const task_it = std::find_if(snapshot.tasks.begin(), snapshot.tasks.end(),
                                    [](auto const& record) { return record.label == "hidden stacktrace reader"; });

  ASSERT_NE(task_it, snapshot.tasks.end());
  EXPECT_TRUE(task_it->creation_site.stacktrace.empty());
  EXPECT_TRUE(task_it->schedule_site.stacktrace.empty());
  EXPECT_TRUE(task_it->last_transition_site.stacktrace.empty());
  EXPECT_TRUE(task_it->last_await_site.stacktrace.empty());
  ASSERT_FALSE(task_it->await_dependencies.empty());
  EXPECT_TRUE(task_it->await_dependencies.front().await_site.stacktrace.empty());

  auto const dot = TaskRegistry::graphviz_dot(snapshot);
  EXPECT_EQ(dot.find("stacktrace:"), std::string::npos);

  sched.schedule(write_value(value.write(), 8));
  sched.run_all();
}
#endif

TEST(TaskRegistryDebugTest, GraphSnapshotCapturesAllAwaiterDependencies)
{
  DebugScheduler sched;
  Async<int> lhs;
  Async<int> rhs(6);
  Async<int> output;

  auto task = sum_all_values(lhs.read(), rhs.read(), output.write());
  task.debug_name("all awaiter kernel");
  sched.schedule(std::move(task));
  sched.run();

  auto const snapshot = TaskRegistry::snapshot();
  auto const task_it = std::find_if(snapshot.tasks.begin(), snapshot.tasks.end(),
                                    [](auto const& record) { return record.label == "all awaiter kernel"; });

  ASSERT_NE(task_it, snapshot.tasks.end());
  auto const read_await_count =
      std::count_if(task_it->await_dependencies.begin(), task_it->await_dependencies.end(),
                    [](auto const& dependency) { return dependency.role == TaskRegistryGraphRole::Reader; });
  EXPECT_GE(read_await_count, 2);

  auto const dot = TaskRegistry::graphviz_dot(snapshot);
  EXPECT_NE(dot.find("all awaiter kernel"), std::string::npos);
  EXPECT_NE(dot.find("co_await read"), std::string::npos);

  sched.schedule(write_value(lhs.write(), 4));
  sched.run_all();
  EXPECT_EQ(output.read().get_wait(sched), 10);
}

TEST(TaskRegistryDebugTest, GraphSnapshotExposesStructuredRecords)
{
  DebugScheduler sched;
  Async<int> input(3);
  Async<int> output;
  input.debug_name("structured input");
  output.debug_name("structured output");

  auto task = copy_value(input.read(), output.write());
  task.debug_name("structured copy");

  auto const snapshot = TaskRegistry::snapshot();
  auto const diagnostics = TaskRegistry::diagnose_snapshot(snapshot);
  auto const find_named_data = [&](std::string_view label) {
    return std::find_if(snapshot.data_nodes.begin(), snapshot.data_nodes.end(),
                        [&](auto const& node) { return node.label == label; });
  };
  auto const has_named_task = std::any_of(snapshot.tasks.begin(), snapshot.tasks.end(), [](auto const& record) {
    return record.label == "structured copy" && record.state == "suspended" && !record.read_dependencies.empty() &&
           !record.write_dependencies.empty();
  });
  auto const input_node = find_named_data("structured input");
  auto const output_node = find_named_data("structured output");

  EXPECT_TRUE(snapshot.snapshot_available);
  EXPECT_TRUE(diagnostics.notes.empty());
  ASSERT_NE(input_node, snapshot.data_nodes.end());
  ASSERT_NE(output_node, snapshot.data_nodes.end());
  EXPECT_FALSE(input_node->type.empty());
  EXPECT_FALSE(input_node->storage_address.empty());
  EXPECT_FALSE(input_node->address.empty());
  EXPECT_EQ(input_node->state, "constructed");
  EXPECT_EQ(input_node->value, "3");
  EXPECT_TRUE(input_node->value_constructed);
  EXPECT_FALSE(output_node->type.empty());
  EXPECT_FALSE(output_node->storage_address.empty());
  EXPECT_EQ(output_node->address, "(unconstructed)");
  EXPECT_EQ(output_node->state, "unconstructed");
  EXPECT_TRUE(output_node->value.empty());
  EXPECT_FALSE(output_node->value_constructed);
  EXPECT_TRUE(has_named_task);

  auto const dot = TaskRegistry::graphviz_dot(snapshot, diagnostics);
  EXPECT_NE(dot.find("structured input"), std::string::npos);
  EXPECT_NE(dot.find("structured output"), std::string::npos);
  EXPECT_NE(dot.find("structured copy"), std::string::npos);
  EXPECT_NE(dot.find("storage=0x"), std::string::npos);
  EXPECT_NE(dot.find("state=unconstructed"), std::string::npos);
  EXPECT_NE(dot.find("value=3"), std::string::npos);
  EXPECT_EQ(dot.find("value=unconstructed"), std::string::npos);
  EXPECT_EQ(dot.find("value=0x"), std::string::npos);
  EXPECT_EQ(dot.find("addr=0x0"), std::string::npos);

  sched.schedule(std::move(task));
  sched.run_all();

  auto const after_snapshot = TaskRegistry::snapshot();
  auto const after_output_node = std::find_if(after_snapshot.data_nodes.begin(), after_snapshot.data_nodes.end(),
                                              [](auto const& node) { return node.label == "structured output"; });
  ASSERT_NE(after_output_node, after_snapshot.data_nodes.end());
  EXPECT_TRUE(after_output_node->value_constructed);
  EXPECT_NE(after_output_node->address, "(unconstructed)");
  EXPECT_EQ(after_output_node->state, "constructed");
  EXPECT_EQ(after_output_node->value, "3");

  EXPECT_EQ(output.read().get_wait(sched), 3);
}

TEST(TaskRegistryDebugTest, GraphSnapshotSummarizesShapeLikeValues)
{
  Async<ShapeLikeValue> tensor(ShapeLikeValue{});
  tensor.debug_name("shape-like value");

  auto const snapshot = TaskRegistry::snapshot();
  auto const tensor_node = std::find_if(snapshot.data_nodes.begin(), snapshot.data_nodes.end(),
                                        [](auto const& node) { return node.label == "shape-like value"; });

  ASSERT_NE(tensor_node, snapshot.data_nodes.end());
  EXPECT_EQ(tensor_node->state, "constructed");
  EXPECT_EQ(tensor_node->value, "shape=(2, 3, 4)");

  auto const dot = TaskRegistry::graphviz_dot(snapshot);
  EXPECT_NE(dot.find("shape-like value"), std::string::npos);
  EXPECT_NE(dot.find("state=constructed"), std::string::npos);
  EXPECT_NE(dot.find("value=shape=(2, 3, 4)"), std::string::npos);
}

TEST(TaskRegistryDebugTest, GraphSnapshotUsesCustomDebugValueHook)
{
  Async<CustomDebugValue> value(CustomDebugValue{7});
  value.debug_name("custom value");

  auto const snapshot = TaskRegistry::snapshot();
  auto const value_node = std::find_if(snapshot.data_nodes.begin(), snapshot.data_nodes.end(),
                                       [](auto const& node) { return node.label == "custom value"; });

  ASSERT_NE(value_node, snapshot.data_nodes.end());
  EXPECT_EQ(value_node->state, "constructed");
  EXPECT_EQ(value_node->value, "custom id=7");

  auto const dot = TaskRegistry::graphviz_dot(snapshot);
  EXPECT_NE(dot.find("custom value"), std::string::npos);
  EXPECT_NE(dot.find("value=custom id=7"), std::string::npos);
}

TEST(TaskRegistryDebugTest, GraphSnapshotOmitsValueForOpaqueConstructedTypes)
{
  Async<OpaqueValue> value(OpaqueValue{});
  value.debug_name("opaque value");

  auto const snapshot = TaskRegistry::snapshot();
  auto const value_node = std::find_if(snapshot.data_nodes.begin(), snapshot.data_nodes.end(),
                                       [](auto const& node) { return node.label == "opaque value"; });

  ASSERT_NE(value_node, snapshot.data_nodes.end());
  EXPECT_EQ(value_node->state, "constructed");
  EXPECT_TRUE(value_node->value.empty());

  auto const dot = TaskRegistry::graphviz_dot(snapshot);
  EXPECT_NE(dot.find("opaque value"), std::string::npos);
  EXPECT_NE(dot.find("state=constructed"), std::string::npos);
  EXPECT_EQ(dot.find("value=constructed"), std::string::npos);
}

TEST(TaskRegistryDebugTest, FormatsEveryConfiguredRealScalar)
{
  auto check = []<typename Scalar>() {
    EXPECT_EQ(uni20::async::detail::scalar_debug_info(Scalar{5} / Scalar{4}), "1.25");
  };

  for (auto precision : uni20::configured_scalar_precisions())
    uni20::visit_scalar_precision(precision, check);
}

TEST(TaskRegistryDebugTest, GraphvizDotDiagnosesMissingWriter)
{
  DebugScheduler sched;
  Async<int> value;
  value.debug_name("unwritten value");

  auto task = wait_for_reader(value.read());
  task.debug_name("blocked reader");
  sched.schedule(std::move(task));
  sched.run();

  auto const snapshot = TaskRegistry::snapshot();
  auto const diagnostics = TaskRegistry::diagnose_snapshot(snapshot);
  EXPECT_EQ(diagnostics.blocked_read_task_ids.size(), 1U);
  EXPECT_EQ(diagnostics.missing_writer_epoch_ids.size(), 1U);
  EXPECT_EQ(diagnostics.missing_writer_node_ids.size(), 1U);

  auto const dot_from_snapshot = TaskRegistry::graphviz_dot(snapshot, diagnostics);
  EXPECT_NE(dot_from_snapshot.find("DAG diagnostics"), std::string::npos);
  EXPECT_NE(dot_from_snapshot.find("missing writer"), std::string::npos);

  auto const dot = TaskRegistry::graphviz_dot();
  EXPECT_NE(dot.find("DAG diagnostics"), std::string::npos);
  EXPECT_NE(dot.find("blocked tasks: 1"), std::string::npos);
  EXPECT_NE(dot.find("missing writer"), std::string::npos);
  EXPECT_NE(dot.find("blocked: read"), std::string::npos);
  EXPECT_NE(dot.find("unwritten value"), std::string::npos);
  EXPECT_NE(dot.find("blocked reader"), std::string::npos);

  sched.schedule(write_value(value.write(), 5));
  sched.run_all();
  EXPECT_EQ(value.read().get_wait(sched), 5);
}

#endif

TEST(TaskRegistryDebugTest, GraphvizDumpFileWritesDot)
{
  auto dir = make_temp_dir("uni20-dag-file-test");
  TaskRegistry::GraphvizDumpOptions options;
  options.output_dir = dir.string();
  options.file_prefix = "registry";
  auto const path = TaskRegistry::default_graphviz_dump_path(options);

  EXPECT_TRUE(TaskRegistry::dump_graphviz_file(path));
  EXPECT_NE(read_file(path).find("digraph uni20_async_dag"), std::string::npos);
  std::filesystem::remove_all(dir);
}

TEST(TaskRegistryDebugTest, DefaultGraphvizDumpOptionsReadEnvironment)
{
  auto dir = make_temp_dir("uni20-dag-default-options-test");
  EnvVarGuard output_dir("UNI20_DEBUG_DAG_OUTPUT_DIR", dir.string());
  EnvVarGuard prefix("UNI20_DEBUG_DAG_FILE_PREFIX", "env-default");

  auto const options = TaskRegistry::default_graphviz_dump_options();
  EXPECT_EQ(options.output_dir, dir.string());
  EXPECT_EQ(options.file_prefix, "env-default");

  auto const path = TaskRegistry::default_graphviz_dump_path();
  EXPECT_NE(path.find(dir.string()), std::string::npos);
  EXPECT_NE(path.find("env-default"), std::string::npos);
  std::filesystem::remove_all(dir);
}

TEST(TaskRegistryDebugTest, DefaultCoroutineExceptionDiagnosticsOptionsReadEnvironment)
{
  EnvVarGuard enabled("UNI20_DEBUG_DAG_DUMP_ON_EXCEPTION", "true");
  EnvVarGuard output_dir("UNI20_DEBUG_DAG_OUTPUT_DIR", "/tmp/uni20-exception-diagnostics-test");
  EnvVarGuard prefix("UNI20_DEBUG_DAG_FILE_PREFIX", "exception-test");

  auto const options = TaskRegistry::default_coroutine_exception_diagnostics_options();
  EXPECT_TRUE(options.enabled);
  EXPECT_TRUE(options.write_graphviz);
  EXPECT_EQ(options.dump_options.output_dir, "/tmp/uni20-exception-diagnostics-test");
  EXPECT_EQ(options.dump_options.file_prefix, "exception-test");
}

TEST(TaskRegistryDebugTest, OriginatingCoroutineExceptionWritesLiveFailedTaskSnapshot)
{
  CoroutineExceptionDiagnosticsOptionsGuard guard;
  auto dir = make_temp_dir("uni20-dag-coroutine-exception-test");
  TaskRegistry::set_coroutine_exception_diagnostics_options({
      .enabled = true,
      .write_graphviz = true,
      .dump_options = {.output_dir = dir.string(), .file_prefix = "coroutine-exception"},
  });

  DebugScheduler scheduler;
  Async<int> intermediate;
  Async<int> output;
  auto producer = fail_writer(intermediate.write());
  producer.debug_name("deliberately failing writer");
  scheduler.schedule(std::move(producer));
  auto consumer = copy_value(intermediate.read(), output.write());
  consumer.debug_name("downstream exception propagation");
  scheduler.schedule(std::move(consumer));
  scheduler.run_all();

  EXPECT_THROW(static_cast<void>(output.get_wait(scheduler)), std::runtime_error);
  std::size_t dot_file_count = 0;
  for (auto const& entry : std::filesystem::directory_iterator(dir))
    if (entry.path().extension() == ".dot") ++dot_file_count;
  EXPECT_EQ(dot_file_count, 1U);
  auto const path = find_dot_file(dir);
  ASSERT_FALSE(path.empty());
  auto const dot = read_file(path);
  EXPECT_NE(dot.find("deliberately failing writer"), std::string::npos);
  EXPECT_NE(dot.find("downstream exception propagation"), std::string::npos);
  EXPECT_NE(dot.find("failed"), std::string::npos);
  EXPECT_NE(dot.find("deliberate task-registry failure"), std::string::npos);
  std::filesystem::remove_all(dir);
}

TEST(TaskRegistryDebugTest, DefaultStacktraceOptionsReadEnvironment)
{
  EnvVarGuard frames("UNI20_DEBUG_DAG_STACKTRACE_FRAMES", "3");
  EnvVarGuard internal_frames("UNI20_DEBUG_DAG_STACKTRACE_INTERNAL_FRAMES", "false");

  auto const options = TaskRegistry::default_stacktrace_options();
  EXPECT_EQ(options.max_frames, 3U);
  EXPECT_FALSE(options.include_internal_frames);
}

TEST(TaskRegistryDebugTest, DefaultStacktraceOptionsAcceptUnlimitedFrames)
{
  EnvVarGuard frames("UNI20_DEBUG_DAG_STACKTRACE_FRAMES", "all");

  auto const options = TaskRegistry::default_stacktrace_options();
  EXPECT_EQ(options.max_frames, std::numeric_limits<std::size_t>::max());
}

TEST(TaskRegistryDebugTest, ResetStacktraceOptionsReadsEnvironment)
{
  StacktraceOptionsGuard guard;
  EnvVarGuard frames("UNI20_DEBUG_DAG_STACKTRACE_FRAMES", "0");

  TaskRegistry::reset_stacktrace_options();
  EXPECT_EQ(TaskRegistry::stacktrace_options().max_frames, 0U);
}

TEST(TaskRegistryDebugTest, RequestedGraphvizDumpIsServicedByCaller)
{
  auto dir = make_temp_dir("uni20-dag-request-test");
  TaskRegistry::GraphvizDumpOptions options;
  options.output_dir = dir.string();
  options.file_prefix = "requested";

  TaskRegistry::request_graphviz_dump();
  EXPECT_TRUE(TaskRegistry::service_debug_requests(options));
  EXPECT_TRUE(wait_for_dot_file(dir, std::chrono::milliseconds(200)));
  std::filesystem::remove_all(dir);
}

TEST(TaskRegistryDebugTest, DebugSchedulerServicesQueuedGraphvizDumpRequests)
{
  auto dir = make_temp_dir("uni20-dag-scheduler-request-test");
  EnvVarGuard output_dir("UNI20_DEBUG_DAG_OUTPUT_DIR", dir.string());
  EnvVarGuard prefix("UNI20_DEBUG_DAG_FILE_PREFIX", "scheduler");

  TaskRegistry::request_graphviz_dump();
  DebugScheduler sched;
  sched.run_all();

  EXPECT_TRUE(wait_for_dot_file(dir, std::chrono::milliseconds(200)));
  std::filesystem::remove_all(dir);
}

namespace
{
std::size_t count_dot_files(std::filesystem::path const& dir)
{
  std::size_t count = 0;
  for (auto const& entry : std::filesystem::directory_iterator(dir))
    count += entry.path().extension() == ".dot";
  return count;
}

bool wait_for_dot_files(std::filesystem::path const& dir, std::size_t count, std::chrono::milliseconds timeout)
{
  auto const deadline = std::chrono::steady_clock::now() + timeout;
  while (std::chrono::steady_clock::now() < deadline)
  {
    if (count_dot_files(dir) >= count) return true;
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
  }
  return false;
}
} // namespace

// The request file is not removed, so that it reaches every process watching
// it, and each change of its modification time asks for a dump.
TEST(TaskRegistryDebugTest, DiagnosticsServiceDumpsOnEachChangeOfTheRequestFile)
{
  TaskRegistry::stop_diagnostics_service();
  auto dir = make_temp_dir("uni20-dag-control-test");
  auto request_file = dir / "dump.request";
  {
    std::ofstream request(request_file);
  }
  auto const created = std::filesystem::last_write_time(request_file);

  TaskRegistry::DiagnosticsServiceOptions options;
  options.dump_options.output_dir = dir.string();
  options.dump_options.file_prefix = "control";
  options.request_file = request_file.string();
  options.poll_interval_ms = 10;

  ASSERT_TRUE(TaskRegistry::start_diagnostics_service(options));
  EXPECT_FALSE(TaskRegistry::start_diagnostics_service(options));
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(count_dot_files(dir), 0u) << "a file that existed before the service started asked for a dump";

  // Explicit times, so that the filesystem's timestamp resolution cannot merge the two changes.
  std::filesystem::last_write_time(request_file, created + std::chrono::seconds(1));
  EXPECT_TRUE(wait_for_dot_files(dir, 1, std::chrono::milliseconds(1000)));
  EXPECT_TRUE(std::filesystem::exists(request_file));
  std::filesystem::last_write_time(request_file, created + std::chrono::seconds(2));
  EXPECT_TRUE(wait_for_dot_files(dir, 2, std::chrono::milliseconds(1000)));
  TaskRegistry::stop_diagnostics_service();
  std::filesystem::remove_all(dir);
}

#if defined(__linux__)
namespace
{
// Each signal probe runs in a child process of its own (death-test style
// "threadsafe" runs the test binary again for just that test). A process gives
// its signal to the service only once, and the child has no thread from an
// earlier test that could take the signal in place of the service thread. A
// probe returns 0 and explains a failure on stderr.

class TaskRegistryDebugServiceDeathTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
      // Re-exec the probe with no environment-started diagnostics worker.
      signal_.unset();
      request_file_.unset();
      GTEST_FLAG_SET(death_test_style, "threadsafe");
    }

  private:
    EnvVarGuard signal_{"UNI20_DEBUG_DAG_SIGNAL"};
    EnvVarGuard request_file_{"UNI20_DEBUG_DAG_REQUEST_FILE"};
};

int probe_failed(char const* reason)
{
  std::fprintf(stderr, "%s\n", reason);
  return 1;
}

TaskRegistry::DiagnosticsServiceOptions signal_service_options(std::filesystem::path const& dir, int signal)
{
  TaskRegistry::DiagnosticsServiceOptions options;
  options.dump_options.output_dir = dir.string();
  options.dump_options.file_prefix = "signal";
  options.signal_number = signal;
  options.poll_interval_ms = 20;
  return options;
}

// Starts the service, sends the signal with `send`, and waits for the DOT file.
template <class Send> int signal_dump_probe(int signal, Send send)
{
  auto dir = make_temp_dir("uni20-dag-signal-test");
  if (!TaskRegistry::start_diagnostics_service(signal_service_options(dir, signal)))
    return probe_failed("the service did not start");
  if (send() != 0) return probe_failed("the signal was not sent");
  bool const dumped = wait_for_dot_file(dir, std::chrono::milliseconds(1000));
  TaskRegistry::stop_diagnostics_service();
  std::filesystem::remove_all(dir);
  return dumped ? 0 : probe_failed("no DOT file after the signal");
}

int signal_to_process_probe(int signal)
{
  return signal_dump_probe(signal, [signal] { return ::kill(::getpid(), signal); });
}

// A thread that exists before the service starts and does not block the
// signal, whatever mask the test was launched with, as a TBB worker would be.
// The signal is sent to it directly, so its default action would end the process.
int signal_to_earlier_thread_probe()
{
  std::promise<void> ready;
  std::jthread bystander([&ready](std::stop_token stop) {
    sigset_t set;
    sigemptyset(&set);
    sigaddset(&set, SIGUSR1);
    ::pthread_sigmask(SIG_UNBLOCK, &set, nullptr);
    ready.set_value();
    std::mutex mutex;
    std::condition_variable_any idle;
    std::unique_lock lock(mutex);
    idle.wait(lock, stop, [] { return false; });
  });
  ready.get_future().wait();
  return signal_dump_probe(SIGUSR1, [&bystander] { return ::pthread_kill(bystander.native_handle(), SIGUSR1); });
}

std::ptrdiff_t thread_count()
{
  return std::distance(std::filesystem::directory_iterator("/proc/self/task"), std::filesystem::directory_iterator());
}

// The only thread blocks the signal, as in a program started with it blocked,
// and the service thread inherits that mask. Only the service thread can then
// receive the signal.
int signal_with_starter_blocking_it_probe()
{
  if (thread_count() != 1) return probe_failed("another thread could receive the signal");
  sigset_t set;
  sigemptyset(&set);
  sigaddset(&set, SIGUSR1);
  if (::pthread_sigmask(SIG_BLOCK, &set, nullptr) != 0) return probe_failed("could not block the signal");
  return signal_to_process_probe(SIGUSR1);
}

void program_handler(int) {}

int install_program_handler(int signal)
{
  struct sigaction handled = {};
  handled.sa_handler = program_handler;
  sigemptyset(&handled.sa_mask);
  return ::sigaction(signal, &handled, nullptr);
}

bool signal_handler_is(int signal, void (*handler)(int))
{
  struct sigaction current = {};
  return ::sigaction(signal, nullptr, &current) == 0 && current.sa_handler == handler;
}

int restores_previous_action_probe()
{
  if (::signal(SIGUSR1, SIG_IGN) == SIG_ERR) return probe_failed("could not ignore the signal");
  TaskRegistry::DiagnosticsServiceOptions options;
  options.signal_number = SIGUSR1;
  if (!TaskRegistry::start_diagnostics_service(options)) return probe_failed("the service did not start");
  TaskRegistry::stop_diagnostics_service();
  return signal_handler_is(SIGUSR1, SIG_IGN) ? 0 : probe_failed("stop did not restore the previous action");
}

// A handler the program installs while the service runs takes the signal back,
// and stopping the service leaves it in place.
int stop_leaves_a_later_handler_probe()
{
  TaskRegistry::DiagnosticsServiceOptions options;
  options.signal_number = SIGUSR1;
  if (!TaskRegistry::start_diagnostics_service(options)) return probe_failed("the service did not start");
  if (install_program_handler(SIGUSR1) != 0) return probe_failed("could not install a handler");
  TaskRegistry::stop_diagnostics_service();
  return signal_handler_is(SIGUSR1, program_handler) ? 0 : probe_failed("stop replaced the program's handler");
}

// A signal the program already handles, for example to checkpoint, stays its
// own, and the service can still take the other one.
int refuses_a_handled_signal_probe()
{
  if (install_program_handler(SIGUSR1) != 0) return probe_failed("could not install a handler");
  TaskRegistry::DiagnosticsServiceOptions options;
  options.signal_number = SIGUSR1;
  if (TaskRegistry::start_diagnostics_service(options))
  {
    TaskRegistry::stop_diagnostics_service();
    return probe_failed("the service took a signal the program handles");
  }
  if (!signal_handler_is(SIGUSR1, program_handler)) return probe_failed("the program's handler was replaced");
  options.signal_number = SIGUSR2;
  if (!TaskRegistry::start_diagnostics_service(options)) return probe_failed("the service did not take SIGUSR2");
  TaskRegistry::stop_diagnostics_service();
  return 0;
}

// One change of the file reaches two processes watching it.
int request_file_reaches_every_process_probe()
{
  auto dir = make_temp_dir("uni20-dag-broadcast-test");
  auto request_file = dir / "request";
  int ready[2];
  if (::pipe(ready) != 0) return probe_failed("pipe failed");
  pid_t watchers[2];
  for (int i = 0; i < 2; ++i)
  {
    watchers[i] = ::fork();
    if (watchers[i] < 0) return probe_failed("fork failed");
    if (watchers[i] == 0)
    {
      auto out = dir / ("watcher" + std::to_string(i));
      std::filesystem::create_directories(out);
      TaskRegistry::DiagnosticsServiceOptions options;
      options.dump_options.output_dir = out.string();
      options.request_file = request_file.string();
      options.poll_interval_ms = 10;
      bool const started = TaskRegistry::start_diagnostics_service(options);
      char const report = started ? '1' : '0';
      bool const reported = ::write(ready[1], &report, 1) == 1;
      bool const dumped = started && reported && wait_for_dot_file(out, std::chrono::milliseconds(2000));
      TaskRegistry::stop_diagnostics_service();
      std::_Exit(dumped ? 0 : 1);
    }
  }
  for (int i = 0; i < 2; ++i)
  {
    char report = '0';
    if (::read(ready[0], &report, 1) != 1 || report != '1') return probe_failed("a watcher did not start");
  }
  {
    std::ofstream request(request_file);
  }
  int failures = 0;
  for (pid_t watcher : watchers)
  {
    int status = 0;
    if (::waitpid(watcher, &status, 0) != watcher || !WIFEXITED(status) || WEXITSTATUS(status) != 0) ++failures;
  }
  std::filesystem::remove_all(dir);
  return failures == 0 ? 0 : probe_failed("a watcher did not dump");
}

int signal_taken_once_probe()
{
  TaskRegistry::DiagnosticsServiceOptions options;
  options.signal_number = SIGUSR1;
  if (!TaskRegistry::start_diagnostics_service(options)) return probe_failed("the first start failed");
  TaskRegistry::stop_diagnostics_service();
  for (int signal : {SIGUSR1, SIGUSR2})
  {
    options.signal_number = signal;
    if (TaskRegistry::start_diagnostics_service(options))
    {
      TaskRegistry::stop_diagnostics_service();
      return probe_failed("a second service took a signal");
    }
  }
  options.signal_number = 0;
  if (!TaskRegistry::start_diagnostics_service(options))
    return probe_failed("a service without a signal did not start");
  TaskRegistry::stop_diagnostics_service();
  return 0;
}

int environment_signal_probe()
{
  {
    EnvVarGuard signal("UNI20_DEBUG_DAG_SIGNAL", std::to_string(SIGFPE));
    if (TaskRegistry::start_diagnostics_service())
      return probe_failed("UNI20_DEBUG_DAG_SIGNAL accepted SIGFPE by number");
  }
  EnvVarGuard signal("UNI20_DEBUG_DAG_SIGNAL", std::to_string(SIGUSR2));
  if (!TaskRegistry::start_diagnostics_service())
    return probe_failed("UNI20_DEBUG_DAG_SIGNAL did not accept SIGUSR2 by number");
  TaskRegistry::stop_diagnostics_service();
  return 0;
}
} // namespace

TEST_F(TaskRegistryDebugServiceDeathTest, DiagnosticsServiceHandlesEitherSignal)
{
  EXPECT_EXIT({ std::_Exit(signal_to_process_probe(SIGUSR1)); }, ::testing::ExitedWithCode(0), "");
  EXPECT_EXIT({ std::_Exit(signal_to_process_probe(SIGUSR2)); }, ::testing::ExitedWithCode(0), "");
}

TEST_F(TaskRegistryDebugServiceDeathTest, DiagnosticsServiceSignalReachesItFromThreadsStartedEarlier)
{
  EXPECT_EXIT({ std::_Exit(signal_to_earlier_thread_probe()); }, ::testing::ExitedWithCode(0), "");
}

TEST_F(TaskRegistryDebugServiceDeathTest, DiagnosticsServiceReceivesTheSignalWhenItsStarterBlocksIt)
{
  EXPECT_EXIT({ std::_Exit(signal_with_starter_blocking_it_probe()); }, ::testing::ExitedWithCode(0), "");
}

TEST_F(TaskRegistryDebugServiceDeathTest, DiagnosticsServiceRestoresThePreviousSignalAction)
{
  EXPECT_EXIT({ std::_Exit(restores_previous_action_probe()); }, ::testing::ExitedWithCode(0), "");
  EXPECT_EXIT({ std::_Exit(stop_leaves_a_later_handler_probe()); }, ::testing::ExitedWithCode(0), "");
}

TEST_F(TaskRegistryDebugServiceDeathTest, DiagnosticsServiceRefusesASignalTheProgramHandles)
{
  EXPECT_EXIT({ std::_Exit(refuses_a_handled_signal_probe()); }, ::testing::ExitedWithCode(0), "");
}

TEST_F(TaskRegistryDebugServiceDeathTest, DiagnosticsServiceRequestFileReachesEveryProcessWatchingIt)
{
  EXPECT_EXIT({ std::_Exit(request_file_reaches_every_process_probe()); }, ::testing::ExitedWithCode(0), "");
}

// "%p" names the process, so that one process among several can be asked.
TEST(TaskRegistryDebugTest, DiagnosticsServiceRequestFileNamesTheProcess)
{
  TaskRegistry::stop_diagnostics_service();
  auto dir = make_temp_dir("uni20-dag-request-pid-test");
  TaskRegistry::DiagnosticsServiceOptions options;
  options.dump_options.output_dir = dir.string();
  options.request_file = (dir / "request.%p.%%").string();
  options.poll_interval_ms = 10;
  ASSERT_TRUE(TaskRegistry::start_diagnostics_service(options));
  {
    std::ofstream other(dir / ("request." + std::to_string(::getpid() + 1) + ".%"));
  }
  std::this_thread::sleep_for(std::chrono::milliseconds(100));
  EXPECT_EQ(count_dot_files(dir), 0u) << "another process's request file asked for a dump";
  {
    std::ofstream request(dir / ("request." + std::to_string(::getpid()) + ".%"));
  }
  EXPECT_TRUE(wait_for_dot_file(dir, std::chrono::milliseconds(1000)));
  TaskRegistry::stop_diagnostics_service();
  std::filesystem::remove_all(dir);
}

TEST_F(TaskRegistryDebugServiceDeathTest, DiagnosticsServiceTakesASignalOncePerProcess)
{
  EXPECT_EXIT({ std::_Exit(signal_taken_once_probe()); }, ::testing::ExitedWithCode(0), "");
}

TEST_F(TaskRegistryDebugServiceDeathTest, DiagnosticsServiceSignalEnvironmentRefusesOtherSignals)
{
  EXPECT_EXIT({ std::_Exit(environment_signal_probe()); }, ::testing::ExitedWithCode(0), "");
}

// A fault signal such as SIGFPE returns to the faulting instruction after a
// handler, which traps again; the others belong to the program. start() refuses
// them before changing anything, so this runs in the test process.
TEST(TaskRegistryDebugTest, DiagnosticsServiceRefusesSignalsOtherThanUsr1AndUsr2)
{
  TaskRegistry::stop_diagnostics_service();
  for (int signal : {SIGFPE, SIGSEGV, SIGBUS, SIGILL, SIGINT, SIGTERM, SIGHUP, SIGCHLD, SIGRTMIN, -1})
  {
    struct sigaction before = {};
    if (signal > 0)
    {
      ASSERT_EQ(::sigaction(signal, nullptr, &before), 0);
    }
    TaskRegistry::DiagnosticsServiceOptions options;
    options.signal_number = signal;
    EXPECT_FALSE(TaskRegistry::start_diagnostics_service(options)) << "signal " << signal;
    TaskRegistry::stop_diagnostics_service();
    struct sigaction after = {};
    if (signal > 0)
    {
      ASSERT_EQ(::sigaction(signal, nullptr, &after), 0);
    }
    EXPECT_EQ(after.sa_handler, before.sa_handler) << "signal " << signal;
  }
}
#endif
#endif
