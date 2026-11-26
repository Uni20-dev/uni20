// Costs of reading and writing an async value once its epoch is ready: the exception checks
// that each access makes (#185, #166) and the scheduling of a short coroutine.
#include <benchmark/benchmark.h>
#include <uni20/async/async.hpp>
#include <uni20/async/awaiters.hpp>
#include <uni20/async/debug_scheduler.hpp>

#include <memory>
#include <vector>

using namespace uni20::async;

namespace
{

// ---------------------------- Reads ----------------------------

// One read through a buffer the caller already holds.
void AccessReadHeldBuffer(benchmark::State& state)
{
  DebugScheduler sched;
  Async<int> x = 1;
  auto reader = x.read();
  for (auto _ : state)
  {
    int value = reader.get_wait(sched);
    benchmark::DoNotOptimize(value);
  }
}
BENCHMARK(AccessReadHeldBuffer);

// One read that takes a new buffer each time.
void AccessReadyGetWait(benchmark::State& state)
{
  DebugScheduler sched;
  Async<int> x = 1;
  for (auto _ : state)
  {
    int value = x.get_wait(sched);
    benchmark::DoNotOptimize(value);
  }
}
BENCHMARK(AccessReadyGetWait);

// Several threads read one epoch, each through its own buffer.
struct SharedEpoch
{
    Async<int> value = 1;
    std::vector<ReadBuffer<int>> readers;
};
std::unique_ptr<SharedEpoch> shared_epoch;

void make_shared_epoch(benchmark::State const& state)
{
  shared_epoch = std::make_unique<SharedEpoch>();
  for (int i = 0; i < state.threads(); ++i)
    shared_epoch->readers.push_back(shared_epoch->value.read());
}

void drop_shared_epoch(benchmark::State const&) { shared_epoch.reset(); }

void AccessReadHeldBufferThreads(benchmark::State& state)
{
  DebugScheduler sched;
  auto const& reader = shared_epoch->readers[state.thread_index()];
  for (auto _ : state)
  {
    int value = reader.get_wait(sched);
    benchmark::DoNotOptimize(value);
  }
}
BENCHMARK(AccessReadHeldBufferThreads)
    ->Setup(make_shared_epoch)
    ->Teardown(drop_shared_epoch)
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->UseRealTime();

// The same, through an owning read proxy that a coroutine holds, which skips the readiness check.
AsyncTask read_through_proxy(ReadBuffer<int> reader, benchmark::State& state)
{
  auto proxy = co_await reader.transfer();
  for (auto _ : state)
  {
    int value = proxy.get();
    benchmark::DoNotOptimize(value);
  }
}

void AccessHeldReadProxyGetThreads(benchmark::State& state)
{
  DebugScheduler sched;
  sched.schedule(read_through_proxy(shared_epoch->readers[state.thread_index()], state));
  sched.run_all();
}
BENCHMARK(AccessHeldReadProxyGetThreads)
    ->Setup(make_shared_epoch)
    ->Teardown(drop_shared_epoch)
    ->Threads(1)
    ->Threads(2)
    ->Threads(4)
    ->UseRealTime();

// ------------------- Writes with a readback --------------------

AsyncTask write_borrowed(WriteBuffer<int> w, int value) { co_await w = value; }

AsyncTask write_owning(WriteBuffer<int> w, int value)
{
  auto proxy = co_await w.transfer();
  proxy = value;
}

AsyncTask write_with_write_to(WriteBuffer<int> w, int value) { co_await write_to(std::move(w), value); }

AsyncTask add_in_place(WriteBuffer<int> w, int value) { co_await w += value; }

// Schedules one writer coroutine per iteration and reads the result back.
template <AsyncTask (*Write)(WriteBuffer<int>, int)> void write_and_read_back(benchmark::State& state)
{
  DebugScheduler sched;
  Async<int> x = 0;
  int i = 0;
  for (auto _ : state)
  {
    sched.schedule(Write(x.write(), ++i));
    sched.run_all();
    int value = x.get_wait(sched);
    benchmark::DoNotOptimize(value);
  }
}
BENCHMARK(write_and_read_back<write_borrowed>)->Name("AccessBorrowedWriteAndReadback");
BENCHMARK(write_and_read_back<write_owning>)->Name("AccessOwningWriteAndReadback");
BENCHMARK(write_and_read_back<write_with_write_to>)->Name("AccessWriteToAndReadback");
BENCHMARK(write_and_read_back<add_in_place>)->Name("AccessInPlaceAddAndReadback");

// ------------- In-place access through a held proxy -------------

AsyncTask update_borrowed_proxy(WriteBuffer<int> w, benchmark::State& state)
{
  auto proxy = co_await w;
  for (auto _ : state)
    benchmark::DoNotOptimize(++proxy.get());
}

AsyncTask update_owning_proxy(WriteBuffer<int> w, benchmark::State& state)
{
  auto proxy = co_await w.transfer();
  for (auto _ : state)
    benchmark::DoNotOptimize(++proxy.get());
}

AsyncTask add_through_proxy(WriteBuffer<int> w, benchmark::State& state)
{
  auto proxy = co_await w;
  for (auto _ : state)
  {
    proxy += 1;
    benchmark::ClobberMemory();
  }
}

// Runs the whole timing loop inside one coroutine that holds the writer.
template <AsyncTask (*Update)(WriteBuffer<int>, benchmark::State&)> void update_held_proxy(benchmark::State& state)
{
  DebugScheduler sched;
  Async<int> x = 0;
  sched.schedule(Update(x.write(), state));
  sched.run_all();
  int value = x.get_wait(sched);
  benchmark::DoNotOptimize(value);
}
BENCHMARK(update_held_proxy<update_borrowed_proxy>)->Name("AccessHeldBorrowedProxyGet");
BENCHMARK(update_held_proxy<update_owning_proxy>)->Name("AccessHeldOwningProxyGet");
BENCHMARK(update_held_proxy<add_through_proxy>)->Name("AccessHeldProxyAdd");

} // namespace
