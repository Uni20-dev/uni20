#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>
#include <uni20/async/tbb_scheduler.hpp>
#if UNI20_ACCOUNTING_PROBE
#include "accounting.hpp"
#endif
#include <chrono>
#include <cstdint>
#include <iostream>
#include <string_view>
#include <vector>

using namespace uni20::async;
struct SelfTransfer
{
    bool await_ready() const noexcept { return false; }
    BasicTask await_suspend(BasicTask task) const noexcept { return task; }
    void await_resume() const noexcept {}
};
std::uint64_t work(std::uint64_t value, unsigned n)
{
  for (unsigned i = 0; i < n; ++i)
    value = value * 6364136223846793005ULL + 1442695040888963407ULL;
  return value;
}
AsyncTask ordinary(WriteBuffer<std::uint64_t> output, unsigned n, unsigned intervals)
{
  auto value = work(9, n);
  for (unsigned i = 0; i < intervals; ++i)
  {
    co_await SelfTransfer{};
    value = work(value, n);
  }
  co_await output = value;
}
#if UNI20_ACCOUNTING_PROBE
namespace ae = uni20::async::accounting_experiment;
ae::Task<> accounted(ae::Options<> opts, WriteBuffer<std::uint64_t> output, unsigned n, unsigned intervals)
{
  (void)opts;
  auto value = work(9, n);
  for (unsigned i = 0; i < intervals; ++i)
  {
    co_await SelfTransfer{};
    value = work(value, n);
  }
  co_await output = value;
}
#endif

template <class Scheduler>
void measure(Scheduler& scheduler, std::string_view scheduler_name, int mode, unsigned work_count, unsigned transfers)
{
  constexpr unsigned batch_size = 256, batches = 16;
  std::uint64_t checksum = 0;
  auto start = std::chrono::steady_clock::now();
  for (unsigned batch = 0; batch < batches; ++batch)
  {
    std::vector<Async<std::uint64_t>> outputs(batch_size);
#if UNI20_ACCOUNTING_PROBE
    std::vector<ae::Receipt> receipts;
    receipts.reserve(batch_size);
#endif
    for (auto& output : outputs)
    {
#if UNI20_ACCOUNTING_PROBE
      if (mode >= 0)
      {
        auto t = accounted({.mode = static_cast<ae::Mode>(mode)}, output.write(), work_count, transfers);
        receipts.push_back(t.receipt());
        scheduler.schedule(std::move(t));
      }
      else
#endif
        scheduler.schedule(ordinary(output.write(), work_count, transfers));
    }
    scheduler.run_all();
    for (auto& output : outputs)
      checksum += output.get_wait(scheduler);
#if UNI20_ACCOUNTING_PROBE
    for (auto const& r : receipts)
      if (r.wait(scheduler).outcome != ae::Outcome::returned) std::terminate();
#endif
  }
  auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count();
  std::cout << UNI20_ACCOUNTING_PROBE << ',' << scheduler_name << ',' << mode << ',' << work_count << ',' << transfers
            << ',' << double(ns) / (batch_size * batches) << ',' << checksum << '\n';
}
int main()
{
  std::cout << "definitions_enabled,scheduler,mode,work_per_interval,self_transfers,ns_per_task,checksum\n";
  DebugScheduler debug;
  TbbScheduler serial(1), parallel(4);
  for (unsigned repeat = 0; repeat < 5; ++repeat)
  {
    for (auto [n, transfers] : {std::pair{0U, 0U}, std::pair{0U, 32U}, std::pair{20000U, 0U}})
    {
      // Rotate mode ordering between repetitions. Ordinary tasks stay identical
      // in both builds; enabled-but-unused is mode -1 in the enabled build.
      constexpr int modes = UNI20_ACCOUNTING_PROBE ? 4 : 1;
      for (int i = 0; i < modes; ++i)
      {
        int mode = (i + repeat) % modes - 1;
        measure(debug, "debug", mode, n, transfers);
        measure(serial, "tbb1", mode, n, transfers);
        measure(parallel, "tbb4", mode, n, transfers);
      }
    }
  }
}
