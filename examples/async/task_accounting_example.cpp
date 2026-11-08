#include <uni20/async/accounted_task.hpp>
#include <uni20/async/tbb_scheduler.hpp>

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string_view>
#include <vector>

using namespace uni20::async;

AccountedTask prepare(AccountingOptions, WriteBuffer<std::uint64_t> output) { co_await output = 7; }

AccountedTask calculate(AccountingOptions, ReadBuffer<std::uint64_t> input, WriteBuffer<std::uint64_t> output,
                        bool fail)
{
  auto const& seed = co_await input;
  auto destination = co_await output;
  std::uint64_t value = seed;
  for (unsigned i = 0; i < 20000; ++i)
    value = value * 6364136223846793005ULL + 1442695040888963407ULL;
  if (fail) throw std::runtime_error("deliberate rejection after computing the candidate");
  destination = value;
}

void print_metric(std::string_view name, std::optional<AccountingDuration> value)
{
  std::cout << ' ' << name << '=';
  if (value)
    std::cout << std::chrono::duration<double, std::micro>(*value).count() << " us";
  else
    std::cout << "unavailable/not requested";
}

void print_report(std::string_view name, AccountingReport const& report)
{
  std::cout << name << ": " << (report.outcome == AccountingOutcome::returned ? "returned" : "failed")
            << ", intervals=" << report.intervals;
  print_metric("elapsed", report.elapsed);
  print_metric("active wall", report.active_wall);
  print_metric("own CPU", report.own_cpu);
  std::cout << '\n';
}

int main()
{
  std::cout << "Selective coroutine accounting: three successful calculations use the same input\n"
               "with elapsed-only, active-wall and CPU-plus-wall measurements. A fourth attempt\n"
               "deliberately fails after computing; its partial measurements remain readable.\n"
               "Consumers are submitted before one shared preparation task. Elapsed includes\n"
               "dependency waiting; active measurements exclude suspension. Preparation is\n"
               "reported once. These timings are illustrative, not an overhead benchmark.\n\n";

  Async<std::uint64_t> input;
  std::vector<Async<std::uint64_t>> outputs(4);
  std::vector<AccountingReceipt> receipts;
  // External data outlive the scheduler, which retires final activations.
  TbbScheduler scheduler(2);
  auto seed = prepare({}, input.write());
  auto preparation = seed.receipt();
  AccountingMode const modes[]{AccountingMode::elapsed, AccountingMode::active_wall, AccountingMode::cpu_and_wall,
                               AccountingMode::cpu_and_wall};
  char const* const labels[]{"elapsed-only", "active-wall", "CPU-plus-wall", "failed attempt"};
  for (std::size_t i = 0; i < outputs.size(); ++i)
  {
    auto task = calculate({.mode = modes[i]}, input.read(), outputs[i].write(), i == 3);
    receipts.push_back(task.receipt());
    scheduler.schedule(std::move(task));
  }
  scheduler.schedule(std::move(seed));

  std::optional<std::uint64_t> expected;
  for (std::size_t i = 0; i < outputs.size(); ++i)
  {
    bool failed = false;
    try
    {
      auto value = outputs[i].get_wait(scheduler);
      if (expected && *expected != value) throw std::logic_error("accounting changed the numerical result");
      expected = value;
      std::cout << labels[i] << " result=" << value << '\n';
    }
    catch (std::runtime_error const& error)
    {
      if (i != 3) throw;
      failed = true;
      std::cout << "Expected numerical failure: " << error.what() << '\n';
    }
    if (failed != (i == 3)) throw std::logic_error("unexpected calculation outcome");
    auto report = receipts[i].wait(scheduler);
    auto outcome = failed ? AccountingOutcome::failed : AccountingOutcome::returned;
    if (report.outcome != outcome) throw std::logic_error("incorrect accounting outcome");
    print_report(labels[i], report);
  }
  print_report("shared preparation", preparation.wait(scheduler));
  std::cout << "\nA timing receipt observes statistics, not frame destruction. This example\n"
               "keeps external resources alive until the scheduler leaves scope.\n";
}
