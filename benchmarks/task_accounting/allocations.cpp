#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>
#if UNI20_ACCOUNTING_PROBE
#include <uni20/async/accounted_task.hpp>
#endif
#include <cstdlib>
#include <iostream>
#include <new>

// Separate executable: allocation instrumentation never enters the throughput
// benchmark. This counts ordinary (not over-aligned) allocations during creation.
namespace
{
bool counting = false;
std::size_t calls = 0, bytes = 0, first = 0;
} // namespace
[[gnu::noinline]] void* operator new(std::size_t size)
{
  if (counting)
  {
    if (calls == 0) first = size;
    ++calls;
    bytes += size;
  }
  if (auto ptr = std::malloc(size ? size : 1)) return ptr;
  throw std::bad_alloc();
}
[[gnu::noinline]] void operator delete(void* ptr) noexcept { std::free(ptr); }
[[gnu::noinline]] void operator delete(void* ptr, std::size_t) noexcept { std::free(ptr); }
using namespace uni20::async;
AsyncTask plain_allocation() { co_return; }
#if UNI20_ACCOUNTING_PROBE
namespace ae = uni20::async;
ae::AccountedTask accounted_allocation(ae::AccountingOptions) { co_return; }
#endif
int main()
{
  // Warm any one-time runtime initialization outside the measured region.
  {
    auto t = plain_allocation();
    t.set_cancel_on_resume();
  }
#if UNI20_ACCOUNTING_PROBE
  {
    auto t = accounted_allocation({});
    t.set_cancel_on_resume();
  }
#endif
  auto measure = [](char const* name, auto make) {
    calls = bytes = first = 0;
    counting = true;
    auto t = make();
    counting = false;
    std::cout << name << ',' << calls << ',' << bytes << ',' << first << '\n';
    t.set_cancel_on_resume();
  };
  std::cout << "task,creation_allocations,total_bytes,first_allocation_bytes\n";
  measure("ordinary", [] { return plain_allocation(); });
#if UNI20_ACCOUNTING_PROBE
  measure("elapsed", [] { return accounted_allocation({.mode = ae::AccountingMode::elapsed}); });
  measure("cpu_and_wall", [] { return accounted_allocation({.mode = ae::AccountingMode::cpu_and_wall}); });
#endif
}
