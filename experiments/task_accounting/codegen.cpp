#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>
#if UNI20_ACCOUNTING_PROBE
#include "accounting.hpp"
#endif

using namespace uni20::async;

AsyncTask ordinary_accounting_comparison(ReadBuffer<int> input, WriteBuffer<int> output)
{
  auto const& value = co_await input;
  co_await output = value + 1;
}
