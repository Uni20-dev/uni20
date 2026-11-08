# Optional task accounting

Include `<uni20/async/accounted_task.hpp>` and return `AccountedTask` to measure
selected host coroutines. Ordinary `AsyncTask` uses its existing promise and has
no accounting state, clock calls or accounting branches, even when this header
is included. A separate promise owns the selected task's measurements.

```cpp
using namespace uni20::async;

AccountedTask calculate(AccountingOptions options, ReadBuffer<double> input,
                        WriteBuffer<double> output)
{
  auto const& x = co_await input;
  co_await output = x * x;
}

// input, output and other resources must outlive the work using them.
auto task = calculate({.mode = AccountingMode::cpu_and_wall},
                      input.read(), output.write());
auto receipt = task.receipt();
scheduler.schedule(std::move(task));
auto report = receipt.wait(scheduler);
```

`AccountingOptions` must be the first coroutine argument; the promise consumes
its mode before initial suspension. The task can also be awaited as a child,
submitted through the global `schedule(...)` helper, or submitted with an
explicit NUMA node. Captureless coroutine lambdas use `static -> AccountedTask`.
Retrieve the receipt before moving or awaiting the task.

## Measurements

| Mode | Measurements | Clock samples |
|---|---|---|
| `elapsed` (default) | Elapsed latency and interval count | Two wall samples |
| `active_wall` | Elapsed latency and active wall time | Above, plus two wall samples per active interval |
| `cpu_and_wall` | Both wall metrics and own thread CPU | Above, plus two CPU samples per active interval |

An active interval starts on coroutine entry or resumption and ends before a
suspending await or finalization. Ready awaits do not split it. Child work is
excluded from the parent's active metrics, including direct coroutine transfers.
CPU samples are subtracted only within an interval on the same OS thread;
a later interval can run on a different thread. Active wall time includes OS
descheduling and ordinary blocking within the interval.

`elapsed` starts during promise construction, after report-state allocation,
and ends immediately before report publication. It includes later queueing and
dependency waits, but excludes initial frame/report allocation, publication and
destruction. It is latency at these boundaries, not automatically an application's
attempt cost. Concurrent elapsed measurements do not sum to campaign wall time.
Subtracting active from elapsed does not separately identify queue and dependency
waiting. An application can retain its own attempt timer when its boundaries differ.

`AccountingReport` contains optional `elapsed`, `active_wall` and `own_cpu`
durations in nanoseconds, `intervals`, `body_entered`, and an `outcome`:

- `returned`: normal coroutine return, without implying scientific acceptance.
- `failed`: an exception escaped the coroutine.
- `cancelled`: the task's explicit cancellation flag was set.
- `interrupted`: destruction before completion without that flag.

`body_entered` means coroutine execution began. A dependency can throw before
an application kernel is invoked, so applications track kernel entry separately
if they need it. Failed or cancelled work retains partial measurements. Missing
or failed clocks produce an absent metric, not measured zero, and do not fail
the calculation. Zero before body entry is a valid measured cost when the
corresponding clock capability is available.

## Reports and lifetime

`AccountingReceipt` retains the report independently of numerical outputs and
the coroutine frame. Copying a receipt shares that observation. `receipt.read()`
returns an ordinary `ReadBuffer<AccountingReport>` that another task can
`co_await`; `receipt.wait(scheduler)` synchronously obtains a report copy.
A failed numerical output does not poison the report.

**Report readiness is not a resource-lifetime join.** Finalization publishes
statistics before coroutine parameters are destroyed. Ordinary successful or
exceptional `co_await child` completion follows child-frame destruction; use that
existing completion mechanism when resources must be released only after the
child is gone. Scheduler routes and external resource owners must remain alive
under the [runtime ownership rules](runtime_model.md). A scheduler owns queued
activations, not suspended tasks held by external awaiters.

## CPU attribution contract

`own_cpu` measures the coroutine's active OS-thread CPU intervals. It does not
attribute work on spawned threads, parallel BLAS workers, subprocesses or devices.
Use single-threaded stages when interpreting it as the stage's CPU cost.

Express cooperative waits through `co_await`. Synchronous nested `get_wait()`,
recursive scheduler execution and nested TBB operations can execute unrelated
work or change resumable stacks without an accounting boundary. They are outside
the supported attribution contract. An ordinary blocking OS call which does not
run other tasks is allowed: its waiting contributes to active wall time, not
thread CPU consumption.

These restrictions are caller responsibilities. The implementation does not
intercept thread creation or detect every unsupported nested operation. Different
threads at interval endpoints invalidate the CPU measurement; matching endpoints
alone do not prove that nothing unsupported occurred in between. Elapsed-only
measurement remains valid for its stated boundaries when other resources are used.

## Cost, clocks and examples

Accounting is selected by task type, so ordinary code needs no global enable
switch. Including the header does not activate instrumentation in `AsyncTask`.
Only selected tasks allocate report state and run timing hooks. There is no
per-resumption allocation, global timing map or thread-local accounting context.
Retained reporting currently adds four creation allocations beyond the ordinary
frame in the measured native configurations; this is not a portable allocation
count guarantee. Prefer meaningful computational stages over tiny coroutines.

`BasicAccountedTask<Clock>` and `BasicAccountingOptions<Clock>` permit deterministic
clock injection. A clock exposes `static constexpr bool has_cpu` and non-throwing
`wall()` / `cpu()` methods returning `std::optional<AccountingDuration>`. Sampling
or clock-state mutation must be safe for the task's execution pattern. Shared or
borrowed clock state must outlive the coroutine. `SystemAccountingClock` uses
`steady_clock` and, when available, `CLOCK_THREAD_CPUTIME_ID`; unavailable thread
CPU timing leaves wall measurements usable.

The [runnable example](../../examples/async/task_accounting_example.cpp) shows
all three modes, direct scheduler submission, shared preparation, dependency
waits and partial costs after failure. Build and run `task_accounting_example`.
The [benchmark tools](../../benchmarks/task_accounting/) compare ordinary tasks
with the accounting header excluded/included, allocation counts and generated
code. They do not turn timing differences on a shared machine into correctness
tests or promise zero overhead for accounted tasks.

The earlier separate-promise experiment is preserved in Git history. Its
mechanism and numerical consumer trial support selective stage accounting;
synchronous nested-wait exclusion, inclusive DAG totals, parallel CPU attribution
and lower-allocation reporting remain separate future work.
