# Async task accounting experiment

This opt-in experiment tests a separate coroutine promise against the existing
Uni20 runtime. It is not an installed API. See the
[design and scope](../../docs/async/task_accounting_design.md).

`accounting.hpp` derives a separate promise from `TaskPromiseBase`, wraps the
base's transformed awaiters, and publishes a retained `Async<Report>`
independently of numerical outputs. Ordinary task frames and awaiters contain
no accounting state or branches. There is no timing registry, thread-local
accounting state, or per-resumption allocation.

The scheduler API now admits any movable `HostTask`, checked against its concrete
promise's host domain. Both ordinary `AsyncTask` and the experimental `Task<>`
use the same initial scheduling implementation. Submit it directly:

```cpp
auto task = measured_stage(options, input.read(), output.write());
auto receipt = task.receipt();
scheduler.schedule(std::move(task)); // or schedule(...) with ScopedScheduler
```

The experimental ownership-envelope adapter has been removed. No extra
coroutine, promise cast, accounting-specific scheduler virtual function or
runtime accounting flag is involved. NUMA selection, pausing and task-registry
events remain part of initial admission. `TbbNumaScheduler::schedule(task, node)`
and the global helper accept the same host task types. Initial host admission
still rejects CUDA and erased `BasicTask` types.

## Measurements and boundaries

| Mode | Measured values | Clock calls |
|---|---|---|
| `elapsed` | Creation-to-sealing latency and interval count | Two wall samples |
| `active_wall` | Elapsed and sum of active interval wall times | Two elapsed samples plus two wall samples per interval |
| `cpu_and_wall` | Both wall metrics and own thread CPU | Above, plus two thread CPU samples per interval |

An interval ends before delegating a suspending await and starts on resumption.
Ready awaits do not split it. Child work reached through `co_await` is excluded
from the parent's active metrics, even with direct coroutine transfer. An
interval never subtracts CPU samples from different OS threads. New intervals
may start on different threads.

Elapsed starts during promise construction, after report-state allocation,
and ends just before report publication. It includes later queueing and
waiting, but excludes coroutine/report allocation, publication, and destruction.
It is **not automatically Bethe's attempt duration**. The application defines
attempt boundaries; a simple external monotonic timer remains sufficient for
Bethe's first cost predictor. Concurrent elapsed reports do not sum to campaign
wall time. CPU/active-wall instrumentation includes its own hook overhead and
cannot measure parallel BLAS work on other threads.

Unavailable clocks yield an absent metric, not zero or numerical failure.
`returned` describes normal coroutine return, not scientific success. An
exception yields `failed`; an explicitly cancelled task yields `cancelled`.
A destroyed parent whose cancellation flag was never set yields `interrupted`.
`body_entered` distinguishes pre-start destruction from partial execution.

**A receipt observes sealed statistics, not resource teardown.** Finalization
can publish a report before coroutine parameters are destroyed. Ordinary
successful/exceptional `co_await child` completion follows destruction of the
child frame. Keep scheduler and external resource owners alive under Uni20's
existing lifetime rules; receipt readiness alone is insufficient.

## Caller contract for CPU attribution

CPU accounting measures this coroutine's active **OS-thread CPU intervals**,
not whole-process CPU consumption or all resources used by an operation.

- Express cooperative waits and child work through `co_await`. A synchronous
  `get_wait()`, recursive scheduler run, or nested TBB operation can execute
  unrelated work or change resumable stacks without an accounting boundary;
  these are outside the supported attribution contract.
- Use single-threaded computational stages when interpreting `own_cpu` as the
  stage's CPU cost. Work on spawned threads, parallel BLAS workers, subprocesses
  and devices is not included. Their CPU/device usage needs separate reporting.
- An ordinary blocking OS call that neither runs other tasks nor changes stacks
  does not charge blocked time to the thread CPU clock; active wall time does
  include that wait. Such wall time must not be interpreted as CPU demand.

These are caller responsibilities, not runtime-enforced restrictions. The
prototype does not inspect arbitrary callees, intercept thread/process creation,
or claim to detect every violation. A different thread at interval endpoints
invalidates the CPU measurement as a diagnostic, but matching endpoints cannot
prove that no unsupported work or stack migration occurred between them.
Elapsed-only timing remains a duration for its documented boundaries even when
the body uses other resources; it does not become a CPU estimate automatically.

The first coroutine argument must be `Options<Clock>`; retrieve the receipt
before submitting or otherwise consuming the task. Coroutines use ordinary
`ReadBuffer` and `WriteBuffer` parameters and captureless `static` coroutine
lambdas. Inclusive DAG accounting and a synchronous-wait exclusion helper remain
outside this experiment.

## Build and run

Use a configured host toolchain and dependencies as described in `AGENTS.md`
and any local overrides. For example:

```sh
mkdir -p build_codex/accounting-release
cmake -S . -B build_codex/accounting-release \
  -DCMAKE_BUILD_TYPE=Release \
  -DUNI20_BUILD_TASK_ACCOUNTING_EXPERIMENT=ON \
  -DUNI20_BUILD_TESTS=ON -DUNI20_BUILD_COMBINED_TESTS=OFF \
  -DUNI20_BUILD_PYTHON=OFF -DUNI20_BUILD_EXAMPLES=OFF \
  -DUNI20_BUILD_BENCH=OFF -DUNI20_ENABLE_LTO=OFF
cmake --build build_codex/accounting-release --target \
  uni20_task_accounting_experiment_tests \
  uni20_accounting_bench_0 uni20_accounting_bench_1 \
  uni20_accounting_codegen_0 uni20_accounting_codegen_1 \
  uni20_accounting_allocations_0 uni20_accounting_allocations_1
ctest --test-dir build_codex/accounting-release \
  -R '^AccountingExperiment\.' --output-on-failure
```

The switch defaults to OFF and adds only experimental targets. It defines no
macro or layout policy in installed headers. Benchmark suffix `0` excludes the
experimental definitions; suffix `1` includes them. Within benchmark `1`, mode
`-1` still runs ordinary tasks, while modes `0`, `1`, `2` select elapsed,
active-wall, and CPU-plus-wall accounting. Each run emits CSV to stdout.

Benchmarks include submission, execution, numerical output retrieval, and (for
accounted tasks) report retrieval. They compare empty tasks, 32 direct
self-transfers, and a single-threaded integer recurrence with an observable
checksum. Batches are bounded at 256 tasks. Run off/on variants separately on
an otherwise quiet machine; do not benchmark during compilation or sanitizer
runs. The allocation executable is separate so replacement `new` does not
perturb throughput. Its counts cover creation only, and its first allocation
is the observed coroutine-frame allocation, not a portable size guarantee.

GCC 13 currently emits its coroutine/atomic `-Wstringop-overflow` warning for
both ordinary and accounted paths at optimization. The local measurement build
uses `-Wno-error=stringop-overflow`, leaving the warning visible. This experiment
does not change repository warning policy. Debug ASan/UBSan runs use the
repository's existing `tests/sanitizers/lsan.supp` for deliberately retained DAG
metadata; compiled oneTBB implementation code retains the existing sanitizer
exclusion. Public oneTBB headers and Uni20 code remain instrumented.
