# Optional CPU and wall-time accounting for Async tasks

**Status: local feasibility experiment, 8 November 2026; no public accounting API.**
Prepared initially by Bethe and refined against Uni20
`6179f39db341349306b6c67769498e717a314a69`. Ian explicitly treats this as an
experiment: a separate promise is preferred **if it works**, not an agreed
architecture to implement regardless of evidence. Proposed type and option
names below are illustrative, not settled API decisions. The isolated probe
first established feasibility without production runtime edits. The subsequent
host-admission refinement generalizes the scheduler entry point; ordinary
promises and awaiters still contain no accounting state or branches. The
`UNI20_BUILD_TASK_ACCOUNTING_EXPERIMENT` build option adds only local experiment
targets and defaults to OFF.

## Purpose and agreed constraints

An adaptive Bethe search can choose expensive state evaluations using both
predicted spectral contribution and observed computational cost. Independent
root solves and determinant stages can run as bounded Async operations; each
completed observation updates subsequent admission decisions. Scientific
priority, exploration and accounting for failed solves belong in Bethe.
Uni20's reusable contribution would be optional execution measurements.

Ian supports build-time disabling because many applications need no timers.
Preserving the existing low coroutine/scheduler overhead is a primary
requirement. He also proposes a separate task type with a derived promise
that owns the accounting state. Both approaches should be evaluated before
choosing the implementation. This note does not assert a new HPX performance
comparison or authorize a Bethe numerical campaign.

Required properties:

- A fully disabled build has no accounting clock calls, extra task state,
  registry entries or accounting branches in ordinary task execution.
- In an enabled build, only explicitly selected tasks need timing. Benchmark
  any overhead imposed on ordinary tasks; do not assume a branch or a larger
  coroutine frame is free.
- No per-resumption allocation, global timing map, global lock, formatted log
  or dependency on expensive task-registry/stacktrace diagnostics.
- Suspension, direct transfers, cancellation and completion preserve the
  current ownership model. Timing must not keep a completed coroutine alive.
- Timing is observational. Clock failure must not alter numerical outcomes;
  missing or incomplete measurements must not masquerade as measured zero.

## First consumer boundary

Bethe's initial consumer needs selected, single-threaded root/determinant
stages, with dependencies expressed through `co_await`. It needs completed
attempt measurements, not live counter polling. Synchronous `get_wait()`
inside an accounted interval, parallel BLAS attribution, subprocess attribution
and automatic inclusive DAG totals are outside this first experiment. An
explicit synchronous-wait exclusion helper is a separate possible experiment,
not a prerequisite for this consumer.

A physical state may require multiple precision or recovery attempts. Preserve
the partial cost of each failed or cancelled attempt as well as successful
ones; report availability and interrupted/unknown outcomes truthfully. A task
that returns normally can still fail Bethe's numerical validation. Task
completion and scientific acceptance must remain distinct, and failure of a
numerical output must not make its accounting report unreadable.

Bethe supplies attempt identifiers and associates reports with model, system
size, state family, precision and recovery path. These are application metadata,
not additional promise fields. Shared preparation is a separately identified
operation counted once; Bethe decides its amortization and distinguishes it
from marginal evaluation cost. This does not require a generic cost ledger or
changes to scheduler policy.

For the first finite-N cost predictor, Ian has selected **monotonic elapsed
attempt time**. The current execution assumption is one available CPU per
single-threaded numerical worker, without oversubscription or meaningful
blocking; BLAS and OpenMP each use one thread. Under those conditions elapsed
time is a practical cost signal, not an assertion that wall and CPU time are
identical under contention. Available worker CPU reports remain separately
reported cross-checks. General per-task CPU attribution is not a prerequisite
for this consumer.

Bethe defines the attempt boundaries and retains costs of failed attempts.
Shared preparation and end-to-end campaign elapsed time remain separate.
Summing concurrent attempt durations may support an application cost budget,
but must not be labelled campaign wall time. A generic task's admission-to-
completion latency is not automatically the same as Bethe's elapsed attempt
time: any queueing, dependency, preparation or communication included by the
chosen boundaries must be explicit.

The existing worker measurements already unblock Bethe's search. This
experiment therefore has no consumer deadline requiring a premature API or
relaxation of ordinary-task overhead checks. Wall-only measurement is useful
independently of general CPU attribution; the separate-promise experiment and
its CPU-accounting questions retain their scope. These requirements do not
select public names, report types or admission APIs.

## What to measure

| Quantity | Proposed meaning | Important limitation |
|---|---|---|
| Own CPU time | Sum of thread CPU deltas during this coroutine's accounted execution intervals | Excludes execution on other threads and in child processes/devices; nested task accounting is described below |
| Active wall time | Sum of monotonic wall deltas over those same intervals | Includes OS descheduling while the task remains active |
| Task elapsed latency | Monotonic time from a documented initial admission point to completion | Includes queue/dependency delays; may differ from the application's attempt boundaries |
| Elapsed attempt time (consumer boundary) | Monotonic duration between Bethe's defined attempt start and end; selected as its initial cost signal | Includes everything inside those boundaries; is neither exclusive CPU nor campaign wall time |
| Execution intervals | Count of entered computational intervals | Makes suspension-related measurement overhead visible |
| Outcome/availability | Completed, failed, cancelled or never started, plus measurement availability | Partial costs of failed/cancelled work remain useful |

These are proposed semantics, not necessarily the final field list. Wall-only
and CPU-plus-wall modes can be separate options. Name the measured interval:
wall-only **elapsed** timing does not mean the sum of active coroutine
intervals. Elapsed-only timing needs boundary samples, not a clock read at
every await; active-wall timing needs the interval hooks. Retaining an interval
count does not itself require sampling a clock at each transition. Neither
wall-only variant should read a CPU clock.

Queue delay and dependency wait need more events to distinguish; do not infer
both from elapsed minus active time. Device timing and whole-operation parallel
resource accounting are later, separate features.

On Linux, `clock_gettime(CLOCK_THREAD_CPUTIME_ID)` measures CPU consumed by
the calling OS thread. `CLOCK_MONOTONIC` measures elapsed time. Sum deltas
within each interval on one thread, then accumulate across resumptions:

```text
own_cpu = sum(thread_cpu_at_exit - thread_cpu_at_entry)
```

Never subtract a start sample on one OS thread from an end sample on another.
OS CPU-core migration within one thread is different and does not invalidate
the thread clock. Uni20's existing `system_run_clock()` in
[`run_context.cpp`](../../src/uni20/common/run_context.cpp) uses
`getrusage(RUSAGE_SELF)` and measures the whole process, so its deltas cannot
attribute overlapping operations. See the
[Linux clock documentation](https://man7.org/linux/man-pages/man3/clock_gettime.3.html).
Use a portable capability boundary for other supported systems; an unavailable
CPU clock leaves wall timing usable and CPU timing explicitly unavailable.

## Current runtime seams

The [ownership model](runtime_model.md#coroutine-frame-ownership) and
[scheduler routing design](scheduler_migration.md) already separate typed
initial admission from promise-neutral continuation routing:

```text
AsyncTask -> AsyncTaskPromise --+
                               +-> TaskPromiseBase
CudaTask  -> CudaTaskPromise  --+

TaskHandle: actual coroutine identity + TaskPromiseBase pointer + domain
BasicTask: owning token used after admission and during suspension
```

Relevant code is in [`async_task.hpp`](../../src/uni20/async/async_task.hpp),
[`async_task_impl.hpp`](../../src/uni20/async/async_task_impl.hpp), and
[`async_task_promise.hpp`](../../src/uni20/async/async_task_promise.hpp).
`AsyncTask` and `AsyncTaskPromise` are currently final. A new concrete promise
would derive from `TaskPromiseBase`, or from a factored policy implementation,
rather than simply subclassing `AsyncTaskPromise`.

Scheduler activations call `TaskPromiseBase::resume_and_track()`. However,
nested task entry and return can transfer directly between coroutines without
another scheduler activation. `BasicTask::await_suspend()`,
`resolve_await_suspend_result()` and `final_suspend()` participate in these
transfers. Timing the outer `resume()` call would conflate several tasks and
cannot reliably recover exclusive per-task costs.

The current `note_running`/`note_suspended` calls are diagnostic transitions,
not already-correct accounting boundaries. Some suspension paths call their
diagnostic hook after publishing the task to an awaiter. Accounting must stop
before that publication: another thread may immediately resume the coroutine.
This is a design constraint, not a new assertion about the correctness of the
existing diagnostic contract.

## Options

| Option | Advantages | Costs and questions |
|---|---|---|
| A. Build-gated fields/hooks in the common promise | One implementation covers existing task types; runtime per-task selection is easy | Enabled builds add fields and checks to ordinary tasks; transition hooks need careful ownership placement |
| B. Separate accounting task/promise with typed awaiter instrumentation | Counters and clock calls can be confined to accounted coroutine frames; ordinary code may remain unchanged even in an enabled build | Needs typed admission and complete awaiter coverage; inheritance alone does not instrument erased runtime calls |
| C. Explicit synchronous-stage timer inside an ordinary task | Smallest immediate Bethe experiment; ordinary Async runtime untouched | Measures selected leaf work only, not arbitrary coroutine execution or dependencies |

**Recommended investigation order:** test the feasibility of B first, while
retaining A as the simpler common-runtime alternative. Both can be build-gated.
Use C if the application needs cost data before generic accounting is ready.
Do not commit to B's zero-overhead claim until code generation, frame sizes and
benchmarks establish it.

### A. Common promise accounting

A provisional `UNI20_ENABLE_ASYNC_ACCOUNTING` option, independent of debug
instrumentation, could compile in an optional state/flag and small hooks.
The running owner updates its own counters; final publication supplies the
synchronization for readers. A nullable external record avoids putting every
counter into ordinary frames, but introduces a pointer/branch and potentially
allocation. Inline fields avoid that allocation but enlarge every enabled
frame. Measure both effects rather than selecting by intuition.

Direct transfer needs outgoing-stop and incoming-start events. Ready awaiters
must not introduce false suspensions. No hook may access a frame after the
owning token is published or the frame is destroyed. Avoid adding a virtual
promise hierarchy solely to dispatch timing hooks.

### B. Accounted task and derived promise

An illustrative `AccountedTask` / `AccountedTaskPromise` retains the host
execution domain and normal dependency semantics. Accounting is not a new
hardware domain and must not force scheduler hops in place of valid direct
transfers.

There are two distinct implementation routes:

1. **Erased runtime hooks:** carry optional accounting state or callbacks
   through the common representation. This is effectively A with typed
   opt-in; it still changes the ordinary enabled-build path.
2. **Compiler-mediated typed hooks:** instrument the accounted promise's
   initial, ordinary-await and final protocol objects. This is the promising
   route for leaving ordinary task execution unchanged.

For the typed route, the intended timeline is:

| Boundary | Accounting action |
|---|---|
| `initial_suspend` awaiter's `await_resume()` | Start the first interval on the executing thread |
| Wrapped ordinary awaiter reports ready | Continue the current interval; do not restart its timer |
| Wrapped awaiter actually suspends | Stop before delegating ownership-transfer/publication |
| Wrapped awaiter's `await_resume()` | Restart on the current thread before returning/throwing into the coroutine |
| Delegated `await_suspend()` throws before publication | Restart in the wrapper's catch path, then rethrow; the language does not call `await_resume()` on this path |
| Final suspension | Stop and seal the record before delegating to frame-destroying runtime logic |
| Permitted destruction before first execution or while suspended | Publish whether the body entered and whether final suspension was reached, retaining partial costs exactly once |

This naturally handles an accounted parent awaiting an ordinary child:
the parent stops before transfer and restarts when its await expression
resumes, even when no scheduler call intervenes. Conversely, an accounted
child starts through its own initial awaiter. Each accounted frame owns only
its own intervals. The four parent/child combinations must all be tested.

The wrappers must preserve the underlying awaiter's lifetime, value category,
return type and exception semantics. Cover `BasicTask` nesting, ordinary
`TaskAwaitable`, factory/join awaiters, empty/ready tasks, and forwarding
awaiters which return another task or the current task. Reuse existing
ownership/routing logic instead of copying it into a second coroutine engine.
Inherited `await_transform` returning the existing base-bound awaiter does
not, by itself, install derived timing hooks.

The wrapping point can be the **result** of the existing base
`await_transform`, rather than a reimplementation of each underlying awaiter.
Preserve a reference result for a passed-through `BasicTask`, and own the
base's value-returned `TaskAwaiter`/`TaskFactoryAwaiter`. This retains existing
exception-sink registration and lvalue/rvalue `await_resume` behavior.
Initial and final suspension still need separate hooks: they do not go
through `await_transform`.

There is a reachable exception path even though ownership-publication methods
are non-throwing: a factory awaiter's `num_awaiters()` may throw before its
`await_suspend(TaskFactory)` is called. The outer wrapper must restart timing
before that exception returns to a handler in the coroutine. See
[the coroutine await rules](https://eel.is/c++draft/expr.await#5.1).
Successful delegation must not subsequently touch the wrapper or promise:
an immediately resumed task, or a zero-owner factory, can already have
destroyed the frame. The exception-recovery path is valid only before
publication; it does not authorize throwing after transferring ownership.

A self-transfer splits accounting intervals even when no different task runs.
Do not interpret the interval counter as a count of scheduler dispatches or
time spent waiting. Ready awaits do not split intervals. A future
bool-returning wrapper would also need to handle `await_suspend() == false`;
the currently supported base transformations return `void` or a transfer
handle instead.

Initial admission now uses the `HostTask`-constrained
`IAsyncScheduler::schedule(task)` template. The concrete promise must declare
the host domain; CUDA and erased `BasicTask` values are excluded. The template
forwards existing ownership to the private virtual
`schedule_host(BasicTask&&)` hook, preserving scheduler-specific initial-submit
behavior without a wrapper coroutine or an ordinary-task ownership envelope.
The actual coroutine identity stays in `TaskHandle`; never reconstruct a
`coroutine_handle<TaskPromiseBase>` from a derived frame's address.

The initial probe uses a small test scheduler with typed submission and the
existing `BasicTask` queue/resumption implementation. It also runs mixed
nesting through the real `DebugScheduler`, using an ordinary outer task.
That outer task is a test bridge, not an admission recommendation: adding a
wrapper coroutine to every accounted root would add a frame and scheduling
work. Before choosing admission, audit the scheduler-specific initial-submit
actions, including NUMA selection and diagnostic events. A new task must not
bypass these merely because existing continuation routing is accessible.

## The difficult boundary: work without a coroutine suspension

Neither typed awaiter wrappers nor ordinary coroutine transition hooks see
everything executed by a synchronous call inside the task:

- A nested `get_wait()` can let TBB execute other work or suspend its stack.
  It can also invalidate the assumption that a thread-clock interval begins
  and ends on the same OS thread. Supporting it requires explicit integration
  with the cooperative wait/resumption path and safe attribution across that
  path; a thread-local current-task pointer alone is insufficient.
- `execute_batch`, nested TBB parallel work or parallel BLAS may execute work
  on several threads. A timer on the caller does not measure the whole
  operation and can include work the caller performs while helping.
- External worker processes and GPU operations have their own resource clocks.
  Time spent waiting for their results is not their CPU/device consumption.

Ian accepts a caller contract of host coroutines whose cooperative waits use
`co_await` and whose accounted computational stages are synchronous,
single-threaded leaf work. This is thread CPU accounting, not whole-process or
whole-operation resource accounting. Uni20 will not intercept or attempt to
detect arbitrary thread/process creation or nested execution. Unsupported
cooperative waits need not be implemented to make this feature useful. Ordinary
untimed Async capabilities remain unchanged.

A blocking OS call that neither executes other tasks nor changes resumable
stacks does not charge blocked time to the thread CPU clock; active wall time
does include it. Work on spawned threads, processes and devices is not part of
`own_cpu`. Callers interpreting a report as the complete stage cost must ensure
that its work stays within the supported scope. Elapsed-only measurements remain
boundary durations, regardless of those other resources.

For Bethe, single-threaded root and determinant stages fit the narrow contract.
Its current subprocess evaluator supplies elapsed attempt measurements for the
initial predictor and process CPU reports as separate cross-checks. These remain
the consumer's measurements until those kernels execute in-process.

The restriction is not automatically detectable by a separate promise. An
arbitrary synchronous function can enter TBB, call a parallel BLAS routine,
or run an ordinary task directly without executing a `co_await` expression.
Without hooks in those paths, the implementation cannot reliably label a
violating interval as incomplete. For the experiment, use explicitly chosen
leaf stages whose execution is known; do not advertise automatic exclusion
of every possible nested operation. Endpoint thread-identity checks can catch
some migration, but cannot detect unrelated work on the same thread.

If a later consumer needs synchronous waits, an accounted-task-only exclusion
helper is worth testing before adding common runtime hooks: explicitly stop
the enclosing interval before a synchronous cooperative wait, then restart
after it returns or throws. The helper can
retain the owning promise directly, without installing a thread-local
current-task pointer. It must remain correct if the stack resumes on another
thread. This would exclude the whole wait/helper call; it would not measure
parallel descendants or make implicit, unmarked waits safe. oneTBB explicitly
allows a resumable task to continue on a different thread in its
[resumable-task contract](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/v1.1-rev-1/elements/onetbb/source/task_scheduler/scheduling_controls/resumable_tasks).

## Statistics lifetime and shared work

The promise can contain interval-start samples, accumulated durations, interval
count and a finalization/availability state. Decide what runtime bookkeeping
and cleanup lie inside those intervals. A useful first definition measures
the coroutine's intervals and excludes final frame deallocation; cancellation
cleanup is not automatically charged as body execution.

Statistics must outlive the promise. Consider an explicitly supplied completion
record or a separately enrolled `Async<TaskAccounting>` result. Publish once
for success, failure, cancellation and never-started destruction. Finalization
must be non-throwing and must precede delegation to the base final awaiter,
which can eagerly destroy the frame. Do not inspect the promise afterward.

Readiness of a numerical output is not automatically readiness of final timing:
a writer can be released before final suspension. The consumer must await the
specified operation/accounting completion, or explicitly join the result and
statistics. Do not use scheduler-wide `run_all()` as the completion contract.
There is no need for live concurrent polling of mutable counters in the first
version, or for locks on every interval. Timing failure should be an availability
status, not a new failure of an otherwise successful calculation.

Two runtime details constrain the terminal state:

- `started_` is set when a task's handle is released for scheduling, which can
  precede execution while a scheduler is paused. Record actual body entry in
  the derived initial awaiter instead of interpreting `has_started()` as
  evidence that CPU time was consumed.
- `destroy_with_continuation()` destroys cancelled continuations without
  setting every ancestor's cancellation flag. A derived destructor can observe
  that final suspension was not reached, but cannot always infer the cause.
  Keep body-entered, normal/failure completion, and destruction-before-completion
  distinct. Do not report an unflagged destroyed parent as successful.

Dropping an ordinary uncancelled, unstarted task currently violates the
ownership contract; it is covered by a death test in
[`test_async_task_lifetime.cpp`](../../tests/async/test_async_task_lifetime.cpp).
The experiment must test explicit cancellation before start rather than
silently changing that lifetime policy. The derived destructor still runs
through destruction of the real erased coroutine handle; it does not require
a virtual base destructor or a per-destruction accounting dispatch.

Prefer a caller-retained record for the first probe, with a clear requirement
that it outlive the task. A later owning report handle could allocate once per
selected task. Neither requires per-interval reference counting. Do not pass
a statistics writer through the ordinary numerical exception-sink path without
checking it: a failed calculation must still yield its partial measurements.
Report completion is about sealed statistics, not scientific convergence.

Publishing statistics before the base final awaiter is safe for the record's
contents but is not permission to destroy the scheduler or other resources:
frame destruction, continuation routing, and activation return may remain.
Specify a separate operation/resource-lifetime boundary. The first prototype
needs no live polling or claim that report readiness joins an entire DAG.

Own-task CPU is not inclusive DAG CPU. Shared ground-state preparation may
serve hundreds of states; charging it to every consumer multiplies the measured
cost. Applications should aggregate unique operation IDs and choose an explicit
amortization policy. Automatic recursive parent/descendant totals are deferred.

## Cost evidence and performance acceptance

A small clock-only probe on Ryzen, 8 November 2026, used GCC 13.3, `-O3`,
affinity CPU0, 500,000 pairs per repeat and five repeats. Results:

| Clock | Median pair cost | Observed min–max |
|---|---:|---:|
| `CLOCK_MONOTONIC` | 47.429 ns | 41.420–47.549 ns |
| `CLOCK_THREAD_CPUTIME_ID` | 503.214 ns | 496.578–508.404 ns |

The loop calls `clock_gettime` twice into separate `timespec` objects, retains
their nanoseconds through a compiler barrier, and divides steady-clock elapsed
time by the pair count. This includes loop/call overhead, is not baseline
subtracted, and is not a Uni20 scheduler benchmark. Scratch reproducer and
outputs are at `../bethe-research/build_codex/async-clock-probe-20261108/`
on Ryzen; this path is host-local rather than a portable dependency.

CPU clock reads alone imply roughly 0.05% overhead per 1 ms execution
interval, but 50% per 1 microsecond interval. Wall and CPU together add both
costs. For interval accounting, cost scales with the number of actual accounted
intervals, not simply task lifetime; elapsed-only boundary timing has a different
cost model. Selecting long tasks does not guarantee low overhead if they
suspend extremely frequently. Sampling selected operations is possible later;
measured versus estimated totals must remain distinguishable.

Extend the existing
[`benchmark_coroutine_overhead.cpp`](../../benchmarks/benchmark_coroutine_overhead.cpp)
with final-result completion and bounded outstanding work. Measure:

1. Baseline versus feature compiled out: ordinary code generation, actual
   coroutine allocation sizes, allocations per task and throughput.
2. Feature compiled in but ordinary tasks only: the same measurements,
   identifying any new common field, branch or indirect call.
3. Accounted ready/suspending tasks, direct parent-child chains and resumed
   tasks on different workers, with wall-only and CPU-plus-wall modes.
4. Single-worker and multiworker throughput, plus substantial CPU-bound leaf
   stages representative of Bethe. Record per-interval overhead and overall
   throughput with repeated, rotated measurements on a quiet host.

Compare wall-only and CPU-plus-wall modes for both small control tasks and
long computational intervals after admission and reporting are present.
State whether each wall-only result measures elapsed duration or active wall
intervals; keep the comparison workloads and completion boundaries explicit.
Include a long-lived task with many short intervals: total task duration must
not hide the cost of repeated clock reads. Retain interval counts in completed
attempt reports, including interrupted attempts.

Do not quote a universal overhead or compare with HPX without a matched
benchmark. No timing-specific regression threshold is selected yet. Any
repeatable regression in the ordinary compiled-out path requires explanation
and resolution; the enabled ordinary path must also have explicit evidence.
Report frame/allocator effects even if clock-free timings look similar.

## Correctness tests and first handoff

Use an injectable fake clock for deterministic accounting checks; real-clock
smoke tests should use broad properties, not fragile exact time thresholds.
Required cases include ready and genuinely suspended awaits, cross-worker
resumption, same-scheduler direct transfer, cross-scheduler continuation,
all accounted/ordinary nesting combinations, forwarding/self-transfer,
joins with immediate concurrent readiness, exceptions during resumed code,
cancellation before start/while suspended, and the existing error policy for
dropping an uncancelled unstarted task.
Verify exactly-once final publication and statistics lifetime after frame
destruction. Discriminative tests must show that another task's execution
while this task waits is excluded from its own CPU and active-wall intervals,
while elapsed measurements retain the intervening delay. Verify that wall-only
measurement makes no CPU-clock calls, including failure and cancellation paths.
Clock failure and unsupported execution paths must leave numerical results
intact and timing status explicit.
In the first consumer cases, independently test normal task return followed
by application-level rejection, a thrown numerical failure, and cancellation
after some measured work. Each attempt must retain a readable report with the
appropriate partial cost. Two readers of one prepared input must not multiply
that preparation's separately reported cost. Resource-lifetime tests must
distinguish report readiness from the completion that permits teardown.

In a narrow prototype, first settle own-task scope, completion semantics and
typed admission; then compare A/B feasibility and ordinary-path code generation.
Only then choose a public type/flag and implement the transition coverage.
Keep feature selection consistent across library and consumers to avoid ODR
or layout mismatches. Decide whether a disabled build omits the accounted
API or reports an explicit unsupported capability; never silently return
apparently valid zero CPU time.

## Isolated feasibility checkpoint

On 8 November, a separate `ProbeTask` / `ProbePromise` was tested against the
unchanged runtime at the revision above. The promise derives directly from
`TaskPromiseBase`, wraps the results of its existing `await_transform`, adds
initial/final hooks, and finalizes interrupted records in its destructor.
The probe uses a manually advanced clock and caller-owned records. It is a
test of transition coverage and frame lifetime, not an implementation of
thread CPU clocks, publication synchronization, or a public accounting API.

Results:

- **13 deterministic tests passed** under GCC 13 and Clang 19 with ASan/UBSan.
  Sanitizers instrument the probe and instantiated runtime headers; the cached
  common/fmt libraries are not instrumented in this isolated check. The common
  sources match the inspected revision. Leak detection was enabled.
- Cases cover ready awaits with references and move-only values, all four
  nesting combinations, direct and queued continuation, cross-scheduler
  return, factory joins, self/forwarding transfers, and a real DebugScheduler
  mixed chain. Explicit pre-start cancellation, destruction of an unflagged
  parent through child cancellation, pre-publication exceptions, immediate
  resume/destruction during publication, and zero-owner factory destruction
  also pass.
- A synthetic clock-domain change while suspended verifies that new intervals
  use a fresh start sample. This is **not** evidence of real cross-thread
  resumption or concurrent publication correctness; those remain untested.
- For one representative ordinary coroutine awaiting a read buffer and writing
  an output, adding the experimental promise definitions produced
  **byte-identical optimized object files** with GCC 13 and Clang 19 (`-O3`,
  no debug information). This supports the possibility of leaving ordinary
  code untouched. It does not measure allocator behavior, throughput, or the
  cost of a future admission adapter/build option. GCC emitted its existing
  atomic `stringop-overflow` warning in both variants; no runtime change or
  warning suppression was added for this probe.

The local reproducer is `build_codex/task-accounting-20261108/`: `probe.cpp`,
`run.py`, compiler/test logs, and `codegen.cpp` with its paired objects and
`codegen-results.txt`. These are scratch artifacts, not shipped test targets.
Run `python3 build_codex/task-accounting-20261108/run.py gcc` or replace `gcc`
with `asan` on this host; the script records its existing cached dependencies.

This initial checkpoint supported continuing option B. It did not establish
that general-purpose accounting is practical. It identified the following
next bounded experiment, now reported in the next section:

1. Typed host admission without a wrapper coroutine and without changing
   ordinary initial-admission behavior or accepting CUDA tasks accidentally.
2. A retained, non-throwing terminal report with separately tested publication
   and lifetime semantics, including numerical failure and cancellation.
3. Real thread-CPU/wall samples on TBB, immediate concurrent readiness, and
   resumption on different workers, using only the declared leaf scope.
4. Build-disabled and enabled-unused object/frame/allocation comparisons,
   followed by bounded throughput and leaf-work benchmarks. Stop or reconsider
   the design if these expose ordinary-path cost or unreliable attribution.

The synchronous-wait exclusion helper is deferred from this sequence: the
first Bethe consumer does not require it. Inclusive DAG accounting and parallel
resource attribution remain separate work as well.

No production runtime changes, public option/type names, or implementation
commit are part of this checkpoint.

## Retained-report and real-scheduler checkpoint

The first retained-report checkpoint on `experiment/async-task-accounting` added an opt-in
[reproducer](../../experiments/task_accounting/) with a separate `Task<Clock>` /
`Promise<Clock>`, three measurement modes, typed initial admission and retained
reports. At that checkpoint, names and packaging remained experimental and no installed
API or runtime header had changed. The later host-admission refinement is
described below.

That checkpoint used an admission adapter passing erased ownership through the ordinary
host scheduler entry point. It adds no trampoline coroutine and preserves
initial scheduling, pause and routing behavior. It accepts only the accounted
host type; compile-time tests exclude ordinary and CUDA tasks from this adapter.
Ordinary tasks continue to use their existing admission API.

Reports are independently retained `Async<Report>` values. The report writer
is never registered as the numerical coroutine's exception sink, so numerical
failure does not poison partial statistics. Publication precedes parameter
teardown: a deterministic destructor test demonstrates this and separately
checks that an ordinary successful `co_await child` resumes after teardown.
No full resource join is claimed, especially for cancellation. The prototype
requires no global registry, timing TLS or per-resumption allocation.

The implemented elapsed interval is **promise construction after report-state
allocation through sealing immediately before publication**. It includes later
queue/dependency waits, excludes allocation/publication/destruction, and is not
a generic interpretation of Bethe's attempt boundaries. Bethe's external
monotonic attempt timer remains the appropriate first predictor under its
one-CPU-per-worker assumptions. Shared preparation is measured separately.

Verification on ryzen (Intel Alder Lake, Linux), GCC 13 and Clang 19, using
oneTBB 2022.3.0 from the cached source:

- **19 experimental tests pass** in GCC Debug, GCC Release, and Clang
  ASan/UBSan Debug. They cover the earlier transition cases using the actual
  retained-report implementation, numerical output failure, pre-start and
  partial cancellation, clock failure, absence of CPU reads in wall-only modes,
  paused admission, reports consumed during TBB execution, and shared
  preparation counted separately from 16 consumers.
- A deterministic real-thread handoff starts on a TBB worker, suspends, then
  resumes on the application thread through DebugScheduler. It verifies thread
  identity changes and that CPU intervals remain available. This complements
  the synthetic different-CPU-epoch test; it does not claim arbitrary nested
  resumable-stack attribution or exhaustive worker-to-worker stress coverage.
- Existing GCC Debug Async tests: **197 pass**. TBB tests: **50 pass, one NUMA
  round-robin test skipped** on the available topology.
- The fresh sanitizer build instruments Uni20 support libraries and template
  instantiations. The compiled oneTBB implementation retains its existing
  sanitizer exclusion. Leak detection uses the repository's existing narrow
  `NodeInfo::create` suppression for intentionally retained debug DAG metadata
  (199 allocations, 11,144 bytes in this run). No new suppression was added.
- The representative ordinary read/write coroutine produces byte-identical
  optimized object files with the experimental definitions absent/present,
  separately with GCC 13 and Clang 19. This is evidence for this implementation
  and coroutine, not a universal guarantee about future admission/API changes.
- GCC's existing optimized coroutine/atomic `stringop-overflow` diagnostic also
  occurs in the **ordinary, definitions-excluded** control. Only the local
  optimized measurement build makes this warning nonfatal; repository warning
  policy is unchanged.

### Initial allocation and timing observations

Release measurements disable debug instrumentation, stacktraces and IPO. The
creation probe measures ordinary `new` calls in a separate executable, keeping
allocation counters out of throughput measurements. An empty ordinary task
uses one 192-byte allocation both with and without the accounting definitions.
The accounted task uses five allocations totalling 816 bytes; its first
allocation (the observed frame) is 360 bytes. Both elapsed and CPU-plus-wall
modes use this same prototype representation. These are compiler/build-specific
observations, not portable frame-size promises.

The following are medians of ten samples per cell from two processes per
variant, with five repetitions per process and mode order rotated. Off/on
process order was off, on, on, off. Each sample submits and retrieves 4,096 tasks
in batches of 256. Numerical outputs and retained reports are both consumed.
Measurements below use DebugScheduler to isolate task/hook/report cost; the
reproducer also measures one- and four-participant TBB scheduling. Units are
microseconds per task, not the report's measured body time.

| Workload | Ordinary, definitions included but unused | Elapsed | Active wall | CPU plus wall |
|---|---:|---:|---:|---:|
| Empty task | 0.351 | 0.689 | 0.718 | 0.970 |
| 32 self-transfers, no numerical work | 0.879 | 1.281 | 2.121 | 10.518 |
| 20,000-step integer recurrence | 17.413 | 17.794 | 17.808 | 18.090 |

The ordinary definitions-excluded controls were 0.349, 0.860 and 17.413 us.
Throughput is noisy, especially with TBB scheduling; it is not evidence of
zero measurable cost in every program. The unchanged object files and ordinary
creation allocation counts are the stronger controls. The recurrence is a
synthetic CPU workload with an observable checksum, not a Bethe numerical
benchmark. No performance threshold is encoded as a unit test.

These results support **selective leaf accounting** with a separate promise.
They do not support timing every small coroutine: retained reports add four
allocations per selected task, and frequent CPU-clock sampling dominates empty
intervals. Wall-only elapsed mode is independently useful and samples only at
its boundaries. An external attempt timer is cheaper and remains sufficient
for Bethe's immediate requirement.

The bounded feasibility checkpoint is complete. Accounting names/packaging, reducing report allocations, or adding a full
completion/join contract need separate design decisions. Direct scheduler
admission has since been selected and is described below. Synchronous nested waits, inclusive DAG totals,
parallel CPU attribution and admission-policy changes remain outside scope.
Raw local logs, paired objects and benchmark CSVs live under
`build_codex/task-accounting-next-20261108/`; build/run instructions and metric
boundaries are in the reproducer's README. The experiment is preserved as a
separate checkpoint from the typed host-admission change.

## Direct host admission and accepted accounting scope

Ian accepts explicit restrictions on code requesting CPU accounting. General
thread/process discovery and synchronous nested-wait attribution are not
prerequisites. `own_cpu` describes active thread CPU intervals; it is not a
whole-process resource ledger. Arbitrary callees may create threads or processes
without a visible coroutine transition. The caller must ensure that the chosen
scope matches the interpretation of its report. The implementation neither
forbids these operations in ordinary Async code nor claims to detect every
violation of the accounting contract.

The experiment now uses `scheduler.schedule(std::move(task))` directly.
`IAsyncScheduler::schedule` and the global helper constrain initial submission
to movable `HostTask` types. The concrete promise must derive from
`TaskPromiseBase` and declare the host domain. An internal virtual
`schedule_host(BasicTask&&)` handles the already-checked task. Debug, TBB and
NUMA schedulers retain their original initial-submit behavior. The explicit
NUMA-node overload accepts the same typed tasks. This removes the experimental
`AsyncTask` envelope without introducing an accounting-specific scheduler hook,
extra frame, timing branch or new promise state in ordinary tasks.

Verification after this refinement:

- **21 accounting tests pass** in GCC 13 Debug/Release and Clang 19 ASan/UBSan,
  using the same existing debug-metadata suppression and oneTBB instrumentation
  boundary as above. Added cases cover global admission and NUMA preference /
  explicit node selection through the new interface.
- **198 ordinary Async tests and 50 TBB tests pass**, with the same one NUMA
  topology skip. The ordinary suite now includes a distinct host promise whose
  body entry, concrete destruction, paused submission and cancellation are
  checked independently of the optional accounting targets.
- Compile-time checks reject CUDA, erased `BasicTask`, lvalue and const
  submissions through the host interface.

The admission change is prepared independently of the accounting API. The accounting type and retained
report implementation remain experimental. The numerical timing table above
records the earlier adapter checkpoint; it has not been replaced by estimates
for the new admission code. No CUDA runtime validation was performed on this
host; no CUDA task execution or device-routing implementation was changed.

### Numerical consumer evidence

A bounded Bethe trial exercised the same accounting promise as a child of an
ordinary completion task. It preserved numerical results across ordinary,
elapsed-only, active-wall and CPU-plus-wall modes, including comparison with a
synchronous reference. Injected numerical failures retained partial costs and
prevented dependent numerical work from running. The explicit completion tokens
and scheduler lifetime remained separate from report readiness.

Active-wall and thread CPU measurements agreed closely with timers inside the
numerical stages. Promise elapsed time also included substantial dependency and
queue delays. This supports selected stage accounting, while reinforcing that
promise latency must not automatically become an application cost predictor.
The trial used the original child-admission boundary, so it adds no evidence
about direct host submission or sub-percent overhead. Its result reduction and
independent review remain consumer-owned; no duplicate numerical run is needed
to package the Uni20 API.

The receiving Uni20 session owns design refinement, runtime changes and their
review/benchmarks. Four owns Bethe's cost-aware admission/search experiment.
This handoff creates no new numerical job, changes no scheduler default, and
does not require completing general task accounting before a Bethe leaf-stage
timing experiment.
