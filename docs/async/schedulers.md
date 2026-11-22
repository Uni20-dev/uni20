# Async Schedulers

Epochs determine when a task is ready. The scheduler runs each ready task on
the appropriate thread and, for CUDA tasks, with the appropriate device selected.

## Choosing a Scheduler

| Workload | Deterministic, on the driving thread | Parallel, using oneTBB |
|---|---|---|
| Host tasks only | `DebugScheduler` (the process default) | `TbbScheduler`, or `TbbNumaScheduler` for per-NUMA-node placement |
| Host and CUDA tasks, or CUDA tasks only | `DebugCudaScheduler` uses one runnable queue for both task types | `TbbCudaScheduler` uses one host arena and one arena per enrolled GPU |

Choose once at application or test setup. Host work can use the default
`DebugScheduler` even in a CUDA-enabled build. When any operation submits a
`CudaTask`, including a child of an ordinary `AsyncTask`, install a CUDA-aware
scheduler that accepts the device. The same instance runs the host work before
and after the CUDA child. Debug scheduling controls coroutine order on the
driving thread. CUDA kernels still run asynchronously on their streams.

For example, `sum_host` on an async CUDA tensor returns a host scalar but
submits a CUDA reduction. `cuda::initialize()` supplies the device resources.
`ScopedScheduler` selects the scheduler used for submission and waiting.

```cpp
#include <uni20/async/debug_cuda_scheduler.hpp>
#include <uni20/backend/cuda/runtime.hpp>
#include <uni20/linalg/async/reductions.hpp>
#include <uni20/tensor/copy.hpp>

int main()
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}});
  uni20::async::DebugCudaScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);

  uni20::Tensor<double, 1> host(2);
  host[0] = 2;
  host[1] = 3;
  uni20::async::Async<uni20::CudaTensor<double, 1>> input = uni20::to_device(host, 0);
  auto sum = uni20::sum_host(input);
  return sum.get_wait() == 5 ? 0 : 1;
}
```

## Task Ownership

A scheduler owns queued runnable activations, not every coroutine whose promise
records that scheduler. Before admission, the typed host task (normally `AsyncTask`) or `CudaTask`
owns the frame. After a running coroutine suspends, its awaiter owns the
`BasicTask` until readiness publishes another activation. The recorded scheduler
pointer is only a non-owning resumption route. Consequently, a scheduler does
not know about a task merely because that task names it, and scheduler
quiescence does not include externally suspended frames.

Current tasks normally retain the scheduler installed when they are first
scheduled. A nested `AsyncTask` with a different scheduler is submitted there,
then returns its continuation to the continuation's recorded scheduler. A
suspended, externally owned task may replace its route with any scheduler that
accepts its domain and CUDA device requirements. A user-facing migration
awaiter remains future work. See
[Scheduler Routing and Task Domains](scheduler_migration.md).

## Global Scheduler Model

Most async code uses the global scheduler helpers from `debug_scheduler.hpp`:

- `set_global_scheduler(...)`
- `get_global_scheduler()`
- `reset_global_scheduler()`
- `schedule(task)` for movable host tasks and, with a CUDA-aware scheduler,
  `CudaTask` (`schedule(task, device)` binds a CUDA device at submission)
- `ScopedScheduler`

If you do not override it, the global scheduler defaults to an internal `DebugScheduler`.

Host admission is constrained by the concrete task's `promise_type`: it must
derive from `TaskPromiseBase` and declare `TaskDomain::host`. This accepts
`AsyncTask` and separately implemented host task/promise types without a wrapper
coroutine. CUDA tasks and already-erased `BasicTask` values cannot enter this
interface. The task must be passed as an rvalue, transferring ownership.
`HostTask` strips cv/ref qualifiers when inspecting type metadata, but
checks ownership conversion on the original argument type: both
`HostTask<AsyncTask>` and `HostTask<AsyncTask&&>` hold, while lvalue
references and const/volatile tasks are rejected.

Scheduler implementations override the private
`schedule_host(BasicTask&&)` hook. The public template checks the typed task
before forwarding the original ownership handle to that hook; normal initial
binding, diagnostic events, pause handling and NUMA selection remain there.
Accounting policies need no scheduler-specific overload. `TbbNumaScheduler`
also accepts these task types in `schedule(task, numa_node)`.

`ScopedScheduler` is the standard way to override scheduler context in tests.
It changes the selected scheduler but does not own it or extend its lifetime.
Any scheduler supplied to it must remain alive until all tasks routed through
that scheduler have completed or can no longer resume. See
[Lifetime and Quiescence](scheduler_migration.md#lifetime-and-quiescence) for
the full contract, including stack-local test schedulers.

## Lightweight Task Batches

`IAsyncScheduler::execute_batch(count, callable)` executes a synchronous batch
of independent ordinary function calls. The free `execute_batch(...)` helper
uses the active global scheduler. On successful return, every index in
`[0, count)` has been invoked exactly once and the callable is no longer
borrowed by the scheduler.

A lightweight batch is deliberately not a collection of `AsyncTask`
coroutines:

- it creates no epoch, dependency edge, coroutine frame, or per-item scheduler
  ownership;
- a stateful or capturing ordinary callable is valid because the call blocks
  until every participating item has finished;
- distinct indices may run concurrently and must not perform unsynchronized
  conflicting accesses; and
- if one item throws, unfinished items may be cancelled and the exception is
  rethrown after participating work joins.

`DebugScheduler` executes indices deterministically using its configured
`fifo`, `reverse`, or seeded `random` ordering. `TbbScheduler` executes every
batch inside its arena, using a serial loop for one participant and a oneTBB
`parallel_for` otherwise. This includes serial batches in the same concurrency
budget as enqueued coroutine activations. An optional
`TbbSchedulerBatchOptions::maximum_concurrency` cap uses that many dynamic
runners over the batch index space, independently
of the arena's coroutine-task concurrency. `TbbNumaScheduler` selects one
managed arena for the batch; placement-aware partitioning across NUMA nodes
remains a later policy.

A batch item may call ordinary scheduler APIs. In particular, scheduling work
and then using `get_wait()` is supported by both the recursive DebugScheduler
progress path and TBB's resumable nested-wait path, including an arena with one
participant. This is a composability guarantee, not the normal batch pattern:
block-level batches should normally call immediate dense operations directly.

## DebugScheduler

Execution model:

- scheduled tasks are stored internally
- `run()` snapshots and executes one runnable batch
- `DebugSchedulerOptions::order` selects `fifo`, `reverse`, or `random` batch
  order; the default is `reverse`, preserving the scheduler's historical
  behavior
- random order is reproducible from `DebugSchedulerOptions::random_seed`,
  which defaults to zero
- synchronous lightweight batches use the same configured index ordering
- `run_all()` is a debugging aid that runs until no task is runnable; it
  returns while tasks are suspended outside the scheduler (CUDA resources,
  I/O), so it does not establish completion. Production code and tests that
  need a result wait on it with `get_wait()` or `wait_for()`
- because whether such a task has resumed by the time `run_all()` returns
  depends on when another thread delivers the wakeup, `DebugScheduler::run_all()`
  aborts with a message if any task began an external wait during the call,
  even on a run where the wakeup happened to arrive in time. A test that
  deliberately leaves such a task suspended, for example while it holds the
  resource the task needs, passes `ExternalWaitPolicy::allow`. A task that was
  already waiting when the call began is not counted. The check fires only on
  calls in which a task suspends, so it converts a timing-dependent wrong result
  into a failure, not into a failure on every run

External wakeups:

- an awaiter that hands a suspended task to another thread, such as a CUDA
  host callback returning a pooled stream or provider handle, calls
  `BasicTask::begin_external_wait(task)` before publishing it and wakes it
  with `BasicTask::reschedule_after_external_wait(task)`; the CUDA and cuBLAS
  resource awaiters do this
- the runnable queue is guarded, so those wakeups may enqueue from any thread;
  tasks still execute only on the driving thread
- `external_waits()` reports the number of registered waits outstanding;
  destroying the scheduler while any remain is an error

Wait/deadlock behavior:

- `help_while_waiting(...)` drives runnable tasks while waiting, and blocks
  without a time limit while the queue is empty but a registered external
  wakeup is pending; external operations may legitimately take minutes
- if the waiting condition is false, no runnable tasks remain, and no external
  wakeup is pending, the wait can never complete: runtime emits
  `TaskRegistry::dump()` and aborts with deadlock diagnostic
- a wait whose wakeup depends on the waiting thread itself, or on work that has
  not been scheduled, still deadlocks; it blocks rather than aborting

## TbbScheduler

Execution model:

- tasks are dispatched into oneTBB `task_arena` + `task_group`
- arenas use oneTBB's fast worker-leave policy so an idle arena bypasses the
  delayed-leave `sched_yield` retention loop
- initial admission and rescheduling register activations with
  `task_group::defer()` and publish them through non-blocking
  `task_arena::enqueue()`
- submitting threads do not enter the arena or wait for arena capacity
- allocation or oneTBB admission failure is a fatal scheduler-infrastructure
  error reported with scheduler and, where applicable, CUDA device context
- ready coroutines resume on threads participating in the arena, including
  oneTBB workers and application threads that enter through `task_arena::execute()`
- `run_all()` resumes if paused, then waits for task-group completion of the
  currently submitted activations
- lightweight batches execute synchronously as a `parallel_for` in the arena;
  an optional dynamic-runner cap limits simultaneous batch items independently
  of arena concurrency, and batches remain separate from coroutine activation
  accounting

The task group owns each scheduled resumption while it runs. If that resumption
suspends the coroutine on an epoch or external event, the TBB task finishes and
the awaiter owns the suspended coroutine until it becomes ready. `run_all()` is
therefore scheduler-activation quiescence, not proof that every coroutine ever
routed through the scheduler has completed.

An admission that happens-before `run_all()` is included because its deferred
task is already registered with the group. If admission and `run_all()` are
unordered and concurrent, either may linearize first; `run_all()` is not a
global barrier against submissions that have not yet been accepted.

Pause/resume:

- `pause()` queues handles without dispatch
- `resume()` drains queued handles and dispatches them

Wait behavior:

- `get_wait()` enters the scheduler arena through `task_arena::execute()`
- the epoch's transition to reader-ready state directly wakes its registered
  suspension point
- oneTBB resumable tasks release the arena slot while dependencies execute
- nested synchronous waits work with one total arena participant
- pending low-level waits require a race-safe notification registration;
  predicate-only calls are accepted only when already ready
- writer waits are notified when their write gate becomes available, then
  recheck acquisition in case another writer took the gate first

There is no internal wait timeout or idle-work bookkeeping. An idle arena can
still receive later work or external completion. Use an attached debugger or
an instrumented `TaskRegistry` diagnostics service to investigate stalled
waits; the latter can consume a signal such as `SIGUSR1` and write a best-effort
DAG snapshot. See [`tbb_execution_primer.md`](tbb_execution_primer.md) for the
execution model, notification contract, and single-participant timeline.

## TbbNumaScheduler

`TbbNumaScheduler` manages multiple `TbbScheduler` arenas, one per NUMA node.

Dispatch policy:

- task with preferred node: dispatch to that node when available
- task without preference: round-robin node selection

Diagnostics:

- `scheduled_count_for(node)` reports dispatch counts used by tests
- tests verify round-robin and preferred-node behavior

## CUDA Schedulers

`DebugCudaScheduler` and `TbbCudaScheduler` implement typed CUDA admission for
`CudaTask`. They share ordinary `BasicTask` rescheduling and continuation
routing with the host schedulers.

- `DebugCudaScheduler` also admits `AsyncTask`. It uses one deterministic queue
  for both domains and selects either the CUDA promise's affinity or the
  scheduler's default device around each CUDA activation. One instance can
  execute tasks for multiple devices and restores the calling thread between
  activations.
- `TbbCudaScheduler` extends `TbbScheduler` with one worker-only arena per
  enrolled CUDA device. The host and device arenas share task-group, pause,
  wait and rescheduling state.
- Each CUDA device arena defaults to one participating worker, so general CUDA
  work has one host submission lane per device. Setting
  `TbbCudaSchedulerOptions::cuda_max_concurrency_per_device` explicitly retains
  the capability to measure or use concurrent submission. This limit is
  independent of stream count and provider-handle pool sizes.
- Each CUDA arena installs a `task_scheduler_observer`. Every participating
  worker saves its previous CUDA device, selects the arena device, and restores
  the previous selection on exit.
- initial CUDA admission and rescheduling enqueue task-group activations without
  making completion-service or publishing threads enter the device arena
- CUDA tasks may begin without device affinity and execute device-neutral work
  through the scheduler's default device arena
- `co_await cuda::set_device(device)` establishes or changes affinity; it
  continues immediately when that device is already current and otherwise
  resubmits the task through the selected device arena
- same-device nested tasks can transfer directly; cross-domain and cross-device
  continuations re-enter through the shared scheduler's routing hook
- Use `get_wait()` on the result to wait for the whole dependency chain,
  including a CUDA child that resumes after acquiring a resource. `run_all()`
  can return while a task is suspended on an external event.

Tasks running in a CUDA scheduler arena must not call `cudaSetDevice` directly.
Cross-device work re-enters the scheduler through the target device arena; this
keeps the observer authoritative for the whole arena activation.

Debug builds verify the thread's current CUDA device before each oneTBB CUDA
activation. Leaf kernel wrappers do not inspect coroutine promises; they check
their CUDA API results and may validate stream and buffer device compatibility.

A child with no scheduler inherits its parent's scheduler if that scheduler
accepts the child's task domain and device. When a host parent awaits a CUDA
child, the scheduler queues the child for CUDA execution and later resumes the
parent in the host domain. See [Nested Entry and Return](scheduler_migration.md#nested-entry-and-return).

If a workload gives different results under the debug and oneTBB schedulers,
check its dependency edges, buffer releases, and object lifetimes.

## Related References

- Runtime semantics: `runtime_model.md`
- oneTBB execution and waiting model: `tbb_execution_primer.md`
- Buffer usage and release patterns: `buffers_and_awaiters.md`
- Deadlock/debug output: `task_registry_debug.md`
- Fast lookup: `quick_reference.md`
