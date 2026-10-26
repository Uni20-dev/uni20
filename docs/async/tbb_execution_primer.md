# oneTBB Execution Primer for Uni20

This document explains the oneTBB concepts used by Uni20's async runtime. It is
not a general oneTBB tutorial. Its purpose is to make the execution and waiting
semantics of `TbbScheduler` understandable, especially when only one thread is
available or synchronous code calls `Async<T>::get_wait()` from inside an async
task.

The most important rule is:

> An arena limits where and how many threads may execute tasks. A task group
> tracks a set of tasks. Neither one automatically makes an ordinary blocking
> function into a scheduler-aware wait.

## Short Mental Model

1. oneTBB schedules **tasks**, not user-selected operating-system threads.
2. A `task_arena` has execution **slots**. Both application threads and
   oneTBB-managed worker threads may occupy slots.
3. `max_concurrency` is the maximum number of occupied arena slots, not the
   number of worker threads.
4. Reserved application-thread slots keep part of an arena available to
   application threads that enter it.
5. `task_arena::execute()` lets an application thread enter an arena and run a
   function there.
6. `task_group::wait()` is a broad scheduler-aware wait for the whole group.
7. `task::suspend()` is a targeted scheduling point that preserves an ordinary
   synchronous call stack while its thread performs other TBB work.

## Vocabulary

| Term | Meaning in this document | Common misunderstanding |
|---|---|---|
| application thread | A thread created by the application, such as `main` | It is not necessarily idle while TBB runs |
| worker thread | A thread managed by oneTBB's scheduler | Arena concurrency is not a worker count |
| task | A schedulable unit of work | It is not an operating-system thread; ordinary execution is non-preemptive |
| arena | A scheduling domain with concurrency and placement constraints | An arena is not a fixed private thread pool |
| slot | Capacity for one thread to execute in an arena | A slot is not an operating-system thread |
| task group | A dynamically populated set of tasks that can be waited for or cancelled together | It does not define arena concurrency |
| resumable task | A TBB task whose synchronous stack is suspended at a `suspend_point` | It is distinct from a C++ coroutine |

Uni20 uses **application thread** and **worker thread** consistently. Current
oneTBB documentation also uses **external thread** for an application thread
that is outside an arena, while older TBB material uses **master thread**. Read
both terms as application thread when consulting upstream documentation.

## Application Threads and Worker Threads

oneTBB owns a process-wide worker pool. An explicit arena does not create a
permanent private set of threads. Instead, threads enter an arena, occupy its
slots, execute eligible tasks, and leave.

An application thread can participate in TBB work. The main example is:

```cpp
arena.execute([&] {
  group.wait();
});
```

The calling thread enters `arena`, executes the function, and may execute tasks
while `group.wait()` is making progress. When the outermost `execute()` returns,
the application thread has left the arena and its previous scheduler context is
restored.

A worker thread and an application thread are different origins for an
operating-system thread, not different kinds of task execution. Either may run
the same task if the arena admits it.

Do not rely on thread-local state surviving a resumable-task suspension. An
ordinary TBB task is non-preemptive, but a suspended task may resume on a
different worker. oneTBB guarantees that an outermost blocking construct such
as `task_arena::execute()` eventually returns on its original calling thread.

### MPFR cache lifetime and deferred cleanup options

With thread-safe MPFR, local caches belong to operating-system threads, not to
Uni20 schedulers or oneTBB arenas. Retaining reusable caches on persistent
threads is acceptable for the usual calculation lifetime. Program-exit cleanup
alone offers little practical benefit; useful motivations would be memory
reclamation during a long run or preventing accumulation when threads terminate
and are replaced during the process lifetime. No new cleanup mechanism is
currently planned for implementation.

`mpfr_free_cache2(MPFR_FREE_LOCAL_CACHE)` frees the calling thread's local caches.
Calling it from the application thread after a scheduler wait does not clear
other threads' caches. Neither scheduler destruction nor submitting one cleanup
task per arena slot guarantees that each participating thread performs cleanup.
See [MPFR's memory-handling contract](https://www.mpfr.org/mpfr-current/mpfr.html#Memory-Handling).

If runtime memory pressure justifies further work, a possible approach is an
explicit cleanup request generation, checked at safe execution boundaries on
each participating thread. A thread would clear its local caches once for each
request it observes. This would be eventual reclamation, not a synchronous
all-thread operation: idle threads might retain caches, and suspended work can
resume on another thread. The integration would need to cover application
threads, worker threads and the provider's auxiliary resumable-task threads,
without assuming that one arena entry covers every later resumption.

A thread-local destructor could complement this by freeing caches when an
enrolled thread actually terminates during the run. Its enrollment coverage,
teardown ordering and execution cost need validation. Clearing caches after
every numerical operation or every arena exit is not proposed; that could
discard useful caches repeatedly.

oneTBB's [scheduler observers](https://uxlfoundation.github.io/oneTBB/main/specification/source/task_scheduler/task_arena/task_scheduler_observer_cls.html)
provide arena entry/exit notifications, not an immediate callback on every pool
thread. [Runtime finalization](https://uxlfoundation.github.io/oneTBB/main/reference/source/task_scheduler/scheduling_controls/task_scheduler_handle_cls.html)
can wait for worker termination under process-wide conditions, but does not
itself perform MPFR cleanup and is unsuitable as a local sub-scheduler cleanup
operation. These limitations should remain explicit in any future API.

Diagnostic-node reclamation has a separate ownership problem; see
[the deferred metadata options](task_registry_debug.md#diagnostic-memory-lifetime-and-deferred-reclamation-options).

### Non-Preemption and Scheduling Points

oneTBB does not time-slice runnable tasks. If several tasks are queued in a
one-slot arena, oneTBB selects one of them without promising FIFO order or
fairness. Ordinary code in that task then runs without TBB preemption.

An arena slot is execution capacity for a participating thread; it is not owned
permanently by a task. Other TBB work can run when the current execution reaches
a scheduler-aware boundary:

- the task returns or completes
- the task waits in a TBB construct such as `task_group::wait()` or a nested
  parallel algorithm, allowing the participating thread to service other work
- the task calls `task::suspend()`, preserving its synchronous stack while TBB
  runs other tasks
- a Uni20 coroutine resumption actually suspends at `co_await` or completes, so
  `resume_and_track()` returns and the surrounding TBB task finishes

An awaitable that is already ready does not suspend merely because the source
contains `co_await`; coroutine execution may continue in the same TBB task.
Likewise, cancellation does not forcibly preempt running user code.

The following are not TBB scheduling points:

- `std::this_thread::yield()`
- sleeping
- blocking on an ordinary condition variable, mutex, file descriptor, or other
  operation unknown to TBB
- a long CPU loop with no TBB wait, task suspension, or coroutine suspension

Those operations retain or block the participating thread. With one arena
participant, no queued TBB task can make progress until control returns to TBB.

## Task Arenas

A `oneapi::tbb::task_arena` defines a scheduling domain:

- tasks submitted from within the arena belong to that arena
- tasks from one arena are not executed in another arena
- the arena limits simultaneous participation through its slot count
- constraints may restrict NUMA node, core type, or threads per core
- construction is lazy; the internal arena is initialized on first use unless
  initialized explicitly

### `execute()`

`task_arena::execute(f)` is the normal way for an application thread to enter
an arena.

- If a slot is available, the calling thread joins the arena and runs `f`.
- If it cannot join, oneTBB may turn `f` into a task, enqueue it, and wait for
  its completion.
- If the caller is already executing in the arena, the call remains in that
  arena.

`execute()` establishes the execution context. It does **not** make arbitrary
blocking code inside `f` scheduler-aware:

```cpp
arena.execute([&] {
  while (!ready())
  {
    // Wrong: this occupies the current stack without giving TBB a scheduling
    // point. With one participating thread, the producer cannot run.
    std::this_thread::yield();
  }
});
```

The body must call a TBB waiting primitive or suspend the current TBB task.

### `enqueue()`

`task_arena::enqueue(f)` submits work without requiring the caller to enter the
arena. It has different progress and exception rules from `task_group::run()`.
In particular, oneTBB may create one worker to service enqueued work even when
`global_control::max_allowed_parallelism` is `1`.

Uni20's host and CUDA TBB schedulers use `task_group::defer()` followed by
`task_arena::enqueue(task_handle)`. Deferring first registers the activation
with the task group's wait set; enqueueing then publishes it without requiring
the caller to enter the arena. This is the oneTBB 2021.x spelling of the newer
`task_arena::enqueue(function, task_group)` convenience overload.
Failure while allocating the deferred task or publishing it into the arena is a
fatal Uni20 scheduler-infrastructure error. The boundary reports structured
scheduler and, for CUDA, device context before terminating.

## Arena Concurrency

### `max_concurrency`

`max_concurrency` limits the total number of threads simultaneously executing
inside an arena. It includes worker and application threads occupying arena
slots.

It is an upper bound, not a promise. The scheduler may use fewer threads due to
the global parallelism limit, unavailable work, affinity restrictions, nested
arenas, or operating-system scheduling.

The value `task_arena::automatic` asks oneTBB to select the concurrency from the
available machine and process configuration.

### Reserved Application-Thread Slots

Reserved slots are arena capacity kept available for application threads.
Workers may use only the non-reserved portion. Application threads enter
through `execute()` and can use reserved capacity.

The public oneTBB documentation normally calls these `reserved_slots`; some
older API declarations and implementation material use
`reserved_for_masters`.

For an arena with maximum concurrency `M` and `R` reserved slots:

- at most `M` threads execute in the arena simultaneously
- at most `M - R` ordinary workers occupy the arena simultaneously
- the reservation does not create application threads
- unused reserved slots do not become ordinary worker capacity

There is an important exception for `task_arena(1, 1)`: oneTBB 2022.3 adds a
mandatory worker slot for enqueued work, allowing the worker and application
thread to execute simultaneously. During that work, `max_concurrency()` can
report two. A process-wide parallelism limit of one does not prevent this
enqueue exception. Use `task_arena(1, 0)` when one simultaneous participant is
required; zero reserved slots does not exclude application threads.

Useful configurations are:

| Construction | Interpretation |
|---|---|
| `task_arena(automatic, 1)` | Automatic total concurrency with one slot available to application threads |
| `task_arena(1, 1)` | Enqueue exception: may execute an application thread and a mandatory worker simultaneously |
| `task_arena(1, 0)` | One simultaneous participant, either an application thread or a worker |
| `task_arena(4, 1)` | At most four participants; at most three ordinary workers, with one slot reserved for application entry |
| `task_arena(4, 0)` | At most four participants and no slot reserved specifically for application entry |
| `task_arena(4, 4)` | No ordinary worker slots; generally unsuitable for asynchronously submitted work |

The current `TbbScheduler(int max_concurrency)` constructor uses:

```cpp
arena_(max_concurrency, /*reserved_for_application_threads=*/max_concurrency == 1 ? 0 : 1)
```

The argument is arena concurrency, not a request for that many worker threads.
The constraints overload makes the same choice for an explicit
`constraints.max_concurrency == 1`. Automatic concurrency retains the default
reservation policy; request one explicitly when that limit is required.

### `global_control::max_allowed_parallelism`

`oneapi::tbb::global_control` applies process-wide while the control object is
alive. For `max_allowed_parallelism = N`, oneTBB limits active worker threads to
`N - 1`. Multiple overlapping controls use the smallest active value.

With `max_allowed_parallelism = 1`, ordinary TBB work is executed serially by
application threads. The documented exception is work submitted with
`task_arena::enqueue()`, for which oneTBB may still run one worker.

This control is different from an arena limit:

- `global_control` bounds worker participation across the process
- `max_concurrency` bounds simultaneous participation in one arena
- reserved application-thread slots partition arena capacity between workers
  and application-thread entry

`TbbScheduler{1}` uses an unreserved sole arena slot, so its single-participant
limit does not depend on a global control. A global limit of one can additionally
restrict ordinary worker participation elsewhere in the process. Neither
setting guarantees OS-thread identity: Uni20 scheduler admission uses
`task_arena::enqueue()`, for which oneTBB may still run one worker. Use
`DebugScheduler` when deterministic calling-thread execution is required.

## Arena Constraints

`task_arena::constraints` combines concurrency with placement preferences or
restrictions.

| Constraint | Purpose |
|---|---|
| `max_concurrency` | Maximum simultaneous arena participation |
| `numa_id` | Select the NUMA node on which arena participants execute |
| `core_type` | Prefer or select a processor core type on supported systems |
| `max_threads_per_core` | Limit simultaneous participants per physical core |

Constraints do not:

- allocate or place tensor storage
- move a task between arenas
- create a fixed set of workers
- replace `EpochQueue` dependency ordering
- guarantee that the requested degree of parallelism is available

`TbbNumaScheduler` uses separate constrained arenas. The scheduler must route a
task to the intended arena; oneTBB does not move tasks between those arenas to
balance load.

When constructing `TbbScheduler` from constraints, set `max_concurrency` on the
constraints object.

### Arena observers and thread-local execution state

`oneapi::tbb::task_scheduler_observer` reports when a worker or application
thread enters or leaves a particular arena. It can establish thread-local state
for the duration of arena participation and restore prior state on exit.

This does not assign permanent workers to the arena. A different worker may
enter the same slot later, and one worker may participate in different arenas
over time. Observer callbacks establish a logical execution context on every
participant.

`TbbCudaScheduler` uses this mechanism for one scheduler arena per device:

- validate the CUDA device before enabling observation;
- select the arena's CUDA device on entry;
- restore the previous CUDA device on exit;
- never throw from observer callbacks;
- keep CUDA streams and provider handles in device-local pools rather than
  attaching them permanently to worker identities.

If an exact persistent set of OS threads is required, `task_arena` is not that
facility; a dedicated-thread executor would be needed. No current Uni20 CUDA
requirement justifies that executor.

## Task Groups

A `task_group` tracks dynamically added tasks. Its main operations are:

- `run(f)`: add and schedule a task, then return
- `defer(f)`: create a task handle without scheduling it yet
- `wait()`: wait until all tasks in the group complete or are cancelled
- `run_and_wait(f)`: schedule one task and wait for the group
- `cancel()`: request cancellation through the group's context

The group is an ownership, completion, and cancellation boundary for submitted
TBB tasks. It is not a scheduling domain. In Uni20, each submitted TBB task is
one coroutine **activation**: it resumes a coroutine and ends when that
coroutine next suspends or completes. `task_group::defer()` registers the
activation before `task_arena::enqueue()` associates and publishes it to the
scheduler arena.

If the coroutine suspends on an epoch or external event, the activation is
complete even though the logical coroutine remains alive. Its awaiter owns the
suspended coroutine handle and submits a later activation when readiness
changes. Consequently, scheduler migration changes the target of the
next activation; it does not transfer persistent task-group membership because
no such membership exists while the coroutine is externally suspended.

`task_group::wait()` is scheduler-aware: the waiting thread may execute
available TBB tasks, including tasks unrelated to that group. It is therefore a
good implementation for `TbbScheduler::run_all()`, whose current contract is
quiescence of activations submitted to that scheduler. It does not prove that
externally suspended coroutines have completed.

The defer-before-enqueue sequence gives accepted admission a clear ordering
boundary. A `run_all()` call sequenced after admission returns must include that
activation. An unordered concurrent `run_all()` may linearize before a new
admission starts and return without it; scheduler quiescence is not a global
submission fence.

It is not a suitable implementation for a targeted `Async<T>::get_wait()`:

- it waits for every task in the group, not the requested epoch
- when called from a task in the same group, the caller itself is unfinished
- unrelated long-running or externally suspended tasks would delay the wait

## C++ Coroutine Suspension and TBB Task Suspension

Uni20 uses two distinct suspension mechanisms.

| Mechanism | What is preserved | How it suspends | How it resumes |
|---|---|---|---|
| C++ coroutine | Coroutine frame containing parameters and live locals | `co_await` returns control through the coroutine protocol | Uni20 scheduler reschedules the coroutine handle |
| TBB resumable task | The current ordinary synchronous TBB task stack | `oneapi::tbb::task::suspend(callback)` | Any thread calls `oneapi::tbb::task::resume(point)` |

They compose. A TBB task may be in the middle of resuming a Uni20 coroutine;
that coroutine may call an ordinary synchronous Krylov routine; Krylov may call
`get_wait()`; and `get_wait()` may suspend the surrounding TBB task stack. No
Krylov coroutine is required.

### `task::suspend()` and `task::resume()`

The intended shape is:

```cpp
oneapi::tbb::task::suspend(
    [&](oneapi::tbb::task::suspend_point point) {
      register_waiter(point);
    });

// Later, after the condition becomes ready:
oneapi::tbb::task::resume(point);
```

The callback receives the continuation token for the current TBB execution
stack. It should register that token and return promptly. Once suspended, the
thread can participate in other TBB work. `resume(point)` signals that the
saved stack may continue and may be called from another thread.

Uni20's waiter contract should require:

- each `suspend_point` is resumed exactly once
- the callback does not block or recursively wait
- readiness is rechecked while registering the waiter, under the same lock used
  by notification, so completion cannot be lost between the initial check and
  registration
- notification removes ready waiters under the lock and calls `resume()` after
  releasing the lock

Nested synchronous function calls are supported because the whole current task
stack is preserved. A resumed stack may later suspend again. Multiple tasks may
also have independent outstanding suspension points. Uni20 does not need two
simultaneous suspension points for the same already-suspended stack.

## The Single-Thread Case

The single-participant case uses `task_arena(1, 0)`, not a fallback that bypasses
TBB. It limits the arena to one simultaneously executing activation, but the
participant may be an application thread or a worker. Serial lightweight
batches also enter that arena. A suspended synchronous wait is not an executing
activation; other work can run while its stack is suspended.

Consider:

```cpp
TbbScheduler scheduler{1};
```

For Uni20's non-blocking admission path, the timeline is:

1. The application thread registers task A with `task_group::defer()` and
   publishes it with `task_arena::enqueue()` without entering the arena.
2. A may execute on a worker without application-thread participation. If the
   requested epoch is not already ready, the application thread calls
   `get_wait()` through `execute()`.
3. `execute()` enters the arena when possible. If its sole slot is occupied,
   oneTBB may delegate the wait body to the arena instead; the application
   thread need not itself participate.
4. If A has not yet made the epoch readable, the wait suspends its synchronous
   TBB execution stack, allowing A to execute and resume the saved suspension
   point when the result becomes readable.
5. Once the wait body completes, `execute()` and the application's `get_wait()`
   return.

This also works when the wait is nested inside a task:

1. A participating thread executes task A while `run_all()` is waiting in
   `group.wait()`.
2. A calls synchronous Krylov code.
3. Krylov schedules task B and calls `get_wait()` for a scalar reduction.
4. A's current synchronous stack is suspended.
5. The sole available arena participant executes B.
6. B publishes the result and resumes A.
7. Krylov and A continue normally.

`execute()` alone is insufficient. If step 4 is replaced by a condition-variable
wait, spin loop, or `std::this_thread::yield()`, the sole participating thread
remains trapped in A and B cannot run.

### Comparison with `DebugScheduler`

`TbbScheduler{1}`, backed by `task_arena(1, 0)`, is similar to
`DebugScheduler` in important ways:

- only one task body executes at a time
- progress is cooperative rather than preemptive
- the configuration exposes code that incorrectly depends on a spare worker

It is not equivalent to `DebugScheduler`:

- `DebugScheduler` uses Uni20's explicit deterministic queue and batch behavior
- `DebugScheduler` executes on its calling thread, while enqueued TBB work may
  use oneTBB's eventual-execution worker
- oneTBB task selection order remains unspecified even with one slot
- a TBB wait may execute unrelated eligible tasks from the arena
- `DebugScheduler` provides direct no-runnable-task deadlock diagnostics
- one-slot TBB exercises arena entry, task-group waiting, and resumable-task
  behavior that `DebugScheduler` does not

Uni20 should test both schedulers. The one-slot TBB configuration is a required
execution mode, not merely a performance setting or a substitute for
`DebugScheduler`.

## Uni20 Targeted-Wait Implementation

`TbbScheduler::wait_for()` follows this conceptual structure:

```cpp
void wait_for(WaitRequest const& request)
{
  if (request.is_ready())
    return;

  require_notification_registration(request);
  arena_.execute([&] {
    while (!request.is_ready()) {
      auto wait = make_suspension_latch();
      request.notify_when_ready(weak_wakeup(wait));
      if (request.is_ready())
        return;
      suspend_and_install_resume_point(wait);
    }
  });
}
```

The initial check keeps ready-value access cheap. `execute()` normalizes callers
that are outside the arena and callers already participating in it. The
suspension is targeted to one readiness predicate rather than all tasks in
`tg_`.

There are three distinct levels of targeting:

1. **Targeted completion:** `get_wait()` returns as soon as its particular
   epoch is readable; it does not wait for scheduler-wide quiescence.
2. **Targeted wakeup:** the transition of that epoch to readable should resume
   the registered suspension point directly, rather than relying on polling or
   completion of every TBB task.
3. **Targeted work selection:** choosing only tasks on the dependency path to
   that epoch.

The first two are appropriate Uni20 contracts. The third is not available from
oneTBB's ordinary arena scheduler because TBB does not know the `EpochQueue`
dependency graph. While the wait is suspended, TBB may select any eligible task
in the arena. If it selects a long-running unrelated task in a one-slot arena,
the target wait is delayed until that task reaches a scheduling point.

Uni20's baseline policy is throughput-oriented and work-conserving. It should
not attempt to reconstruct the dependency path and assign higher TBB priority
to tasks merely because an application thread is waiting for one epoch. Krylov
scalar reductions are commonly phase boundaries at which the machine is already
working on the contributing matrix-vector and inner-product operations. Running
other eligible work while that wait is outstanding is acceptable.

The debug `TaskRegistry` can expose an approximate explicit DAG, but that graph
is diagnostic data. It may be disabled, incomplete, or expensive to maintain.
Normal scheduling behavior must not depend on debug instrumentation. Critical-
path prioritization would require always-available dependency metadata and
evidence from profiling that the latency benefit outweighs scheduling overhead
and possible starvation. It is not part of the current design.

`IAsyncScheduler::WaitRequest` carries a readiness predicate and a targeted
wakeup-registration function. For a pending TBB wait, registration is required;
an unsatisfied predicate-only request throws `async_state_error`. Ready-value
access still returns immediately. Polling schedulers such as `DebugScheduler`
can ignore the registration.

Registration must not block waiting for readiness, and must close the race
between checking the predicate and subscribing to its next change. It may invoke
the supplied non-throwing callback inline. The suspension latch handles a
notification before or after installation of the resume point and resumes that
point at most once. Providers may retain callbacks: a weak reference makes late
notifications harmless after the waiting stack returns.

Reader waits subscribe to the epoch's mutex-protected transition to `Reading`.
Writer waits subscribe to availability of the exclusive write gate, and acquire
it through their readiness predicate. Notification does not reserve the gate:
if another writer acquires it first, the waiter checks again and registers a
fresh notification. Epoch callbacks run outside the epoch mutex.

### Diagnosing stalled waits

TBB waits have no internal deadline, idle-work counter, or execution-scope
bookkeeping. They remain suspended until notified, including while the arena
has no runnable tasks. Later submissions and external completions can still
satisfy the dependency. Application and worker threads use the same path;
no thread-local execution chain is carried across a resumable-stack switch.

Use an attached debugger or optional `TaskRegistry` diagnostics to investigate
stalled work. The diagnostics service can consume a signal such as `SIGUSR1`
and write a best-effort DAG snapshot without terminating the process. External
test timeouts bound tests of progress; timeout-based deadlock detection is not
a scheduler feature.

## What Suspend/Resume Does Not Fix

Resumable tasks provide progress for valid dependency graphs. They do not make
invalid graphs valid.

- Waiting for an epoch that depends on the current task is still a cycle.
- Acquiring incompatible read and write epochs for aliased storage may still
  deadlock.
- A paused scheduler cannot make progress until resumed.
- A suspension point that is never resumed is a leaked blocked computation.
- Blocking inside the suspension callback can still exhaust scheduler progress.

Use `DebugScheduler` and `TaskRegistry` diagnostics to find dependency cycles.
Use the TBB single-thread configuration to find waits that accidentally depend
on spare workers.

## Uni20 Design Rules

1. Treat arena concurrency as slot capacity, not worker count.
2. Enter the scheduler's arena with `execute()` before an application-thread
   targeted wait.
3. Use `task_group::wait()` for scheduler-wide quiescence, not for one async
   value.
4. Use resumable-task suspension for nested synchronous waits under
   `TbbScheduler`.
5. Do not spin or use `std::this_thread::yield()` while waiting for TBB work.
6. Keep suspend callbacks non-blocking and make waiter registration race-free.
7. Do not rely on thread identity or thread-local state across a suspension.
8. Test scheduler changes with one total participant as well as several
   workers.
9. Register activations with `task_group::defer()` before non-blocking
   `task_arena::enqueue()` publication so `run_all()` accounting cannot miss an
   accepted submission.
10. Remember that `EpochQueue` defines dependency legality; TBB only executes
    work that those dependencies make ready.

## Recommended Tests

The TBB scheduler test suite should include:

- top-level `get_wait()` with `max_allowed_parallelism = 1` and
  `TbbScheduler{1}`
- nested `get_wait()` from an async task under the same configuration
- two sequential suspensions of the same resumed synchronous stack
- several tasks suspended independently and resumed in different orders
- readiness becoming true during waiter registration
- repeated ready-value access that never enters the suspension path
- a multi-worker stress test for lost wakeups and duplicate resume attempts
- an idle wait satisfied by a later submission
- writer waits behind preceding readers or an active writer
- concurrent coroutine waits nested inside lightweight batches
- unsatisfied predicate-only requests rejected before entering the arena
- initial admission and rescheduling from outside a saturated arena
- `run_all()` ordered after accepted admission waiting for that activation

Single-participant tests should confirm progress without relying on OS-thread
identity. Saturated-arena tests should use latches or barriers to establish
ordering and use timeouts only as deadlock escape paths.

## Official References

These specification pages define the contracts summarized above:

- [oneTBB task arena specification](https://uxlfoundation.github.io/oneTBB/main/specification/source/task_scheduler/task_arena/task_arena_cls.html)
- [oneTBB task scheduler specification](https://uxlfoundation.github.io/oneTBB/main/specification/source/task_scheduler.html)
- [oneTBB task group specification](https://uxlfoundation.github.io/oneTBB/main/specification/source/task_scheduler/task_group/task_group_cls.html)
- [oneTBB global control specification](https://uxlfoundation.github.io/oneTBB/main/specification/source/task_scheduler/scheduling_controls/global_control_cls.html)
- [oneTBB resumable tasks specification](https://uxlfoundation.github.io/oneTBB/main/specification/source/task_scheduler/scheduling_controls/resumable_tasks.html)

## Related Uni20 Documentation

- [Async schedulers](schedulers.md)
- [Async runtime model](runtime_model.md)
- [Buffers and awaiters](buffers_and_awaiters.md)
- [Task registry debugging](task_registry_debug.md)
- [Coroutine primer](coroutines_primer.md)
