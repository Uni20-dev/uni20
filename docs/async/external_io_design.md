# Async I/O and external completion

**Status: future design direction.** This note records use cases and the
boundary between I/O and Uni20's computational DAG. It does not introduce an
implemented I/O API, select a provider, or commit to an implementation schedule.

The existing [runtime ownership model](runtime_model.md#coroutine-frame-ownership)
already describes a coroutine suspended on an external event: its awaiter
owns the task until readiness schedules another activation. The
[scheduler roadmap](../roadmap.md#3-harden-async-operation-composition) also
anticipates CUDA/MPI completion notifications. General file, socket and
subprocess I/O would extend this direction with concrete operation lifetimes,
progress and failure handling.

The existing [MPI persistent-store proposal](../backends/mpi/persistent_dispatch.md)
is a concrete related direction: its proposed `co_load()` suspends while data
is materialized from disk, MPI or a storage rank, then returns a pinned
resident value. A general I/O adapter could supply part of that progress
mechanism. Persistent identity, cache policy and durability remain concerns
of the store; this note does not replace that design.

## Motivating workloads

| Workload | Useful overlap | Required boundary |
|---|---|---|
| Checkpoint output | Write a completed snapshot while the calculation advances | A consistent snapshot, retained payloads and an explicit publication policy |
| Input streaming and prefetch | Read the next batch while computing the current one | Bounded buffers and dependencies before consumers access the data |
| Remote computation | Continue other work while a remote result is pending | Request identity, failure reporting and bounded outstanding requests |
| Observable output | Stream completed results to storage or visualization | Explicit buffering, ordering and backpressure |
| External programs | Compose subprocess results with numerical operations | Pipe framing, process lifetime and operation completion |

These uses are independent of any Python interface. Python event-loop
integration would be another adapter at the boundary, with its own interpreter
and callback lifetime requirements.

The experimental Bethe search illustrates persistent subprocess communication,
but its intended integrated C++ path will call numerical kernels directly.
Async I/O is not a prerequisite for that search or its Async integration.

## Proposed division of responsibility

The compute scheduler continues to run **ready** coroutine activations. An
I/O progress service observes operating-system readiness or completion and
returns suspended tasks to their recorded scheduler routes. It should serve
many outstanding operations without reserving a compute worker for each wait.

An operation's awaiter/state would retain its task, buffers, resource handles
and completion outcome until ownership can safely transfer. Higher-level
adapters would implement byte-stream framing, file formats or remote request
protocols. A socket becoming readable is not necessarily a complete message;
short reads and writes belong in those operation/protocol layers.

Completed data would enter ordinary Async dependencies through the existing
value and buffer mechanisms. An external callback should not create competing
read/write epochs on a shared value without respecting the
[queue enrollment contract](runtime_model.md#queue-enrollment-threading-contract).
Establish the required buffer capabilities through a causally ordered path;
then let completion make their operation ready. The progress service may need
internal synchronization for its concurrent operation registry. That is a
different concern from ordering numerical access through Async epochs.

Public naming remains open: direct I/O awaitables, operations producing
`Async<T>` values, or both could serve different layers. This note does not
define signatures for file reads, writes or subprocess requests.

## Lifetimes, cancellation and shutdown

The future adapter needs a race-safe handoff between registration and
completion, including immediate completion during registration. Success,
failure, timeout and cancellation must resolve an operation once, without
losing or duplicating its task ownership. Callbacks must not retain a route to
a scheduler that has already been destroyed.

Cancelling a waiter and cancelling an underlying operation are separate
choices, particularly when several consumers share a result. A cancellation
request is not proof that the OS or provider has stopped accessing a buffer.
Resources must remain alive until completion or confirmed cancellation, even
if the caller has stopped waiting. The implementation must map outcomes into
the established [exception and cancellation model](exceptions_and_cancellation.md).

Shutdown needs an explicit order: stop admitting operations, finish or cancel
accepted work, retire callbacks and their resources, then destroy the progress
service and any scheduler routes they could use. User code should wait for its
final result or operation group. As the [scheduler documentation](schedulers.md)
explains, `run_all()` establishes activation quiescence; it does not prove
that externally suspended operations have completed.

Existing host-side waiting also needs an explicit integration plan. For
example, `DebugScheduler::help_while_waiting()` currently aborts if the
requested result is unavailable and no task is runnable. A future adapter
cannot assume that every current scheduler wait already supports independent
I/O progress. Define how supported wait paths receive external notifications,
and how deterministic tests drive completion, before exposing the adapter.

## Buffers, bounds and persistence

I/O lifetime must also protect the **contents** of a buffer. An asynchronous
write can retain a read capability until the provider finishes using its
memory, delaying later mutation of that storage. Alternatively, copying into
a bounded staging buffer or retaining an immutable version can allow the
calculation to advance sooner. Completion semantics should distinguish when
the source can be reused from when the output operation has finished.

Similarly, a read into destination storage must retain exclusive access until
the provider stops writing; dependent consumers become ready only after the
operation publishes its outcome. Set bounds on pending requests, buffers and
staging bytes. Waiting for capacity should suspend the producer, allowing
other runnable work to continue.

Checkpointing requires an application-defined consistent snapshot of values
and algorithm state. An arbitrary coroutine suspension is not automatically
a checkpoint boundary. Completing a write also does not, by itself, specify
durability or atomic checkpoint publication. Those guarantees belong to the
persistence layer and must be explicit when it uses the I/O adapter.

## Provider choices and a possible first step

Keep the completion/ownership boundary separate from provider selection.
Plausible alternatives include integration with an existing event loop, a
Uni20-owned readiness/completion service, and a bounded pool for unavoidable
blocking I/O calls. Such a pool would be separate from numerical workers;
it should not grow by one thread per pending operation. Readiness-based stream
handling and regular-file I/O need separate capability decisions; a readiness
notification alone is not a guarantee that a file operation cannot block.

A useful first implementation could be one bounded operation on a pipe or
socket, backed by a controllable test provider. It should establish ownership,
completion routing, cancellation and progress when no compute task is
runnable before extending to files or remote request pools. Keep Linux as
the primary platform and macOS portability as an intended target; choose
provider dependencies and public API shape in a later implementation proposal.

Validation should cover completion before and after suspension, partial
transfers, end-of-stream and errors, cancellation/completion races, resource
lifetime during shutdown, and backpressure. Measure CPU usage while waiting,
throughput, latency and retained memory under bounded concurrency. This would
show whether the adapter composes correctly and efficiently with numerical
work without conflating I/O overlap with faster numerical kernels.
