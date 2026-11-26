# Exceptions and Cancellation

This document explains how failures move through async coroutines and epoch contexts.

The key distinction is:

- access errors (uninitialized read/write)
- cancellation signals
- unhandled exceptions escaping coroutine bodies

## Exception Hierarchy

Defined in `src/uni20/async/async_errors.hpp`.

| Type | Meaning |
|---|---|
| `async_error` | base class for async runtime failures |
| `async_state_error` | invalid state/access class |
| `async_storage_missing` | storage metadata unexpectedly missing |
| `async_value_uninitialized` | direct async value access before initialization |
| `async_cancellation` | cancellation-related base class |
| `task_cancelled` | coroutine/task cancellation signal |
| `buffer_error` | buffer-specific state failure |
| `buffer_cancelled` | read buffer cancelled intentionally |
| `buffer_read_uninitialized` | read attempted before any value was constructed |
| `buffer_write_uninitialized` | mutable write ref requested before construction |

## Access-Path Behavior

### Read side

- normal read path on missing value: `buffer_read_uninitialized`
- `or_cancel()` path on missing value: `task_cancelled`
- every read path propagates a stored writer exception unchanged; `or_cancel()`
  only changes the missing-value case

### Write side

- awaiting an unconstructed writer still returns `WriteAccessProxy<T>`
- requesting `get()`, `operator->`, or conversion to `T&` from that proxy: `buffer_write_uninitialized`
- preferred fix: `co_await writer = value` (or proxy `emplace(...)`)
- consuming a value through `take()`, `take_release()`, or `move_from_wait()`
  rethrows the producer's exception before checking for a constructed value.
  Empty storage with no producer failure raises `buffer_write_uninitialized`.
- in-place updates also rethrow a failure that the epoch inherited, so an
  update never builds on a failed value. That covers value access through a
  write proxy (`get()`, `operator->`, conversion to `T&`, and compound
  assignment such as `+=`) and `Async<T>`'s compound operators.
- assignment, `emplace(...)`, `storage()` and an async alias's descriptor are
  not checked, because overwrites use them too. Linalg kernels reach their
  outputs through `storage()` or a descriptor, so they do not check. Code that
  updates through `storage()` can call `WriteBuffer::check_inherited_failure()`
  first.

## Suspension Registration

Uni20 awaiters that receive `BasicTask` or `TaskFactory` in
`await_suspend(...)` must make that function `noexcept`. At that point the
coroutine is suspended and its ownership is being transferred to the awaiter.
If registration threw after moving one or more ownership claims, unwinding
could not reliably undo waiter registration or recover every claim.

Perform fallible preparation before acquiring task ownership when possible.
Failures discovered after the task becomes ready should be reported by
`await_resume()`, where they follow normal coroutine exception propagation.
The `TaskAwaitable`, `TaskFactoryChildAwaitable`, and `TaskFactoryAwaitable`
concepts enforce the non-throwing registration rule.

`try_await(...)` changes only unavailable readiness into an empty `optional`.
It does not suppress failures: exceptions from the readiness probe or the
child's `await_resume()` propagate through the awaiting coroutine normally.

## Unhandled Exception Flow in Coroutines

When a coroutine throws and does not catch:

1. promise catches in `unhandled_exception()`
2. if exception is `task_cancelled`, promise sets cancel-on-resume flag
3. otherwise the debug registry marks the task failed and, when enabled, captures
   live exception diagnostics before the task leaves the registry
4. promise stores `std::exception_ptr`
5. promise forwards exception to registered epoch sinks

Set `UNI20_DEBUG_DAG_DUMP_ON_EXCEPTION=1`, or use
`TaskRegistry::set_coroutine_exception_diagnostics_options(...)`, to emit the
live registry report and optional DOT artifact. This policy is off by default
because an exception escaping one coroutine may still be expected and observed
at a downstream `Async<T>` boundary.

At final suspension, the stored exception goes to the awaiting coroutine, which
receives it at `co_await child`.

## Caught Upstream Failures

A coroutine sees a producer's failure only through a value it reads or takes,
and ordinary C++ `catch` rules apply. If the coroutine catches the failure and
returns normally, its parent continues past `co_await child`. The outcome is
the same whether the value was ready when the coroutine awaited it or the epoch
resumed the coroutine after the producer failed (#166).

- The source epoch keeps the failure, so its other consumers still receive it.
- A writer on an epoch that inherited a failure still runs. `take()` and
  in-place access through a write proxy rethrow the failure. A writer that
  only overwrites the value never sees it. Readers of that epoch still receive
  the inherited failure.
- If an exception escapes the coroutine, the parent receives that exception,
  not an upstream failure the coroutine caught earlier.

An epoch that resumes a coroutine with a stored failure only notes the failure
for task diagnostics. The debug registry uses that note to tell an exception
the coroutine passed on from one it raised.

## Exception Sink Registration

### Automatic path

`WriteBuffer` coroutine parameters are auto-registered as sinks.

This covers the normal kernel pattern where output buffers should receive failures
from the coroutine computing them. Explicitly releasing a writer ends exception
delivery through that writer. A later failure still reaches the coroutine's
remaining output writers. An output can also fail while waiting for an earlier
epoch to finish. A successful predecessor preserves that failure.

### Owning writers

A `WriteBuffer` parameter stays registered as a sink when an owning form takes
its writer: `co_await buffer.transfer()`, `buffer.transfer().storage()`, an
owning `take()`, or `write_to(buffer.transfer(), value)`. An exception thrown
while one of these holds the writer therefore reaches the epoch. That includes
an exception from constructing the value
([#169](https://github.com/Uni20-dev/uni20/issues/169)).

An owning holder that finishes normally, at the end of its scope or when
`write_to` has emplaced its value, ends exception delivery to the epoch, as
`release()` does. So does a `write_to` dropped without being awaited, which
publishes the previous value. A later exception escaping the coroutine then leaves that
value alone.

A coroutine that receives an owning holder by value, as in
`co_await child(co_await buffer.transfer())`, destroys its parameters after the
promise has handled an escaping exception. When an exception has escaped the
coroutine body, its frame is destroyed while that exception is rethrown, so the
holder counts as destroyed by the exception, wherever it sits among the
parameters. A failure that the body caught does not count. The failure then reaches
the epoch from the caller's `WriteBuffer` sink, as described next.

An owning holder that an exception destroys publishes the epoch at that point.
That happens before the promise reaches `unhandled_exception()` and delivers the
failure ([#163](https://github.com/Uni20-dev/uni20/issues/163)).
- Readers that still hold the epoch when the failure arrives see it, and so
  does a successor that starts after them.
- A successor that starts in between keeps its own value, and a reader on
  another thread can see the previous value.
- A failure that arrives after the epoch has finished is dropped. Builds with
  task diagnostics (`UNI20_DEBUG_ASYNC_TASKS`) still record the exception in the
  task registry when it escapes the coroutine.
- If the coroutine catches that exception, delivery stays open. A later
  exception escaping the coroutine can still fail the epoch while it is being
  read.

If a failing producer must always fail its value, hold the writer through the
non-owning `co_await buffer` instead. That form keeps the writer until the
coroutine finishes.

### Explicit path

Use:

```cpp
co_await propagate_exceptions_to(buf1, buf2, ...);
```

This is useful when errors should be routed to additional buffers (including read buffers in advanced flows).

### Explicit sink lifetime guard

If an explicit sink is destroyed during stack unwinding before promise
`unhandled_exception()` runs, and the promise has no other sink for that epoch,
the runtime aborts intentionally (`CHECK`).

- A copy of a registered buffer registers with the same promise. Destroying the
  copy while the original remains loses nothing and does not abort.
- A failed coroutine's parameters are destroyed during a rethrow of its
  exception (see "Owning writers" above). Moving the only registered sink into
  a child coroutine that fails therefore aborts, even if the parent catches the
  exception.

This fail-fast behavior prevents silent loss of exception routing in hard-to-debug cases.

## Cancellation Semantics

Cancellation state is represented through promise flagging (`set_cancel_on_resume()`).

Effects:

- cancelled tasks follow cancellation cleanup path at final suspend
- continuations can be destroyed recursively when cancellation is active
- downstream await paths may throw `task_cancelled` depending on awaiter mode

Cancellation and failure are deliberately distinct:

- an absent value observed through `or_cancel()` cancels the current task
- a genuine exception from an upstream epoch remains an exception, including
  when the consumer used `or_cancel()`
- cancelling a task releases its unwritten output buffers, allowing absence to
  propagate through further cancellation-aware consumers
- destroying an unobserved `Async<T>` result discards its value or stored
  failure once no buffers or aliases retain the state; destruction does not
  turn the result into cancellation

This last rule permits demand-driven graph pruning. In particular, an unseeded
reverse-mode gradient branch may disappear without reporting an error, while a
failure on a branch that is actually consumed remains observable.

## Practical Guidelines

- use proxy assignment (`co_await writer = value`) for first write to default `Async<T>`
- treat `or_cancel()` as a control-flow choice, not just a convenience API
- do not use cancellation to suppress a genuine upstream failure
- register explicit sinks only when routing intent is clear
- keep explicit sink objects alive until coroutine completion

## Testing Focus Areas

Important tests in `tests/async/test_async_basic.cpp` cover:

- handled vs unhandled exception propagation
- automatic routing to write parameters
- explicit routing with `propagate_exceptions_to(...)`
- duplicate explicit registration no-op behavior
- fail-fast path for destroyed explicit sinks during unwind

`tests/async/test_async_caught_failure.cpp` runs caught and uncaught upstream
failures in both scheduling orders.

## Related References

- Buffer API details: `buffers_and_awaiters.md`
- Runtime semantics: `runtime_model.md`
- Debug dump triage: `task_registry_debug.md`
- Fast lookup: `quick_reference.md`
