# Python extensions and asynchronous numerical values

Status: design proposal, 2026-10-30. This document records a proposed direction
and alternatives for discussion; it does not describe an implemented Python
API. Implementation is deferred. All Python examples below are
illustrative, including names and keyword arguments.

The aim is to make Uni20 useful from ordinary scientific Python: configure
execution resources, call numerical libraries, combine results, and inspect
values when needed. A Bethe-ansatz extension and a small NumPy-like array
interface would exercise the common design before larger MPS and PEPS bindings.

## Reading guide

- [Direction and present implementation](#direction-and-present-implementation)
- [What a calculation should look like](#what-a-calculation-should-look-like)
- [Core and extensions](#core-and-extensions)
- [Templates at the Python boundary](#templates-at-the-python-boundary)
- [Dtypes, scalars and precision](#dtypes-scalars-and-precision)
- [Asynchronous values and observation](#asynchronous-values-and-observation)
- [Execution and lifetime](#execution-and-lifetime)
- [NumPy compatibility and array ownership](#numpy-compatibility-and-array-ownership)
- [Numerical results and errors](#numerical-results-and-errors)
- [Optional Python coroutine integration](#optional-python-coroutine-integration)
- [Prototype sequence and evidence](#prototype-sequence-and-evidence)
- [Open decisions](#open-decisions)

## Direction and present implementation

The discussion has converged on these goals:

1. **An extension-based Python library.** A shared core supports separately
   developed modules such as `uni20.bethe`, `uni20.mps` and `uni20.peps`.
2. **Native execution controlled from Python.** Python can select a scheduler,
   resource limits and operation budgets without using Python coroutines.
3. **Immediate submission with deferred observation.** Numerical objects can
   represent pending values. Arithmetic composes native dependencies; numerical
   observation waits for the required result.
4. **Familiar numerical types.** Standard Python and NumPy scalars and dtype
   specifications should work naturally. Extended precision must survive the
   binding boundary without silent narrowing.
5. **The existing C++ semantics remain authoritative.** Bindings preserve
   native ownership, accessors, symmetry, dispatch and Async dependencies.

Specific Python class names, return-type policies, packaging and shutdown rules
are recommendations or open decisions below. They are not settled by these goals.

Today, [the Python binding](bindings.md) is a nanobind smoke module exposing a
greeting and build information. It does not bind tensors, numerical operations
or Async values. The [dtype promotion note](dtype_promotion.md) is also a design
proposal. The present extension is named `uni20`; there is no packaged core and
extension system yet.

The native building blocks already include `Async<T>`, scheduler-controlled
operations, typed tensors and views, and precision-aware scalars. Bethe already
has owning numerical results, asynchronous scans, bounded observable grids and
shared prepared backgrounds; see [Bethe's Async interface](https://uni20-dev.github.io/bethe/docs/async-execution/).
These are starting points, not proof that the
Python contracts below already work.

## What a calculation should look like

### A Bethe reference value

```python
import numpy as np
import uni20
from uni20 import bethe

with uni20.Execution(threads=8) as execution:
    energy = bethe.xxx_energy_async(128, dtype=np.float64,
                                   execution=execution)
    energy_per_site = energy / 128
    shifted = energy_per_site + 0.25
    print(f"{shifted:.12f}")
```

Submission returns a numerical object immediately, subject to admission limits.
The two arithmetic operations register dependencies and return pending results.
Formatting requests a numerical value and waits. No explicit `Future.result()`
or Python event loop is needed in this example.

For this proposed convenience function, the documented default would be the
periodic spin-1/2 XXX ground state with `J=1`, zero field, and total energy in
the unshifted Heisenberg convention. The division converts to energy per site;
the final addition changes its reference. Other boundary conditions and sectors
must be explicit, and solver restrictions must remain visible.

The detailed interface should also expose a state with roots, quantum numbers,
momentum and convergence information. It must not reduce every solver to a
single successful-looking energy.

### Array computations

```python
import uni20
import uni20.numpy as unp

with uni20.Execution(threads=8) as execution:
    a = unp.asarray(input_array, execution=execution)
    b = a @ a.T
    c = unp.exp(b)
    total = unp.sum(c)
    print(total)
```

The first prototype would implement a specified dense-array subset. Matrix
products and reductions would submit ordinary Uni20 operations. A reduction and
a Bethe energy should produce compatible asynchronous numerical values, even
though they originate in different extensions.

### Shared preparation and tensor-network comparisons

A second Bethe example should submit a Hubbard background and immediately
submit several dispersion curves reading it. Preparation runs once; consumers
wait through the native dependency graph. The Python caller observes only the
final curves. This tests more than launching independent functions in parallel.

Eventually an MPS energy and a Bethe reference could be subtracted before either
has finished. Both calculations use the same execution resources and scalar
policy. Their energy and Hamiltonian conventions still need to agree explicitly.

## Core and extensions

The proposed public organization is:

```text
uni20                 shared numerical types and execution interface
uni20.numpy           supported NumPy-like dense-array operations
uni20.bethe           Bethe-ansatz models and observables
uni20.mps             MPS/MPO algorithms
uni20.peps            PEPS algorithms
```

These are Python namespaces, not a requirement to put all code in one repository
or binary. Bethe remains an independent C++ library with its bindings maintained
alongside its models. Importing one optional extension should not require all
the others.

| Shared Uni20 responsibility | Extension responsibility |
| --- | --- |
| Scalar conversion, dtype interpretation and precision policy | Model parameters, physical conventions and numerical controls |
| Python ownership of execution resources and native Async state | Submission of the extension's native operations |
| Common tensor identity, metadata and interoperability | Supported rank/symmetry/operation instantiations |
| Exception transport, cancellation plumbing and diagnostics hooks | Model-specific convergence and result fields |
| Binding registration and compatibility checks | Declaring required capabilities and supported signatures |

The binding support layer adapts these concepts to Python. It does not create
another tensor storage system, kernel dispatcher or numerical scheduler.

### Packaging and binary compatibility

A conventional `uni20` Python package containing a compiled `_core` is a
reasonable starting point. Two plausible extension layouts are:

| Layout | Advantage | Cost or unresolved work |
| --- | --- | --- |
| Distributions install owned subpackages such as `uni20/bethe` | Direct correspondence between installation and import paths | Define package ownership so distributions never overwrite shared files; test editable installs and removal |
| Separate packages such as `uni20_bethe`, exposed through a Uni20 extension loader | Clear distribution ownership and optional discovery | Specify how dotted imports are provided and how discovery behaves; entry points alone do not implement `import uni20.bethe` |

For a first prototype, build core and extension modules against one compatible
Uni20 build. Separate importable modules do not require independently built
wheels immediately. Later packaging should declare compatibility with the
Uni20 runtime, scalar configuration, compiler/C++ ABI and nanobind ABI. Python's
stable extension ABI alone does not make arbitrary C++ objects interchangeable.

Nanobind can share registered types across compatible extensions. That facility
does not automatically give independently linked modules one Uni20 scheduler,
configuration state or diagnostics service. The common runtime services need
explicit ownership and a supported access path. See the
[nanobind compatibility guidance](https://nanobind.readthedocs.io/en/latest/faq.html#what-does-the-term-abi-compatible-mean-in-this-project).

A shared native runtime and compatible C++ binding SDK is the preferred initial
direction. A narrow versioned capsule/C interface is an alternative if
independent binary distribution later requires stronger isolation. It would
cost additional adapters and should not be designed before the required
cross-extension operations are understood.

## Templates at the Python boundary

Python selects among compiled native types and operations using runtime
metadata. The numerical computation remains in the selected C++ instantiation.
Rank, scalar type, symmetry and leg orientation retain their native meaning.

The core should not precompile the Cartesian product of every rank, symmetry,
precision and algorithm. An extension contributes the instantiations it needs
and declares which operation signatures it supports. A new Python argument
cannot manufacture an arbitrary uncompiled C++ template specialization.

Two representations deserve comparison:

- **A Python class for each bound native type**, with shared methods/protocols
  and factories. This follows native types closely, but public class names and
  cross-extension registration can become cumbersome.
- **A common Python facade retaining a typed native object**, with a registry
  selecting binding adapters. This gives a simpler public interface, but needs
  disciplined type erasure, diagnostics and registration. It must not introduce
  a second runtime-rank numerical tensor implementation.

Prefer a common user-facing interface, while leaving the precise class layout
open until a small prototype measures binding complexity and cross-module
interchange. Register a shared native type once. An MPS and a PEPS extension
must not create incompatible Python identities for the same native object.

The registry described here is proposed binding infrastructure, not a second
backend-selection mechanism. After validating a signature, adapters call the
normal native operation and dispatch path. Unavailable configured kernels must
cross the binding boundary as an exception, using the existing
[dynamic dispatch contract](../architecture/kernel_dispatch.md).

The first NumPy-style prototype can restrict dense rank and scalar combinations.
Higher ranks and symmetry-bearing tensors can be added by extensions without
changing scalar conversion or execution semantics. Plain dense NumPy export
does not carry symmetry metadata; block export or explicit densification must
make that loss of structure visible.

## Dtypes, scalars and precision

### Familiar standard types

Accept standard dtype spellings wherever supported:

```python
bethe.xxx_energy_async(128, dtype=float, execution=execution)
bethe.xxx_energy_async(128, dtype=np.float64, execution=execution)
bethe.xxx_energy_async(128, dtype=np.dtype("float64"), execution=execution)
```

These select the same binary64 calculation. Ordinary Python and NumPy scalar
arguments should not require wrapping in a separate Uni20 number class. If
`uni20.float64` is exposed, it should be a genuine alias or interoperable dtype
descriptor, not an incompatible scalar merely sharing the name. Prefer existing
NumPy scalar classes for fixed formats that NumPy represents faithfully.

Separate three concepts:

| Concept | Example | Meaning |
| --- | --- | --- |
| Numerical representation | `np.dtype("float64")` | The scalar format |
| Pending numerical object | The value returned by `xxx_energy_async` | Native storage, readiness and dependencies |
| Materialized scalar | Python `float` or NumPy scalar | A value available for ordinary Python consumption |

A pending scalar can have NumPy's binary64 dtype without being an instance of
Python `float` or `np.float64`. Exact-type checks in third-party code cannot be
made transparent in every case. Ordinary arithmetic and supported conversions
should remain natural.

Use the [promotion proposal](dtype_promotion.md) as the starting point for
tensor arithmetic: distinguish weak Python scalars from explicitly typed NumPy
values, keep in-place storage types fixed, and report algorithmic working
precision separately. Extensions should not invent independent promotion tables.

### Extended and runtime precision

| Format | Proposed interoperability policy |
| --- | --- |
| Binary32/64 and corresponding complex types | Standard NumPy dtypes and scalar types |
| Native extended precision | Accept `np.longdouble` only after checking the actual platform format; preserve values without an intermediate Python float |
| IEEE binary128 | A distinct supported representation where NumPy's native type does not match; never infer IEEE binary128 from the name `np.float128` |
| MPFR | Explicit working precision and lossless interchange with suitable Python types; representation remains an open decision |

On common x86 Linux builds, `np.float128` is the padded storage spelling of
80-bit `long double`. It is not IEEE binary128. Format checks must use actual
precision and representation, not storage size alone. NumPy documents this
[extended-precision distinction](https://numpy.org/doc/stable/user/basics.types.html#extended-precision).

Investigate `gmpy2.mpfr` interoperability before designing a wholly independent
Python MPFR ecosystem. Reuse and conversion are different options. Native
Uni20 `mpreal` can hold exact rationals as well as finite-precision MPFR values;
it also has explicit precision-agreement rules. A direct alias to another
library's scalar must not silently replace those semantics with ambient-context
rounding. See [native MPFR semantics](../tensor/mpreal.md) and
[gmpy2's MPFR interface](https://gmpy2.readthedocs.io/en/latest/mpfr.html).

Working bits belong to each request and retained numerical value. Constructing
an input from an exact integer, rational or decimal string must not first round
it through binary64. Independent requests with different precisions must coexist
on one scheduler. More working bits do not imply convergence or remove mesh and
iteration controls.

For materialization, `.item()` could preserve the selected scalar format, while
`float(value)` explicitly requests a Python float and may narrow. Numeric
formatting should use native precision directly. A custom NumPy dtype for
binary128 or MPFR is a possible later project; registering a dtype would not
by itself supply all NumPy ufuncs or linear-algebra operations for it.

## Asynchronous values and observation

### Execution model

The preferred model is immediate native submission with deferred observation.
Operations begin when dependencies and resources permit. Arithmetic on pending
values submits further native operations. This is different from postponing
all computation until a later graph-wide `compute()` call.

True deferred graph construction remains a plausible advanced mode, useful for
batch planning or optimization. It changes start time, error timing and resource
lifetime; it should be explicit rather than accidentally introduced by the
Python wrapper. No graph optimizer or automatic fusion is required for the
first interface. [JAX's asynchronous dispatch](https://docs.jax.dev/en/latest/async_dispatch.html)
provides a useful precedent for the user experience, not an implementation to
copy wholesale.

### Observation rules

The following is a proposed contract, not current behaviour:

| Python operation | Synchronization and result |
| --- | --- |
| `other = value` | Binds the same Python object; no numerical observation |
| Arithmetic, supported ufuncs, reductions | Submit dependencies; return pending numerical objects |
| Known dtype, rank and shape | Metadata only; no numerical wait |
| Shape determined by data, such as a selected-element count | May need a dependency or wait; specify this per operation |
| `float(value)` / `complex(value)` | Wait and explicitly convert to the requested built-in type |
| `.item()` | Wait and return a scalar, preserving supported precision |
| Numeric formatting or `str(value)` | Wait and render a bounded numerical preview |
| `repr(value)` | Prefer metadata/readiness without waiting |
| Scalar truth test | Wait for a boolean; a multi-element array retains NumPy's ambiguous-truth error |
| `.wait()` | Wait for the relevant value/operation without converting it |
| Copied import from an external mutable NumPy array | Finish reading external memory before returning; subsequent native computation can be async |
| Ordinary NumPy materialization | Wait for an exportable CPU value; follow the ownership rules below |

For `if energy < 0:`, comparison may return an async boolean and its truth test
waits. Python cannot decide the branch without a concrete answer. Chained
comparisons and Python `and`/`or` consequently introduce observations too.
Returning the default truthiness of a pending wrapper would be incorrect.

Python's `__float__`, arithmetic, formatting and `__bool__` hooks support these
behaviours; accessing a variable name itself has no universal materialization
hook. An extension demanding an exact built-in scalar can still require explicit
conversion. See the [Python data model](https://docs.python.org/3/reference/datamodel.html).

Representation policy has a real tradeoff. A nonblocking `repr()` makes IDE and
notebook inspection safe from surprising waits, but a bare notebook expression
may show status instead of numbers. Alternatives are a blocking bounded preview
or an explicit rich-display request. Prefer the nonblocking default initially,
with `print`, numeric formatting and an explicit preview for values.

### Scalar object or rank-zero array

Two viable designs are a dedicated async scalar facade over `Async<Real>` and
a rank-zero async tensor used for every numerical scalar. The former maps
directly to Bethe results and native scalar operations; the latter unifies
reductions and array protocols. Neither should force a model result into a
different numerical algorithm merely to fit the wrapper.

Prototype both at the binding level and choose a consistent public protocol.
Bethe energies and array reductions must compose without materialization even
if their native storage types differ. Explicit result containers may retain
`Async<State>` internally and expose pending field projections, such as energy,
without copying the complete state or waiting in an attribute getter.

### Function naming

| Choice | Benefit | Cost |
| --- | --- | --- |
| Blocking `xxx_energy` and pending `xxx_energy_async` | Submission is explicit at the model boundary | Two entry points even though returned pending values behave numerically |
| Async-by-default `xxx_energy` | Uniform with an async array interface | Hidden waits and delayed errors need particularly clear documentation |
| A mode/context that changes return behaviour | Short spelling in either mode | Identical source can change return type and synchronization based on ambient state |

Prefer explicit `_async` Bethe entry points for the first prototype, while the
dedicated array namespace can consistently return async objects. Reconsider the
default model spelling after using both. A mode that silently changes return
types is the least attractive option.

## Execution and lifetime

### Scheduler ownership and routing

An execution context owns, or safely retains, native scheduler resources.
Accepted work retains everything needed to complete: input owners, options,
precision, operation state and scheduler lifetime. Native scheduler pointers
are borrowed; the Python binding must provide the missing ownership guarantee.

Derived operations must have a defined execution route. Prefer inheriting the
context from their pending operands; immediate Python scalars add no competing
route. For operands from different contexts, initially require an explicit
destination/import policy or reject the combination. Later native support may
permit safe cross-context dependencies. Do not silently pick whichever process
global scheduler happens to be active.

Constructors and model calls without pending inputs need an execution-selection
policy. The examples pass an explicit context to root operations; the proposed
prototype does not make `Execution.__enter__` install an implicit default.
Calls omitting execution would initially require a separately configured
default or raise a clear error. A later scoped Python default
could select where new roots are submitted, but the route must be captured at
submission. Changing that default must not reroute existing work or replace the
process-wide native scheduler. Explicit execution requests and conflicting
operand contexts still need the cross-context policy above. Whether to create a
library-owned default automatically, its worker count, and its shutdown policy
remain open.

This requires an implementation audit. Some native scalar Async operator
helpers currently call the active scheduler through `schedule(...)`; a wrapper
carrying a context is not sufficient to make those helpers inherit that context.
Use or extend the native submission mechanisms so the binding contract is real.

`threads=8` should mean the scheduler's concurrency limit, including application
thread participation in cooperative waits. It does not promise eight dedicated
workers plus a calling thread. It is also distinct from `max_in_flight`, whose
scope and credit-release event must be specified. Bethe's ordered grids retain
credits through collection, including completed-but-uncollected points; a
general Python execution context need not use that same accounting. Retained
user-visible results need a separate storage policy. Keeping a completed array
alive must not consume an execution credit forever and prevent admission of an
operation that uses it.

Bounded admission must preserve progress of accepted dependency chains and
child tasks. In particular, a worker cannot block submitting a required child
while holding the last credit that child needs. Distinguish root admission,
internal task scheduling and retained-output storage; settle their precise
budget interfaces in the prototype. Existing BLAS/OpenMP threading must also
be coordinated to avoid multiplying parallelism unexpectedly.

Synchronous observations wait for their own result and admitted child work.
They must not use scheduler-wide `run_all()` as their completion boundary.
Waiting may participate in native execution and should release the GIL for the
native work, reacquiring it before Python conversion or exception delivery.

### Scope exit and abandoned handles

Prefer a structured execution scope for the first prototype: scope exit stops
new submissions, observes all operations owned by that scope, and releases its
resources after they finish. This is an operation-ownership boundary, not a
drain of unrelated work on a shared scheduler. Materialized results remain
usable after exit; submitting new dependent work on a closed context requires
an explicit supported route.

An alternative is a context retained by pending objects that continues beyond
the `with` block. It is convenient for returned pending results but makes scope
exit and error reporting less predictable. This remains an open policy choice;
it must be settled before publishing context-manager examples as supported API.

Dropping a Python handle must not invalidate accepted work or implicitly cancel
other readers. Cancellation is explicit and cooperative: stop admission where
supported, finish already active kernels, then report the operation's terminal
state. Cancelling one consumer must not cancel a shared preparation or sibling
consumers without an explicit policy.

Owned operations must record execution exceptions even if the caller never
inspects their values. Prefer reporting otherwise-unobserved errors at explicit
scope close, while preserving any original Python exception already unwinding
the scope. Multiple failures need a documented aggregation/reporting rule.
Garbage collection and interpreter teardown are not reliable substitutes for
structured completion.

### Python objects and native workers

Prefer native-owned request data and native collectors initially. Retaining a
Python owner for borrowed memory requires an interpreter-safe release path;
the last reference cannot casually be decremented by an arbitrary worker after
Python shutdown has begun. GIL release alone does not solve this problem.

Arbitrary Python callbacks inside native worker loops are a separate feature.
Progress can initially be queried or delivered in batches on a Python-safe
path. Callback exceptions, reentrancy and shutdown must be specified before
allowing per-point Python callbacks in parallel grids.

## NumPy compatibility and array ownership

### Two complementary interfaces

A module such as `uni20.numpy` can provide familiar constructors and operations
with uniform async behaviour. NumPy's `__array_ufunc__` and `__array_function__`
protocols can also route supported existing `np.*` calls on Uni20 objects to
native work without first converting them into ndarrays. These are separate
features: replacing an import helps constructors too, while dispatch protocols
only apply where NumPy supports the relevant interception.

An initial compatibility table should name operations and supported arguments,
including dtype, axis, broadcasting and output rules. For example: dense array
creation, elementary arithmetic, matrix products, transpose/basic views, and
selected reductions. `out=`, `where=`, advanced indexing, mutation and unusual
strides must not appear supported simply because the function name exists.

Prefer clear rejection of unsupported dispatched operations. A later explicit
fallback mode could materialize and call NumPy, but an automatic fallback may
silently synchronize, change backend or lose symmetry/precision. Third-party
code that calls `np.asarray` or demands an exact ndarray may remain usable only
through a deliberate materialization boundary.

See [NumPy's interoperability protocols](https://numpy.org/doc/stable/user/basics.interoperability.html).
Nanobind's [array exchange support](https://nanobind.readthedocs.io/en/latest/ndarray.html)
provides useful adapters; it does not enforce Uni20's epochs or accessor
semantics for us.

### Copies, borrowed storage and mutation

For the first prototype, prefer copying external mutable array inputs into
native-owned storage and exporting a ready NumPy snapshot. These copies create
simple, explicit ownership boundaries; they are not the intended final limit
of interoperability. A required copy must be reported or rejected when the
caller explicitly requested `copy=False`.

A copied import must finish reading external mutable memory before the call
returns. The caller must keep that memory stable during the import, then may
mutate the original array without affecting accepted native work. Merely queuing
the copy would not establish this boundary. A later asynchronous import needs
an explicit borrowing and completion contract. Export snapshots similarly
retain native read protection until their copy has completed.

Zero-copy exchange needs a stronger contract:

- An owner keeps the storage alive, and the actual accessor, dtype, device,
  layout, alignment and strides permit the intended direct access.
- A native read lease remains valid for the duration of external reading, or
  the storage is immutable/versioned so future writes cannot race that view.
- External mutation must not race submitted native reads or writes. A scoped
  borrow/lease or explicit ownership transfer is a possible interface.
- Marking a NumPy view read-only prevents writes through that view; it does not
  prevent a native writer or another NumPy alias from modifying the allocation.

Calling `get_wait()` and returning a raw view after releasing all dependency
protection is insufficient for mutable storage. Similarly, keeping a NumPy
input alive does not stop the user from changing it while a native task reads.

Native slicing, reshaping, conjugation and in-place writes must retain the
existing [value/view and accessor contracts](../tensor/operations.md) and
[Async storage semantics](../async/storage.md). An accessor that applies a
transformation cannot be exported as if its handle described ordinary raw
elements. Symmetry-bearing tensors require explicit block interchange or a
named dense conversion.

### Performance expectations

Async execution can overlap independent operations and let Python submit ahead
of native computation. It does not make every operation faster. Per-element
tasks, many tiny scalar tasks, excessive copies or repeated observations can
dominate useful work. Use native operation granularity and bounded admission.
Admission backpressure may itself delay submission when resource limits are
reached; deferred numerical observation does not imply unbounded queues.

Benchmark completed calculations, not just submission time. Compare direct C++,
blocking Python and async Python with identical data, precision and numerical
controls. Include export and synchronization costs explicitly. Implicit-wait
diagnostics, optionally reporting where materialization occurred, would help
users understand lost concurrency without changing numerical semantics.

## Numerical results and errors

Model results retain convergence status, residuals, iteration/refinement
diagnostics and physically relevant metadata. A completed Async operation is
not necessarily a converged numerical answer.

Prefer a strict convenience energy API: a failed solve becomes a numerical
convergence exception when that energy is consumed, and dependent arithmetic
must propagate that failure. The detailed state API can return the documented
last iterate and diagnostics without treating it as a converged state. A later
explicit policy may permit using failed estimates; it must not be implicit.

Validate immediate user input before entering native invariant-assuming paths.
Failures that depend on pending inputs occur during execution and surface on
observation or scope close. Preserve the distinction between invalid arguments,
unsupported configured operations, numerical nonconvergence, cancellation and
execution errors. Exceptions need operation/model context so an error reported
at `print(energy)` still identifies the failing solve.

The implementation should retain native exceptions in operation state and
translate them on the Python-facing path. It must not let worker exceptions
escape into unrelated code or represent a failed root as zero spectral weight.

## Optional Python coroutine integration

Python `asyncio` integration is an additional observation mechanism. It is not
required for native concurrency, numerical dependency composition or resource
control. The same object could eventually support both:

```python
energy.wait()                  # Blocking observation from ordinary Python.
value = await energy           # Optional integration from a Python coroutine.
```

An await adapter should arrange completion notification through the event loop,
using its thread-safe entry points when called from native workers. It must not
call a blocking native wait on the event-loop thread or allocate one blocked
Python thread per pending numerical value. Python-owned future objects and
callbacks need the lifetime protections discussed above. See
[asyncio's cross-thread rules](https://docs.python.org/3/library/asyncio-dev.html#concurrency-and-multithreading).

Await cancellation also needs a choice: cancelling a Python waiter may merely
detach that waiter, or request cancellation of an explicitly owned native
operation. Prefer detaching by default for shared values and retaining a
separate native cancellation request. The adapter must preserve the native
operation's eventual cleanup even after the Python waiter is cancelled.

## Prototype sequence and evidence

These are bounded future experiments, not a commitment to implement the whole
design now. Each should leave a usable interface and a reviewable decision.

| Stage | Deliverable | What it establishes |
| --- | --- | --- |
| 1. Small core plus Bethe module | XXX ground-state state/energy, binary64 and explicit MPFR, build capabilities and typed exceptions | Module separation, physical conventions, scalar conversion and precision preservation |
| 2. Native async numerical value | Explicit execution context, pending energy, arithmetic, formatting, `.wait()` and scope lifetime | Deferred observation and native dependency composition without Python coroutines |
| 3. Small dense array interface | Selected rank-zero/vector/matrix operations, reductions and safe NumPy import/export | Common scalar/array behaviour and template dispatch across modules |
| 4. Shared prepared calculation | One Hubbard preparation feeding multiple curves; combine a model scalar with an array result | Retained dependencies, shared work, failure isolation and bounded output |
| 5. Independent extension packaging | Install/import core and Bethe independently against a compatible build | Type identity, runtime sharing, missing-extension diagnostics and ABI/configuration checks |
| 6. Optional await adapter | The same values observed through `asyncio` | Event-loop responsiveness, cancellation and interpreter-safe notification |

Standard dtype coverage can grow incrementally, but stage 1 should exercise an
extended-precision path so the interface is not accidentally hardwired to
Python `float`. Stage 3 is a substrate for a later MPS prototype; it does not
require exposing every rank or symmetry in advance.

The discriminating checks should include:

- A deliberately held input proves that submission and dependent arithmetic
  return before completion, while numerical observation waits correctly.
- Single-worker execution and two independent roots establish progress and
  composition; both consumers observe one shared preparation.
- Saturated admission with a single worker still allows a dependent chain and
  required child tasks to finish; retaining completed Python values does not
  deadlock new submissions.
- Dtype, exact input conversion and mixed-precision rejection agree with native
  contracts; formatting does not silently pass through binary64.
- Deleting submission-local inputs and handles does not invalidate work;
  context exit, cancellation and failure observe accepted children.
- Native views and in-place mutations preserve dependency ordering; external
  array snapshots cannot race later native writes. Mutating a NumPy input after
  copied import returns cannot change the native result.
- Separate extensions exchange the same registered objects and use the same
  execution service. Incompatible builds fail before crossing unsafe boundaries.
- End-to-end timings include waits and demonstrate a worthwhile workload, with
  no claim that async submission alone constitutes a numerical speedup.

## Open decisions

| Decision | Current preference | Alternative worth evaluating |
| --- | --- | --- |
| Pending scalar representation | One common numerical protocol | Dedicated scalar facade versus rank-zero tensor |
| Model function naming | Explicit `_async` initially | Async-by-default model interface |
| Standard dtype exposure | Accept NumPy spellings and avoid duplicate scalar classes | Whether Uni20 also re-exports dtype aliases |
| Extended scalar interoperability | Preserve Uni20 precision/exactness; investigate existing Python types | Native wrappers, lossless adapters, or later custom NumPy dtypes |
| Public tensor class layout | Common interface over typed native objects | Individual bound classes versus one facade |
| Execution scope exit | Structured completion of scope-owned work | Handles retaining an open context beyond scope exit |
| Default execution resources | Explicit contexts in the first experiments | Lazily created default context, with documented concurrency and shutdown |
| Admission and memory budgets | Separate root admission, child progress and retained storage | Exact limits and credit-release events need prototype evidence |
| Cross-context operands | Require a defined explicit route initially | Safe automatic composition where native scheduling permits it |
| Display | Nonblocking `repr`, numerical formatting waits | Blocking bounded notebook preview |
| Array interchange | Safe copied boundaries first | Immutable/versioned views, scoped leases or ownership transfer |
| Unsupported NumPy operations | Clear rejection | Explicit materializing fallback mode |
| Distribution layout | Modular imports, first built together | Owned subpackages versus external packages and a loader |
| Await cancellation | Detach a waiter from a shared value | Propagate cancellation for explicitly owned operations |

The first review should decide the public numerical-value protocol, context
lifetime and dtype conversion policy. Those choices constrain both Bethe and
the array prototype. Packaging details and Python coroutine support can then
be developed against concrete users rather than an assumed monolithic binding.
