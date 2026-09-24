# Shared run context, provenance, and output sessions

Status: design proposal, 2026-09-24. The configuration resolver, run context and
output session described here are not implemented. This is the canonical shared
design; consumer adoption plans should link here rather than maintain another
copy.

Scope: [Uni20 issue #55](https://github.com/Uni20-dev/uni20/issues/55), following
the CLI/presentation and data-table work. CLI11 is already the selected parser;
this proposal does not revisit that choice.

## 1. Recommendation and motivation

Build a small, parser-independent application-support layer in Uni20, and
incrementally replace Bethe's equivalent plumbing. Start with reusable typed
values and run documents, then shared configuration resolution; extract output
sessions after the metadata path has a working consumer.

Bethe already has working CLI and typed table output throughout its frontends.
This work is not a prerequisite for another model and does not improve a solver's
accuracy or speed. Its value is a single authoritative description of a run,
clear output lifetimes, and fewer independently maintained application helpers.

For Bethe, this provides:

- One definition of resolved parameters for screen reports and file metadata,
  retaining native scalar types until each output is formatted.
- Reusable resolution of declared CLI, option-file, attribute, environment and
  default sources, with the winning source recorded alongside each value.
- A useful initial preamble before a long calculation, followed by a concise
  final summary with explicitly distinguished computation CPU and elapsed time.
- Common provenance and output behavior across models, reusable by a future
  C++ or Python driver without constructing command-line arguments.

For Uni20, it fills the gap between its existing presentation/data-table
facilities and a usable scientific executable. Examples, benchmarks and other
consumers can reuse the same mechanisms without adopting Bethe's physics or
linking a parser into numerical libraries.

The cost is an API and associated ownership, portability and failure semantics.
Moving a large Bethe header upstream unchanged would not justify that cost. The
extraction should be driven by the boundaries below and tested with a small
non-Bethe example as well as the real consumer.

## 2. Starting point

Current Uni20 behavior is documented in [Command-line Applications](command_line.md),
[Presentation Formatting](presentation.md), and [Typed Data Tables](data_tables.md).
The Bethe implementation links below identify the extraction baseline. The
important implementation boundaries are:

| Existing component | Reuse or change |
| --- | --- |
| Uni20 `presentation::program_info`, report documents and display layer | Reuse identity, styles, width handling and routing; do not add an ANSI formatter. |
| Uni20 `data_table`, snapshots and sinks | Reuse typed rows, optional retention, replay, failure reports and explicit finalization. |
| Uni20 `build_info::current()` | Extend the existing provenance provider rather than introducing a competing build-information system. |
| Bethe [program-options.hpp](https://github.com/Uni20-dev/bethe/blob/6394f6a369f2b09dbd0f9bc9c10c0342f28579d0/apps/program-options.hpp) | Keep scientific option registration and reference selection here; use optional Uni20 adapters for generic options. |
| Bethe [result-output.hpp](https://github.com/Uni20-dev/bethe/blob/6394f6a369f2b09dbd0f9bc9c10c0342f28579d0/apps/result-output.hpp) | Replace the conversion from rendered overview fields back into export metadata. |
| Bethe [data-output.hpp](https://github.com/Uni20-dev/bethe/blob/6394f6a369f2b09dbd0f9bc9c10c0342f28579d0/apps/data-output.hpp) | Source of tested session, file ownership, comment-dialect and named-document behavior to extract. |
| Bethe model-specific report/output headers | Keep physical conventions, column schemas, root interpretation and convergence policy here. |

The main metadata problem is semantic, not an established loss of digits:
`ResultOutput` currently reads the already formatted `report_builder::fields()`
and copies them into a flat string map. Human labels become export keys, grouping
is lost, and timing/status are identified by particular label strings. A
carefully formatted string can preserve fp128 digits, but it no longer carries
its numeric type or independent display intent.

The table cells already retain their native types. They do not need redesigning.

## 3. Ownership and dependencies

Four objects have distinct responsibilities. Names below are provisional;
`program_info` is existing API, while the other objects are proposed API.

| Object | Owns | Does not own |
| --- | --- | --- |
| `program_info` | Application identity, help content and optional reference provider | A calculation's parameters, results or files |
| Configuration resolver | Declared source bindings, precedence, checked conversion and resolved values with their origins | Model physics, persistent-store implementation or report/file ownership |
| `run_context` | Owned invocation/provenance, ordered typed metadata, timing samples and application-supplied outcome | Parsing, model validation, numerical algorithms or file streams |
| `output_session` | Destination policy, owned files, sink bindings, document lifecycle and run-report emission | Scientific schemas, eigenstate ranking or the definition of convergence |

The data flow is:

```text
CLI / option file / input attributes / environment / defaults
                           |
                  configuration resolver
                           |
                resolved values + origins
                    /             \
     application configuration   run_context
               |                  /       \
             solver      human documents   immutable snapshots
                                                   |
application-owned typed tables + output_session -> selected sinks
```

The owning metadata value layer and source resolver must be usable without a
renderer, CLI11, output session or persistent store. Direct C++ callers can
supply explicit values without constructing argv; a later Python adapter can
use the same boundary. Run contexts and document builders belong alongside
Uni20's parser-independent common/presentation facilities. Session/file helpers
should be opt-in headers or an application-support target, with no CLI11 includes.
CLI11 registration, option-file syntax and parser-specific source/invocation
adapters belong in `uni20_cli`.
Numerical targets must acquire neither a parser dependency nor runtime output
state. Exact header/target names can follow a prototype; the dependency direction
is a requirement, not an open decision.

### Object attributes, run records and history

MPToolkit provides a useful distinction that the shared design should preserve:

| Concept | Meaning and lifetime | Relationship to this proposal |
| --- | --- | --- |
| Object attributes | Metadata describing the current saved object; selected fields may supply inputs to a later calculation | Can reuse typed metadata values, but belong to the object rather than the run or output session |
| Run context | Resolved configuration, provenance, timing and outcome of one invocation | Owns the current run's data and produces immutable snapshots |
| Object history | Successive operations and annotations associated with an object | May later contain run snapshots or references to them; independent of terminal or file-report output |
| Output session | Destinations and lifetime of reports and table documents | Projects run data without becoming its persistent owner |

In MPToolkit, `interface/attributes.h` stores string attributes, while
`interface/history.h` stores timestamped commands and notes. Both are serialized
with `wavefunction/mpwavefunction.h`. These attributes affect computation:
`mp/mp-dmrg-2site.cpp` can obtain its Hamiltonian from the input wavefunction,
and `mp/mp-itdvp.cpp` reads saved time, beta and evolution Hamiltonian attributes.
They are not merely display labels. MPToolkit's string storage and serialization
format are not requirements for the new typed representation.

Share owned values and snapshot machinery across these uses, not an entire
run-context object with clocks, invocation and presentation policy. Applications
declare source bindings and select policy; shared code performs precedence
resolution and checked conversion. It records the winning source so an
application need not reconstruct origins when publishing resolved values.
Input identifiers and attribute keys can identify those origins; a pathname is
not treated as durable object identity.

An immutable run snapshot must be obtainable without an output session and must
not retain live clocks, streams or application references. This permits a later
persistence adapter to use it as an operation record. A snapshot taken during a
run is not a final outcome. Saving an object/checkpoint and successfully closing
its reports are separate events; a future persistence layer decides when to
attach history to a published object. Persistent attributes, history storage,
checkpoint/restart and legacy MPToolkit conversion remain separate work.

MPToolkit also has per-step benchmark files, repeatable verbosity, precision
settings, flushing controls and process timing. Sections 6 and 7 retain the
useful separation of these concerns without adopting global logger registries,
environment-variable conventions or historical command-line spellings.

### Shared configuration resolution

Bind additional sources to an option already declared on `CLI::App`, rather
than declaring its names, conversion and help text in a second option language.
The CLI adapter connects those declarations to parser-independent resolution.
This is illustrative proposed syntax, not an implemented API:

```cpp
auto* h = app.add_option("-H,--Hamiltonian", hamiltonian);
config.bind(h).attribute("Hamiltonian").metadata("hamiltonian");
```

The binding identifies which input attribute supplies the option and which
stable metadata ID receives its resolved value. With multiple input objects,
the application selects the attribute provider explicitly; do not use whichever
object happened to load last. A provider exposes application-supplied attributes
through a lookup interface, so an in-memory map or a future object-store adapter
works without linking persistence into the resolver. Resolving sources does not
interpret a Hamiltonian expression or validate a wavefunction's physics.

The default precedence is highest first:

| Priority | Source | Purpose |
| --- | --- | --- |
| 1 | Explicit CLI option, or explicit value from a direct API caller | Override for this invocation |
| 2 | Explicit option file | Settings selected for this calculation |
| 3 | Input-object attribute | Settings carried by the selected input object |
| 4 | Explicitly named environment variable | Ambient defaults |
| 5 | Application default | Final fallback |

The application opts each option into its applicable sources; there is no
automatic import of every attribute or environment variable. Common cache,
thread, formatting and verbosity settings are useful environment bindings.
Physics parameters may use them when explicitly declared. An ambient setting
must not replace a saved Hamiltonian under the default precedence. Any policy
override must be declared and visible in the binding rather than implemented
as a separate ad hoc conditional in each application.

Reuse CLI11's [environment bindings](https://cliutils.github.io/CLI11/class_c_l_i_1_1_option.html)
and [option-file support](https://cliutils.github.io/CLI11/book-config.html),
including its TOML/INI reader. Start with explicitly named environment variables
and one explicit `--config FILE`; automatic user/project file discovery, profiles
and merging multiple option files are deferred. Existing initialized application
variables can supply fallback defaults without repeating their values in the
binding. Precision-dependent defaults are evaluated after precision selection.

CLI11 support for these sources does not itself implement the full precedence
above. The adapter must preserve distinct source candidates until attributes
are available; a CLI11 option count after parsing is not proof that the value
came from argv. In particular, an environment fallback must not hide a
higher-priority attribute, and an attribute must not overwrite an explicit CLI
or option-file value. Do not implement this by changing advertised defaults
after parsing or replaying option callbacks with side effects.

Resolution must preserve these contracts:

- Presence is explicit. Zero, `false`, an empty string and a missing source are
  different states. An empty value is converted or rejected under the option's
  contract; it does not silently activate the next fallback.
- Every source uses the same checked value conversion and validation. A present
  winning value that cannot be converted is an error identifying the option and
  its source, not permission to try a lower-priority value. Invalid option-file
  syntax and an explicitly requested unreadable file are errors as well.
- Retain textual real values until the resolved precision is known, then use
  native `parse_real<Real>` and checks in that type. Preserve already typed
  attribute/API values through checked conversions without a `double` or
  presentation-string intermediate. Half-integer inputs retain exact semantics.
- Return owned values and origin descriptions together. Origins distinguish CLI,
  API, file/key, input identifier/attribute, environment name and application
  default. They must not borrow parser buffers, environment pointers or a loaded
  object's attribute storage. Publishing them remains application-selected.
- Using an attribute as an input does not mutate its source object. The
  application can explicitly copy selected resolved settings into output-object
  attributes when saving; reading fallback values and writing attributes are
  separate operations.

### Staged parsing and required values

Input attributes may become available only after opening an input object. The
shared coordinator therefore separates collecting sources from final required
value and cross-option validation:

1. Handle help/version/references/build-info before opening option files or
   input objects. Keep ordinary syntax handling in CLI11.
2. Read the explicitly selected option file and collect declared environment
   sources. Resolve bootstrap settings such as input locations from the sources
   already available. A file locator cannot depend on attributes from the file
   it is needed to open, or on that file's own contents.
3. Ask the application to load input attributes through its provider. This does
   not initialize a solver or create output files.
4. Resolve precision and remaining options, convert selected values, then check
   requirements and application-level consistency across options. A required
   Hamiltonian supplied by an attribute satisfies the requirement just as a CLI
   value does. Missing values fail before numerical work begins.

The binding adapter must defer requirements on attribute-backed values to this
final resolution stage; leaving CLI11's early `required()` check active would
reject valid input before the attribute is loaded. Bootstrap options keep their
own earlier requirements. Help should describe a value as required from its
allowed sources, without loading an object to guess its default. Applications
with no attribute sources can use a single resolution step.

## 4. Typed run metadata

A run contains ordered groups of fields. Each field has:

- A unique stable identifier, separate from its human label.
- Group membership and insertion order; groups also have stable order.
- An owned value, optional units/description and a human display policy.
- Compact/detail visibility, independent of whether a field is published.
- Optional origin information, separate from the value: supplied with a resolved
  option by the shared resolver, or provided by the application for other fields.

Typical groups are Model, Numerics, Provenance, Background and Summary. These
are application-selected labels, not mandatory physics concepts built into
Uni20. A field such as `energy_convention` must not depend on its displayed
spelling, and moving it between groups must not change its identifier.

Use the existing table scalar support as the initial value vocabulary:
integers, booleans, strings, native reals, exact `basic_half_int<T>` and supported
optional numeric values. Copy borrowed text on insertion. A templated field
builder should retain the actual value type; an owning type-erased field with
type-specific formatting operations is a suitable implementation. Do not use a
fixed `double`-based numeric variant, require MPLAPACK in fp64-only builds, or
allow callbacks to retain references to temporary application variables.
Presentation groups and display policies belong to the run's document view;
reusing the value container for object attributes must not require that view.

Reuse the scalar formatting machinery from data tables. Human display can use
shorter precision; machine projection must round-trip the stored finite real
without intermediate narrowing. Half-integers use exact decimal spelling from
integer storage, not a conversion through `double`. Missing values remain
distinct from zero and from a present nonfinite value inside the context.

Identifiers follow the existing data-column identifier grammar and are unique
across the run. Duplicate declarations are errors. Explicit replacement of a
field preserves its identity, position and declared type; parameter resolution
must select the scalar type before inserting the value. This is not a second
options-definition language: applications explicitly publish the configuration
they actually use, including defaults and derived values.

### Snapshots and compatibility

The first implementation projects typed fields into the existing
`table_metadata` string map. It does **not** change Uni20's table JSON schema.
That projection preserves numeric text precision, but does not preserve field
types, group ordering or typed nulls in the exported metadata map. Those remain
available in the run context; a future structured metadata format is separate
work and would need an explicit schema.

For new consumers, use stable IDs as metadata keys. During Bethe migration,
allow an explicit legacy export-key mapping so existing keys such as `Program`
and `CPU time` do not change accidentally. Check projected-key collisions;
never silently overwrite one field with another. The compatibility mapping is
a view of the same fields, not a second set of values.

Create an immutable metadata snapshot when a table begins output. Subsequent
context changes cannot rewrite already emitted headers. A late sink replaying
that table receives its original metadata, not the current global context.
Background information obtained before table creation belongs in its initial
snapshot; information obtained later belongs in a later table or a summary.

## 5. Invocation and build provenance

Invocation capture is optional and explicitly selected by the application.
When supplied, store invocation as owned tokens, not just one shell-command
string. For a non-CLI caller, invocation can instead identify the calling API;
do not invent an executable command. Resolved values are authoritative for what
was computed; original spelling helps explain how that configuration was
requested. An omitted invocation does not prevent reports, tables or provenance
for the rest of the run.

Derive a quoted command for display from the token list, consolidating with
Uni20's existing shell-quoting helpers. Label the quoting convention (initially
POSIX); it is a display/export operation, never a command to execute. Escape
control characters for the destination separately from shell quoting. A newline
inside an argument must not become a spurious comment, terminal control sequence
or data record.

Applications explicitly supply metadata and invocation details suitable for
publication. The run context does not discover fields by inspecting an entire
configuration or automatically copy raw process arguments. An application that
uses credentials can omit those fields and omit invocation capture; compact/detail
visibility is only a presentation choice, not a way to keep a published field
private. There is no generic credential detection, token-to-option redaction
machinery or special redacted field state in the initial implementation. Those
features are deferred until a consumer has a concrete need.

Capture one UTC start timestamp for the run. Extend Uni20's existing build-info
mechanism with an application-supplied identity/revision descriptor and a small
consumer CMake helper, informed by Bethe's
[current helper](https://github.com/Uni20-dev/bethe/blob/6394f6a369f2b09dbd0f9bc9c10c0342f28579d0/cmake/WriteBuildInfo.cmake). Record the actual application and
Uni20 sources used to build, including local overrides, tracked-dirty and
unknown states. Source archives without revision information must say so.
Never infer the executable's revision from its runtime working directory.

Default provenance should be a bounded selection: application/version,
application and Uni20 revisions, start time and invocation when supplied. Detailed
compiler/provider information can come from `build_info::current()` through
an explicit selection policy. Do not dump its complete cache/environment maps,
absolute build paths, environment variables, host identity or user identity
into every result file. Build provenance aids reproducibility; it is not a
complete fingerprint of untracked files, inputs or runtime libraries.

## 6. Run lifecycle and human output

The normal lifecycle is:

1. Capture a timing anchor; parse or receive configuration.
2. Handle help, references, version and build information without reading option
   files or input objects, creating output files or starting a calculation.
3. Collect declared configuration sources and input attributes using staged
   resolution. Resolve precision/defaults, validate the model and known
   destination/table selections, and construct the context from publishable
   resolved values and origins.
4. Start the output session and emit one compact human preamble before expensive
   numerical work. No preamble goes to machine stdout or quiet stdout.
5. Perform background/solver work, add derived metadata and publish typed tables
   from appropriate context snapshots.
6. Finalize tables and emit a final run summary; explicitly finalize the output
   session and report any required-output failure before returning success.

The initial document should show identity, model/conventions, precision and
important numerical controls. Long hashes and the full invocation belong in
detailed human output; exported metadata includes them when supplied by the
application. A solved background can have a later, separately headed section;
do not repeat the entire preamble to add one result.

Construct these documents with Uni20's presentation layer. Plain/styled forms
carry equivalent information, respect existing glyph/width policy, and never
truncate or split numeric tokens to fit a terminal. Machine sinks consume
values and metadata projections directly, never a rendered screen report.

When a session owns the human run preamble, its terminal table sinks suppress
their own repeated run metadata using the existing `show_metadata` option.
Standalone table sinks keep their existing default behavior. Table-specific
titles and necessary local metadata remain visible through the session's
documents. This suppression is explicit, not a global display side effect.

Reference selection and whether ordinary help includes a bibliography remain
application policy. Bethe keeps its separate `--references` action; the shared
layer does not remove Uni20's existing optional help reference provider.

### Reports, results and diagnostics

Keep four independent choices in the application/session policy:

- Human report visibility and compact/detail selection.
- Result destinations, formats and column selection.
- Diagnostic destination and verbosity.
- Human display precision versus machine export precision.

For the initial CLI adapter, quiet suppresses session stdout, including machine
stdout when selected, while requested file exports and stderr warnings/errors
remain enabled. Quiet must not change a table's schema or hide numerical columns.
Applications translate their verbosity controls into solver-specific settings;
this proposal does not introduce a general logger or change solver interfaces.

Every display adapter must preserve an event's destination. A session's plain
table rendering must not install a process-wide router that sends unrelated
stderr events to stdout or captures another session's output. In particular,
Bethe's baseline plain streaming adapter must be corrected rather than copied
unchanged: it ignores `display::event::destination`. Test a warning emitted
while a plain table is active, and test machine stdout with diagnostics on
stderr. Reuse [display routing](display_layer.md); durable structured diagnostics
and async event queues remain in the [logging design](logging_plan.md).

### Timing and outcome

Keep three quantities distinct:

| Quantity | Meaning |
| --- | --- |
| Compute CPU | Process CPU accumulated over application-marked computation intervals, excluding interleaved rendering/export. |
| Run CPU | Process CPU since the run timing anchor, including setup and output work up to the final summary sample. |
| Elapsed | Monotonic wall time between the same anchor and final sample. |

Store durations as values with units, not strings containing `s`. Process CPU
is not thread CPU or wall time; multithreaded CPU can exceed elapsed time, and
subprocess/device time is not silently included. Unavailable or invalid clock
samples produce an unavailable duration, not zero. Nested compute scopes must
not double-count the same interval; the initial accumulator can reject nesting
and concurrent mutation rather than imply parallel profiling support.

These are measurements of process CPU over marked intervals, not attribution
to individual tasks. A caller timing asynchronous work must include its
completion in the measured interval; merely submitting work does not time the
calculation. Cumulative timing across saved-object histories is separate from
the current run's timing.

The final sample precedes rendering/flushing its own summary, so it cannot
include all finalization work. Document that boundary rather than calling it
exact process lifetime. Test clocks should be injectable without real sleeps.

The application supplies scientific outcome fields and its numerical exit
decision. Uni20 tracks output completion independently. A document can be fully
written while containing unconverged estimates; conversely, a converged solve
can fail to save its results. No generic `complete` flag means convergence or
spectral completeness. A required I/O failure must prevent a success exit even
if the numerical outcome was successful.

## 7. Output sessions

The session is a thin owner/coordinator around existing table sinks, not a
second table implementation. It supports a single table or an explicitly named
collection of sequential tables with different schemas. Table names and schemas
are supplied by the application; available names are validated before opening
files where they are known in advance.

Retain Bethe's current destination policy: independent human/machine stdout,
repeatable CSV/TSV/JSON files, named-table selection for delimited exports,
quiet, streaming, retention and explicit overwrite controls. CLI declarations
are optional adapters to a parser-independent options structure. Non-tabular
applications need not advertise any of these options.

### Retention and late attachment

There is one table API. Retention is enabled by default; disabling it is an
explicit construction policy, not a separate streaming-table type.

Do not freeze sink attachment when output begins. A later sink can replay the
retained table and then follow new rows, including receiving the final summary
when attaching to a finished retained table. Without retained history, a late
sink must explicitly select future-only delivery and publish its starting row
offset. Full-history requests must fail before writing misleading output.

Late file destinations use the same preflight rules as initial destinations.
A full-run export requested after an earlier table's history was discarded
cannot silently omit that table. The session need not retain a heterogeneous
archive: callers supply retained tables for replay, or explicitly request a
partial export. The first implementation can reject unsupported document-wide
replay while still supporting ordinary per-table late attachment.

Retaining rows and retaining solver state are different. A sorted excitation
scan may need solver storage even with output retention disabled. Uni20 must
not promise live sorted results or constant-memory computations on this basis.

### Streams, files and failure

Prefer a scoped `write_table`-style convenience that binds, fills and explicitly
finishes a caller-owned typed table while session-owned streams are alive.
An advanced attachment handle may expose late binding, but must keep stream
ownership alive until the binding is finished or detached. Do not make stream
lifetime depend on the caller remembering a fragile declaration order.

All requested output destinations are required by default. Reuse Uni20's sink
failure reports and accepted-row semantics:

- Attempt delivery to every active sink before reporting failures. Required
  failures prevent success; optional failures remain visible in the delivery
  report rather than being silently discarded.
- Never retry an append after a delivery exception: that row may already be
  accepted and written to healthy sinks.
- Disable/quarantine failed sinks; finalize healthy sinks best-effort with the
  appropriate aborted status when delivery cannot continue.
- Explicit finalization checks write, flush and close failures, with destination
  names attached to diagnostics. Preserve the original failure and collect
  cleanup failures without turning an error into success.
- Destructors provide nonthrowing cleanup only. They cannot certify successful
  output or replace an explicit `finish()`/abort path.

Preflight existing targets and aliases, including symlinks, hardlinks and
redirected stdout. Without explicit overwrite permission, use exclusive
creation rather than a check-then-truncate sequence. Audit Bethe's POSIX-specific
timestamp and file-identity code during extraction; unsupported platform checks
must have a documented policy rather than silently disappearing.

These checks prevent ordinary mistakes, not adversarial filesystem races or
transactional multi-file failure. Partial files can remain after an error, and
an explicitly overwritten file cannot be restored. Atomic replacement, crash
recovery and transactional groups are outside this proposal.

### Streaming visibility and flushing

Streaming delivers rows incrementally; it does not by itself make buffered file
output immediately visible. Existing CSV/TSV and JSON sinks explicitly flush
at finish. Preserve buffered file output by default and add an explicit session
flush operation so applications can publish progress at useful boundaries, such
as the end of a DMRG sweep. A destination may opt into flushing each completed
row; flushing policy is independent of retention and display precision.

A flush neither finishes a table nor writes its summary. It attempts all active
session destinations before reporting failures and follows the same required/
optional failure policy as row delivery. A failed destination is disabled;
accepted rows are not replayed as a retry. A session cannot flush arbitrary
unrelated sinks attached directly by the caller.

Flushing concerns stream visibility, not `fsync`, atomic publication or crash
durability. A JSON document can remain syntactically incomplete until finish
even after a successful flush. MPToolkit-style append-to-file across separate
runs is deferred: repeating headers or concatenating JSON documents requires an
explicit format contract and must not be inferred from overwrite permission.

### Metadata and summaries in files

Keep generic CSV/TSV writers rectangular. Metadata comments are an explicitly
selected adapter with escaped initial and trailing comments, as in Bethe today.
`--no-preamble` continues to disable both sets of delimited comments, not JSON
metadata or human reports. No terminal banners or ANSI sequences enter exports.

There are two finalization levels. A table's summary is fixed when that table
finishes; the final run summary is available only after the whole calculation.
Do not insert provisional whole-run times into every table or retroactively
rewrite an earlier table's summary.

The proposed session supports final run summaries as follows:

- Human output: one final Summary section, not one copy per table.
- Commented CSV/TSV files: keep the owned stream open for trailing, clearly
  run-scoped summary comments. Strict delimited files intentionally omit them.
- Named JSON documents: an optional outer `summary` member at document finish,
  alongside `tables` and document `status`. Inner tables keep Uni20's existing
  schema and their own summaries.
- Single-table JSON: when the table spans the run, include run fields in its
  existing final summary. A table finalized earlier cannot acquire later fields;
  use a document envelope when a separate run summary is needed.

The outer JSON summary is an additive format extension, not a silent consequence
of moving code. First extract the current formats unchanged, then explicitly
enable and document this extension in a separate Bethe checkpoint. Preserve
Hubbard dispersion's existing single-table shape. Exact completion remains
subject to final I/O checks: bytes already written cannot be amended if a later
flush/close fails, so a file's `complete` marker alone does not prove successful
delivery to every destination.

## 8. Incremental delivery

Each step should leave a usable application; no all-frontends rewrite is needed
to establish the design.

1. **Typed context and document builders.** Implement reusable owning values,
   immutable snapshots usable without a session, provenance with optional
   invocation capture, and timing without new table wire formats.
   Add a small parser-free Uni20 example. Pilot in Bethe's Hubbard dispersion
   frontend: its background solve, long point loop and precision choices exercise
   more of the lifecycle than a trivial one-row program.
2. **Shared configuration resolution.** Add source bindings, precedence, owned
   origins and staged required-value checks, with CLI11 adapters for declared
   environment variables and an explicit option file. Exercise attribute
   fallback through an in-memory provider in a Uni20 example; no persistent
   store is required. Adopt option-file/environment bindings in the Bethe pilot.
3. **Bethe metadata adoption.** Replace formatted-overview-to-metadata copying
   with typed fields, one model family at a time. Preserve existing machine keys
   through the compatibility projection. Build human overview and exports from
   that one source; remove the old bridge once no callers remain.
4. **Session extraction.** Move generic file, comment and document coordination
   into the parser-independent Uni20 layer, followed by optional CLI helpers.
   Port generic failure/replay tests, add destination-preservation and explicit
   flushing tests, then replace Bethe's adapter with a thin consumer. Exercise
   both single-table dispersion and a named-table model.
5. **User-facing completion.** Enable staged preamble/final summaries and the
   explicitly reviewed run-summary export extension. Update output docs and
   regression fixtures. Only then assess issue #55 against its full checklist.

Do not couple these checkpoints to new Bethe physics. Keep model schemas,
convergence tests and citation registries in Bethe. Generic lifecycle tests
belong in Uni20; consumer integration tests should remain in Bethe as well.

## 9. Acceptance tests

| Area | Required evidence |
| --- | --- |
| Ownership | Temporary strings/argument buffers can disappear; copied snapshots survive later context edits; bindings cannot outlive owned streams. |
| Independent metadata | Owned values, source resolution and immutable run snapshots work without CLI11, an output session or a persistent store; rendering origin annotations does not rerun resolution. |
| Source resolution | Each adjacent precedence pair and a full-source conflict select the declared winner; options use only declared sources; zero/false/empty remain distinct from absence; invalid winning values report their origin and never silently fall back. |
| Staged validation | An attribute satisfies a required option after input loading; missing values fail before solver/output initialization; help/version need no option file or input object; bootstrap locations do not depend on unavailable attributes. |
| Source ownership | Origins survive parser/provider destruction; native real and half-integer conversions are consistent across sources; explicit CLI/file values beat attributes and attributes beat environment; reading options does not mutate input attributes. |
| Native values | fp64, long-double and enabled fp128 metadata round-trip; exact positive/negative half-integers; missing/nonfinite values; formatting never narrows through `double`. |
| Metadata | Stable identifiers, group/order preservation in documents, explicit replacement, duplicate/collision rejection and unchanged legacy export keys. |
| Provenance | One run timestamp across tables; actual overridden dependency revision; dirty/unknown/archive cases; no dependence on runtime cwd. |
| Publication | Omitted invocation stays absent in every report, snapshot and replay; only supplied fields are exported; supplied empty/control-containing tokens are preserved as data and escaped for each destination. |
| Lifecycle | Help/references/version/build-info read no option files or input objects and create no outputs; preamble before a slow solve; one final summary; table snapshots do not change retrospectively. |
| Routing | Warnings remain on stderr during plain streaming; machine stdout contains only the selected result format; quiet leaves file exports and stderr diagnostics enabled; one session does not capture another's events. |
| Timing | Deterministic injected samples distinguish compute CPU, run CPU and elapsed; invalid clocks and nested scopes follow policy. |
| Replay | Early/late/finished-table attachment; retained replay exactly once; explicit future-only offsets; rejected unavailable full histories. |
| I/O | Existing/aliased paths, stdout aliases, late-open errors, required begin/row/finish/flush/close failures, healthy sink cleanup and no accepted-row retries. |
| Streaming visibility | A buffered test stream observes explicit and per-row flushes without finishing the table; all healthy destinations are attempted when a flush fails; subsequent appends neither retry accepted rows nor reuse a failed sink. |
| Formats | Existing single/named JSON shapes during extraction; opt-in outer summary separately tested; rectangular comment-free CSV/TSV; no ANSI or banners in machine output. |
| Integration | Multiple Bethe models retain numerical results, precision, null/failed-state semantics and exit behavior; parser-free Uni20 example builds without CLI11. |

Use Bethe's existing [output tests](https://github.com/Uni20-dev/bethe/blob/6394f6a369f2b09dbd0f9bc9c10c0342f28579d0/tests/test_output.cpp) as extraction
fixtures, especially accepted-row failure, final-flush failure, late replay,
future-only offsets and named-document abort. Do not substitute success-only
golden files for these lifecycle tests. Human golden tests should fix display
policy and normalize nondeterministic provenance/times, not weaken value checks.

## 10. Deferred work and decisions before implementation

Not included: a general application framework, solver registry, global run
singleton, model expression language, automatic user/project configuration
discovery, profiles or merging multiple option files,
asynchronous sinks, generic credential detection/redaction, distributed or
cumulative historical timing, persistent object attributes/history,
checkpoint/restart, multi-run file append, Python bindings, or a replacement
for Uni20's table format. The parser-independent design permits a later Python
adapter; it does not implement one.

Resolve these API details in a small prototype before freezing public names:

- Namespace/header/target placement and the smallest owning field interface.
  Prefer common/presentation for context/documents, with opt-in I/O helpers.
- The source-binding interface and CLI11 collection hooks that preserve origins
  and defer attribute-backed requirements without duplicating option syntax.
- The advanced table-binding handle's ownership contract. Start with the scoped
  convenience, but prove ordinary late attachment remains possible.
- Human defaults for the extra timing fields and detailed provenance selection.
  Prefer a compact compute-CPU/elapsed summary, with run CPU in detailed output.
- The explicit rollout of stable metadata IDs and outer JSON run summaries.
  Prefer compatibility keys for existing Bethe exports until a deliberate,
  documented format change is warranted.

The first checkpoint is worthwhile on its own. If it fails to simplify Bethe's
metadata or requires a broad application framework, reconsider the abstraction
before extracting more machinery.
