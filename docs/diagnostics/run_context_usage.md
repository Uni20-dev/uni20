# Using run contexts and output sessions

The implementation lives in `uni20/common/metadata.hpp`, `configuration.hpp`,
`run_context.hpp`, and `output_session.hpp`; link `uni20_common`. Optional CLI11
adapters live in `uni20/cli/configuration.hpp` and `output.hpp`; link `uni20_cli`.
The [design and remaining adoption work](run_context.md) explains the boundaries.

## Native metadata and snapshots

```cpp
uni20::run_context run({.name = "solver", .version = "1", .revision = application_revision});
run.metadata().group("model", "Model");
run.metadata().add("model", "interaction", uni20::parse_real<long double>("4.000000000000000001"),
                   {.label = "U", .unit = "t"});
run.metadata().add("model", "spin", uni20::half_int::parse("3/2"),
                   {.label = "Spin", .display = {.fractions = true}});
auto initial = run.snapshot();
auto report = uni20::metadata_report(initial);
```

`metadata_value` owns its native scalar, including optional numeric values, and
copies borrowed text. `get<T>()` requires the exact declared type; `as<T>()`
performs a checked conversion or parses a complete text token directly as `T`.
Integers and half-integers reject narrowing overflow; real conversion permits
rounding but rejects finite overflow. Floating-point values never implicitly
round to integers or half-integers. The supported vocabulary matches data tables;
optional strings, vectors and arbitrary object types are not included.

A `metadata_document` orders groups and fields. Field IDs are globally unique
within the document and use the data-column identifier grammar. Labels, units,
descriptions, display policies, detail visibility and source origins are separate
from values. `replace(id, value)` preserves type, position and identity.
`append(document)` imports disjoint groups, rejecting collisions before mutation.
Copies share immutable scalar storage and own independent document structure.

Human rendering honors field display policies. `document.strings()` uses native
round-trip formatting and exact half-integer decimals for the existing table
metadata format. An explicit mapping, such as `{{"program", "Program"}}`, changes
export keys; unknown IDs and key collisions are errors. This projection loses
native type, grouping, order and the distinction between missing and empty text.
It does not change the typed document. A typed metadata JSON schema is future work.

## Configuration without a parser

```cpp
uni20::configuration config;
config.add("steps", 20).file("steps").attribute("state", "Steps").environment("SOLVER_STEPS");
config.add<std::string>("hamiltonian").required().attribute("state", "Hamiltonian");
config.attributes("state", {{"Steps", 30}, {"Hamiltonian", "Hubbard"}});
config.option_file("job.toml", {{"steps", "40"}}); // An application-selected reader supplied these values.
config.set("steps", 50);                            // Explicit API input.
auto resolved = config.snapshot();
int steps = resolved.find("steps")->value.get<int>();
run.metadata().append(resolved);
```

Precedence is explicit CLI/API input, explicit option file, selected input-object
attribute, selected environment variable, then initialized application default.
Bindings opt into each fallback; nothing scans all environment variables or
imports all attributes. Empty strings, zero and false are present values. An
invalid winning value fails with its field, source location and key; it never
falls through to a lower-priority source. File keys must be declared and unique.

Use `add<T>(id).required()` for a field with no default. `without_default()` removes
an initialized fallback; `check<T>(callback)` validates the resolved native value.
Resolve a bootstrap input path before loading its selected attributes, then take
the final snapshot. Each resolve is a new resolution, so capture the snapshot once
for the solver and its reports when a stable configuration is needed. Attribute
maps and returned origins are owned. `copy_attributes(output, {{id, attribute}})`
is an explicit operation for saving selected resolved values; reading fallback
attributes never modifies an input object.

## Reusing CLI11 declarations

```cpp
unsigned steps = 20;
std::string hamiltonian;
CLI::App app;
uni20::cli::configure(app, program);
uni20::cli::configuration config(app);
config.bind(uni20::cli::add_count_option(app, "--steps", steps)
                ->check(CLI::PositiveNumber)->capture_default_str(), steps, "steps")
      .file("steps").environment("SOLVER_STEPS");
config.bind(app.add_option("-H,--Hamiltonian", hamiltonian)->required(), hamiltonian, "hamiltonian")
      .attribute("state", "Hamiltonian");
config.option_file(); // --config; one explicitly selected TOML/INI file.

auto result = config.parse(argc, argv);
if (result.requested != uni20::cli::action::run) {
  uni20::display::emit(config.result_report(program, result), result.destination);
  return result.exit_code;
}
config.load_option_file();
// Resolve bootstrap fields, load the input, then config.values().attributes(...).
auto resolved = config.resolve(); // Assign variables and return owned metadata.
```

Bind after declaring defaults, validators and transforms. Initial values are
captured once; `capture_default_str()` advertises them. A native CLI11 `required()`
option has no implicit initialized fallback and is checked after attributes are
available. A bound `envname()` is handed to the resolver so attributes beat the
environment, including when the environment value is invalid or empty.

The adapter uses CLI11's existing conversions, validators, transforms and file
reader. Explicit CLI arguments run that chain once; fallback values enter the
same option using text or lossless native formatting. No conversion passes
through `double`. Keep real literals in string-bound options until the precision
choice is resolved, then parse with `parse_real<Real>` and publish the native
value in the run document. The complete example demonstrates this sequence.

The adapter is single-use. The app, options and bound variables must outlive it.
The initial binding interface supports non-optional scalar options and boolean
flags. Bind independent options: CLI11 `needs`/`excludes`, required option groups,
subcommands and vector-valued configuration are not supported by staged
resolution. Check cross-field constraints after `resolve()`. Use the adapter's
`option_file()` rather than `app.set_config()`. Automatic file discovery,
multiple-file merging and profile inheritance remain deferred. Resolution may
assign some variables before another field fails; do not start a calculation
unless the whole resolution and application validation succeed.

`cli::add_output_options(app, options)` registers format, output path, repeatable
exports, quiet, plain, preamble, overwrite and per-row-flush controls.
`options.session(initial, metadata_keys)` constructs an unopened output session;
the optional mapping applies to initial metadata and final summaries on every
destination. These options are intentionally not copied
to scientific metadata automatically. Help/version/build-info create no output
files, read no option/input files, and do not resolve environment fallbacks.

`--format` selects the primary stdout representation; `--output=PATH` redirects
that primary destination to a file. Each `--export=FORMAT:PATH` adds an independent
file destination, leaving the primary destination in place. Split at the first
colon; subsequent colons and spaces belong to the path (quote the argument when
needed). The available formats are `terminal`, `csv`, `tsv`, `commented-csv`,
`commented-tsv`, `json` and `named-json`. All files are required destinations and
share the overwrite, preamble and flush controls. Quiet suppresses stdout while
file exports continue. For example:

```sh
run_cli_example --export=csv:results.csv --export=named-json:results.json
run_cli_example --quiet --export=commented-csv:results.csv --export=json:results.json
```

## Timing and provenance

```cpp
auto energy = run.measure([&] { return solve_and_wait_for_completion(); });
// Rendering and checkpoint I/O here are outside the computation interval.
auto const& summary = run.finish(uni20::run_outcome::success);
```

Alternatively use `auto scope = run.computation()` and explicitly `scope.finish()`.
Scopes must remain inside the run context's lifetime; overlapping scopes are
rejected. Exception unwinding ends an interval. Explicit completion reports clock
exceptions; destructor cleanup cannot throw and marks the accumulated timing
unavailable if sampling fails.

The final document contains numeric `compute_cpu_seconds`, `run_cpu_seconds`,
`elapsed_seconds`, and the application-selected `outcome` (`success`, `partial`,
`failed`, or `cancelled`). CPU is process-wide; elapsed time uses a monotonic clock.
Unavailable/invalid CPU timing is missing, not zero. Async submission alone does
not measure completed computation. Final timing is sampled before rendering and
flushing the summary. Inject clock/UTC functions for deterministic tests.

The run records UTC start, application identity/revision and bounded Uni20 build
information. Invocation capture is opt-in with owned argument tokens in
`run_context_options::invocation`. The display spelling uses POSIX shell quoting;
control bytes are escaped in human metadata and comments. Applications select
publishable fields. Generic credential detection/redaction is deferred.

`build_info::current().revision` identifies the actual Uni20 source tree at build
time, including tracked dirty changes. Archives report `unknown`, unless the
build sets `UNI20_SOURCE_REVISION`. A consumer can reuse the same generator:

```cmake
uni20_target_provenance(my_solver HEADER my_solver_revision.hpp
  NAMESPACE my_solver_build SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
```

Include that header and pass `std::string(my_solver_build::source_revision)` into
`program_info::revision`. `REVISION` supplies an explicit archive/package identity.
The helper never consults the executable's runtime working directory. Untracked
files do not affect Git's dirty flag.

## Streaming and owned output sessions

```cpp
uni20::output_session output(run.snapshot());
output.standard_output(uni20::output_format::terminal);
output.file("results.tsv", uni20::output_format::tsv);
output.file("results.json", uni20::output_format::named_json);
output.open(); // Emit preamble and discover opening failures before expensive work.
output.attach(table, "dispersion");
for (...) table.append(...);
table.finish({{"converged", "true"}}); // Immutable table summary, distinct from run summary.
output.finish(run.finish(uni20::run_outcome::success));
```

The session owns file streams. Table subscriptions share that ownership, so
changing declaration order or destroying the session cannot leave a table with a
dangling stream. Shared stream owners can be supplied with `stream(...)`; an
optional close callback participates in explicit finalization. Call `finish()`
to certify output completion. Destructors provide resource cleanup only.

Rows remain in the existing `data_table` by default. Late `attach` replays rows
and subscribes to new rows. `write_table(table, name, summary)` is a one-shot
snapshot: it exports retained rows without copying them, closes only its own
subscriptions, and leaves the source table appendable and its other sinks active.
An already-finished source keeps its original summary.

Late `attach` also replays an already-finished table's summary. To attach only a new destination, pass
its returned ID: `attach(table, "name", sink_replay::all, {new_id})`.
No-retention tables require `future_only` after rows have been discarded; rejected
full-history replay writes nothing. JSON records the first-row offset. Strict CSV
and TSV have no metadata channel for offsets. Document-wide history/replay is not
retained. Sequential named tables must finish before starting the next table.

Each destination has its own projection, metadata export-key mapping, human
policy, required/optional status, and flush policy. Terminal streams use instance-local routing, leaving unrelated
stderr diagnostics and the process-wide display router intact. Plain output is
an explicit policy. Quiet suppresses session stdout, including machine formats;
files and caller-owned diagnostics continue. No-preamble affects the initial
human report and commented-file metadata, not machine rows or the human final
summary.

| Format | Contents |
| --- | --- |
| `terminal` | Initial report, streaming tables and final run report; numeric tokens stay intact. |
| `csv`, `tsv` | One rectangular table, no comments or summaries. |
| `commented_csv`, `commented_tsv` | One delimited table with escaped initial/table/final `# key: value` comments. Both use the same comment handling and CSV-style field quoting. |
| `json` | One existing data-table JSON document, including its table summary. |
| `named_json` | Sequential `{name, data}` tables with an outer run summary and status. |

The named envelope is `{"tables":[{"name":"dispersion","data":...}],
"summary":{"outcome":"partial",...},"status":"complete"}`. Initial run metadata is projected into
each table's metadata. Run/table key collisions are rejected. A single-table JSON
stream is closed at table finish; use `named_json` to carry a later run summary
separately. Explicit finalization writes `status: "complete"` independently of
scientific outcome, including partial or failed calculations. Unconverged results
can therefore form a complete document. This marker describes the document,
not successful flush/close or durable storage: callers must still check
`finish()` for I/O failures. Interruption or write failure can leave incomplete
JSON; no destructor or automatic abort path repairs it or certifies completion.

Export-key mappings apply to run metadata wherever the field is present. A
single mapping can cover disjoint initial and summary fields; entries absent
from a particular snapshot are ignored. Human labels and the source metadata
remain unchanged, and table-local metadata and summaries keep their own keys.
Collisions between mapped fields (including an unmapped field with the same key),
or between mapped run metadata and table metadata, disable that destination and
are reported through the ordinary required/optional failure policy. Other
destinations are still attempted.

```cpp
output.file("results.json", uni20::output_format::named_json,
            {.metadata_keys = {{"program", "Program"}, {"outcome", "Outcome"}}});
```

The application supplies these mappings. Uni20 does not encode consumer-specific
attribute or export names, and the unmapped default uses canonical field IDs.

Files use exclusive creation unless overwrite is explicit. Preflight rejects
aliases by path, symlink, hard link, shared stream buffer and redirected stdout
before opening new destinations. It is not a filesystem transaction or an
adversarial race defense. Opening multiple destinations can leave partial files.
Machine files buffer output by default; human terminal output flushes its
preamble and each displayed row so live results are visible immediately.
`flush()` attempts all active streams without
ending tables or writing summaries; it does not promise `fsync` or crash recovery.
Per-row flushing is opt-in.

A row is accepted before output delivery. All sinks are attempted; a failing sink
is disabled. `data_table::append` reports optional failures or throws
`data_delivery_error` after all sinks have been attempted. Session operations
return an accumulated `output_report`, or throw `output_error` if any required
output failed. Each failure retains destination, operation and exception.
After a session flush failure, that destination rejects its next table callback
without writing. Finish healthy tables and call session `finish()` to collect
additional flush/close errors. Repeated finish returns/rethrows the stored report
without repeating writes. Never retry an accepted row automatically.

## Runnable examples

All examples are in [examples/presentation](../../examples/presentation/).

| Target | Demonstrates |
| --- | --- |
| `run_metadata_example` | Parser-free metadata, snapshots, display mapping and timing. |
| `configuration_sources_example` | Five source levels, origins and explicit output-attribute copying. |
| `run_cli_example` | Staged CLI/file/input/environment resolution, precision and output controls. |
| `output_live_example DIRECTORY` | Live terminal display plus owned TSV and named JSON files. |
| `output_named_tables_example` | Two schemas and a separate final run summary. |
| `output_replay_example` | Attach a new destination after rows already exist. |
| `output_no_retention_example` | Future-only output with an explicit offset. |
| `output_failure_example` | Disable an optional display while required TSV succeeds. |

Each example prints a preamble explaining its sample values and expected
behavior. Examples with JSON or TSV on stdout send that explanation to stderr,
so redirecting stdout still produces a valid data file. The CLI example honors
`--quiet` and `--no-preamble` for this explanation too.

`output_live_example` deliberately selects only step and energy for its live
table, while the TSV and JSON files retain residuals as well, at full numerical
precision. Its preamble explains the missing first residual and the deliberately
partial final outcome. `output_failure_example` similarly identifies its
injected optional-display failure as expected behavior.

A small `job.toml` for the CLI example:

```toml
points = 3
precision = "long-double"
[model]
U = "4.000000000000000001"
spin = "3/2"
```

An illustrative `state.ini` attribute source can contain `U = "3"` and
`Spin = "1/2"`. Try:

```sh
run_cli_example --config=job.toml --input=state.ini --format=named-json
run_cli_example --config=job.toml --U=5 --format=tsv
UNI20_EXAMPLE_U=2 run_cli_example --input=state.ini --format=commented-tsv
run_cli_example --quiet --format=named-json --output=results.json
```

This input fixture demonstrates an attribute provider, not a persistent-object
file format. Bethe's real object attributes, model validation and solver remain
consumer responsibilities.
