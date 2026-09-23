# Typed Data Tables and Output Adapters

**Status:** implemented: typed rows with checked insertion, native half-integers,
optional retention, report snapshots, CSV/TSV and typed JSON export, and
synchronous terminal/CSV/TSV/JSON sinks with replay. Include
`<uni20/common/data_table.hpp>` for tables and delimited snapshots,
`<uni20/common/data_table_json.hpp>` for JSON, or
`<uni20/common/data_table_sinks.hpp>` for streaming adapters. Link against
`uni20_common`. Document/notebook adapters and asynchronous output remain future
extensions.

## Current API

```cpp
#include <uni20/common/data_table.hpp>

namespace p = uni20::presentation;
auto results = p::make_data_table(
    "Results",
    p::data_column<unsigned>("index").label("Index"),
    p::data_column<long double>("energy").label("Energy").fixed(6),
    p::data_column<std::optional<long double>>("gap").missing("unavailable"));
results.append(1, -6.25L, std::nullopt);
results.append(2, -6.125L, 0.125L);

auto display = p::to_report_table(results);
p::write_csv(output_stream, results);
p::write_tsv(output_stream, results,
             {.precision = p::data_export_precision::display}); // Deliberate real rounding.
```

The table owns its schema and retained rows. It is move-only, so subscriptions
cannot be duplicated by copying. Sink IDs follow a moved table; use moved-from
tables only for destruction. Do not move a table during a sink callback. `columns()` returns the immutable schema;
`rows()` returns a read-only span of typed tuples. Spans/references to rows can
be invalidated by a subsequent append. `title()` and `size()` provide metadata
and accepted row count. `retained_size()` reports stored rows; the counts differ
with retention disabled. `data_table<Ts...>::make_row(...)` performs the same
checked conversion as `append` without requiring retained history.

Supported columns are the standard signed/unsigned integer storage types,
`bool`, Uni20 `Real` types, `basic_half_int<T>`, `std::string`, and optional
numeric/boolean/half-integer values. Plain character types, borrowed string
views, optional strings and complex values are not column types. String views
are accepted as insertion inputs and copied; null C string pointers are rejected.
Row arity and unsupported conversions fail at compile time. Integer range
failures throw `std::overflow_error`; invalid schema identifiers, duplicate
identifiers and invalid display precisions throw `std::invalid_argument`.
Identifiers follow `[A-Za-z_][A-Za-z0-9_]*`.

Real output supports `float`, `double`, `long double`, and the MPLAPACK native
`_Float128` provider of `uni20::float128`. Other extension providers require a
lossless formatter before their tables can be rendered/exported; those writer
instantiations fail at compile time instead of narrowing through `long double`.
Formatting is locale independent, including the native binary128 C-library path,
which temporarily selects the C numeric locale for the calling thread only.

A writer checks stream failures even when stream exceptions are disabled and
throws `std::ios_base::failure`. It does not flush or close the caller-owned
stream; the caller must explicitly flush/close and check for delayed file errors.
A failed write may leave partial output, but never modifies the retained table.

See the runnable [data table example](../../examples/presentation/data_table_example.cpp)
for terminal, CSV and TSV output from the same table.

## Purpose and Existing Boundary

A calculation should construct a table once, retain its numerical values, and
use it for terminal display, scientific export and eventual Python plotting.
Column formatting can be specified at construction time, but is applied only
when an output adapter consumes the table.

The current [presentation layer](presentation.md) provides `report_table` with
styled-text cells, column spans, separators and layout controls. Numeric inputs
are converted to text when inserted. This remains the appropriate model for
arbitrary human-facing reports; it does not retain numerical types or distinguish
a missing number from text such as `"unavailable"`.

The Bethe application currently uses `uni20::format_real` before inserting
numbers into report cells, preserving the selected scalar's round-trip digits.
That preserves text precision but loses the distinction between numbers and
strings. A typed table should remove the need to preformat those values and
support multiple outputs from the same result rows.

This is a small presentation/export facility, not a dataframe engine, tensor
serializer or persistent object store. Sorting, fitting and numerical analysis
remain application or analysis-library operations.

## Data Model

| Part | Contents | Rules |
|---|---|---|
| Schema | Ordered columns with unique stable identifiers and value types | Identifiers such as `energy_per_site` are independent of display labels |
| Rows | One typed value per column; optional values where declared | No spanning cells, separators or embedded layout rows |
| Metadata | Table title, column labels, optional units and descriptions | Describes the meaning of the values |
| Presentation | Numeric notation/digits, alignment, missing-value label and future optional emphasis | Does not alter stored values or export precision |

Use actual scalar types, including `long double` and enabled extension types.
Do not force numbers through `double`, a fixed numeric cell variant, or formatted
strings. Strings are data when the declared column type is a string. Boolean,
integer, real, `basic_half_int<T>` and optional numeric columns belong in the
first implementation. Complex and general rational columns require explicit
writer support later; half-integers do not depend on a general rational type.

The first representation is an owning `data_table<Ts...>` with a fixed
heterogeneous schema and rows equivalent to `std::tuple<Ts...>`. A typed column
factory can infer the table's template arguments. This keeps scalar formatting
and export dispatch templated on the actual value type; it does not prescribe
the physical container layout permanently.

Retention is a construction-time policy of this API: retain all rows by default,
or select `data_table_options{.retain = retention::none}` for large/long-running
output. Changing
retention mid-run and bounded partial histories are deferred. Separate schema
and checked row construction from history storage now, so adding streaming does
not require a second user-facing table type.

The table owns textual metadata and copies/moves retained row values, including
string cells. It must not retain temporary `string_view` values accidentally.
Without retention, a completed row still owns its cell values during synchronous
delivery. Rendering and writing are synchronous operations, with no concurrent
mutation or reentrant append/attachment from callbacks.
Invalid schema identifiers or row shape/type mismatches should be rejected at
construction/insertion rather than rediscovered by each writer.

### Insertion and Conversions

The schema supplies the destination types. `append` accepts values convertible
to those types rather than requiring an exact type match: an `int` can populate
a representable unsigned integer or real column, a value or `std::nullopt` can
populate an optional column, and string literals/views are copied into owned
string cells. Engaged optional inputs undergo the same element conversion.

Integer-to-integer conversion must check representability before casting,
including negative inputs to unsigned columns. Reject an out-of-range value;
never wrap it. Complete conversion and validation of the row before inserting
it, so a rejected cell does not leave a partial row. Float-to-integer conversion
and non-boolean inputs to boolean columns need explicit policies before being
supported; convenient insertion is not a reason to inherit accidental truth
conversions or unchecked truncation.

Numeric conversion to the declared column type happens at insertion. A chosen
`double` column can round a higher-precision input to `double`; subsequent
machine export preserves the stored value. Conversely, an extended-precision
column cannot recover precision already lost in a supplied `double` literal.
The exactly representable literals in the worked example are safe for each
supported real precision. Display precision never participates in insertion.

### Exact Half-Integer Values

Store `basic_half_int<T>` itself, including inside optional columns. Half-integer
storage conversions preserve the exact doubled integer and check narrowing;
integral inputs represent whole numbers and require range checking before
doubling or narrowing. Do not automatically route floating inputs through the
half-integer constructor's rounding behavior. A caller choosing rounding must
construct that value explicitly before insertion.

Display may select fractions such as `3/2` or exact decimals such as `1.5`.
CSV/TSV numerical export uses exact integer or `.5` decimal spellings derived
from `twice()`, never `to_double()`. Handle negative halves and the underlying
integer's full supported range without negating its minimum value. Typed JSON
records the half-integer type and its doubled integer as an exact decimal
string. The column remains a half-integer quantity, rather than a user-managed
`twice_spin` column or a string containing a fraction.

A later borrowed row/column adapter can expose existing application results
without copying them. Its lifetime and mutation rules must be explicit. Writers
should use a small typed traversal interface so ownership is not built into the
format algorithms. Dynamic Python schemas and type erasure are later adapter
work; neither justifies narrowing the initial C++ value model.

## Formatting and Writer Policies

Display configuration is attached to columns at construction time. A
`table_projection` selects columns in an independent order and overrides whole
`data_column_display` values by identifier. Empty selection means all columns.
The projection applies to report snapshots and each terminal/delimited sink,
without mutating the table. Unknown/duplicate selected columns and overrides
for unselected columns are rejected before output begins. Reuse the scalar I/O customization boundary
where it is correct, rather than introducing another scalar formatting system.

State digit semantics explicitly: general notation uses significant digits;
fixed notation uses digits after the decimal point; scientific notation uses
digits after the decimal point in the mantissa. A setting of three scientific
fractional digits therefore displays four significant digits.

`-1` selects the stored real type's `max_digits10` as the digit count in all
three notations. Both column helpers and projection overrides accept it,
including an override that chooses a notation and leaves precision at its
default. Values below `-1` are rejected. Zero digits is valid for fixed and
scientific notation, but general notation requires a positive count or `-1`.
This default digit count is not a universal round-trip guarantee: for `double`,
`.fixed(-1)` uses 17 fractional digits and renders `1e-20` as zero.

Use `.round_trip()` on a real or optional-real column to request text from which
the original finite value can be recovered, including the sign of zero:

```cpp
p::data_column<long double>("energy").round_trip();
```

This selects general notation, `max_digits10`, and preservation of negative
zero. Ordinary `.fixed(...)`, `.scientific(...)` and `.general(...)` helpers
normalize negative zero; calling one after `.round_trip()` replaces that mode
with its display format. To apply round-trip formatting to one projection, copy
the desired column's `.round_trip().display()` settings into the projection's
`display` map. Formatting never changes stored values, and NaN payload bits are
outside text round-tripping's scope.

A future conditional-emphasis adapter will receive the original typed value,
including missingness, before display rounding. It returns semantic presentation intent rather than
ANSI, HTML or LaTeX fragments. The terminal adapter can lower that intent to
existing styled text. Unsupported emphasis may be omitted by a writer without
changing values.

Machine export uses a separate policy. The default is
locale-independent decimal output with enough digits to round-trip the original
finite scalar type, no intermediate narrowing, and preserved negative zero.
The current scalar formatting default normalizes negative zero, so export must
override that choice. Reusing scalar I/O also requires checking the enabled
provider's conversion path and locale behavior, not merely its precision option.
Preservation of NaN payload bits is outside text export's scope.

The explicit `{.precision = data_export_precision::display}` export option
applies the selected real display precision to CSV/TSV. It still uses machine
field encoding and omits styles, units suffixes and other decorations. Such export is intentionally
allowed to round and has no blanket lossless guarantee; a column explicitly
configured with `.round_trip()` retains that precision even in display mode.

## Output Adapters

| Adapter | Status | Input and behavior |
|---|---|---|
| Terminal/plain display | Implemented | Format typed cells and build an existing `report_table`; reuse glyph, width, style and border handling |
| CSV/TSV | Implemented | Write one rectangular table, stable column identifiers and machine scalar values; ignore display layout |
| JSON | Implemented | Write schema, useful metadata and typed value encodings; optionally group named tables in an application result |
| Markdown | Planned | Apply display settings and emit a table with target-specific escaping and limited emphasis |
| LaTeX | Planned | Apply display settings, escape ordinary text and support explicitly declared mathematical labels |
| HTML/notebook | Planned | Emit semantic table markup with headings, values and styles; do not translate terminal whitespace or ANSI |

Writers consume the typed table directly. Only the terminal adapter needs the
intermediate `report_table` for a snapshot; live display uses the existing
`display::streaming_table`. There is no reverse conversion from rich report
cells to numerical data. A rich report containing spans or commentary is not
automatically a CSV dataset; the application selects the data table to export.

A one-shot write exports the current retained table without changing its rows
or attached sinks. It does not subscribe to subsequent appends or finalize the
calculation. With retention disabled, full-history snapshot/export must fail
explicitly rather than emit an apparently complete empty table after rows have
been discarded. A live sink can still write subsequent rows in that mode.

Separate writing from file ownership: a writer accepts a caller-owned output
stream and reports encoding/I/O failures. A later save-to-path convenience can
open the destination and invoke the writer. Stream output is incremental, not
transactional; an I/O error can leave partial output. Export must not depend on
TTY detection, terminal width, color environment variables or the display sink.
The existing [display layer](display_layer.md) continues to own human-output
routing, rather than acquiring responsibility for numerical export files.

### Delimited Files

Use UTF-8 text, a header of stable identifiers, `.` as the decimal separator,
and explicit comma or tab delimiters. Writers preserve supplied text bytes;
they do not validate or transcode UTF-8. Export one table per file/stream invocation
without interspersed diagnostics, metadata prose or tables with different schemas.
Both writers use CSV-style quoting: fields containing the chosen delimiter,
a double quote, CR or LF are enclosed in double quotes, and interior double
quotes are doubled. Records end with LF. Empty string values are quoted as `""`.
Missing numeric/boolean/half-integer values are empty fields; a one-column
missing row is quoted as `""` to preserve its record instead of emitting a blank
line that readers might skip. Quoting does not establish a separate null type.
The basic [IANA TSV registration](https://www.iana.org/assignments/media-types/text/tab-separated-values)
forbids embedded tabs and supplies no null encoding or CSV-style quoting rule.
A tab-delimited CSV dialect can support quoted fields, but that behavior must
be documented rather than assumed of every TSV reader.

There is no universal missing-value token in TSV. This API uses empty fields;
readers must apply the known schema to interpret those fields as missing values.
A configurable export token is future work; `.missing(...)` changes display only.
W3C's
[tabular metadata vocabulary](https://www.w3.org/TR/tabular-metadata/#inherited-properties)
uses an empty string as the default null marker and permits declared alternatives;
that is a reader/schema convention, not an intrinsic TSV value type.

Use `true`/`false` for booleans. Non-finite reals may use documented `nan`, `inf`
and `-inf` spellings; a missing value remains distinct from a present NaN.
Optional string columns need a separate missing-value policy: generic readers
do not reliably distinguish missing strings from empty strings. A token such
as `NA` or `\N` can also collide with real string data, so a token alone does not
solve the problem without an agreed escaping/reader contract. CSV/TSV alone is
not a fully typed, lossless interchange format; JSON retains `null` versus `""`.

A whitespace numeric writer with `#` comments can be added for plotting tools.
It is distinct from general TSV: treating repeated whitespace as one delimiter
does not preserve empty fields. Applications must choose a missing-value token
and select numeric columns for that route.

### JSON Encoding

Use an ordered column schema and row arrays so column order and metadata are
explicit. For a precision-preserving mode, encode all real columns as decimal
strings, with the scalar radix and precision recorded in the schema. This avoids
switching a column's JSON representation when moving from fp64 to fp128 and
avoids relying on the reader's default numeric precision. Schema describes the
actual arithmetic type, not merely the spelling `long double`.

Use JSON booleans and `null` for booleans and missing values. A real column's
encoding can explicitly admit `"nan"`, `"inf"` and `"-inf"`, distinct from
`null`. Exact large integer columns also need a decimal-string encoding when
their range exceeds the interoperable integer range; small bounded indices can
use JSON numbers. Complex columns would use separately encoded real/imaginary
components; rational columns would retain numerator/denominator rather than
display fractions or approximations. These latter mappings are not part of the
first implementation.

For `basic_half_int<T>`, describe `type: "half_int"`, the signed storage width,
and `encoding: "twice_decimal_string"`. A value of `3/2` is encoded as `"3"`;
an optional missing value remains `null`. This follows the existing JSON
schema/rows structure while retaining exact quantum numbers.

The ordered schema, row arrays and precision-preserving decimal strings are
implemented by `write_json` and `json_sink`. Integer columns whose type has
more than 53 value bits always use decimal strings, independent of the current
cell value; smaller types use JSON numbers. Text must be valid UTF-8, and
control bytes are escaped. Invalid UTF-8 throws `std::invalid_argument`.
Initial `metadata` and final `summary` are string-to-string application maps.
Applications should use `format_real` when storing precise numeric metadata.
An unfinished snapshot omits `summary`; a finished empty summary is `{}`.
`first_row` records the zero-based index where an output begins: zero for
snapshots/full replay, or the current accepted count for future-only attachment.
A streaming JSON document is complete only after explicit finalization; it is
not a checkpoint or interrupted-run recovery format.
Extended mappings for complex/rational values remain future work.
JSON forbids bare NaN/infinity tokens and
does not guarantee that consumers retain extended numeric precision; see
[RFC 8259, numbers](https://www.rfc-editor.org/rfc/rfc8259#section-6).

### Labels and Rich Output

Keep a plain label independent of its machine identifier. Later allow a
separately declared mathematical label, for example plain `Energy/site` and
math `E/L`. Ordinary labels and string cells must be escaped for every target;
they must never become executable markup implicitly. A policy for explicitly
trusted math markup can be added with the LaTeX/notebook adapters without
designing a general document language now.

HTML is useful as the representation returned by a future Python
`_repr_html_()` or MIME display adapter; users need not manage HTML files.
Plain `repr` and notebook display should be bounded previews, with elision
visible. File export is exhaustive unless the caller explicitly selects rows.
Python column access should expose typed values independently of any rendered
representation. Plot choices such as axes, connected series, error bars and
branch grouping remain explicit application choices.

### Streaming to Multiple Sinks

A table accepts synchronous sinks at any time. Default attachment begins output,
replays all retained rows, then receives future appends in order. A one-shot
write remains an independent snapshot. Each adapter receives the original typed
row before any display formatting. There are no worker threads or queues.

```cpp
#include <uni20/common/data_table_sinks.hpp>
namespace p = uni20::presentation;
auto table = p::make_data_table(
    "Dispersion", {.metadata = {{"model", "Hubbard"}, {"U", "4"}}},
    p::data_column<unsigned>("branch"),
    p::data_column<long double>("energy").fixed(6));
table.append(1, -6.25L);
// Both sinks first replay the row already present.
auto screen = table.attach(p::terminal_sink({
    .projection = {.columns = {"energy"}},
    .destination = uni20::display::stream::err}));
auto file = table.attach(p::tsv_sink(output_stream));
table.append(2, -6.125L);
p::write_json(snapshot_stream, table); // No final summary yet.
table.finish({{"status", "complete"}, {"cpu_seconds", "1.125"}});
```

The table owns attached adapters; streams are borrowed and must outlive their
attachment. A `data_attachment` contains a table-local `id` and a failure
`report`. IDs remain meaningful after the table is moved, but must not be used
with other tables. `finish_sink(id, summary)` finalizes one sink and stops its
subscription without ending the table. `finish(summary)` ends appends,
finalizes every remaining sink, and preserves both rows and summary. A late
attachment to a finished table replays history and receives the summary without
subscribing. Repeated finish with no new summary (or the same summary) writes
nothing; an attempt to replace a finalized summary throws `std::logic_error`.
Repeated table finish re-reports any failure from its first finalization.

Explicit finalization is required to discover flush errors. Destruction releases
adapters without calling their output methods or inventing a final summary.
CSV/TSV/JSON adapters flush their borrowed stream at finish, but do not close it;
the owner must check close errors. The terminal adapter uses
`display::streaming_table` with incremental fit widths. The default display
router flushes each emission and reports stdio failures; custom display routers
must report their own failures. Terminal metadata/summary display can be
disabled with `.show_metadata = false`; generic CSV/TSV always omits them to keep
the output rectangular. JSON records them separately from rows.

#### Delivery and failure contract

`append` converts and validates the whole row once. Invalid input is neither
retained nor delivered. Once accepted, the row is retained when enabled and the
accepted count increases, then active sinks run in attachment order. Mutation
from inside a callback is rejected; concurrent access and moving a table during
callbacks are not supported.

A sink exception disables that sink. Other active sinks still receive the row.
After all attempts, any required failure throws `data_delivery_error`, whose
`report()` contains **all failures from that operation**, including optional
ones. Each `data_sink_failure` records the attachment ID, required flag,
`begin`/`row`/`finish` phase, and original `exception_ptr`.

If only optional sinks fail, `append`, `finish_sink` and `finish` return a
`data_delivery_report`. `attach` puts it in the `data_attachment`. Callers using
optional sinks **must inspect these reports**; required-only calculations can
rely on exceptions. To make a display optional:

```cpp
auto screen = table.attach(p::terminal_sink(), {.required = false});
report_optional_failures(screen.report); // Application reporting policy.
auto delivery = table.append(3, -6.0L);
report_optional_failures(delivery);
```

Required sinks are the default. A failed begin or replay disables only the new
attachment and reports the exact phase; it does not touch existing sinks or
retained rows. A failed finish still allows other sinks to finalize. Failures
from prior append/attachment operations are not emitted again by later
operations; `sink_state(id)` remains `failed`. Reports are not a success status
for the calculation: the application decides how to respond to already reported
failures and what final summary to supply.

Delivery is not atomic across outputs. Output failure does not roll back a row,
and **no append, replay or finalization is retried automatically**. A caller
must not retry the same append after `data_delivery_error`, since the row has
already been accepted. File output may be partial after a failure.

#### Disabling retention

```cpp
auto table = p::make_data_table("Long run", {.retain = p::retention::none},
                                p::data_column<double>("energy"));
table.attach(p::tsv_sink(output_stream)); // Before the first row: full history is available.
table.append(-6.25);
// Explicitly acknowledge that earlier rows cannot be replayed.
table.attach(p::terminal_sink(), {.replay = p::sink_replay::future_only});
table.append(-6.125);
table.finish();
```

With `retention::none`, row storage stays empty; each accepted row lives only
through synchronous delivery. `rows()` and all full-history snapshot/render
functions throw before writing anything, even for an empty no-retention table.
A late full-replay request fails before sink output begins. Explicit
`future_only` attachment is available in either retention mode; JSON marks the
starting row index. CSV/TSV remain rectangular, so applications must record any
partial-history provenance separately when saving such output.

#### Custom adapters

A sink object supplies these methods (which may be templated on schema/row):

```cpp
void begin(std::string const& title, Schema const& schema,
           p::table_metadata const& metadata, p::data_sink_start start);
void row(Schema const& schema, Row const& row);
void finish(p::table_metadata const& summary);
```

Schema, metadata and row arguments are borrowed for the duration of the callback;
a sink must copy anything it retains. A sink can be move-only. The table owns it
until successful finish, failure or destruction. Custom sinks must propagate
output errors and must not rely on their destructor to perform fallible output.
The table keeps no row history on their behalf.

Asynchronous delivery, backpressure, bounded partial histories and concurrent
appends remain deferred. They do not require another user-facing table API.

## Worked Bethe Table

Consider an excitation table with momentum, energy, an optional gap and a
convergence flag. The two rows below are invented, exactly representable test
fixtures, not Bethe solver predictions. The failed row illustrates missing data;
applications decide whether failed candidates belong in their exported table.

The owning-table portion is implemented:

```cpp
namespace p = uni20::presentation;

template <uni20::Real Real> auto example_levels()
{
  auto table = p::make_data_table(
      "Excitation levels",
      p::data_column<std::uint32_t>("level").label("Level"),
      p::data_column<uni20::half_int>("spin").label("Spin").fractional(),
      p::data_column<Real>("momentum").label("P").fixed(3),
      p::data_column<Real>("energy").label("Energy").fixed(4),
      p::data_column<std::optional<Real>>("gap").label("E-E0").fixed(4),
      p::data_column<bool>("converged").label("Converged"));

  table.append(1, uni20::from_twice(1),  0,   -6.25,  0,            true);
  table.append(2, uni20::from_twice(-3), 0.5, -6.125, std::nullopt, false);
  return table;
}

auto table = example_levels<double>();
auto report_table = p::to_report_table(table);
p::write_tsv(output_stream, table); // Machine precision by default.
```

Terminal/plain display, omitting optional borders and styling:

```text
Level  Spin      P   Energy    E-E0  Converged
    1   1/2  0.000  -6.2500  0.0000  true
    2  -3/2  0.500  -6.1250       —  false
```

TSV export (fields below are separated by literal tabs; the second gap is empty):

```tsv
level	spin	momentum	energy	gap	converged
1	0.5	0	-6.25	0	true
2	-1.5	0.5	-6.125		false
```

Precision-preserving JSON shape for the same `double` table (optional metadata
fields such as empty units/descriptions and false nullability omitted here for
brevity):

```json
{
  "title": "Excitation levels",
  "metadata": {},
  "first_row": 0,
  "columns": [
    {"id": "level", "label": "Level", "type": "uint32", "encoding": "number"},
    {"id": "spin", "label": "Spin", "type": "half_int", "storage_bits": 64,
     "encoding": "twice_decimal_string"},
    {"id": "momentum", "label": "P", "type": "real", "radix": 2,
     "precision_bits": 53, "encoding": "decimal_string"},
    {"id": "energy", "label": "Energy", "type": "real", "radix": 2,
     "precision_bits": 53, "encoding": "decimal_string"},
    {"id": "gap", "label": "E-E0", "type": "real", "radix": 2,
     "precision_bits": 53, "encoding": "decimal_string", "nullable": true},
    {"id": "converged", "label": "Converged", "type": "bool"}
  ],
  "rows": [
    [1, "1", "0", "-6.25", "0", true],
    [2, "-3", "0.5", "-6.125", null, false]
  ]
}
```

Changing the display from four to eight decimal places leaves default TSV and
JSON values unchanged. Exporters need not preserve the display settings. A
Python consumer can read the TSV into named columns or decode the JSON's decimal
strings using a suitable numeric type. Converting to binary64 for a plot is a
consumer decision, not an export-time loss of precision.

### First Consumer: Hubbard Dispersion

Bethe's `apps/bethe-hubbard-dispersion.cpp` currently collects its points, formats
them into string cells, and selects report or CSV/TSV output. Its solver loops
already produce points in the desired output order. With the table API available, migrate that frontend to append typed results without
changing the solver; use optional live sink attachment where useful.

This exercises native `Real` precision, exact spin labels, missing energies,
infinite endpoint rapidities, diagnostic columns and final CPU timing. The
terminal may select branch/momentum/energy/status while a file keeps every
diagnostic. Initial model metadata and the final summary must survive the
migration without being inserted as irregular CSV rows. Bethe's existing
comment-prefixed metadata output is an application dialect, not the generic
CSV writer's contract.

## Implementation Stages and Verification

1. **Implemented:** the owning typed table, shared schema/checked row
   construction, optional values, native half-integers and column display
   settings. The existing rich table remains unchanged; row validation is
   independent of retained history.
2. **Implemented:** the terminal/report-table adapter and CSV/TSV writers, with
   integer, boolean, real, half-integer and string columns plus optional numeric
   and boolean values. Unsupported column/writer combinations fail explicitly.
3. **Implemented:** the Bethe-shaped example and focused tests. Integrate an
   actual Bethe result table in a separate consumer change, retaining its
   selected arithmetic precision and scientific status information.
4. **Implemented:** typed JSON, synchronous sink attachment/replay, per-output
   column projection/display overrides, initial metadata and final summaries,
   and construction-time `retention::none` in the same table API.
5. Markdown, LaTeX, notebook HTML, conditional emphasis and borrowed/dynamic
   adapters are subsequent output/access extensions.

Required evidence includes display-format changes leaving default exports
unchanged; real values round-tripping in their original enabled types; values
that would lose information if narrowed to `double`; preserved negative zero;
missing versus non-finite values; and correct delimiter/quote/newline escaping.
Check convenient insertion from ordinary literals, engaged/disengaged optionals
and temporary strings; integer boundaries in both signedness directions; and
rejection without row insertion when any cell is out of range.
For half-integers, test positive and negative halves, whole values, optional
values, storage conversions, integer-to-half range checks and doubled values
beyond binary64's exact integer precision. Check exact decimal/fraction output
and JSON doubled-integer reconstruction without a floating intermediate.
Test zero rows, invalid/duplicate identifiers, stream failures and independence
from terminal/color settings. Validate JSON with an independent parser. Reader defaults that narrow values are not exporter tests.

Streaming tests cover attachment before/after rows, replay order, column
projections, independent sink finalization, summary delivery, failed replay and
required/optional sink failures. Demonstrate that no-retention mode has bounded
row memory and refuses unavailable full-history export, and that late future-only
attachment makes its lack of history explicit. Tests must distinguish row
validation failure from output failure after a row has been retained/delivered.

The first slice needs neither CUDA access nor Async acquisition, Python,
plotting dependencies, a persistence backend or a generic serialization
framework. Applications provide completed result values; this layer must not
silently read device memory or wait on unresolved calculations.

## Agreed Initial Conversion and Format Boundaries

- CSV/TSV use empty missing numeric fields and CSV-style quoting with the
  corresponding delimiter. Optional string columns remain unsupported.
- Float-to-integer and non-boolean-to-boolean insertion are rejected. Callers
  must explicitly choose and perform those conversions before insertion.
- Floating-to-half-integer insertion is rejected; exact integer and half-integer
  inputs are checked for representability before storing the row.
- Streaming is synchronous and non-atomic across sinks. Attempt all sinks, disable
  failures, then throw for required failures or return optional failures explicitly.
