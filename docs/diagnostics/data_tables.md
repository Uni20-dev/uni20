# Typed Data Tables and Output Adapters

**Status:** the first owning-table slice is implemented in
`<uni20/common/data_table.hpp>`: checked typed insertion, native half-integers,
column display settings, report-table snapshots and CSV/TSV export. Link against
`uni20_common`. JSON and optional streaming sinks with replay remain follow-on
work within the same table API. Sections below distinguish current behavior
from the agreed extension direction.

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

The table owns its schema and rows. `columns()` returns the immutable schema;
`rows()` returns a read-only span of typed tuples. Spans/references to rows can
be invalidated by a subsequent append. `title()` and `size()` provide metadata
and retained row count. `data_table<Ts...>::make_row(...)` performs the same
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

The current table retains all rows. With streaming, retention will become a
construction-time policy of this same API: retain all rows by default, or select
`retention::none` for large/long-running output. Changing
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

Display configuration is attached to columns at construction time. Per-render
overrides are future work; their precedence will be per-render override, column
setting, then library default, without mutating the table. Reuse the scalar I/O customization boundary
where it is correct, rather than introducing another scalar formatting system.

State digit semantics explicitly: general notation uses significant digits;
fixed notation uses digits after the decimal point; scientific notation uses
digits after the decimal point in the mantissa. A setting of three scientific
fractional digits therefore displays four significant digits.

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
rounded and must not be described as lossless.

## Output Adapters

| Adapter | Status | Input and behavior |
|---|---|---|
| Terminal/plain display | Implemented | Format typed cells and build an existing `report_table`; reuse glyph, width, style and border handling |
| CSV/TSV | Implemented | Write one rectangular table, stable column identifiers and machine scalar values; ignore display layout |
| JSON | Planned | Write schema, useful metadata and typed value encodings; optionally group named tables in an application result |
| Markdown | Planned | Apply display settings and emit a table with target-specific escaping and limited emphasis |
| LaTeX | Planned | Apply display settings, escape ordinary text and support explicitly declared mathematical labels |
| HTML/notebook | Planned | Emit semantic table markup with headings, values and styles; do not translate terminal whitespace or ANSI |

Writers consume the typed table directly. Only the terminal adapter needs the
intermediate `report_table` for a snapshot; planned live display will use the existing
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

The ordered schema, row arrays and precision-preserving decimal strings in the
worked example are agreed design choices, not an implemented Uni20 file format.
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

### Streaming to Multiple Sinks (Follow-on)

The same table can later attach one or more sinks: for example, pretty terminal
output and a full-precision TSV file. Split implementation responsibilities into
schema/checked row construction, optional retained history, and output adapters.
Numerical fan-out occurs before the terminal adapter converts values to
presentation cells. A collector is internal history storage, not another table
API that users must choose instead of streaming.

Start synchronously. Convert and validate each entire row once; retain it when
enabled, then deliver a read-only typed row to sinks in attachment order. Each
sink independently chooses columns, display formatting and export precision.
An invalid row is neither retained nor delivered. Sinks cannot change the
canonical row or each other's selections. Threads and queues are unnecessary
for this facility.

**Attachment is allowed after rows exist.** With default retention, attaching a
sink begins its output, replays all retained rows in order and then subscribes
it to future appends. Synchronous attachment finishes replay before another
append can begin, so successful delivery neither omits nor duplicates a row.
With `retention::none`, a late sink can receive future rows but cannot recover
discarded history. Report that limitation at attachment; a request for complete
replay must fail rather than silently become future-only output. The schema
remains fixed after the first row or output begins; sink membership does not.

Provisional lifecycle sketch:

```cpp
auto table = make_data_table(/* typed columns; retain rows by default */);
table.append(/* first completed result */);
auto screen = table.attach(terminal_sink(/* selected columns */));
auto file = table.attach(tsv_sink(output_stream)); // Both replay the first row.
table.append(/* next result: retained and delivered to both */);
write_json(snapshot_stream, table);              // Independent current snapshot.
table.finish(/* final summary */);
```

Give each sink a `begin(schema, initial_metadata) -> rows -> finish(summary)`
lifecycle. Bethe can supply U, density and conventions at the beginning, while
CPU time and overall success arrive at the end. Generic CSV/TSV stays
rectangular: summaries belong in separate metadata output or an explicitly
documented application format. JSON can include a final summary after its row
array; an in-progress snapshot must not claim a final success status.

Closing/finalizing one sink does not discard retained rows or prevent other
sinks from following the calculation. Table-level finish supplies the final
summary and finalizes active outputs; retained data remains inspectable and
exportable. Define finished-table attachment to replay the retained data and
deliver the recorded summary without subscribing to new rows. Exact ownership
of attachment handles and the behavior of repeated finish calls belong in the
streaming API implementation contract.

Output failures must be visible. A required data-file failure propagates to the
caller; an explicitly optional display sink may be disabled with a reported
failure while calculation continues. Delivery is not atomic across sinks, and
output failure does not roll back a retained row. Do not automatically retry a
partly delivered append or replay: doing so can duplicate data. Explicit
finalization must report flush/close failures; destruction cannot be the only
way to discover them. Settle the precise failure result and whether later sinks
are attempted after a required failure before implementing streaming.

Reuse `display::streaming_table` for live human output. Its width policy cannot
depend on unseen rows; use declared widths or explicit incremental layout.
Asynchronous delivery, backpressure, partial histories and concurrent appends
are deferred. These extensions do not require different retained/streaming
table APIs or block the first owning-table implementation.

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

Agreed precision-preserving JSON shape for the same `double` table:

```json
{
  "title": "Excitation levels",
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
already produce points in the desired output order. After the owning-table
slice is available, migrate that frontend to append typed results without
changing the solver; add optional live sink attachment when streaming lands.

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
4. **Next:** add JSON using the agreed schema and synchronous sink attachment/
   replay with the same table API. Expose construction-time `retention::none`
   with streaming; it is not available in the initial snapshot-only slice.
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
from terminal/color settings. Validate JSON with an independent parser when its
writer is added. Reader defaults that narrow values are not exporter tests.

When streaming is added, test attachment before/after rows, replay order, column
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
- JSON and streaming API sketches are provisional; no compatibility aliases are
  needed when implementing those extensions.
