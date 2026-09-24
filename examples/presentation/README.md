# Presentation Examples

- `cli_example.cpp` demonstrates CLI11 help, version/build information, exact
  half-integers and deferred native-precision conversion followed by typed-table
  display or CSV/TSV/JSON export. Build `presentation_cli_example` with
  `UNI20_BUILD_CLI=ON`; see [Command-line Applications](../../docs/diagnostics/command_line.md).
- `table_layout_example.cpp` demonstrates report composition, table borders,
  spans, separators, decimal alignment, width pressure, and numeric formatting.
  It builds both `presentation_example` and `presentation_table_layout_example`.
- `glyph_policy_example.cpp` demonstrates Unicode, emoji, ASCII, charset, and
  color policies controlled explicitly or through the environment.
- `text_art_example.cpp` demonstrates indentation and semantic-glyph tensor
  network sketches.
- `diagnostic_region_example.cpp` renders width-aware annotated source regions.
- `mdspan_example.cpp` demonstrates full and abbreviated rank-N mdspan output.
- `display_example.cpp` demonstrates semantic status messages, report builders,
  and streaming tables.
- `data_table_example.cpp` stores typed results, including exact half-integers
  and missing values, then renders a report or writes CSV/TSV/JSON. Run
  `presentation_data_table_example [pretty|csv|tsv|json|stream|stream-no-retention]`.
  Streaming modes send selected display columns to stderr and full-precision TSV
  to stdout. `stream` demonstrates late attachment/replay; `stream-no-retention`
  attaches before the first row and keeps no history.
- `presentation_example_common.hpp` contains shared example-only policy and
  styling helpers; it is not an executable target.
- `run_metadata_example.cpp` demonstrates native metadata, snapshots, display
  mapping, provenance and explicit computation timing without CLI11.
- `configuration_sources_example.cpp` resolves API, file, attribute, environment
  and default sources, displays their origins, and copies selected output attributes.
- `run_cli_example.cpp` combines CLI11 source bindings, an explicit TOML/INI file,
  input attributes, deferred precision and shared output controls.
- `output_live_example.cpp` displays live results while writing owned TSV and
  JSON files. Pass an existing output directory; files are created exclusively.
- `output_named_tables_example.cpp` writes different schemas and a final run
  summary in the explicit named JSON envelope.
- `output_replay_example.cpp` adds a JSON destination after rows already exist.
- `output_no_retention_example.cpp` streams future rows with an explicit offset
  after earlier rows were discarded.
- `output_failure_example.cpp` reports an optional display failure while
  delivering every result to the required TSV destination.

Build targets match these eight filenames without `.cpp`. See
[Using Run Contexts](../../docs/diagnostics/run_context_usage.md) for recipes,
sample configuration/input files, output contracts and remaining boundaries.

See the [examples index](../), [Presentation Formatting](../../docs/diagnostics/presentation.md),
and the [Display Layer](../../docs/diagnostics/display_layer.md).

See [Typed Data Tables](../../docs/diagnostics/data_tables.md) for the API and export contracts.
