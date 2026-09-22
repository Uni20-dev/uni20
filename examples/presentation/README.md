# Presentation Examples

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
  and missing values, then renders a report or writes CSV/TSV. Run
  `presentation_data_table_example [pretty|csv|tsv]`; machine output contains
  only the rectangular table.
- `presentation_example_common.hpp` contains shared example-only policy and
  styling helpers; it is not an executable target.

See the [examples index](../), [Presentation Formatting](../../docs/diagnostics/presentation.md),
and the [Display Layer](../../docs/diagnostics/display_layer.md).

See [Typed Data Tables](../../docs/diagnostics/data_tables.md) for the API and export contracts.
