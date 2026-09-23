# Command-line applications

Uni20 uses CLI11 for option parsing and the presentation layer for help,
application banners, version/build information and usage errors. Applications
declare options directly on `CLI::App`; there is no second option-definition
language. Numerical libraries do not depend on CLI11.

This implements the first stage of [issue #53](https://github.com/Uni20-dev/uni20/issues/53):
single-command frontends, including CLI11 option groups. Shared run metadata,
output-file sessions and migration of application frontends are subsequent work.
An expression language for models/operators is a separate parser concern.

## Build and ownership

Enable `UNI20_BUILD_CLI=ON` and link the executable to `uni20_cli`:

```cmake
set(UNI20_BUILD_CLI ON CACHE BOOL "Build application helpers")
add_subdirectory(path/to/uni20)
add_executable(my_solver main.cpp)
target_link_libraries(my_solver PRIVATE uni20_cli)
```

The option defaults to `OFF`, including when Uni20 is the top-level project.
Neither `uni20`, `uni20_deps` nor any numerical module links `uni20_cli`.
`uni20_common` supplies the parser-independent documents in
`<uni20/common/program.hpp>` without requiring CLI11 headers.

CLI11 2.7.2 or newer is required. An existing `CLI11::CLI11` target is reused;
otherwise `UNI20_USE_SYSTEM_CLI11=AUTO` searches for a compatible package and
falls back to the pinned v2.7.2 source. `ON` requires a system package; `OFF`
selects the pinned source. Parent-supplied targets remain the parent's
responsibility; the header also checks the version when compiling the adapter.
The FetchContent revision is `cbd58a3696887b34c70949aef21a71735a0c2ad5`.

## Parse, decide, then run

```cpp
#include <uni20/cli/cli.hpp>

namespace p = uni20::presentation;
namespace c = uni20::cli;

int main(int argc, char** argv)
{
  p::program_info program{.name = "solver", .description = "Calculate a spectrum",
                          .version = "0.1"};
  CLI::App app;
  c::configure(app, program);
  std::string interaction;
  unsigned points = 101;
  uni20::half_int spin = uni20::from_twice(1);
  app.add_option("--U", interaction, "Interaction in hopping units")
      ->required()->type_name("REAL")->group("Model");
  c::add_count_option(app, "--points", points, "Number of grid points")
      ->check(CLI::PositiveNumber)->capture_default_str()->group("Numerics");
  c::add_half_int_option(app, "--spin", spin, "Exact spin")
      ->capture_default_str()->group("Model");

  auto const result = c::parse(app, argc, argv);
  if (result.requested != c::action::run)
  {
    uni20::display::emit(c::result_report(app, program, result), result.destination);
    return result.exit_code;
  }
  // Resolve precision, validate the model, open outputs, then initialize the solver.
}
```

`configure` registers `-h,--help`, `--version` and `--build-info`. Call it once,
before parsing. `parse` returns an owned outcome; it neither prints nor exits.
The default outcomes are:

| Invocation | Outcome | Destination | Status |
| --- | --- | --- | --- |
| `--help` or `-h` | full help | stdout | 0 |
| `--version` | application identity | stdout | 0 |
| `--build-info` | application and Uni20 build information | stdout | 0 |
| no arguments | full help | stderr | 1 |
| invalid arguments | concise usage error | stderr | 1 |
| valid arguments | run | executable decides | executable decides |

Use `parse_policy` to change no-argument behavior, destination or usage status.
For a program which can run entirely from defaults, use
`{.empty = c::no_arguments::run}`. This still performs normal validation.
`result_report` rejects a `run` outcome, which has no informational document.
Construction errors and unrelated application exceptions propagate to the caller.

CLI11's information callbacks run before ordinary option validation, requirements
and callbacks. Thus `--help --precision=unavailable` works even when `--U` is
required. This uses CLI11's parse stages, not an independent scan of `argv`:
`--label=--help` is a value and `-- --help` can be a positional argument.
Errors encountered while gathering arguments, such as a missing option value,
can still precede information callbacks. If several information flags are given,
CLI11's callback ordering applies; full help takes precedence.

Application option callbacks should only convert and validate configuration.
Opening output files or initializing a solver belongs after the `run` outcome.
Callbacks explicitly moved ahead of help with CLI11's priority controls can
defeat this lifecycle; do not use those priorities for application side effects.
Parsing is not transactional: an error can leave some bound variables changed.
Discard the configuration after an error, or initialize it before another parse.

## Help and identity

`presentation::program_info` owns the program name, description, version,
revision, copyright, authors, project URL, license, examples and convention
notes. Populate version/revision from the application's build, not from an
assumed Uni20 version. Its optional reference callback returns entries from the
application's authoritative citation registry; only full help evaluates it.
Version output and error diagnostics do not print a bibliography.

`help_groups(app)` reads the actual CLI11 names, descriptions, type/validator
labels, captured defaults, requirements, dependencies, exclusions and environment
names. Ordinary `.group(...)` declarations and unnamed `add_option_group(...)`
groups are supported. Named subcommands and custom CLI11 formatter layouts are
outside this first adapter's scope. CLI11 itself remains available for those
applications; the semantic document model is independent of it.

`program_report`, `help_report` and `build_info_report` return ordinary
`report_builder` documents. `display::emit` applies existing color, glyph,
character-set and terminal-width policy. Applications can also render the same
document with `render_plain`, `render_terminal` or their own display sink.
There is no second ANSI formatter. Help and build-information tables enable
`report_table::preserve_tokens()`: wrapping occurs at whitespace, while long
executable names, option values, numbers and links remain intact. The table may
exceed the requested width when those tokens cannot fit side by side.

Build information comes from the existing
[configured Uni20 snapshot](../development/build_information.md). This stage
does not introduce a run-provenance schema or Git revision discovery mechanism.

## Exact and precision-aware conversion

`add_count_option` accepts an unsigned integral destination (other than `bool`).
It checks the entire decimal token and rejects negative, overflowing and partly
parsed values. Zero is permitted unless an application adds a positivity
validator. `add_half_int_option` binds `basic_half_int<T>` directly, accepting
the exact forms supported by its parser, including `-3/2` and `-1.5`.
Non-half-integral input and values outside the doubled storage range are
rejected rather than rounded or wrapped. Both return a native `CLI::Option*`,
and the bound destination must outlive parsing and help requests which evaluate
defaults.

### Defaults from initialized variables

Prefer initializing the variable once and using CLI11's `capture_default_str()`
to display that value:

```cpp
unsigned count = 12;
uni20::half_int spin = uni20::from_twice(3);
c::add_count_option(app, "--count", count)->capture_default_str();
c::add_half_int_option(app, "--spin", spin)->capture_default_str();
```

Omitted options leave these initialized values unchanged, and help advertises
`12` and `1.5`. A supplied argument replaces the value. The captured default is
a snapshot taken at registration, so help continues to show the original default
after parsing. This works for ordinary `app.add_option(...)` bindings too.
Capturing is display-only: it neither reassigns the variable nor runs validators.
Application validation of the resolved configuration belongs after parsing.

If an explicit default is useful, `->default_val(12)` or
`->default_val("1.5")` instead converts, validates and assigns the value during
option registration, just like CLI11's ordinary variable-binding overload.
Invalid or overflowing counts and non-half-integral defaults throw
`CLI::ValidationError`; the bound value stays unchanged. Attach any additional
`.check(...)` validators before calling `default_val(...)` so they also check
that default. These are registration errors, not errors returned by `parse`.

### Deferred real conversion

Real-valued model parameters should be bound to **strings**, not to `double` or
CLI11 floating-point validators. After parsing the precision choice, call
`parse_real<Real>(token)` and validate the value in that same `Real` type. This
preserves all digits regardless of the order of `--precision` and parameter
options. Compute omitted precision-dependent defaults inside the templated run
function; the example uses `100 * numeric_limits<Real>::epsilon()`.
Conversion failures after dispatch are application errors and can be rendered
with the same error document.

Native CLI11 rules apply to positionals, `--key=value`, negative arguments,
validators, requirements, exclusions and repeated options. Choose repetition
explicitly: e.g. a vector with `->expected(1)->take_all()` collects one filename
per repeated option. Scalar options reject duplicates by default. CLI11's
diagnostics identify the option/token but do not supply source spans for the
presentation layer's annotated-source renderer.

## Runnable example

With `UNI20_BUILD_EXAMPLES=ON`, build `presentation_cli_example`. It inspects
Hubbard-style parameters and emits one typed row; it deliberately has no solver:

```sh
presentation_cli_example --help
presentation_cli_example --U=4 --spin=-3/2 --format=tsv
presentation_cli_example --U=4.000000000000000001 --precision=long-double --format=json
```

The same row can be rendered as a terminal table or exported as CSV/TSV/JSON.
Machine output contains only the table; errors go to stderr. `fp128` is offered
only in builds with native binary128 support. Stream attachment, replay and
failure handling follow the existing [data-table contracts](data_tables.md);
a shared application output-session wrapper is not implemented here.
