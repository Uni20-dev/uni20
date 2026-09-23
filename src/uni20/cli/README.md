# Command-line adapters

`cli.hpp` provides CLI11 configuration, explicit parse outcomes, exact scalar
converters and semantic help/build-information documents. Link application
frontends to the optional `uni20_cli` target (`UNI20_BUILD_CLI=ON`).

Options remain ordinary CLI11 declarations. Keep numerical initialization and
output-file creation in the executable after a successful run outcome.
Parser-independent identity and help content belong in
[`common/program.hpp`](../common/program.hpp), not in this module.

See [Command-line applications](../../../docs/diagnostics/command_line.md) for
the lifecycle, precision policy, dependency selection and scope.
