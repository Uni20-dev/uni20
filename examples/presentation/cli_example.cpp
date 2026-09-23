#include <uni20/cli/cli.hpp>
#include <uni20/common/data_table_json.hpp>
#include <uni20/core/scalar_io.hpp>

#include <iostream>

namespace
{
namespace p = uni20::presentation;
namespace c = uni20::cli;

struct arguments
{
    std::string interaction;
    std::string density = "0.5";
    std::string tolerance = {}; // Resolve the default in the selected scalar type.
    std::string precision = "fp64";
    std::string format = "pretty";
    unsigned points = 11;
    uni20::half_int spin = uni20::from_twice(1);
};

template <typename Real> void run(arguments const& args)
{
  auto const interaction = uni20::parse_real<Real>(args.interaction);
  auto const density = uni20::parse_real<Real>(args.density);
  auto const tolerance = args.tolerance.empty() ? Real(100) * uni20::numeric_limits<Real>::epsilon()
                                                : uni20::parse_real<Real>(args.tolerance);
  // Comparisons also reject NaN. An upper bound rejects infinities without a double conversion.
  auto const maximum = uni20::numeric_limits<Real>::max();
  if (!(interaction >= Real(0) && interaction <= maximum))
    throw std::invalid_argument("--U must be finite and nonnegative");
  if (!(density > Real(0) && density <= Real(1))) throw std::invalid_argument("--density must lie in (0, 1]");
  if (!(tolerance > Real(0) && tolerance <= maximum))
    throw std::invalid_argument("--tolerance must be finite and positive");

  auto table =
      p::make_data_table("Resolved Hubbard-style parameters (no solver)", p::data_column<Real>("U").round_trip(),
                         p::data_column<Real>("density").round_trip(), p::data_column<Real>("tolerance").round_trip(),
                         p::data_column<unsigned>("points"), p::data_column<uni20::half_int>("spin").fractional());
  table.append(interaction, density, tolerance, args.points, args.spin);
  if (args.format == "json")
    p::write_json(std::cout, table);
  else if (args.format == "csv")
    p::write_csv(std::cout, table);
  else if (args.format == "tsv")
    p::write_tsv(std::cout, table);
  else
  {
    p::report_builder report;
    report.table("") = p::to_report_table(table);
    uni20::display::emit(std::move(report), uni20::display::stream::out);
  }
  std::cout.flush();
  if (!std::cout) throw std::ios_base::failure("output flush failed");
}
} // namespace

int main(int argc, char** argv)
{
  p::program_info const program{
      .name = "presentation_cli_example",
      .description = "Inspect Hubbard-style parameters using native precision; this example does not run a solver.",
      .project_url = "https://github.com/Uni20-dev/uni20",
      .examples = {{"presentation_cli_example --U=4 --spin=-3/2 --format=tsv",
                    "Exact spin and machine-readable output"},
                   {"presentation_cli_example --U=4.000000000000000001 --precision=long-double",
                    "Deferred real conversion"}},
      .notes = {"U is in hopping units. Density is particles per site, restricted here to (0, 1].",
                "Help and version requests never initialize a calculation. Machine output contains only the table."}};
  arguments args;
  CLI::App app;
  c::configure(app, program);
  app.add_option("--U", args.interaction, "Interaction in hopping units")
      ->required()
      ->type_name("REAL")
      ->group("Model");
  app.add_option("--density", args.density, "Particles per site")
      ->type_name("REAL")
      ->capture_default_str()
      ->group("Model");
  c::add_half_int_option(app, "--spin", args.spin, "Exact half-integer quantum number")
      ->capture_default_str()
      ->group("Model");
  c::add_count_option(app, "--points", args.points, "Number of requested points")
      ->check(CLI::PositiveNumber)
      ->capture_default_str()
      ->group("Numerics");
  std::vector<std::string> precisions{"fp64", "long-double"};
#if UNI20_HAS_FLOAT128
  precisions.push_back("fp128");
#endif
  app.add_option("--precision", args.precision, "Real scalar type")
      ->check(CLI::IsMember(precisions))
      ->capture_default_str()
      ->group("Numerics");
  app.add_option("--tolerance", args.tolerance, "Positive tolerance; default: 100 * epsilon of the selected precision")
      ->type_name("REAL")
      ->group("Numerics");
  app.add_option("--format", args.format, "Result representation")
      ->check(CLI::IsMember({"pretty", "csv", "tsv", "json"}))
      ->capture_default_str()
      ->group("Output");
  auto const result = c::parse(app, argc, argv);
  if (result.requested != c::action::run)
  {
    uni20::display::emit(c::result_report(app, program, result), result.destination);
    return result.exit_code;
  }
  try
  {
    if (args.precision == "fp64")
      run<double>(args);
    else if (args.precision == "long-double")
      run<long double>(args);
#if UNI20_HAS_FLOAT128
    else if (args.precision == "fp128")
      run<uni20::float128>(args);
#endif
    return 0;
  }
  catch (std::exception const& error)
  {
    uni20::display::emit(c::result_report(app, program, {.requested = c::action::error, .message = error.what()}));
    return 1;
  }
}
