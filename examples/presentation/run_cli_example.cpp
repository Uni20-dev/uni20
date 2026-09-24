#include <iostream>
#include <run_cli_revision.hpp>
#include <uni20/cli/configuration.hpp>
#include <uni20/cli/output.hpp>
#include <uni20/common/output_session.hpp>

namespace p = uni20::presentation;
namespace c = uni20::cli;

struct arguments
{
    std::string input;
    std::string u = "4";
    std::string precision = "long-double";
    uni20::half_int spin = uni20::half_int::parse("1/2");
    unsigned points = 4;
    c::output_options output;
};

template <uni20::Real Real>
void calculate(arguments const& args, p::program_info const& program, uni20::metadata_document const& resolved)
{
  auto interaction = uni20::parse_real<Real>(args.u); // Convert only after selecting Real.
  uni20::run_context run(program);
  run.metadata().group("parameters", "Parameters");
  for (auto const& group : resolved.groups())
    for (auto const& field : group.fields)
      run.metadata().add("parameters", field.id, field.id == "u" ? uni20::metadata_value(interaction) : field.value,
                         field.options);
  auto output = args.output.session(run.snapshot());
  if (!args.output.quiet && !args.output.no_preamble)
  {
    auto& explanation = args.output.format == "terminal" && args.output.path.empty() ? std::cout : std::cerr;
    explanation << "Example: staged configuration, selected numerical precision, and result export.\n"
                   "Values are resolved from CLI > option file > input attributes > environment > defaults.\n"
                   "After selecting precision, U is parsed and sample rows use energy = U/(point+1),\n"
                   "with the selected spin repeated in each row. These illustrate output, not a physical solver.\n"
                   "The terminal shows energy to six decimal places; exports preserve numerical precision.\n"
                << "Points: " << args.points << "; precision: " << args.precision
                << "; output format: " << args.output.format << ".\n";
    if (args.output.path.empty())
      explanation << "Primary results are written to stdout.\n";
    else
      explanation << "Primary results are written to " << std::filesystem::path(args.output.path) << ".\n";
    for (auto const& specification : args.output.exports)
      explanation << "Additional export (format:path): " << specification << '\n';
    explanation << '\n' << std::flush;
  }
  auto table =
      p::make_data_table("Illustrative dispersion", p::data_column<unsigned>("point"),
                         p::data_column<uni20::half_int>("spin").fractional(), p::data_column<Real>("energy").fixed(6));
  output.attach(table, "dispersion"); // Preamble precedes the calculation.
  for (unsigned i = 0; i < args.points; ++i)
  {
    Real energy = run.measure([&] { return interaction / Real(i + 1); });
    table.append(i, args.spin, energy);
  }
  table.finish({{"points", std::to_string(args.points)}});
  output.finish(run.finish(uni20::run_outcome::success));
}

int main(int argc, char** argv)
{
  p::program_info program{
      .name = "run_cli_example",
      .description = "Staged configuration and scientific output",
      .version = "1",
      .revision = std::string(run_cli_build::source_revision),
      .examples = {{"run_cli_example --U=4.000000000000000001 --format=named-json", "Native long-double output"},
                   {"run_cli_example --config=job.toml --input=state.ini", "File and saved-attribute fallbacks"},
                   {"run_cli_example --export=csv:results.csv --export=named-json:results.json",
                    "Live terminal output with two full-precision file exports"}}};
  arguments args;
  CLI::App app;
  c::configure(app, program);
  c::configuration config(app);
  config.bind(app.add_option("--input", args.input, "Optional illustrative attribute file"), args.input, "input")
      .file("input");
  config
      .bind(app.add_option("--U", args.u, "Interaction; parsed after precision selection")->capture_default_str(),
            args.u, "u")
      .file("model.U")
      .attribute("state", "U")
      .environment("UNI20_EXAMPLE_U")
      .metadata({.label = "U"});
  config.bind(c::add_half_int_option(app, "--spin", args.spin)->capture_default_str(), args.spin, "spin")
      .file("model.spin")
      .attribute("state", "Spin");
  config
      .bind(c::add_count_option(app, "--points", args.points)->check(CLI::Range(1, 1000))->capture_default_str(),
            args.points, "points")
      .file("points")
      .environment("UNI20_EXAMPLE_POINTS");
  std::vector<std::string> precisions{"fp64", "long-double"};
#if UNI20_HAS_FLOAT128
  precisions.push_back("fp128");
#endif
  config
      .bind(app.add_option("--precision", args.precision)->check(CLI::IsMember(precisions))->capture_default_str(),
            args.precision, "precision")
      .file("precision");
  config.option_file();
  c::add_output_options(app, args.output);
  try
  {
    auto result = config.parse(argc, argv, {.empty = c::no_arguments::run});
    if (result.requested != c::action::run)
    {
      uni20::display::emit(config.result_report(program, result), result.destination);
      return result.exit_code;
    }
    config.load_option_file();
    // Bootstrap the input path before loading the selected object's attributes.
    auto input = config.values().get<std::string>("input");
    if (!input.empty())
    {
      uni20::attribute_map attributes;
      for (auto const& field : app.get_config_formatter()->from_file(input))
      {
        if (field.name == "++" || field.name == "--") continue;
        if (field.inputs.size() != 1) throw std::invalid_argument("example attributes must be scalar");
        if (!attributes.emplace(field.fullname(), field.inputs.front()).second)
          throw std::invalid_argument("duplicate input attribute");
      }
      config.values().attributes("state", std::move(attributes));
    }
    auto resolved = config.resolve(); // All declared values and their source provenance.
    if (args.precision == "fp64")
      calculate<double>(args, program, resolved);
    else if (args.precision == "long-double")
      calculate<long double>(args, program, resolved);
#if UNI20_HAS_FLOAT128
    else
      calculate<uni20::float128>(args, program, resolved);
#endif
  }
  catch (std::exception const& e)
  {
    uni20::display::emit(c::result_report(app, program, {.requested = c::action::error, .message = e.what()}),
                         uni20::display::stream::err);
    return 1;
  }
}
