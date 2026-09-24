#include <iostream>
#include <uni20/common/output_session.hpp>

int main(int argc, char** argv)
{
  if (argc != 2)
  {
    std::cerr << "Usage: output_live_example OUTPUT_DIRECTORY\n"
                 "Use an existing directory; results.tsv and results.json must not already exist.\n";
    return 1;
  }
  namespace p = uni20::presentation;
  try
  {
    std::filesystem::path directory(argv[1]);
    std::cout << "Example: one result table, different output columns and formatting.\n"
                 "Four illustrative rows use energy = -1/(step+1); no solver is run.\n"
                 "The live table selects only step and energy, with energy shown to four decimal places.\n"
                 "The TSV and JSON files retain all three columns: step, energy and residual,\n"
                 "using full numerical precision. Residual is missing at step 0 (empty in TSV,\n"
                 "null in JSON), then equals 1/(10*step).\n"
                 "The final 'partial' scientific outcome illustrates results without convergence;\n"
                 "the JSON document status is still 'complete' because all output was written.\n"
              << "TSV file: " << directory / "results.tsv" << '\n'
              << "JSON file: " << directory / "results.json" << "\n\n"
              << std::flush;
    uni20::run_context run({.name = "live-sweep"});
    run.metadata().group("model");
    run.metadata().add("model", "interaction", 4.L);
    uni20::output_session output(run.snapshot());
    output.standard_output(uni20::output_format::terminal, {.projection = {.columns = {"step", "energy"}}});
    output.file(directory / "results.tsv", uni20::output_format::tsv);
    output.file(directory / "results.json", uni20::output_format::named_json);
    output.open(); // Preamble and file failures precede the expensive calculation.
    auto table =
        p::make_data_table("Sweep", p::data_column<int>("step"), p::data_column<long double>("energy").fixed(4),
                           p::data_column<std::optional<long double>>("residual"));
    output.attach(table, "sweep");
    for (int step = 0; step < 4; ++step)
    {
      auto energy = run.measure([&] { return -1.L / (step + 1); });
      table.append(step, energy, step == 0 ? std::nullopt : std::optional<long double>(1.L / (step * 10)));
    }
    table.finish({{"converged", "false"}});
    output.finish(run.finish(uni20::run_outcome::partial));
  }
  catch (std::exception const& e)
  {
    std::cerr << e.what() << '\n';
    return 1;
  }
}
