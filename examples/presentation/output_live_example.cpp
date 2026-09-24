#include <iostream>
#include <uni20/common/output_session.hpp>

int main(int argc, char** argv)
{
  if (argc != 2)
  {
    std::cerr << "Usage: output_live_example NEW_OUTPUT_DIRECTORY\n";
    return 1;
  }
  namespace p = uni20::presentation;
  try
  {
    std::filesystem::path directory(argv[1]);
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
