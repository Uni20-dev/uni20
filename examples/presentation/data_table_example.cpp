#include <uni20/common/data_table_sinks.hpp>

#include <iostream>
#include <optional>
#include <string_view>

int main(int argc, char** argv)
{
  namespace p = uni20::presentation;
  std::string_view const format = argc > 1 ? argv[1] : "pretty";
  try
  {
    bool const no_history = format == "stream-no-retention";
    bool const streaming = format == "stream" || no_history;
    auto table = p::make_data_table("Excitation levels (illustrative data)",
                                    {.retain = no_history ? p::retention::none : p::retention::all,
                                     .metadata = {{"source", "invented test fixtures"}}},
                                    p::data_column<unsigned>("level").label("Level"),
                                    p::data_column<uni20::half_int>("spin").label("Spin").fractional(),
                                    p::data_column<long double>("momentum").label("P").fixed(3),
                                    p::data_column<long double>("energy").label("Energy").fixed(4),
                                    p::data_column<std::optional<long double>>("gap").label("E-E0").fixed(4),
                                    p::data_column<bool>("converged").label("Converged"));
    auto attach = [&] {
      // Keep stdout rectangular; independently format selected columns on stderr.
      table.attach(p::terminal_sink(
          {.projection = {.columns = {"level", "spin", "energy"}}, .destination = uni20::display::stream::err}));
      table.attach(p::tsv_sink(std::cout));
    };
    if (no_history) attach();
    table.append(1, uni20::from_twice(1), 0, -6.25, 0, true);
    if (streaming && !no_history) attach(); // Replays the first retained row.
    table.append(2, uni20::from_twice(-3), 0.5, -6.125, std::nullopt, false);
    table.finish({{"rows", "2"}}); // Explicitly flushes active sinks and reports failures.

    if (format == "csv")
      p::write_csv(std::cout, table);
    else if (format == "tsv")
      p::write_tsv(std::cout, table);
    else if (format == "json")
      p::write_json(std::cout, table);
    else if (format == "pretty")
    {
      p::report_builder report;
      report.table("") = p::to_report_table(table);
      std::cout << p::render_terminal(report, p::terminal_policy(stdout));
    }
    else if (!streaming)
    {
      std::cerr << "Usage: presentation_data_table_example [pretty|csv|tsv|json|stream|stream-no-retention]\n";
      return 1;
    }
    std::cout.flush();
    if (!std::cout) throw std::ios_base::failure("output flush failed");
  }
  catch (std::exception const& error)
  {
    std::cerr << error.what() << '\n';
    return 1;
  }
}
