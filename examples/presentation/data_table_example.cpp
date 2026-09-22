#include <uni20/common/data_table.hpp>

#include <iostream>
#include <optional>
#include <string_view>

int main(int argc, char** argv)
{
  namespace p = uni20::presentation;
  auto table =
      p::make_data_table("Excitation levels (illustrative data)", p::data_column<unsigned>("level").label("Level"),
                         p::data_column<uni20::half_int>("spin").label("Spin").fractional(),
                         p::data_column<long double>("momentum").label("P").fixed(3),
                         p::data_column<long double>("energy").label("Energy").fixed(4),
                         p::data_column<std::optional<long double>>("gap").label("E-E0").fixed(4),
                         p::data_column<bool>("converged").label("Converged"));
  table.append(1, uni20::from_twice(1), 0, -6.25, 0, true);
  table.append(2, uni20::from_twice(-3), 0.5, -6.125, std::nullopt, false);

  std::string_view const format = argc > 1 ? argv[1] : "pretty";
  try
  {
    if (format == "csv")
      p::write_csv(std::cout, table);
    else if (format == "tsv")
      p::write_tsv(std::cout, table);
    else if (format == "pretty")
    {
      p::report_builder report;
      report.table("") = p::to_report_table(table);
      std::cout << p::render_terminal(report, p::terminal_policy(stdout));
    }
    else
    {
      std::cerr << "Usage: presentation_data_table_example [pretty|csv|tsv]\n";
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
