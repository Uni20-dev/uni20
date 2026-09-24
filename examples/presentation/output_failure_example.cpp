#include <iostream>
#include <sstream>
#include <uni20/common/output_session.hpp>

int main()
{
  namespace p = uni20::presentation;
  std::cerr << "Example: continuing required output after an optional display fails.\n"
               "A write failure is deliberately injected into the optional display.\n"
               "The following 'Disabled: optional display' message is expected.\n"
               "The required TSV on stdout still receives the sample answer 42; exit status 0\n"
               "means this deliberate failure was handled as intended.\n\n";
  uni20::output_session output;
  auto broken_display = std::make_shared<std::ostringstream>();
  broken_display->setstate(std::ios::badbit); // Simulate an unavailable optional display.
  output.stream("optional display", broken_display, uni20::output_format::terminal, {.required = false});
  output.standard_output(uni20::output_format::tsv); // The required result still receives every row.
  auto table = p::make_data_table("Results", p::data_column<int>("answer"));
  auto begin = output.attach(table, "results");
  for (auto const& failure : begin.failures)
    std::cerr << "Disabled: " << failure.name << '\n';
  table.append(42);
  table.finish();
  auto final = output.finish();
  return final.failures.empty() ? 1 : 0; // The optional failure is visible without failing scientific output.
}
