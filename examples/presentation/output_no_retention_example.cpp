#include <iostream>
#include <uni20/common/output_session.hpp>

int main()
{
  namespace p = uni20::presentation;
  std::cerr << "Example: streaming without retaining earlier rows.\n"
               "Row 0 is produced and discarded before an output is attached. The late JSON\n"
               "sink receives only rows 1 and 2; first_row = 1 records the missing prefix.\n"
               "Compare with output_replay_example, which retains and replays earlier rows.\n"
               "Stdout contains only the JSON document.\n\n";
  auto table = p::make_data_table("Future rows", {.retain = p::retention::none}, p::data_column<int>("step"));
  table.append(0); // Intentionally discarded: a late sink cannot reconstruct this row.
  uni20::output_session output;
  output.standard_output(uni20::output_format::json);
  output.attach(table, "steps", p::sink_replay::future_only); // JSON records first_row=1.
  table.append(1);
  table.append(2);
  table.finish();
  output.finish();
}
