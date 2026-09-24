#include <sstream>
#include <uni20/common/output_session.hpp>

int main()
{
  namespace p = uni20::presentation;
  auto table = p::make_data_table("Retained history", p::data_column<int>("step"));
  uni20::output_session output;
  auto archive = std::make_shared<std::ostringstream>();
  output.stream("archive", archive, uni20::output_format::tsv);
  table.append(0);
  output.attach(table, "steps"); // Replays row 0 to the archive, then follows.
  table.append(1);
  auto late = output.standard_output(uni20::output_format::json);
  output.attach(table, "steps", p::sink_replay::all, {late}); // Only the newly attached destination.
  table.append(2);
  table.finish({{"note", "late sink saw every row once"}});
  output.finish();
}
