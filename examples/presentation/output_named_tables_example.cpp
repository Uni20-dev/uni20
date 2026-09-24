#include <iostream>
#include <uni20/common/output_session.hpp>

int main()
{
  namespace p = uni20::presentation;
  uni20::run_context run({.name = "named-tables"});
  uni20::output_session output(run.snapshot());
  output.standard_output(uni20::output_format::named_json);
  auto spectrum =
      p::make_data_table("Spectrum", p::data_column<uni20::half_int>("spin"), p::data_column<long double>("energy"));
  spectrum.append(uni20::half_int::parse("1/2"), -0.75L);
  spectrum.append(uni20::half_int::parse("3/2"), 0.25L);
  output.write_table(spectrum, "spectrum");
  auto diagnostics =
      p::make_data_table("Diagnostics", p::data_column<std::string>("method"), p::data_column<bool>("converged"));
  diagnostics.append("illustrative", true);
  output.write_table(diagnostics, "diagnostics");
  output.finish(run.finish(uni20::run_outcome::success));
}
