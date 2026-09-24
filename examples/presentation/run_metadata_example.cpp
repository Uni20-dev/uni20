#include <iostream>
#include <uni20/common/run_context.hpp>

int main()
{
  namespace p = uni20::presentation;
  uni20::run_context run({.name = "native-driver", .version = "1"});
  run.metadata().group("model", "Model");
  run.metadata().add("model", "interaction", uni20::parse_real<long double>("4.000000000000000001"),
                     {.label = "U", .unit = "t"});
  run.metadata().add("model", "spin", uni20::half_int::parse("3/2"), {.label = "Spin", .display = {.fractions = true}});
  auto before = run.snapshot(); // Native values, not strings copied from a report.
  std::cout << p::render_plain(uni20::metadata_report(before, "Before calculation"));
  auto answer = run.measure([] { return 42.L; });
  uni20::metadata_document result;
  result.group("result", "Result");
  result.add("result", "answer", answer);
  auto const& summary = run.finish(uni20::run_outcome::success, std::move(result));
  std::cout << p::render_plain(uni20::metadata_report(summary));
  // Explicit projection for an existing string-metadata consumer. Original values remain typed.
  auto legacy = before.strings({{"program", "Program"}});
  return legacy.at("Program") == "native-driver" ? 0 : 1;
}
