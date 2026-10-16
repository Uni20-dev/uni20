#include <iostream>
#include <uni20/common/data_table_json.hpp>
#include <uni20/common/metadata.hpp>

int main()
{
  using namespace uni20;
  namespace p = uni20::presentation;
  auto table =
      p::make_data_table("Exact fractions and runtime precision", p::data_column<mpreal>("fraction").fractional(),
                         p::data_column<mpreal>("decimal").fixed(6), p::data_column<mpreal>("approximate"));
  metadata_value input("1/3");
  auto exact = input.as<mpreal>(Precision::exact());
  for (auto bits : {128, 256})
    table.append(exact, exact, input.as<mpreal>(Precision::bits(bits)));

  // CSV chooses fraction or decimal formatting for the exact values; approximate
  // values retain enough digits to round-trip at their own precision.
  p::write_csv(std::cout, table);
  // JSON is a separate document; it retains exactness and per-value precision,
  // irrespective of display settings. Real applications use separate files.
  p::write_json(std::cout, table);
  std::cout.flush();
  return std::cout ? 0 : 1;
}
