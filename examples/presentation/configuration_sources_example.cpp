#include <iostream>
#include <uni20/common/configuration.hpp>
#include <uni20/common/run_context.hpp>

int main()
{
  // A C++ or Python driver need not construct argv or link CLI11.
  uni20::configuration config([](std::string const& name) -> std::optional<std::string> {
    if (name == "SOLVER_STEPS") return "20";
    return std::nullopt;
  });
  config.add("steps", 10).file("steps").attribute("groundstate", "Steps").environment("SOLVER_STEPS");
  config.add<std::string>("hamiltonian").required().attribute("groundstate", "Hamiltonian");
  config.add("spin", uni20::half_int::parse("1/2")).attribute("groundstate", "Spin");
  config.attributes("groundstate", {{"Steps", 30}, {"Hamiltonian", "Hubbard"}, {"Spin", "3/2"}});
  config.option_file("job.toml", {{"steps", "40"}}); // Parsed by the caller's chosen file adapter.
  config.set("steps", 50); // Explicit API input wins over file, attributes, environment and default.
  auto snapshot = config.snapshot();
  std::cout << uni20::presentation::render_plain(uni20::metadata_report(snapshot, "Resolved configuration", true));
  uni20::attribute_map output_attributes;
  config.copy_attributes(output_attributes, {{"hamiltonian", "Hamiltonian"}, {"spin", "Spin"}});
  return config.get<int>("steps") == 50 && output_attributes.size() == 2 ? 0 : 1;
}
