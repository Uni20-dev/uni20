#include <iostream>
#include <uni20/common/configuration.hpp>
#include <uni20/common/run_context.hpp>

int main()
{
  std::cout << "Example: configuration precedence and the origin of each selected value.\n"
               "All sources are constructed in memory; no file is read or environment changed.\n"
               "For steps, the candidates are: API 50 > file 40 > attribute 30 > environment 20 > default 10.\n"
               "Expect steps = 50. The Source rows identify the winners; Hamiltonian and spin\n"
               "come from the groundstate attributes. Only those two fields are copied to output attributes.\n\n"
            << std::flush;
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
  std::cout << "\nCopied output attributes:\n";
  for (auto const& [name, value] : output_attributes)
    std::cout << "  " << name << " = " << value.text({}, true) << '\n';
  return config.get<int>("steps") == 50 && output_attributes.size() == 2 ? 0 : 1;
}
