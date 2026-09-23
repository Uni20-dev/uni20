#include "cli.hpp"
#include <algorithm>
#include <uni20/buildinfo.hpp>

namespace uni20::cli
{
namespace
{
struct build_info_requested : CLI::Success
{};

bool visible_option_group(CLI::App const* app)
{
  return app->get_name().empty() && !app->get_group().empty() && app->get_group().front() != '+';
}

void collect_groups(CLI::App const& app, std::vector<presentation::help_group>& groups)
{
  for (auto const& name : app.get_groups())
  {
    if (name.empty()) continue; // CLI11 uses an empty group to hide options.
    auto heading = name;
    if (app.get_parent() && app.get_name().empty())
      heading = app.get_group() + (name == CLI::OptionDefaults{}.get_group() ? "" : " / " + name);
    presentation::help_group group{.heading = std::move(heading)};
    if (app.get_parent() && app.get_name().empty()) group.description = app.get_description();
    auto const minimum = app.get_require_option_min();
    auto const maximum = app.get_require_option_max();
    auto note = [&](std::string text) {
      if (!group.description.empty()) group.description += "\n";
      group.description += text;
    };
    if (app.get_required()) note("This group is required.");
    if (minimum > 0) note("Minimum selected options: " + std::to_string(minimum));
    if (maximum > 0) note("Maximum selected options: " + std::to_string(maximum));
    for (auto const* option : app.get_options())
    {
      if (option->get_group() != name) continue;
      presentation::help_option entry{.names = option->get_name(false, true), .description = option->get_description()};
      if (option->get_type_size_max() != 0)
      {
        auto const label = option->get_option_text().empty() ? option->get_type_name() : option->get_option_text();
        if (!label.empty()) entry.names += " " + label;
        if (option->get_expected_max() > 1) entry.names += " ...";
      }
      if (option->get_required()) entry.attributes.push_back("required");
      if (!option->get_default_str().empty()) entry.attributes.push_back("default: " + option->get_default_str());
      if (!option->get_envname().empty()) entry.attributes.push_back("environment: " + option->get_envname());
      if (option->get_expected_min() > 1)
        entry.attributes.push_back("minimum values: " + std::to_string(option->get_expected_min()));
      auto append_related = [&](std::string const& prefix, auto const& related) {
        std::vector<std::string> names;
        for (auto const* other : related)
          names.push_back(other->get_name());
        std::ranges::sort(names);
        for (auto const& other : names)
          entry.attributes.push_back(prefix + other);
      };
      append_related("requires: ", option->get_needs());
      append_related("excludes: ", option->get_excludes());
      group.options.push_back(std::move(entry));
    }
    if (!group.options.empty()) groups.push_back(std::move(group));
  }
  // CLI11 implements composable option groups as unnamed subcommands.
  for (auto const* child : app.get_subcommands(visible_option_group))
    collect_groups(*child, groups);
}

void append_positionals(CLI::App const& app, std::string& usage)
{
  for (auto const* option : app.get_options())
  {
    if (!option->get_positional() || option->get_group().empty()) continue;
    auto name = option->get_name(true);
    if (option->get_expected_max() > 1) name += " ...";
    usage += option->get_required() ? " " + name : " [" + name + "]";
  }
  for (auto const* child : app.get_subcommands(visible_option_group))
    append_positionals(*child, usage);
}
} // namespace

void configure(CLI::App& app, presentation::program_info const& program)
{
  app.name(program.name);
  app.description(program.description);
  app.set_help_flag("-h,--help", "Show full help and references")->group("Program");
  app.set_version_flag("--version", presentation::program_identity(program), "Show application version")
      ->group("Program");
  app.add_flag_callback(
         "--build-info", [] { throw build_info_requested{}; }, "Show application and Uni20 build information")
      ->group("Program")
      ->configurable(false)
      ->callback_priority(CLI::CallbackPriority::First);
}

parse_result parse(CLI::App& app, int argc, char const* const* argv, parse_policy policy)
{
  if (argc <= 1 && policy.empty == no_arguments::help)
    return {.requested = action::help, .destination = policy.empty_destination, .exit_code = policy.empty_exit_code};
  try
  {
    app.parse(argc, argv);
    return {};
  }
  catch (CLI::CallForHelp const&)
  {
    return {.requested = action::help};
  }
  catch (CLI::CallForVersion const&)
  {
    return {.requested = action::version};
  }
  catch (CLI::CallForAllHelp const&)
  {
    return {.requested = action::help};
  }
  catch (build_info_requested const&)
  {
    return {.requested = action::build_info};
  }
  catch (CLI::ParseError const& error)
  {
    return {.requested = action::error,
            .destination = display::stream::err,
            .exit_code = policy.error_exit_code,
            .message = error.what()};
  }
}

std::vector<presentation::help_group> help_groups(CLI::App const& app)
{
  std::vector<presentation::help_group> groups;
  collect_groups(app, groups);
  return groups;
}

presentation::report_builder help_report(CLI::App const& app, presentation::program_info const& program)
{
  auto usage = program.name + " [options]";
  append_positionals(app, usage);
  return presentation::help_report(program, usage, help_groups(app));
}

presentation::report_builder build_info_report(presentation::program_info const& program)
{
  auto report = presentation::program_report(program);
  auto const info = build_info::current();
  auto& build = report.table("Uni20 build");
  build.column("Setting", presentation::table_alignment::left).column("Value", presentation::table_alignment::left);
  build.row("Compiler", std::string(info.cxx_compiler_id) + " " + std::string(info.cxx_compiler_version));
  build.row("Compiler path", info.cxx_compiler_path);
  build.row("Build type", info.build_type);
  build.row("Generator", info.generator);
  build.row("System", info.system_name);
  build.row("Processor", info.system_processor);
  for (auto const& [heading, entries] :
       {std::pair{"Build options", info.build_options}, std::pair{"Detected dependencies", info.detected_environment}})
  {
    auto& table = report.table(heading);
    table.column("Setting", presentation::table_alignment::left).column("Value", presentation::table_alignment::left);
    for (auto const& entry : entries)
      table.row(entry.key, entry.value);
  }
  return report;
}

presentation::report_builder result_report(CLI::App const& app, presentation::program_info const& program,
                                           parse_result const& result)
{
  switch (result.requested)
  {
    case action::help:
      return help_report(app, program);
    case action::version:
      return presentation::report_builder(presentation::program_identity(program));
    case action::build_info:
      return build_info_report(program);
    case action::error:
    {
      presentation::report_builder report(program.name);
      report.status(presentation::semantic_glyph::failure, result.message);
      report.field("Usage", "Run " + program.name + " --help for options and references.");
      return report;
    }
    case action::run:
      throw std::logic_error("a run outcome has no informational document");
  }
  throw std::logic_error("invalid CLI parse outcome");
}
} // namespace uni20::cli
