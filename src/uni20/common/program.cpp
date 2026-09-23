#include "program.hpp"

namespace uni20::presentation
{
std::string program_identity(program_info const& program)
{
  auto text = program.name;
  if (!program.version.empty()) text += " " + program.version;
  if (!program.revision.empty()) text += " (" + program.revision + ")";
  return text;
}

report_builder program_report(program_info const& program)
{
  report_builder report(program_identity(program));
  if (!program.description.empty()) report.field("Description", program.description);
  if (!program.copyright.empty()) report.field("Copyright", program.copyright);
  if (!program.authors.empty()) report.field("Authors", program.authors);
  if (!program.license.empty()) report.field("License", program.license);
  if (!program.project_url.empty()) report.field("Project", program.project_url);
  return report;
}

report_builder help_report(program_info const& program, std::string_view usage, std::span<help_group const> groups)
{
  auto report = program_report(program);
  report.field("Usage", usage);
  for (auto const& group : groups)
  {
    if (group.options.empty()) continue;
    auto& table = report.table(group.heading);
    table.column("Option", table_alignment::left).column("Description", table_alignment::left);
    if (!group.description.empty()) table.row({table_cell(group.description, 2)});
    for (auto const& option : group.options)
    {
      styled_text description;
      description.append(option.description);
      for (auto const& attribute : option.attributes)
      {
        if (!description.empty()) description.append("\n");
        description.append(attribute, terminal::TerminalStyle(std::string_view("LightGray")));
      }
      table.row(style("Cyan;Bold")(option.names), std::move(description));
    }
  }
  if (!program.examples.empty())
  {
    auto& table = report.table("Examples");
    table.column("Invocation", table_alignment::left).column("Purpose", table_alignment::left);
    for (auto const& example : program.examples)
      table.row(example.command, example.description);
  }
  if (!program.notes.empty())
  {
    auto& table = report.table("Conventions and limitations");
    table.column("Notes", table_alignment::left);
    for (auto const& note : program.notes)
      table.row(note);
  }
  if (program.references)
  {
    auto references = program.references();
    if (!references.empty())
    {
      auto& table = report.table("References");
      table.column("Key", table_alignment::left).column("Reference", table_alignment::left);
      for (auto const& reference : references)
      {
        auto text = reference.citation;
        if (!reference.link.empty()) text += "\n" + reference.link;
        if (!reference.applicability.empty()) text += "\n" + reference.applicability;
        table.row(reference.key, text);
      }
    }
  }
  return report;
}
} // namespace uni20::presentation
