/**
 * \file output.hpp
 * \brief Optional common CLI output controls; scientific applications choose whether to expose them.
 */
#pragma once
#include "cli.hpp"
#include <uni20/common/output_session.hpp>

namespace uni20::cli
{
struct output_options
{
    std::string format = "terminal";
    std::string path = {};
    bool quiet = false;
    bool plain = false;
    bool no_preamble = false;
    bool overwrite = false;
    bool flush_each_row = false;

    /// \brief Construct an unopened session. Quiet suppresses stdout only; files remain required.
    [[nodiscard]] output_session session(metadata_document initial = {}) const
    {
      static std::map<std::string, output_format> const formats{
          {"terminal", output_format::terminal}, {"csv", output_format::csv},
          {"tsv", output_format::tsv},           {"commented-tsv", output_format::commented_tsv},
          {"json", output_format::json},         {"named-json", output_format::named_json}};
      auto found = formats.find(format);
      if (found == formats.end()) throw std::invalid_argument("unknown output format: " + format);
      output_session result(std::move(initial), quiet);
      output_destination_options options{.overwrite = overwrite,
                                         .flush_each_row = flush_each_row,
                                         .preamble = !no_preamble,
                                         .human_policy = plain || !path.empty()
                                                             ? presentation::plain_policy()
                                                             : presentation::terminal_policy(stdout)};
      if (path.empty())
        result.standard_output(found->second, std::move(options));
      else
        result.file(path, found->second, std::move(options));
      return result;
    }
};

/// \brief Register output options on CLI11; call after initializing any application-specific defaults.
/// \details The options object must outlive parsing. This opens no files and changes no display router.
inline void add_output_options(CLI::App& app, output_options& options)
{
  auto* group = app.add_option_group("Output");
  group->add_option("--format", options.format, "Result representation")
      ->check(CLI::IsMember({"terminal", "csv", "tsv", "commented-tsv", "json", "named-json"}))
      ->capture_default_str();
  group->add_option("--output", options.path, "Output file; omitted means stdout");
  group->add_flag("-q,--quiet", options.quiet, "Suppress session stdout, including machine output");
  group->add_flag("--plain", options.plain, "Use plain human output");
  group->add_flag("--no-preamble", options.no_preamble, "Omit the human preamble and commented-file metadata");
  group->add_flag("--overwrite", options.overwrite, "Permit replacing an existing output file");
  group->add_flag("--flush-each-row", options.flush_each_row, "Flush result streams after every accepted row");
}
} // namespace uni20::cli
