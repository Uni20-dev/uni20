/**
 * \file output.hpp
 * \brief Optional common CLI output controls; scientific applications choose whether to expose them.
 */
#pragma once
#include "cli.hpp"
#include <cstdio>
#include <filesystem>
#include <map>
#include <stdexcept>
#include <string>
#include <uni20/common/output_session.hpp>
#include <utility>
#include <vector>

namespace uni20::cli
{
namespace detail
{
inline output_format parse_output_format(std::string const& name)
{
  static std::map<std::string, output_format> const formats{{"terminal", output_format::terminal},
                                                            {"csv", output_format::csv},
                                                            {"tsv", output_format::tsv},
                                                            {"commented-csv", output_format::commented_csv},
                                                            {"commented-tsv", output_format::commented_tsv},
                                                            {"json", output_format::json},
                                                            {"named-json", output_format::named_json}};
  auto found = formats.find(name);
  if (found == formats.end()) throw std::invalid_argument("unknown output format: " + name);
  return found->second;
}

inline std::pair<output_format, std::filesystem::path> parse_file_export(std::string const& specification)
{
  auto separator = specification.find(':');
  if (separator == std::string::npos || separator + 1 == specification.size())
    throw std::invalid_argument("export must be FORMAT:PATH with a nonempty path");
  return {parse_output_format(specification.substr(0, separator)), specification.substr(separator + 1)};
}
} // namespace detail

struct output_options
{
    std::string format = "terminal";
    std::string path = {};
    /// \brief Additional FORMAT:PATH destinations, independent of the primary stdout or file destination.
    std::vector<std::string> exports = {};
    bool quiet = false;
    bool plain = false;
    bool no_preamble = false;
    bool overwrite = false;
    bool flush_each_row = false;

    /// \brief Construct an unopened session. Quiet suppresses stdout only; files remain required.
    /// \param metadata_keys Export keys for initial metadata and final summaries on all destinations.
    [[nodiscard]] output_session session(metadata_document initial = {},
                                         std::map<std::string, std::string> metadata_keys = {}) const
    {
      auto primary_format = detail::parse_output_format(format);
      output_session result(std::move(initial), quiet);
      output_destination_options options{.overwrite = overwrite,
                                         .flush_each_row = flush_each_row,
                                         .preamble = !no_preamble,
                                         .metadata_keys = std::move(metadata_keys),
                                         .human_policy = plain || !path.empty()
                                                             ? presentation::plain_policy()
                                                             : presentation::terminal_policy(stdout)};
      if (path.empty())
        result.standard_output(primary_format, options);
      else
        result.file(path, primary_format, options);
      options.human_policy = presentation::plain_policy();
      for (auto const& specification : exports)
      {
        auto [export_format, export_path] = detail::parse_file_export(specification);
        result.file(std::move(export_path), export_format, options);
      }
      return result;
    }
};

/// \brief Register output options on CLI11; call after initializing any application-specific defaults.
/// \details The options object must outlive parsing. This opens no files and changes no display router.
inline void add_output_options(CLI::App& app, output_options& options)
{
  auto* group = app.add_option_group("Output");
  group->add_option("--format", options.format, "Result representation")
      ->check(CLI::IsMember({"terminal", "csv", "tsv", "commented-csv", "commented-tsv", "json", "named-json"}))
      ->capture_default_str();
  group->add_option("--output", options.path, "Output file; omitted means stdout");
  group->add_option("--export", options.exports, "Additional file export; repeat for multiple destinations")
      ->type_name("FORMAT:PATH")
      ->expected(1)
      ->multi_option_policy(CLI::MultiOptionPolicy::TakeAll)
      ->check([](std::string const& specification) {
        try
        {
          (void)detail::parse_file_export(specification);
          return std::string{};
        }
        catch (std::invalid_argument const& error)
        {
          return std::string(error.what());
        }
      });
  group->add_flag("-q,--quiet", options.quiet, "Suppress session stdout, including machine output");
  group->add_flag("--plain", options.plain, "Use plain human output");
  group->add_flag("--no-preamble", options.no_preamble, "Omit the human preamble and commented-file metadata");
  group->add_flag("--overwrite", options.overwrite, "Permit replacing an existing output file");
  group->add_flag("--flush-each-row", options.flush_each_row, "Flush result streams after every accepted row");
}
} // namespace uni20::cli
