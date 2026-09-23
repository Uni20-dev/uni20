/**
 * \file program.hpp
 * \brief Parser-independent application identity and help documents.
 */
#pragma once

#include "presentation.hpp"
#include <functional>
#include <span>

namespace uni20::presentation
{
/// \brief One invocation and its purpose, supplied by the application.
struct program_example
{
    std::string command;
    std::string description = {};
};

/// \brief A literature entry supplied by the application's authoritative citation registry.
struct program_reference
{
    std::string key;
    std::string citation;
    std::string link = {};
    std::string applicability = {};
};

/// \brief Application identity and explanatory content, independent of any option parser.
/// \details References are requested only while constructing full help. Strings and callbacks are owned.
struct program_info
{
    std::string name;
    std::string description = {};
    std::string version = {};
    std::string revision = {};
    std::string copyright = {};
    std::string authors = {};
    std::string project_url = {};
    std::string license = {};
    std::vector<program_example> examples = {};
    std::vector<std::string> notes = {};
    std::function<std::vector<program_reference>()> references = {};
};

/// \brief One option's semantic help content, projected from the parser's declarations.
struct help_option
{
    std::string names;
    std::string description = {};
    std::vector<std::string> attributes = {};
};

/// \brief An ordered group of options in a help document.
struct help_group
{
    std::string heading;
    std::string description = {};
    std::vector<help_option> options = {};
};

/// \brief Concise application identity for version output and report headings.
[[nodiscard]] std::string program_identity(program_info const& program);

/// \brief Construct the application's banner without evaluating its reference provider.
[[nodiscard]] report_builder program_report(program_info const& program);

/// \brief Construct a full help document shared by plain and styled renderers.
[[nodiscard]] report_builder help_report(program_info const& program, std::string_view usage,
                                         std::span<help_group const> groups);
} // namespace uni20::presentation
