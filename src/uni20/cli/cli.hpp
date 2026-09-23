/**
 * \file cli.hpp
 * \brief CLI11 option adapters, explicit parse outcomes and semantic application help.
 */
#pragma once

#include <CLI/CLI.hpp>
#include <charconv>
#include <concepts>
#include <uni20/common/display.hpp>
#include <uni20/common/half_int.hpp>
#include <uni20/common/program.hpp>

#if CLI11_VERSION_MAJOR < 2 || (CLI11_VERSION_MAJOR == 2 && CLI11_VERSION_MINOR < 7) ||                                \
    (CLI11_VERSION_MAJOR == 2 && CLI11_VERSION_MINOR == 7 && CLI11_VERSION_PATCH < 2)
#error "Uni20 requires CLI11 2.7.2 or newer"
#endif

namespace uni20::cli
{
/// \brief The executable decides whether to calculate or render an informational/error result.
enum class action
{
  run,
  help,
  version,
  build_info,
  error
};
/// \brief Whether an empty invocation displays help or parses application defaults.
enum class no_arguments
{
  help,
  run
};

/// \brief Executable policy for empty invocations and usage failures.
struct parse_policy
{
    no_arguments empty = no_arguments::help;
    display::stream empty_destination = display::stream::err;
    int empty_exit_code = 1;
    int error_exit_code = 1;
};

/// \brief Owned parse outcome; parsing itself does not print, open files or terminate the process.
struct parse_result
{
    action requested = action::run;
    display::stream destination = display::stream::out;
    int exit_code = 0;
    std::string message = {};
};

/// \brief Register common help/version/build-info actions on a single-command application.
/// \details Call once before parsing. Ordinary options retain normal CLI11 registration. Application callbacks
///          must not initialize solvers or open outputs during parsing; do that after a run outcome.
void configure(CLI::App& app, presentation::program_info const& program);

/// \brief Parse argc/argv, preserving CLI11's argument boundaries and help precedence.
/// \details Catches CLI11 parse outcomes only; construction and unrelated application exceptions propagate.
[[nodiscard]] parse_result parse(CLI::App& app, int argc, char const* const* argv, parse_policy policy = {});

/// \brief Project CLI11 option metadata into parser-independent semantic help groups.
[[nodiscard]] std::vector<presentation::help_group> help_groups(CLI::App const& app);

/// \brief Build full help from the same options used for parsing, plus application-supplied content.
[[nodiscard]] presentation::report_builder help_report(CLI::App const& app, presentation::program_info const& program);

/// \brief Build application and existing Uni20 build information without running numerical initialization.
[[nodiscard]] presentation::report_builder build_info_report(presentation::program_info const& program);

/// \brief Build the selected informational/error document; a run outcome has no document and is rejected.
[[nodiscard]] presentation::report_builder result_report(CLI::App const& app, presentation::program_info const& program,
                                                         parse_result const& result);

namespace detail
{
template <typename T, typename Converter>
CLI::Option* converted_option(CLI::App& app, std::string names, T& target, std::string description, Converter converter)
{
  return app.add_option_function<std::string>(
      names,
      [&target, converter, names](std::string const& token) {
        try
        {
          target = converter(token);
        }
        catch (std::invalid_argument const& error)
        {
          throw CLI::ValidationError(names, error.what());
        }
        catch (std::out_of_range const& error)
        {
          throw CLI::ValidationError(names, error.what());
        }
        catch (std::runtime_error const& error)
        {
          throw CLI::ValidationError(names, error.what());
        }
      },
      std::move(description));
}
} // namespace detail

/// \brief Bind an unsigned decimal count with complete-token, negative and overflow checks.
/// \details Bound storage must outlive parsing. The returned option supports ordinary CLI11 modifiers.
template <std::unsigned_integral T>
  requires(!std::same_as<T, bool>)
CLI::Option* add_count_option(CLI::App& app, std::string names, T& value, std::string description = {})
{
  auto* option =
      detail::converted_option(app, std::move(names), value, std::move(description), [](std::string const& text) {
        T parsed{};
        auto const result = std::from_chars(text.data(), text.data() + text.size(), parsed);
        if (result.ec != std::errc{} || result.ptr != text.data() + text.size())
          throw std::invalid_argument("expected a representable nonnegative decimal integer: " + text);
        return parsed;
      });
  return option->type_name("COUNT")->default_function([&value] { return fmt::format("{}", value); });
}

/// \brief Bind an exact half-integer, accepting integer, .0/.5 decimal and denominator-two fractional forms.
/// \details Bound storage must outlive parsing. Non-half-integral values are rejected, never rounded.
template <std::signed_integral T>
CLI::Option* add_half_int_option(CLI::App& app, std::string names, basic_half_int<T>& value,
                                 std::string description = {})
{
  auto* option = detail::converted_option(app, std::move(names), value, std::move(description),
                                          [](std::string const& text) { return basic_half_int<T>::parse(text); });
  return option->type_name("HALF_INT")->default_function([&value] { return uni20::to_string(value); });
}
} // namespace uni20::cli
