/**
 * \file configuration.hpp
 * \brief Staged configuration using existing CLI11 scalar option declarations.
 */
#pragma once
#include "cli.hpp"
#include <algorithm>
#include <concepts>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <uni20/common/configuration.hpp>
#include <utility>
#include <vector>

namespace uni20::cli
{
/// \brief Collect CLI input first, then resolve declared fallback sources after loading input attributes.
/// \details The app, options and bound variables must outlive this adapter. It is single-use and supports
///          scalar options/boolean flags, not subcommands or CLI11 needs/excludes constraints on bound options.
///          Declare cross-field checks after resolve(). Ordinary unbound CLI11 options keep their semantics.
class configuration {
  public:
    explicit configuration(
        CLI::App& app,
        uni20::configuration::environment_reader environment =
            [](std::string const& name) -> std::optional<std::string> {
          if (auto p = std::getenv(name.c_str())) return std::string(p);
          return std::nullopt;
        })
        : app_(app), values_(std::move(environment))
    {}
    configuration(configuration const&) = delete;
    configuration& operator=(configuration const&) = delete;

    /// \brief Bind a declared CLI11 option to a stable metadata id, capturing its initialized value as default.
    /// \details Conversion and validators remain on the original CLI11 option. Fallback values cross that
    ///          textual boundary using native round-trip formatting, including exact half-integers.
    /// \pre target is the variable assigned by this option's original CLI11 callback.
    template <presentation::DataTableValue T>
      requires(!presentation::data_table_detail::optional_traits<T>::optional)
    configuration_binding& bind(CLI::Option* option, T& target, std::string id)
    {
      if (parsed_) throw std::logic_error("configuration bindings must precede parsing");
      if (!option) throw std::invalid_argument("null CLI option");
      if constexpr (!std::same_as<T, bool>)
        if (option->get_type_size_max() == 0)
          throw std::invalid_argument("only boolean flags can be configuration bindings");
      if (std::ranges::any_of(options_, [&](auto const& b) { return b.option == option; }))
        throw std::invalid_argument("CLI option already bound");
      auto& binding = values_.add<T>(id, target);
      if (option->get_required()) binding.required().without_default();
      if (!option->get_envname().empty()) binding.environment(option->get_envname());
      option->envname("")->configurable(false);
      binding.convert_ = [option, &target](metadata_value const& value, value_origin const& origin) {
        // CLI11 owns its validators, transforms and callback. Do not reimplement them.
        // Explicit CLI input has already run that chain once; transforms need not be idempotent.
        if (origin.source == configuration_source::command_line)
        {
          target = value.get<T>();
          return value;
        }
        if (value.missing()) throw std::invalid_argument("missing value for CLI scalar option");
        auto text = [&] {
          if constexpr (std::same_as<T, std::string>)
            return value.text({}, true);
          else
            return value.type() == typeid(std::string) ? value.get<std::string>()
                                                       : metadata_value(value.as<T>()).text({}, true);
        }();
        option->clear();
        option->add_result(text);
        option->run_callback();
        return metadata_value(target);
      };
      options_.push_back({option, std::move(id), [&target] { return metadata_value(target); }});
      return binding;
    }

    /// \brief Register an explicit option-file path; help never opens the file.
    CLI::Option* option_file(std::string names = "--config")
    {
      if (parsed_ || file_option_) throw std::logic_error("option file must be declared once before parsing");
      file_option_ = app_.add_option(std::move(names), file_path_, "Read selected options from a TOML/INI file");
      file_option_->configurable(false);
      return file_option_;
    }

    [[nodiscard]] parse_result parse(int argc, char const* const* argv, parse_policy policy = {})
    {
      if (parsed_) throw std::logic_error("configuration parser is single-use");
      if (app_.get_config_ptr())
        throw std::invalid_argument("use configuration::option_file instead of CLI11 set_config");
      parsed_ = true;
      for (auto const& b : options_)
      {
        for (auto const* other : app_.get_options())
          if (other->get_needs().contains(b.option) || other->get_excludes().contains(b.option))
            throw std::invalid_argument("resolve cross-option constraints after configuration sources");
        if (!b.option->get_needs().empty() || !b.option->get_excludes().empty() || b.option->get_trigger_on_parse() ||
            b.option->get_force_callback() || b.option->get_expected_max() > 1 || b.option->get_type_size_max() > 1)
          throw std::invalid_argument("bound configuration options must be independent scalar options");
      }
      std::vector<bool> required;
      for (auto const& b : options_)
      {
        required.push_back(b.option->get_required());
        if (b.option->get_required()) values_.binding(b.id).required().without_default();
        if (!b.option->get_envname().empty())
        {
          values_.binding(b.id).environment(b.option->get_envname());
          b.option->envname("");
        }
        b.option->required(false);
      }
      auto restore = [&] {
        for (std::size_t i = 0; i < options_.size(); ++i)
          options_[i].option->required(required[i]);
      };
      parse_result result;
      try
      {
        result = cli::parse(app_, argc, argv, policy);
      }
      catch (...)
      {
        restore();
        throw;
      }
      restore();
      if (result.requested != action::run) return result;
      for (auto const& b : options_)
        if (b.option->count())
        {
          values_.set(b.id, b.read());
          values_.cli_.insert(b.id);
        }
      ready_ = true;
      return result;
    }

    /// \brief Read the explicitly requested file using CLI11's configured TOML/INI reader.
    /// \details Call after handling information actions and before bootstrap/input resolution.
    void load_option_file()
    {
      if (!ready_) throw std::logic_error("configuration is not ready for resolution");
      if (!file_option_ || !file_option_->count()) return;
      attribute_map fields;
      for (auto const& item : app_.get_config_formatter()->from_file(file_path_))
      {
        if (item.name == "++" || item.name == "--") continue;
        if (item.inputs.size() != 1) throw std::invalid_argument("configuration file requires scalar values");
        if (!fields.emplace(item.fullname(), item.inputs.front()).second)
          throw std::invalid_argument("duplicate option-file key: " + item.fullname());
      }
      values_.option_file(file_path_, std::move(fields));
    }

    [[nodiscard]] uni20::configuration& values() { return values_; }
    /// \brief Include declared fallback sources in help, without evaluating or opening them.
    [[nodiscard]] presentation::report_builder result_report(presentation::program_info const& program,
                                                             parse_result const& result) const
    {
      auto report = cli::result_report(app_, program, result);
      if (result.requested != action::help) return report;
      auto& table = report.table("Configuration sources (highest priority first)");
      table.column("Option").column("Sources after explicit CLI/API input").preserve_tokens();
      for (auto const& option : options_)
      {
        auto const& b = values_.binding(option.id);
        std::string sources;
        auto append = [&](std::string text) {
          if (!sources.empty()) sources += "; ";
          sources += text;
        };
        if (b.file_) append("file key " + *b.file_);
        if (b.attribute_) append("attribute " + b.attribute_->first + ":" + b.attribute_->second);
        if (b.environment_) append("environment " + *b.environment_);
        if (b.default_)
          append("default " + b.default_->text());
        else if (b.required_)
          append("required after loading sources");
        table.row(option.option->get_name(), sources);
      }
      return report;
    }
    /// \brief Resolve all bindings, assigning native CLI variables and returning owned metadata.
    [[nodiscard]] metadata_document resolve() const
    {
      if (!ready_) throw std::logic_error("configuration is not ready for resolution");
      return values_.snapshot();
    }

  private:
    struct bound_option
    {
        CLI::Option* option;
        std::string id;
        std::function<metadata_value()> read;
    };
    CLI::App& app_;
    uni20::configuration values_;
    std::vector<bound_option> options_;
    CLI::Option* file_option_ = nullptr;
    std::string file_path_;
    bool parsed_ = false;
    bool ready_ = false;
};
} // namespace uni20::cli
