/**
 * \file configuration.hpp
 * \brief Explicit source bindings and checked configuration resolution.
 */
#pragma once
#include "metadata.hpp"
#include <algorithm>
#include <cstdlib>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace uni20
{
namespace cli
{
class configuration;
}
using attribute_map = std::map<std::string, metadata_value>;

struct resolved_value
{
    metadata_value value;
    value_origin origin;
};

/// \brief A single declared scalar, with application-selected fallback sources.
class configuration_binding {
  public:
    configuration_binding& attribute(std::string repository, std::string key)
    {
      attribute_ = std::pair{std::move(repository), std::move(key)};
      return *this;
    }
    configuration_binding& environment(std::string name)
    {
      environment_ = std::move(name);
      return *this;
    }
    configuration_binding& file(std::string key)
    {
      file_ = std::move(key);
      return *this;
    }
    configuration_binding& required(bool value = true)
    {
      required_ = value;
      return *this;
    }
    /// \brief Require an actual source instead of treating the initialized variable as a fallback.
    configuration_binding& without_default()
    {
      default_.reset();
      return *this;
    }
    configuration_binding& metadata(metadata_field_options options)
    {
      metadata_ = std::move(options);
      return *this;
    }
    /// \brief Validate the final native value, after source selection and conversion.
    template <presentation::DataTableValue T, typename F> configuration_binding& check(F&& check)
    {
      checks_.push_back([fn = std::forward<F>(check)](metadata_value const& v) { fn(v.get<T>()); });
      return *this;
    }
    [[nodiscard]] std::string const& id() const { return id_; }
    [[nodiscard]] std::optional<std::string> const& file_key() const { return file_; }

  private:
    friend class configuration;
    friend class cli::configuration;
    std::string id_;
    std::optional<metadata_value> default_;
    std::optional<std::pair<std::string, std::string>> attribute_;
    std::optional<std::string> environment_;
    std::optional<std::string> file_;
    bool required_ = false;
    metadata_field_options metadata_;
    std::function<metadata_value(metadata_value const&, value_origin const&)> convert_;
    std::vector<std::function<void(metadata_value const&)>> checks_;
};

/// \brief Parser-independent configuration with CLI/API > file > attribute > environment > default precedence.
/// \details A present empty/false/zero value wins. Invalid winners never fall back. Each resolve call
///          makes a new owned result, permitting bootstrap resolution before input attributes are loaded.
///          The environment reader is injectable and returns nullopt only for absence.
class configuration {
  public:
    using environment_reader = std::function<std::optional<std::string>(std::string const&)>;
    explicit configuration(environment_reader reader = [](std::string const& name) -> std::optional<std::string> {
      if (auto p = std::getenv(name.c_str())) return std::string(p);
      return std::nullopt;
    })
        : environment_(std::move(reader))
    {}

    template <presentation::DataTableValue T> configuration_binding& add(std::string id, T initial)
    {
      auto& binding = this->add<T>(std::move(id));
      binding.default_ = metadata_value(std::move(initial));
      return binding;
    }
    template <presentation::DataTableValue T> configuration_binding& add(std::string id)
    {
      metadata_detail::dt::validate_identifier(id);
      if (bindings_.contains(id)) throw std::invalid_argument("duplicate configuration field: " + id);
      auto binding = std::make_unique<configuration_binding>();
      binding->id_ = id;
      binding->convert_ = [](metadata_value const& value, value_origin const&) {
        return metadata_value(value.as<T>());
      };
      auto* result = binding.get();
      bindings_.emplace(id, std::move(binding));
      order_.push_back(std::move(id));
      return *result;
    }

    void set(std::string const& id, metadata_value value)
    {
      (void)this->binding(id);
      explicit_.insert_or_assign(id, std::move(value));
      cli_.erase(id);
    }
    void attributes(std::string repository, attribute_map values)
    {
      attributes_.insert_or_assign(std::move(repository), std::move(values));
    }

    /// \brief Supply one already parsed option file; reject undeclared or ambiguous keys before use.
    void option_file(std::string path, attribute_map values)
    {
      for (auto const& [key, value] : values)
      {
        auto count = std::ranges::count_if(bindings_, [&](auto const& item) { return item.second->file_ == key; });
        if (count != 1) throw std::invalid_argument("unknown or ambiguous option-file key: " + key);
      }
      file_path_ = std::move(path);
      file_ = std::move(values);
    }
    [[nodiscard]] configuration_binding const& binding(std::string const& id) const
    {
      auto found = bindings_.find(id);
      if (found == bindings_.end()) throw std::invalid_argument("unknown configuration field: " + id);
      return *found->second;
    }
    [[nodiscard]] configuration_binding& binding(std::string const& id)
    {
      return const_cast<configuration_binding&>(std::as_const(*this).binding(id));
    }
    [[nodiscard]] std::optional<resolved_value> resolve(std::string const& id) const
    {
      auto const& b = this->binding(id);
      std::optional<resolved_value> selected;
      auto take = [&](attribute_map const& map, std::string const& key, value_origin origin) {
        if (selected) return;
        if (auto it = map.find(key); it != map.end()) selected.emplace(it->second, std::move(origin));
      };
      take(explicit_, id,
           {cli_.contains(id) ? configuration_source::command_line : configuration_source::explicit_input, {}, id});
      if (b.file_) take(file_, *b.file_, {configuration_source::option_file, file_path_, *b.file_});
      if (b.attribute_)
      {
        auto const& [repository, key] = *b.attribute_;
        if (auto it = attributes_.find(repository); it != attributes_.end())
          take(it->second, key, {configuration_source::attribute, repository, key});
      }
      if (!selected && b.environment_)
        if (auto value = environment_(*b.environment_))
          selected.emplace(metadata_value(std::move(*value)),
                           value_origin{configuration_source::environment, {}, *b.environment_});
      if (!selected && b.default_)
        selected.emplace(*b.default_, value_origin{configuration_source::application_default, {}, id});
      if (!selected)
      {
        if (b.required_) throw std::invalid_argument("missing required configuration field: " + id);
        return std::nullopt;
      }
      try
      {
        selected->value = b.convert_(selected->value, selected->origin);
        for (auto const& check : b.checks_)
          check(selected->value);
      }
      catch (std::exception const& e)
      {
        throw std::invalid_argument("configuration field '" + id + "' from " +
                                    std::string(to_string(selected->origin.source)) + " '" + selected->origin.location +
                                    "' key '" + selected->origin.key + "': " + e.what());
      }
      return selected;
    }
    template <presentation::DataTableValue T> [[nodiscard]] T get(std::string const& id) const
    {
      auto result = this->resolve(id);
      if (!result) throw std::invalid_argument("configuration field has no value: " + id);
      return result->value.as<T>();
    }
    [[nodiscard]] metadata_document snapshot(std::string group = "configuration") const
    {
      metadata_document document;
      document.group(group, "Configuration");
      for (auto const& id : order_)
        if (auto value = this->resolve(id))
        {
          auto options = this->binding(id).metadata_;
          options.origin = value->origin;
          document.add(group, id, std::move(value->value), std::move(options));
        }
      return document;
    }
    /// \brief Copy only explicitly selected resolved fields to output-object attributes.
    void copy_attributes(attribute_map& output, std::map<std::string, std::string> const& keys) const
    {
      attribute_map additions;
      for (auto const& [id, key] : keys)
      {
        auto value = this->resolve(id);
        if (!value) throw std::invalid_argument("no value for output attribute: " + id);
        if (!additions.emplace(key, std::move(value->value)).second)
          throw std::invalid_argument("duplicate output attribute: " + key);
      }
      for (auto& [key, value] : additions)
        output.insert_or_assign(key, std::move(value));
    }

  private:
    friend class cli::configuration;
    environment_reader environment_;
    std::map<std::string, std::unique_ptr<configuration_binding>> bindings_;
    std::vector<std::string> order_;
    attribute_map explicit_;
    std::set<std::string> cli_;
    attribute_map file_;
    std::string file_path_;
    std::map<std::string, attribute_map> attributes_;
};
} // namespace uni20
