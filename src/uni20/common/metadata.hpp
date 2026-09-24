/**
 * \file metadata.hpp
 * \brief Owned native scalar metadata, independent of parsers and output streams.
 */
#pragma once
#include "data_table.hpp"
#include <any>
#include <charconv>
#include <typeindex>

namespace uni20
{
namespace metadata_detail
{
namespace dt = presentation::data_table_detail;

template <typename T> T parse(std::string_view text)
{
  if constexpr (std::same_as<T, std::string>)
    return std::string(text);
  else if constexpr (std::same_as<T, bool>)
  {
    if (text == "true" || text == "1") return true;
    if (text == "false" || text == "0") return false;
    throw std::invalid_argument("expected true, false, 1 or 0");
  }
  else if constexpr (dt::integer<T>)
  {
    T value{};
    auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), value);
    if (error == std::errc::result_out_of_range) throw std::out_of_range("integer out of range");
    if (text.empty() || error != std::errc{} || end != text.data() + text.size())
      throw std::invalid_argument("expected a complete integer literal");
    return value;
  }
  else if constexpr (dt::half_integer<T>)
    return T::parse(text);
  else
    return parse_real<T>(text);
}
} // namespace metadata_detail

/// \brief An immutable owned value that retains its native scalar and optional type.
/// \details Copies share immutable storage. Text views and C strings are copied on entry.
class metadata_value {
  public:
    template <presentation::DataTableValue T>
    metadata_value(T value) : storage_(std::make_shared<model<T>>(std::move(value)))
    {}
    metadata_value(std::string_view text) : metadata_value(std::string(text)) {}
    metadata_value(char const* text) : metadata_value(copy_text(text)) {}

    [[nodiscard]] std::type_index type() const { return storage_->type(); }
    [[nodiscard]] bool missing() const { return !storage_->scalar().has_value(); }
    [[nodiscard]] std::string text(presentation::data_column_display const& format = {}, bool machine = false) const
    {
      return storage_->text(format, machine);
    }

    /// \brief Access the exact declared type; a mismatch throws std::bad_any_cast.
    template <typename T> [[nodiscard]] T const& get() const { return std::any_cast<T const&>(storage_->value()); }

    /// \brief Convert without a double intermediate; textual numbers use the destination's parser.
    /// \details Integer and half-integer narrowing is checked. Real rounding follows the target type;
    ///          finite overflow is rejected. A real is never implicitly rounded to an integer.
    template <presentation::DataTableValue T> [[nodiscard]] T as() const
    {
      namespace dt = metadata_detail::dt;
      if (auto p = std::any_cast<T>(&storage_->value())) return *p;
      if constexpr (dt::optional_traits<T>::optional)
      {
        if (this->missing()) return std::nullopt;
        return this->as<typename dt::optional_traits<T>::value_type>();
      }
      else
      {
        if (this->missing()) throw std::invalid_argument("missing value for a non-optional field");
        auto const& scalar = storage_->scalar();
        if (auto p = std::any_cast<T>(&scalar)) return *p;
        if (auto p = std::any_cast<std::string>(&scalar)) return metadata_detail::parse<T>(*p);
        std::optional<T> result;
        auto try_type = [&]<typename U>() {
          if constexpr (dt::accepts<T, U>())
            if (auto p = std::any_cast<U>(&scalar))
            {
              if constexpr (Real<T> && Real<U>)
                if constexpr (numeric_limits<U>::max_exponent > numeric_limits<T>::max_exponent ||
                              (numeric_limits<U>::max_exponent == numeric_limits<T>::max_exponent &&
                               numeric_limits<U>::digits >= numeric_limits<T>::digits))
                  if (uni20::isfinite(*p) &&
                      (*p > static_cast<U>(numeric_limits<T>::max()) || *p < -static_cast<U>(numeric_limits<T>::max())))
                    throw std::out_of_range("real conversion overflow");
              auto converted = dt::convert<T>(*p);
              result = std::move(converted);
            }
        };
        [&]<typename... U>(std::tuple<U...>*) {
          (try_type.template operator()<U>(), ...);
        }(static_cast<
            std::tuple<signed char, unsigned char, short, unsigned short, int, unsigned int, long, unsigned long,
                       long long, unsigned long long, float, double, long double, basic_half_int<std::intmax_t>>*>(
            nullptr));
#if UNI20_HAS_FLOAT128
        try_type.template operator()<uni20::float128>();
#endif
        if (!result) throw std::invalid_argument("incompatible metadata scalar types");
        return std::move(*result);
      }
    }

  private:
    struct interface
    {
        virtual ~interface() = default;
        virtual std::type_index type() const = 0;
        virtual std::any const& value() const = 0;
        virtual std::any const& scalar() const = 0;
        virtual std::string text(presentation::data_column_display const&, bool) const = 0;
    };
    template <typename T> struct model final : interface
    {
        explicit model(T value) : value_(std::move(value))
        {
          auto const& v = std::any_cast<T const&>(value_);
          if constexpr (metadata_detail::dt::optional_traits<T>::optional)
          {
            if (v) this->set_scalar(*v);
          }
          else
            this->set_scalar(v);
        }
        std::type_index type() const override { return typeid(T); }
        std::any const& value() const override { return value_; }
        std::any const& scalar() const override { return scalar_; }
        std::string text(presentation::data_column_display const& format, bool machine) const override
        {
          return metadata_detail::dt::cell_text(std::any_cast<T const&>(value_), format, machine);
        }
        std::any value_;
        std::any scalar_;
        template <typename V> void set_scalar(V const& value)
        {
          if constexpr (metadata_detail::dt::half_integer<V>)
            scalar_ = basic_half_int<std::intmax_t>(value);
          else
            scalar_ = value;
        }
    };
    static std::string copy_text(char const* text)
    {
      if (!text) throw std::invalid_argument("null metadata text");
      return text;
    }
    std::shared_ptr<interface const> storage_;
};

enum class configuration_source
{
  explicit_input,
  command_line,
  option_file,
  attribute,
  environment,
  application_default
};

[[nodiscard]] inline std::string_view to_string(configuration_source source)
{
  switch (source)
  {
    case configuration_source::explicit_input:
      return "explicit input";
    case configuration_source::command_line:
      return "command line";
    case configuration_source::option_file:
      return "option file";
    case configuration_source::attribute:
      return "attribute";
    case configuration_source::environment:
      return "environment";
    case configuration_source::application_default:
      return "default";
  }
  throw std::invalid_argument("invalid configuration source");
}

/// \brief Identifies the winning source, without retaining discarded candidates.
struct value_origin
{
    configuration_source source = configuration_source::application_default;
    std::string location = {};
    std::string key = {};
};

struct metadata_field_options
{
    std::string label = {};
    std::string unit = {};
    std::string description = {};
    presentation::data_column_display display = {};
    bool detail = false;
    std::optional<value_origin> origin = std::nullopt;
};

struct metadata_field
{
    std::string id;
    metadata_value value;
    metadata_field_options options = {};
};

struct metadata_group
{
    std::string id;
    std::string label;
    std::vector<metadata_field> fields = {};
};

/// \brief Ordered, uniquely identified metadata. Copies are independent snapshots.
class metadata_document {
  public:
    void group(std::string id, std::string label = {})
    {
      metadata_detail::dt::validate_identifier(id);
      if (std::ranges::any_of(groups_, [&](auto const& g) { return g.id == id; }))
        throw std::invalid_argument("duplicate metadata group: " + id);
      if (label.empty()) label = id;
      groups_.push_back({std::move(id), std::move(label)});
    }
    void add(std::string const& group, std::string id, metadata_value value, metadata_field_options options = {})
    {
      metadata_detail::dt::validate_identifier(id);
      if (this->find(id)) throw std::invalid_argument("duplicate metadata field: " + id);
      auto found = std::ranges::find(groups_, group, &metadata_group::id);
      if (found == groups_.end()) throw std::invalid_argument("unknown metadata group: " + group);
      if (options.label.empty()) options.label = id;
      found->fields.push_back({std::move(id), std::move(value), std::move(options)});
    }
    void replace(std::string const& id, metadata_value value)
    {
      for (auto& group : groups_)
        for (auto& field : group.fields)
          if (field.id == id)
          {
            if (field.value.type() != value.type()) throw std::invalid_argument("metadata replacement changes type");
            field.value = std::move(value);
            return;
          }
      throw std::invalid_argument("unknown metadata field: " + id);
    }
    [[nodiscard]] metadata_field const* find(std::string_view id) const
    {
      for (auto const& group : groups_)
        for (auto const& field : group.fields)
          if (field.id == id) return &field;
      return nullptr;
    }
    [[nodiscard]] std::vector<metadata_group> const& groups() const { return groups_; }
    /// \brief Append disjoint groups from an owned snapshot, rejecting all collisions before mutation.
    void append(metadata_document const& other)
    {
      auto combined = *this;
      for (auto const& group : other.groups())
      {
        combined.group(group.id, group.label);
        for (auto const& field : group.fields)
          combined.add(group.id, field.id, field.value, field.options);
      }
      *this = std::move(combined);
    }

    /// \brief Explicitly project to the table format's flat string metadata, checking export-key collisions.
    [[nodiscard]] presentation::table_metadata strings(std::map<std::string, std::string> const& keys = {}) const
    {
      for (auto const& [id, key] : keys)
        if (!this->find(id)) throw std::invalid_argument("unknown metadata export field: " + id);
      presentation::table_metadata result;
      for (auto const& group : groups_)
        for (auto const& field : group.fields)
        {
          auto key = keys.contains(field.id) ? keys.at(field.id) : field.id;
          if (!result.emplace(key, field.value.text({}, true)).second)
            throw std::invalid_argument("duplicate metadata export key: " + key);
        }
      return result;
    }

  private:
    std::vector<metadata_group> groups_;
};
} // namespace uni20
