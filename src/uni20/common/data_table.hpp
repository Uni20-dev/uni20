/**
 * \file data_table.hpp
 * \brief Typed result tables with deferred presentation and numerical export.
 */
#pragma once

#include "half_int.hpp"
#include "presentation.hpp"

#include <array>
#include <concepts>
#include <cstdint>
#include <optional>
#include <ostream>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20::presentation
{
namespace data_table_detail
{
template <typename T> struct optional_traits
{
    static constexpr bool optional = false;
    using value_type = T;
};
template <typename T> struct optional_traits<std::optional<T>>
{
    static constexpr bool optional = true;
    using value_type = T;
};
template <typename T> using value_type = typename optional_traits<T>::value_type;
template <typename T> inline constexpr bool half_integer = false;
template <std::signed_integral T> inline constexpr bool half_integer<basic_half_int<T>> = true;

// Use integer storage types, including int8_t, without treating character text as numbers.
template <typename T>
concept integer =
    std::same_as<T, signed char> || std::same_as<T, unsigned char> || std::same_as<T, short> ||
    std::same_as<T, unsigned short> || std::same_as<T, int> || std::same_as<T, unsigned int> || std::same_as<T, long> ||
    std::same_as<T, unsigned long> || std::same_as<T, long long> || std::same_as<T, unsigned long long>;

template <typename To, typename From> consteval bool accepts()
{
  using F = std::remove_cvref_t<From>;
  if constexpr (optional_traits<To>::optional)
  {
    if constexpr (std::same_as<F, std::nullopt_t>)
      return true;
    else if constexpr (optional_traits<F>::optional)
      return accepts<value_type<To>, decltype(*std::declval<From>())>();
    else
      return accepts<value_type<To>, From>();
  }
  else if constexpr (std::same_as<To, std::string>)
    return std::constructible_from<std::string, From> && !std::same_as<F, std::nullptr_t>;
  else if constexpr (std::same_as<To, bool>)
    return std::same_as<F, bool>;
  else if constexpr (integer<To>)
    return integer<F>;
  else if constexpr (half_integer<To>)
    return half_integer<F> || integer<F>;
  else if constexpr (Real<To>)
    return (Real<F> || integer<F>) && std::constructible_from<To, From>;
  else
    return false;
}

template <typename To, typename From>
  requires(accepts<To, From>())
To convert(From&& input)
{
  using F = std::remove_cvref_t<From>;
  if constexpr (optional_traits<To>::optional)
  {
    if constexpr (std::same_as<F, std::nullopt_t>)
      return std::nullopt;
    else if constexpr (optional_traits<F>::optional)
    {
      if (!input) return std::nullopt;
      return To{convert<value_type<To>>(*std::forward<From>(input))};
    }
    else
      return To{convert<value_type<To>>(std::forward<From>(input))};
  }
  else if constexpr (integer<To>)
  {
    if (!std::in_range<To>(input)) throw std::overflow_error("data table integer conversion out of range");
    return static_cast<To>(input);
  }
  else if constexpr (half_integer<To>)
  {
    if constexpr (half_integer<F>)
      return To(input);
    else
    {
      using Storage = typename To::value_type;
      // Check in the source type's range before any narrowing or doubling.
      if (std::cmp_less(input, uni20::numeric_limits<Storage>::min() / 2) ||
          std::cmp_greater(input, uni20::numeric_limits<Storage>::max() / 2))
        throw std::overflow_error("data table half-integer conversion out of range");
      return To(static_cast<Storage>(input));
    }
  }
  else if constexpr (std::same_as<To, std::string>)
  {
    if constexpr (std::is_pointer_v<F>)
      if (input == nullptr) throw std::invalid_argument("null data table string");
    return std::string(std::forward<From>(input));
  }
  else
    return static_cast<To>(std::forward<From>(input));
}

void validate_identifier(std::string_view identifier);
void write_field(std::ostream& out, std::string_view field, char delimiter, bool quote_empty = false);
void check_output(std::ostream& out);

// Only C-library scalar providers need a thread-local C locale. Never change the process locale.
class classic_numeric_locale {
  public:
    classic_numeric_locale();
    ~classic_numeric_locale();
    classic_numeric_locale(classic_numeric_locale const&) = delete;
    classic_numeric_locale& operator=(classic_numeric_locale const&) = delete;

  private:
    void* previous_;
};

template <Real T> std::string real_text(T value, scalar_format_options options)
{
#if UNI20_HAS_FLOAT128 && UNI20_FLOAT128_PROVIDER_MPLAPACK && defined(MPLAPACK_BINARY128_MODE) &&                      \
    (MPLAPACK_BINARY128_MODE == MPLAPACK_BINARY128_MODE_FLOAT128)
  if constexpr (std::same_as<T, uni20::float128>)
  {
    classic_numeric_locale locale;
    return uni20::format_real(value, options);
  }
  else
#endif
  {
    static_assert(std::floating_point<T>, "data table export needs a lossless formatter for this scalar provider");
    return uni20::format_real(value, options);
  }
}
} // namespace data_table_detail

/// \brief Types retained directly by data tables, without numerical narrowing or text conversion.
/// \details Optional string columns are excluded until their delimited null encoding is specified.
template <typename T>
concept DataTableValue =
    std::same_as<T, std::remove_cvref_t<T>> &&
    (data_table_detail::integer<data_table_detail::value_type<T>> || Real<data_table_detail::value_type<T>> ||
     std::same_as<data_table_detail::value_type<T>, bool> ||
     data_table_detail::half_integer<data_table_detail::value_type<T>> || std::same_as<T, std::string>);

/// \brief Column display settings. These never change stored values or default export precision.
struct data_column_display
{
    scalar_format_options numeric = {};
    table_alignment alignment = table_alignment::right;
    std::string missing = "—";
    bool fractions = false;
};

/// \brief Typed column schema with owned metadata and deferred display formatting.
template <DataTableValue T> class data_column {
  public:
    using value_type = T;
    explicit data_column(std::string identifier) : identifier_(std::move(identifier)), label_(identifier_)
    {
      data_table_detail::validate_identifier(identifier_);
      if constexpr (std::same_as<T, std::string> || std::same_as<data_table_detail::value_type<T>, bool>)
        display_.alignment = table_alignment::left;
      else
        display_.alignment = table_alignment::decimal;
    }

    data_column& label(std::string text)
    {
      label_ = std::move(text);
      return *this;
    }
    data_column& unit(std::string text)
    {
      unit_ = std::move(text);
      return *this;
    }
    data_column& description(std::string text)
    {
      description_ = std::move(text);
      return *this;
    }
    data_column& alignment(table_alignment value)
    {
      display_.alignment = value;
      return *this;
    }
    data_column& missing(std::string text)
    {
      display_.missing = std::move(text);
      return *this;
    }

    /// \brief Use digits after the decimal point for real display.
    data_column& fixed(int digits)
      requires Real<data_table_detail::value_type<T>>
    {
      return this->set_numeric(real_format_notation::fixed, digits);
    }
    /// \brief Use digits after the decimal point in the mantissa for real display.
    data_column& scientific(int digits)
      requires Real<data_table_detail::value_type<T>>
    {
      return this->set_numeric(real_format_notation::scientific, digits);
    }
    /// \brief Use significant digits for real display; -1 selects round-trip precision.
    data_column& general(int digits = -1)
      requires Real<data_table_detail::value_type<T>>
    {
      return this->set_numeric(real_format_notation::general, digits);
    }
    /// \brief Display exact fractions instead of exact decimals for half-integers.
    data_column& fractional(bool enabled = true)
      requires data_table_detail::half_integer<data_table_detail::value_type<T>>
    {
      display_.fractions = enabled;
      return *this;
    }

    [[nodiscard]] std::string const& identifier() const noexcept { return identifier_; }
    [[nodiscard]] std::string const& label() const noexcept { return label_; }
    [[nodiscard]] std::string const& unit() const noexcept { return unit_; }
    [[nodiscard]] std::string const& description() const noexcept { return description_; }
    [[nodiscard]] data_column_display const& display() const noexcept { return display_; }

  private:
    data_column& set_numeric(real_format_notation notation, int digits)
    {
      if (digits < 0 && !(notation == real_format_notation::general && digits == -1))
        throw std::invalid_argument("negative data table display precision");
      if (notation == real_format_notation::general && digits == 0)
        throw std::invalid_argument("general display precision must be positive");
      display_.numeric.notation = notation;
      display_.numeric.precision = digits;
      return *this;
    }
    std::string identifier_;
    std::string label_;
    std::string unit_;
    std::string description_;
    data_column_display display_;
};

/// \brief Owning heterogeneous result table with a fixed schema and checked whole-row insertion.
/// \details All mutation and output is synchronous. Const rows preserve the column's original scalar type.
template <DataTableValue... Ts> class data_table {
  public:
    static_assert(sizeof...(Ts) > 0, "data tables require at least one column");
    using row_type = std::tuple<Ts...>;
    using schema_type = std::tuple<data_column<Ts>...>;

    data_table(std::string title, data_column<Ts>... columns)
        : title_(std::move(title)), columns_(std::move(columns)...)
    {
      std::apply(
          [](auto const&... column) {
            std::array names{std::string_view(column.identifier())...};
            for (std::size_t i = 0; i < names.size(); ++i)
              for (std::size_t j = 0; j < i; ++j)
                if (names[i] == names[j]) throw std::invalid_argument("duplicate data table column identifier");
          },
          columns_);
    }

    /// \brief Convert an entire row before storing it; failure leaves existing rows unchanged.
    /// \throws std::overflow_error An integer or half-integer value is out of range.
    template <typename... Us>
      requires(sizeof...(Ts) == sizeof...(Us) && (data_table_detail::accepts<Ts, Us>() && ...))
    void append(Us&&... values)
    {
      rows_.push_back(this->make_row(std::forward<Us>(values)...));
    }

    /// \brief Construct a validated, owning typed row independently of retained history.
    template <typename... Us>
      requires(sizeof...(Ts) == sizeof...(Us) && (data_table_detail::accepts<Ts, Us>() && ...))
    [[nodiscard]] static row_type make_row(Us&&... values)
    {
      return row_type{data_table_detail::convert<Ts>(std::forward<Us>(values))...};
    }

    [[nodiscard]] std::string const& title() const noexcept { return title_; }
    [[nodiscard]] schema_type const& columns() const noexcept { return columns_; }
    [[nodiscard]] std::span<row_type const> rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t size() const noexcept { return rows_.size(); }

  private:
    std::string title_;
    schema_type columns_;
    std::vector<row_type> rows_;
};

/// \brief Deduce an owning table's value types from its column declarations.
template <DataTableValue... Ts> [[nodiscard]] auto make_data_table(std::string title, data_column<Ts>... columns)
{
  return data_table<Ts...>(std::move(title), std::move(columns)...);
}

namespace data_table_detail
{
template <typename T> std::string cell_text(T const& value, data_column_display const& display, bool machine)
{
  if constexpr (optional_traits<T>::optional)
  {
    if (!value) return machine ? "" : display.missing;
    return cell_text(*value, display, machine);
  }
  else if constexpr (std::same_as<T, std::string>)
    return value;
  else if constexpr (std::same_as<T, bool>)
    return value ? "true" : "false";
  else if constexpr (half_integer<T>)
    return !machine && display.fractions ? uni20::to_string_fraction(value) : uni20::to_string(value);
  else if constexpr (integer<T>)
    return fmt::format("{}", value);
  else
  {
    scalar_format_options options = machine ? scalar_format_options{} : display.numeric;
    if (machine) options.normalize_negative_zero = false;
    return real_text(value, options);
  }
}

template <typename Schema, typename Row, typename Function>
void visit_cells(Schema const& schema, Row const& row, Function&& function)
{
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    (function(std::get<I>(schema), std::get<I>(row)), ...);
  }(std::make_index_sequence<std::tuple_size_v<Schema>>{});
}
} // namespace data_table_detail

/// \brief Format a snapshot using column display settings and the existing rich table model.
template <DataTableValue... Ts> [[nodiscard]] report_table to_report_table(data_table<Ts...> const& table)
{
  report_table result(table.title());
  std::apply([&](auto const&... column) { (result.column(column.label(), column.display().alignment), ...); },
             table.columns());
  for (auto const& row : table.rows())
  {
    std::vector<std::string> cells;
    cells.reserve(sizeof...(Ts));
    data_table_detail::visit_cells(table.columns(), row, [&](auto const& column, auto const& value) {
      cells.push_back(data_table_detail::cell_text(value, column.display(), false));
    });
    result.row(std::move(cells));
  }
  return result;
}

/// \brief Precision for delimited numerical output; display mode deliberately rounds real values.
enum class data_export_precision
{
  round_trip,
  display
};

/// \brief Delimited output policy, independent of terminal styles and width.
struct delimited_options
{
    data_export_precision precision = data_export_precision::round_trip;
};

namespace data_table_detail
{
template <typename T>
std::string export_text(T const& value, data_column_display const& display, delimited_options options)
{
  if constexpr (optional_traits<T>::optional)
  {
    if (!value) return "";
    return export_text(*value, display, options);
  }
  else
    return cell_text(value, display, !Real<T> || options.precision == data_export_precision::round_trip);
}

template <DataTableValue... Ts>
void write_delimited(std::ostream& out, data_table<Ts...> const& table, char delimiter, delimited_options options)
{
  check_output(out);
  bool first = true;
  std::apply(
      [&](auto const&... column) {
        auto heading = [&](auto const& c) {
          if (!std::exchange(first, false)) out.put(delimiter);
          write_field(out, c.identifier(), delimiter);
        };
        (heading(column), ...);
      },
      table.columns());
  out.put('\n');
  for (auto const& row : table.rows())
  {
    first = true;
    visit_cells(table.columns(), row, [&](auto const& column, auto const& value) {
      if (!std::exchange(first, false)) out.put(delimiter);
      using T = std::remove_cvref_t<decltype(value)>;
      auto text = export_text(value, column.display(), options);
      // A single empty field needs quotes so readers do not skip it as a blank record.
      write_field(out, text, delimiter, std::same_as<T, std::string> || sizeof...(Ts) == 1);
    });
    out.put('\n');
    check_output(out);
  }
  check_output(out);
}
} // namespace data_table_detail

/// \brief Write a CSV snapshot with stable headers, quoted strings and empty missing numeric fields.
/// \details Writes all rows without modifying the table or flushing/closing the caller-owned stream.
///          Finite reals default to round-trip precision in their stored type. Output errors may leave partial data.
/// \throws std::ios_base::failure A stream write fails, even when stream exception flags are disabled.
template <DataTableValue... Ts>
void write_csv(std::ostream& out, data_table<Ts...> const& table, delimited_options options = {})
{
  data_table_detail::write_delimited(out, table, ',', options);
}

/// \brief Write tab-delimited CSV using CSV quoting, not whitespace-token splitting.
/// \details Otherwise follows write_csv, including preservation of empty fields and numeric precision.
template <DataTableValue... Ts>
void write_tsv(std::ostream& out, data_table<Ts...> const& table, delimited_options options = {})
{
  data_table_detail::write_delimited(out, table, '\t', options);
}

} // namespace uni20::presentation
