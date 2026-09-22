/**
 * \file data_table_json.hpp
 * \brief Precision-preserving JSON snapshots of typed result tables.
 */
#pragma once
#include "data_table.hpp"

namespace uni20::presentation
{
/// \brief Ordered JSON column selection. Empty selects every column; display precision is never applied.
struct data_json_options
{
    std::vector<std::string> columns = {};
};

namespace data_table_detail
{
// JSON writes must ignore the caller's locale, field width and numeric stream flags.
struct json_output
{
    std::ostream& stream;
    void put(char c) { stream.put(c); }
    json_output& operator<<(std::string_view text)
    {
      stream.write(text.data(), static_cast<std::streamsize>(text.size()));
      return *this;
    }
    json_output& operator<<(char c)
    {
      this->put(c);
      return *this;
    }
};

// Validates UTF-8 and escapes JSON syntax/control bytes without changing numerical text.
void write_json_string(std::ostream& out, std::string_view value);

template <typename T> void write_json_value(json_output& out, T const& value)
{
  if constexpr (optional_traits<T>::optional)
  {
    if (value)
      write_json_value(out, *value);
    else
      out << "null";
  }
  else if constexpr (half_integer<T>)
    write_json_string(out.stream, fmt::format("{}", value.twice()));
  else if constexpr (std::same_as<T, std::string>)
    write_json_string(out.stream, value);
  else if constexpr (Real<T>)
    write_json_string(out.stream, cell_text(value, {}, true));
  else if constexpr (integer<T> && uni20::numeric_limits<T>::digits > 53)
    write_json_string(out.stream, cell_text(value, {}, true));
  else
    out << cell_text(value, {}, true);
}

template <DataTableValue T> void write_json_column(json_output& out, data_column<T> const& column)
{
  using V = value_type<T>;
  out << "{\"id\":";
  write_json_string(out.stream, column.identifier());
  out << ",\"label\":";
  write_json_string(out.stream, column.label());
  out << ",\"unit\":";
  write_json_string(out.stream, column.unit());
  out << ",\"description\":";
  write_json_string(out.stream, column.description());
  out << ",\"type\":";
  if constexpr (half_integer<V>)
  {
    out << "\"half_int\",\"storage_bits\":"
        << fmt::format("{}", uni20::numeric_limits<typename V::value_type>::digits + 1)
        << ",\"encoding\":\"twice_decimal_string\"";
  }
  else if constexpr (Real<V>)
  {
    out << "\"real\",\"radix\":" << fmt::format("{}", uni20::numeric_limits<V>::radix)
        << ",\"precision_bits\":" << fmt::format("{}", uni20::numeric_limits<V>::digits)
        << ",\"encoding\":\"decimal_string\"";
  }
  else if constexpr (integer<V>)
  {
    write_json_string(out.stream, fmt::format("{}int{}", std::is_signed_v<V> ? "" : "u",
                                              uni20::numeric_limits<V>::digits + (std::is_signed_v<V> ? 1 : 0)));
    out << ",\"encoding\":";
    write_json_string(out.stream, uni20::numeric_limits<V>::digits > 53 ? "decimal_string" : "number");
  }
  else if constexpr (std::same_as<V, bool>)
    out << "\"bool\"";
  else
    out << "\"string\"";
  out << ",\"nullable\":" << (optional_traits<T>::optional ? "true" : "false") << '}';
}

template <typename Schema, typename Row>
void write_json_row(json_output& out, Schema const& schema, Row const& row, resolved_projection const& selection)
{
  out.put('[');
  bool first = true;
  visit_selected(schema, row, selection, [&](auto const&, auto const& value, auto const&) {
    if (!std::exchange(first, false)) out.put(',');
    write_json_value(out, value);
  });
  out.put(']');
  check_output(out.stream);
}

inline void write_json_metadata(json_output& out, table_metadata const& metadata)
{
  out.put('{');
  bool first = true;
  for (auto const& [key, value] : metadata)
  {
    if (!std::exchange(first, false)) out.put(',');
    write_json_string(out.stream, key);
    out.put(':');
    write_json_string(out.stream, value);
  }
  out.put('}');
}

template <typename Schema>
void write_json_begin(json_output& out, std::string const& title, Schema const& schema, table_metadata const& metadata,
                      resolved_projection const& selection, data_sink_start start)
{
  check_output(out.stream);
  out << "{\"title\":";
  write_json_string(out.stream, title);
  out << ",\"metadata\":";
  write_json_metadata(out, metadata);
  out << ",\"first_row\":" << fmt::format("{}", start.first_row) << ",\"columns\":[";
  bool first = true;
  for (auto index : selection.indices)
    visit_column(schema, index, [&](auto const& column) {
      if (!std::exchange(first, false)) out.put(',');
      write_json_column(out, column);
    });
  out << "],\"rows\":[";
  check_output(out.stream);
}

inline void write_json_end(json_output& out, table_metadata const* summary)
{
  out.put(']');
  if (summary)
  {
    out << ",\"summary\":";
    write_json_metadata(out, *summary);
  }
  out << "}\n";
  check_output(out.stream);
}
} // namespace data_table_detail

/// \brief Write an ordered schema and typed rows, preserving real and integer precision.
/// \details Reals and wide integers are decimal strings; half-integers use their exact doubled integer.
///          UTF-8 text is validated. The caller owns the stream and must check flush/close errors.
///          In-progress snapshots omit the final summary. No-retention tables reject snapshots before writing.
template <DataTableValue... Ts>
void write_json(std::ostream& stream, data_table<Ts...> const& table, data_json_options const& options = {})
{
  auto rows = table.rows();
  auto selection = data_table_detail::resolve_projection(table.columns(), {.columns = options.columns});
  data_table_detail::json_output out{stream};
  data_table_detail::write_json_begin(out, table.title(), table.columns(), table.metadata(), selection, {});
  bool first = true;
  for (auto const& row : rows)
  {
    if (!std::exchange(first, false)) out.put(',');
    data_table_detail::write_json_row(out, table.columns(), row, selection);
  }
  data_table_detail::write_json_end(out, table.summary() ? &*table.summary() : nullptr);
}
} // namespace uni20::presentation
