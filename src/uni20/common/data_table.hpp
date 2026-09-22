/**
 * \file data_table.hpp
 * \brief Typed result tables with deferred presentation and numerical export.
 */
#pragma once

#include "half_int.hpp"
#include "presentation.hpp"

#include <algorithm>
#include <array>
#include <concepts>
#include <cstdint>
#include <exception>
#include <map>
#include <memory>
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

/// \brief Ordered output columns and independent display overrides. Empty columns selects the full schema.
struct table_projection
{
    std::vector<std::string> columns = {};
    std::map<std::string, data_column_display> display = {};
};

/// \brief Application metadata is owned text, independent of numerical rows.
using table_metadata = std::map<std::string, std::string>;

/// \brief Whether accepted rows remain available for snapshots and replay.
enum class retention
{
  all,
  none
};

/// \brief Construction-time table policy and initial application metadata.
struct data_table_options
{
    retention retain = retention::all;
    table_metadata metadata = {};
};

/// \brief Full replay is the default. Future-only attachment explicitly opts out of history.
enum class sink_replay
{
  all,
  future_only
};
struct data_sink_options
{
    bool required = true;
    sink_replay replay = sink_replay::all;
};

/// \brief The index of the first row this attachment can emit (zero for full replay).
struct data_sink_start
{
    std::size_t first_row = 0;
};

/// \brief A table-local attachment identifier, stable when its table is moved.
struct data_sink_id
{
    std::size_t value;
    bool operator==(data_sink_id const&) const = default;
};
enum class data_sink_state
{
  active,
  closed,
  failed
};
enum class data_sink_operation
{
  begin,
  row,
  finish
};

/// \brief Original exception and attachment identity for one disabled sink.
struct data_sink_failure
{
    data_sink_id sink;
    bool required;
    data_sink_operation operation;
    std::exception_ptr exception;
};

/// \brief Newly encountered failures. Callers using optional sinks must inspect this result.
struct data_delivery_report
{
    std::vector<data_sink_failure> failures = {};
};

/// \brief Required output failed after all eligible sinks were attempted; retained rows are not rolled back.
class data_delivery_error : public std::runtime_error {
  public:
    explicit data_delivery_error(data_delivery_report report)
        : std::runtime_error("required data table output failed"), report_(std::move(report))
    {}
    [[nodiscard]] data_delivery_report const& report() const noexcept { return report_; }

  private:
    data_delivery_report report_;
};

/// \brief Attachment identity and any optional begin/replay/finalization failure.
struct data_attachment
{
    data_sink_id id;
    data_delivery_report report;
};

/// \brief Typed result table with synchronous sinks and optional retained history.
/// \details Move-only: copying live outputs would duplicate delivery. Call finish explicitly to observe flush errors.
///          Mutation is single-caller and non-reentrant; sink callbacks receive borrowed const schema/rows.
template <DataTableValue... Ts> class data_table {
  public:
    static_assert(sizeof...(Ts) > 0, "data tables require at least one column");
    using row_type = std::tuple<Ts...>;
    using schema_type = std::tuple<data_column<Ts>...>;

    data_table(std::string title, data_column<Ts>... columns)
        : data_table(std::move(title), data_table_options{}, std::move(columns)...)
    {}
    data_table(std::string title, data_table_options options, data_column<Ts>... columns)
        : title_(std::move(title)), columns_(std::move(columns)...), options_(std::move(options))
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
    data_table(data_table const&) = delete;
    data_table& operator=(data_table const&) = delete;
    data_table(data_table&&) = default;
    data_table& operator=(data_table&&) = delete;

    /// \brief Validate once, retain when enabled, then deliver to every active sink in attachment order.
    /// \details Conversion failure leaves the table unchanged. Output failure occurs after row acceptance;
    ///          inspect optional failures in the returned report and never retry a failed delivery automatically.
    template <typename... Us>
      requires(sizeof...(Ts) == sizeof...(Us) && (data_table_detail::accepts<Ts, Us>() && ...))
    data_delivery_report append(Us&&... values)
    {
      mutation_guard guard(busy_);
      if (finished_) throw std::logic_error("cannot append to a finished data table");
      auto row = this->make_row(std::forward<Us>(values)...);
      auto report = this->new_report();
      if (options_.retain == retention::all) rows_.push_back(std::move(row));
      ++size_;
      auto const& accepted = options_.retain == retention::all ? rows_.back() : row;
      for (auto& sink : sinks_)
        this->attempt(sink, data_sink_operation::row, report, [&] { sink.output->row(columns_, accepted); });
      return this->checked_report(std::move(report));
    }

    /// \brief Construct a checked, owning row independently of storage or output.
    template <typename... Us>
      requires(sizeof...(Ts) == sizeof...(Us) && (data_table_detail::accepts<Ts, Us>() && ...))
    [[nodiscard]] static row_type make_row(Us&&... values)
    {
      return row_type{data_table_detail::convert<Ts>(std::forward<Us>(values))...};
    }

    /// \brief Own a sink, begin its output and replay history before subscribing to future rows.
    /// \details Sink must provide begin(title, schema, metadata, start), row(schema, row), and finish(summary).
    ///          Streams/resources borrowed by a sink must outlive its attachment. Failed sinks are disabled.
    template <typename Sink> data_attachment attach(Sink&& output, data_sink_options options = {})
    {
      mutation_guard guard(busy_);
      if (options.replay == sink_replay::all && options_.retain == retention::none && size_ != 0)
        throw std::logic_error("data table history unavailable; request future_only attachment explicitly");
      auto owned = std::make_unique<sink_model<std::remove_cvref_t<Sink>>>(std::forward<Sink>(output));
      data_delivery_report report;
      report.failures.reserve(1);
      data_sink_id id{sinks_.size()};
      sinks_.push_back({id, options.required, data_sink_state::active, std::move(owned)});
      auto& sink = sinks_.back();
      this->attempt(sink, data_sink_operation::begin, report, [&] {
        sink.output->begin(title_, columns_, options_.metadata, {options.replay == sink_replay::all ? 0 : size_});
      });
      if (options.replay == sink_replay::all)
        for (auto const& row : rows_)
        {
          if (sink.state != data_sink_state::active) break;
          this->attempt(sink, data_sink_operation::row, report, [&] { sink.output->row(columns_, row); });
        }
      if (finished_) this->close(sink, *summary_, report);
      return {id, this->checked_report(std::move(report))};
    }

    /// \brief Finalize and unsubscribe one sink without ending the table or discarding rows.
    data_delivery_report finish_sink(data_sink_id id, table_metadata const& summary = {})
    {
      mutation_guard guard(busy_);
      auto report = this->new_report();
      this->close(this->find_sink(id), summary, report);
      return this->checked_report(std::move(report));
    }

    /// \brief End row insertion and finalize all active outputs, retaining the summary for later replay.
    /// \details Repeated finish with no new summary (or the same summary) does not write again.
    ///          Destruction never substitutes for explicit finalization; snapshots remain available afterwards.
    data_delivery_report finish(table_metadata summary = {})
    {
      mutation_guard guard(busy_);
      if (finished_)
      {
        if (!summary.empty() && summary != *summary_) throw std::logic_error("data table summary already finalized");
        return this->checked_report(final_report_);
      }
      auto report = this->new_report();
      summary_ = std::move(summary);
      finished_ = true;
      for (auto& sink : sinks_)
        this->close(sink, *summary_, report);
      final_report_ = std::move(report);
      return this->checked_report(final_report_);
    }

    [[nodiscard]] std::string const& title() const noexcept { return title_; }
    [[nodiscard]] schema_type const& columns() const noexcept { return columns_; }
    [[nodiscard]] table_metadata const& metadata() const noexcept { return options_.metadata; }
    [[nodiscard]] std::optional<table_metadata> const& summary() const noexcept { return summary_; }
    [[nodiscard]] bool finished() const noexcept { return finished_; }
    [[nodiscard]] retention retention_policy() const noexcept { return options_.retain; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t retained_size() const noexcept { return rows_.size(); }
    /// \brief Require complete retained history before inspecting or exporting rows.
    [[nodiscard]] std::span<row_type const> rows() const
    {
      if (options_.retain == retention::none) throw std::logic_error("data table does not retain history");
      return rows_;
    }
    [[nodiscard]] data_sink_state sink_state(data_sink_id id) const { return this->find_sink(id).state; }

  private:
    struct mutation_guard
    {
        bool& busy;
        explicit mutation_guard(bool& flag) : busy(flag)
        {
          if (busy) throw std::logic_error("data table mutation is not reentrant");
          busy = true;
        }
        ~mutation_guard() { busy = false; }
    };
    struct sink_interface
    {
        virtual ~sink_interface() = default;
        virtual void begin(std::string const&, schema_type const&, table_metadata const&, data_sink_start) = 0;
        virtual void row(schema_type const&, row_type const&) = 0;
        virtual void finish(table_metadata const&) = 0;
    };
    template <typename Sink> struct sink_model final : sink_interface
    {
        Sink output;
        explicit sink_model(Sink value) : output(std::move(value)) {}
        void begin(std::string const& title, schema_type const& schema, table_metadata const& metadata,
                   data_sink_start start) override
        {
          output.begin(title, schema, metadata, start);
        }
        void row(schema_type const& schema, row_type const& row) override { output.row(schema, row); }
        void finish(table_metadata const& summary) override { output.finish(summary); }
    };
    struct sink_record
    {
        data_sink_id id;
        bool required;
        data_sink_state state;
        std::unique_ptr<sink_interface> output;
    };
    data_delivery_report new_report() const
    {
      data_delivery_report report;
      report.failures.reserve(sinks_.size());
      return report;
    }
    static data_delivery_report checked_report(data_delivery_report report)
    {
      if (std::any_of(report.failures.begin(), report.failures.end(), [](auto const& f) { return f.required; }))
        throw data_delivery_error(std::move(report));
      return report;
    }
    template <typename Function>
    void attempt(sink_record& sink, data_sink_operation operation, data_delivery_report& report, Function&& function)
    {
      if (sink.state != data_sink_state::active) return;
      try
      {
        function();
      }
      catch (...)
      {
        sink.state = data_sink_state::failed;
        report.failures.push_back({sink.id, sink.required, operation, std::current_exception()});
        sink.output.reset();
      }
    }
    void close(sink_record& sink, table_metadata const& summary, data_delivery_report& report)
    {
      this->attempt(sink, data_sink_operation::finish, report, [&] { sink.output->finish(summary); });
      if (sink.state == data_sink_state::active)
      {
        sink.state = data_sink_state::closed;
        sink.output.reset();
      }
    }
    sink_record const& find_sink(data_sink_id id) const
    {
      if (id.value >= sinks_.size()) throw std::invalid_argument("unknown data table sink");
      return sinks_[id.value];
    }
    sink_record& find_sink(data_sink_id id) { return const_cast<sink_record&>(std::as_const(*this).find_sink(id)); }

    std::string title_;
    schema_type columns_;
    data_table_options options_;
    std::vector<row_type> rows_;
    std::vector<sink_record> sinks_;
    std::size_t size_ = 0;
    bool busy_ = false;
    bool finished_ = false;
    std::optional<table_metadata> summary_;
    data_delivery_report final_report_;
};

/// \brief Deduce column types with default retained history.
template <DataTableValue... Ts> [[nodiscard]] auto make_data_table(std::string title, data_column<Ts>... columns)
{
  return data_table<Ts...>(std::move(title), std::move(columns)...);
}

/// \brief Deduce column types with an explicit retention policy and initial metadata.
template <DataTableValue... Ts>
[[nodiscard]] auto make_data_table(std::string title, data_table_options options, data_column<Ts>... columns)
{
  return data_table<Ts...>(std::move(title), std::move(options), std::move(columns)...);
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
struct resolved_projection
{
    std::vector<std::size_t> indices;
    std::vector<data_column_display> display;
};

template <typename Schema, typename Function>
void visit_column(Schema const& schema, std::size_t index, Function&& function)
{
  [&]<std::size_t... I>(std::index_sequence<I...>) {
    ((I == index ? (void)function(std::get<I>(schema)) : (void)0), ...);
  }(std::make_index_sequence<std::tuple_size_v<Schema>>{});
}

template <typename Schema> resolved_projection resolve_projection(Schema const& schema, table_projection const& options)
{
  resolved_projection result;
  std::vector<std::string> names;
  std::apply([&](auto const&... column) { (names.push_back(column.identifier()), ...); }, schema);
  auto add = [&](std::string const& name) {
    auto found = std::find(names.begin(), names.end(), name);
    if (found == names.end()) throw std::invalid_argument("unknown data table output column: " + name);
    auto index = static_cast<std::size_t>(found - names.begin());
    if (std::find(result.indices.begin(), result.indices.end(), index) != result.indices.end())
      throw std::invalid_argument("duplicate data table output column: " + name);
    result.indices.push_back(index);
    visit_column(schema, index, [&](auto const& c) { result.display.push_back(c.display()); });
    if (auto override = options.display.find(name); override != options.display.end())
      result.display.back() = override->second;
  };
  for (auto const& name : options.columns.empty() ? names : options.columns)
    add(name);
  for (auto const& [name, format] : options.display)
  {
    auto found = std::find(names.begin(), names.end(), name);
    if (found == names.end() ||
        std::find(result.indices.begin(), result.indices.end(), found - names.begin()) == result.indices.end())
      throw std::invalid_argument("display override requires a selected data table column: " + name);
    auto const& numeric = format.numeric;
    if (numeric.precision < -1 || (numeric.notation == real_format_notation::general && numeric.precision == 0))
      throw std::invalid_argument("invalid data table output precision");
  }
  return result;
}

template <typename Schema, typename Row, typename Function>
void visit_selected(Schema const& schema, Row const& row, resolved_projection const& selection, Function&& function)
{
  for (std::size_t p = 0; p < selection.indices.size(); ++p)
    [&]<std::size_t... I>(std::index_sequence<I...>) {
      ((I == selection.indices[p] ? (void)function(std::get<I>(schema), std::get<I>(row), selection.display[p])
                                  : (void)0),
       ...);
    }(std::make_index_sequence<std::tuple_size_v<Schema>>{});
}
} // namespace data_table_detail

/// \brief Format a snapshot using column display settings and the existing rich table model.
template <DataTableValue... Ts>
[[nodiscard]] report_table to_report_table(data_table<Ts...> const& table, table_projection const& projection = {})
{
  auto rows = table.rows();
  report_table result(table.title());
  auto selection = data_table_detail::resolve_projection(table.columns(), projection);
  for (std::size_t p = 0; p < selection.indices.size(); ++p)
    data_table_detail::visit_column(table.columns(), selection.indices[p], [&](auto const& column) {
      result.column(column.label(), selection.display[p].alignment);
    });
  for (auto const& row : rows)
  {
    std::vector<std::string> cells;
    cells.reserve(sizeof...(Ts));
    data_table_detail::visit_selected(table.columns(), row, selection,
                                      [&](auto const&, auto const& value, auto const& display) {
                                        cells.push_back(data_table_detail::cell_text(value, display, false));
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
    table_projection projection = {};
};

namespace data_table_detail
{
template <typename T>
std::string export_text(T const& value, data_column_display const& display, delimited_options const& options)
{
  if constexpr (optional_traits<T>::optional)
  {
    if (!value) return "";
    return export_text(*value, display, options);
  }
  else
    return cell_text(value, display, !Real<T> || options.precision == data_export_precision::round_trip);
}

template <typename Schema>
void write_delimited_header(std::ostream& out, Schema const& schema, resolved_projection const& selection,
                            char delimiter)
{
  check_output(out);
  bool first = true;
  for (auto index : selection.indices)
    visit_column(schema, index, [&](auto const& column) {
      if (!std::exchange(first, false)) out.put(delimiter);
      write_field(out, column.identifier(), delimiter);
    });
  out.put('\n');
  check_output(out);
}

template <typename Schema, typename Row>
void write_delimited_row(std::ostream& out, Schema const& schema, Row const& row, resolved_projection const& selection,
                         char delimiter, delimited_options const& options)
{
  bool first = true;
  visit_selected(schema, row, selection, [&](auto const&, auto const& value, auto const& display) {
    if (!std::exchange(first, false)) out.put(delimiter);
    using T = std::remove_cvref_t<decltype(value)>;
    auto text = export_text(value, display, options);
    write_field(out, text, delimiter, std::same_as<T, std::string> || selection.indices.size() == 1);
  });
  out.put('\n');
  check_output(out);
}

template <DataTableValue... Ts>
void write_delimited(std::ostream& out, data_table<Ts...> const& table, char delimiter,
                     delimited_options const& options)
{
  auto rows = table.rows();
  auto selection = resolve_projection(table.columns(), options.projection);
  write_delimited_header(out, table.columns(), selection, delimiter);
  for (auto const& row : rows)
    write_delimited_row(out, table.columns(), row, selection, delimiter, options);
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
