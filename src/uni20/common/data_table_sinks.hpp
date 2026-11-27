/**
 * \file data_table_sinks.hpp
 * \brief Synchronous CSV, TSV, JSON and terminal adapters for typed table subscriptions.
 */
#pragma once
#include "data_table_json.hpp"
#include "display.hpp"

namespace uni20::presentation
{
namespace data_table_detail
{
class delimited_sink {
  public:
    delimited_sink(std::ostream& out, char delimiter, delimited_options options)
        : out_(out), delimiter_(delimiter), options_(std::move(options))
    {}
    template <typename Schema>
    void begin(std::string const&, Schema const& schema, table_metadata const&, data_sink_start)
    {
      selection_ = resolve_projection(schema, options_.projection);
      write_delimited_header(out_, schema, selection_, delimiter_);
    }
    template <typename Schema, typename Row> void row(Schema const& schema, Row const& row)
    {
      write_delimited_row(out_, schema, row, selection_, delimiter_, options_);
    }
    void finish(table_metadata const&)
    {
      out_.flush();
      check_output(out_);
    }

  private:
    std::ostream& out_;
    char delimiter_;
    delimited_options options_;
    resolved_projection selection_;
};

class json_sink_adapter {
  public:
    json_sink_adapter(std::ostream& out, data_json_options options) : out_(out), options_(std::move(options)) {}
    template <typename Schema>
    void begin(std::string const& title, Schema const& schema, table_metadata const& metadata, data_sink_start start)
    {
      selection_ = resolve_projection(schema, {.columns = options_.columns});
      json_output out{out_};
      write_json_begin(out, title, schema, metadata, selection_, start);
    }
    template <typename Schema, typename Row> void row(Schema const& schema, Row const& row)
    {
      json_output out{out_};
      if (!std::exchange(first_, false)) out.put(',');
      write_json_row(out, schema, row, selection_);
    }
    void finish(table_metadata const& summary)
    {
      json_output out{out_};
      write_json_end(out, &summary);
      out_.flush();
      check_output(out_);
    }

  private:
    std::ostream& out_;
    data_json_options options_;
    resolved_projection selection_;
    bool first_ = true;
};
} // namespace data_table_detail

/// \brief Stream CSV rows to a borrowed stream, flushing on explicit sink/table finish.
/// \details Generic delimited output omits metadata and summaries to remain rectangular.
[[nodiscard]] inline auto csv_sink(std::ostream& out, delimited_options options = {})
{
  return data_table_detail::delimited_sink(out, ',', std::move(options));
}

/// \brief Stream tab-delimited CSV rows to a borrowed stream, flushing on explicit finish.
[[nodiscard]] inline auto tsv_sink(std::ostream& out, delimited_options options = {})
{
  return data_table_detail::delimited_sink(out, '\t', std::move(options));
}

/// \brief Stream a JSON document. Explicit finish closes its row array, writes summary and flushes.
/// \details An interrupted or failed JSON stream can be incomplete; it is not a checkpoint format.
[[nodiscard]] inline auto json_sink(std::ostream& out, data_json_options options = {})
{
  return data_table_detail::json_sink_adapter(out, std::move(options));
}

/// \brief Independent terminal projection, widths, and destination using the existing display router.
struct data_terminal_options
{
    table_projection projection = {};
    display::stream destination = display::stream::out;
    std::size_t minimum_column_width = 10;
    std::optional<std::size_t> wrap_width = std::nullopt;
    bool show_metadata = true;
    display::sink output = {};
    std::optional<output_policy> policy = std::nullopt;
};

namespace data_table_detail
{
template <typename T> display::formatted_cell terminal_cell(T const& value, data_column_display const& format)
{
  if constexpr (optional_traits<T>::optional)
  {
    if (value) return terminal_cell(*value, format);
    return {.text = styled_text{}.append(format.missing), .decimal_exception = true};
  }
  else
  {
    bool candidate = false;
    bool exception = false;
    if constexpr (Real<T>)
    {
      candidate = uni20::isfinite(value);
      exception = !candidate;
      if constexpr (runtime_real<T>)
        if (value.is_exact() && format.fractions) candidate = false;
    }
    else if constexpr (integer<T>)
      candidate = true;
    else if constexpr (half_integer<T>)
      candidate = !format.fractions;
    return {.text = styled_text{}.append(cell_text(value, format, false)),
            .decimal_candidate = candidate,
            .decimal_exception = exception,
            .keep_together = Real<T> || integer<T> || half_integer<T>};
  }
}

// Decimal digits of the largest exponent a native real type can print, at least two.
template <typename T> std::size_t exponent_digits()
{
  if constexpr (requires { uni20::numeric_limits<T>::max_exponent10; })
  {
    std::size_t digits = 0;
    for (int e = uni20::numeric_limits<T>::max_exponent10; e > 0; e /= 10)
      ++digits;
    return std::max<std::size_t>(digits, 2);
  }
  else
    return 4; // binary128 reaches e+4932; a wider mpreal exponent still grows its column
}

// Display cells for the widest text a real column's format gives at this value's precision, so that a
// streaming column is as wide as its widest value from its first row: the header and the rows already
// written cannot be laid out again.  General notation is a sign, `digits` significant digits, a point and
// an exponent "e-ddd" (its fixed form, "-0.000ddd", is never wider); scientific notation has `digits`
// digits after one leading digit.  A runtime-precision column takes its digits from its format, or else
// from an approximate first value.  Zero, meaning no bound, for fixed notation, exact fractions, a
// runtime-precision column with neither, and columns that are not real.
template <typename V> std::size_t reserved_width([[maybe_unused]] V const* value, data_column_display const& display)
{
  if constexpr (!Real<V>)
    return 0;
  else
  {
    auto const& numeric = display.numeric;
    if (numeric.notation == real_format_notation::fixed) return 0;
    std::size_t digits = 0;
    if constexpr (runtime_real<V>)
    {
#if UNI20_HAS_MPREAL
      if (value && value->is_exact() && display.fractions) return 0; // a fraction has no digit bound
      if (numeric.precision >= 0)
        digits = static_cast<std::size_t>(numeric.precision);
      else if (value && !value->is_exact())
        digits = mpfr_get_str_ndigits(10, value->precision().bit_count());
      else
        return 0; // neither the format nor a missing or exact first value gives a digit count
#else
      return 0;
#endif
    }
    else
      digits = numeric.precision >= 0 ? static_cast<std::size_t>(numeric.precision)
                                      : static_cast<std::size_t>(uni20::numeric_limits<V>::max_digits10());
    std::size_t const exponent = 2 + exponent_digits<V>();
    return numeric.notation == real_format_notation::general ? 1 + digits + 1 + exponent : 3 + digits + exponent;
  }
}

class terminal_sink_adapter {
  public:
    explicit terminal_sink_adapter(data_terminal_options options) : options_(std::move(options)) {}
    template <typename Schema>
    void begin(std::string const& title, Schema const& schema, table_metadata const& metadata, data_sink_start)
    {
      selection_ = resolve_projection(schema, options_.projection);
      table_.emplace(title, options_.destination);
      columns_declared_ = false;
      if (options_.output) table_->output(options_.output, options_.policy.value_or(plain_policy()));
      if (options_.wrap_width) table_->wrap_width(*options_.wrap_width);
      this->emit_metadata(metadata);
    }
    template <typename Schema, typename Row> void row(Schema const& schema, Row const& row)
    {
      // The columns are declared with the first row, whose values give the precision of runtime reals.
      if (!columns_declared_)
      {
        visit_selected(schema, row, selection_, [&](auto const& column, auto const& value, auto const& format) {
          using cell_type = std::remove_cvref_t<decltype(value)>;
          using real_type = data_table_detail::value_type<cell_type>;
          real_type const* present = nullptr;
          if constexpr (optional_traits<cell_type>::optional)
          {
            if (value) present = &*value;
          }
          else
            present = &value;
          auto const reserved = reserved_width<real_type>(present, format);
          table_->column(column.label(), display::width::fit(std::max(options_.minimum_column_width, reserved)),
                         format.alignment);
        });
        columns_declared_ = true;
      }
      std::vector<display::formatted_cell> cells;
      cells.reserve(selection_.indices.size());
      visit_selected(schema, row, selection_, [&](auto const&, auto const& value, auto const& format) {
        cells.push_back(terminal_cell(value, format));
      });
      table_->row(std::move(cells));
    }
    void finish(table_metadata const& summary) { this->emit_metadata(summary); }

  private:
    void emit_metadata(table_metadata const& metadata)
    {
      if (!options_.show_metadata) return;
      for (auto const& [key, value] : metadata)
      {
        auto text = styled_text{}.append(key + ": " + value);
        if (options_.output)
          options_.output(display::event{.destination = options_.destination,
                                         .content = std::move(text),
                                         .newline = true,
                                         .context = {},
                                         .where = std::source_location::current()});
        else
          display::emit(std::move(text), options_.destination);
      }
    }
    data_terminal_options options_;
    resolved_projection selection_;
    std::optional<display::streaming_table> table_;
    bool columns_declared_ = false;
};
} // namespace data_table_detail

/// \brief Stream deferred display formatting through display::streaming_table.
/// \details Uses fit widths and the current display sink; no table history is retained by the adapter. Real
///          columns start as wide as the widest text their format can give, so earlier rows stay aligned.
[[nodiscard]] inline auto terminal_sink(data_terminal_options options = {})
{
  return data_table_detail::terminal_sink_adapter(std::move(options));
}
} // namespace uni20::presentation
