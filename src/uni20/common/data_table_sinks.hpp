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

class terminal_sink_adapter {
  public:
    explicit terminal_sink_adapter(data_terminal_options options) : options_(std::move(options)) {}
    template <typename Schema>
    void begin(std::string const& title, Schema const& schema, table_metadata const& metadata, data_sink_start)
    {
      selection_ = resolve_projection(schema, options_.projection);
      table_.emplace(title, options_.destination);
      if (options_.output) table_->output(options_.output, options_.policy.value_or(plain_policy()));
      if (options_.wrap_width) table_->wrap_width(*options_.wrap_width);
      for (std::size_t p = 0; p < selection_.indices.size(); ++p)
        visit_column(schema, selection_.indices[p], [&](auto const& column) {
          table_->column(column.label(), display::width::fit(options_.minimum_column_width),
                         selection_.display[p].alignment);
        });
      this->emit_metadata(metadata);
    }
    template <typename Schema, typename Row> void row(Schema const& schema, Row const& row)
    {
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
};
} // namespace data_table_detail

/// \brief Stream deferred display formatting through display::streaming_table.
/// \details Uses incremental fit widths and the current display sink; no table history is retained by the adapter.
[[nodiscard]] inline auto terminal_sink(data_terminal_options options = {})
{
  return data_table_detail::terminal_sink_adapter(std::move(options));
}
} // namespace uni20::presentation
