/**
 * \file output_session.hpp
 * \brief Owned output destinations layered on typed table subscriptions.
 */
#pragma once
#include "data_table_sinks.hpp"
#include "run_context.hpp"
#include <filesystem>
#include <fstream>
#include <set>

namespace uni20
{
enum class output_format
{
  terminal,
  csv,
  tsv,
  commented_tsv,
  json,
  named_json
};
enum class output_operation
{
  open,
  begin,
  row,
  table_finish,
  flush,
  finish,
  close
};

struct output_destination_options
{
    bool required = true;
    bool overwrite = false;
    bool flush_each_row = false;
    bool preamble = true;
    presentation::table_projection projection = {};
    presentation::output_policy human_policy = presentation::plain_policy();
};

struct output_failure
{
    std::size_t destination;
    std::string name;
    bool required;
    output_operation operation;
    std::exception_ptr exception;
};
struct output_report
{
    std::vector<output_failure> failures = {};
};
class output_error : public std::runtime_error {
  public:
    explicit output_error(output_report report)
        : std::runtime_error("required run output failed"), report_(std::move(report))
    {}
    [[nodiscard]] output_report const& report() const { return report_; }

  private:
    output_report report_;
};

namespace output_detail
{
struct destination
{
    std::size_t id;
    std::string name;
    output_format format;
    output_destination_options options;
    std::filesystem::path path = {};
    std::shared_ptr<std::ostream> stream = {};
    std::function<void()> close = {};
    bool opened = false;
    bool active_table = false;
    bool closed = false;
    bool suppressed = false;
    std::set<std::string> tables = {};
    std::vector<output_failure> failures = {};
    presentation::table_metadata initial = {};

    void check() const;
    void fail(output_operation operation, std::exception_ptr error);
    void flush();
    void comments(presentation::table_metadata const& fields);
    void preamble(metadata_document const& metadata);
};

/// One adapter is owned by each table subscription; it keeps the destination alive independently of the session.
class table_sink {
  public:
    table_sink(std::shared_ptr<destination> output, std::string name);
    template <typename Schema>
    void begin(std::string const& title, Schema const& schema, presentation::table_metadata const& metadata,
               presentation::data_sink_start start)
    {
      this->attempt(output_operation::begin, [&] {
        auto& d = *output_;
        if (d.active_table) throw std::logic_error("previous output table is still active");
        auto combined = d.initial;
        for (auto const& [key, value] : metadata)
          if (!combined.emplace(key, value).second) throw std::invalid_argument("table/run metadata collision: " + key);
        if (d.format == output_format::named_json)
        {
          if (d.tables.size() > 1) *d.stream << ',';
          *d.stream << "{\"name\":";
          presentation::data_table_detail::write_json_string(*d.stream, name_);
          *d.stream << ",\"data\":";
        }
        if (d.format == output_format::commented_tsv && d.options.preamble)
        {
          d.comments(metadata);
          if (start.first_row) d.comments({{"first_row", std::to_string(start.first_row)}});
        }
        d.active_table = true;
        std::visit([&](auto& sink) { sink.begin(title, schema, combined, start); }, sink_);
      });
    }
    template <typename Schema, typename Row> void row(Schema const& schema, Row const& row)
    {
      this->attempt(output_operation::row, [&] {
        std::visit([&](auto& sink) { sink.row(schema, row); }, sink_);
        if (output_->format == output_format::terminal || output_->options.flush_each_row) output_->flush();
      });
    }
    void finish(presentation::table_metadata const& summary);

  private:
    template <typename F> void attempt(output_operation operation, F&& fn)
    {
      output_->check();
      try
      {
        fn();
        presentation::data_table_detail::check_output(*output_->stream);
      }
      catch (...)
      {
        output_->fail(operation, std::current_exception());
        throw;
      }
    }
    using adapter = std::variant<presentation::data_table_detail::delimited_sink,
                                 presentation::data_table_detail::json_sink_adapter,
                                 presentation::data_table_detail::terminal_sink_adapter>;
    static adapter make_adapter(std::shared_ptr<destination> const& output);
    std::shared_ptr<destination> output_;
    std::string name_;
    adapter sink_;
};
} // namespace output_detail

/// \brief An explicitly finalized set of owned result streams, with no global display changes.
/// \details Single-caller. Finish tables before finishing the session. Destruction only releases resources;
///          it never certifies successful output. Failures disable the destination, all other destinations
///          are attempted, and required failures are thrown after delivery. Rows are never retried.
class output_session {
  public:
    explicit output_session(metadata_document initial = {}, bool quiet = false)
        : initial_(std::move(initial)), quiet_(quiet)
    {}
    output_session(output_session const&) = delete;
    output_session& operator=(output_session const&) = delete;
    output_session(output_session&&) = default;
    output_session& operator=(output_session&&) = delete;

    std::size_t file(std::filesystem::path path, output_format format, output_destination_options options = {});
    /// \brief Retain a shared stream owner. An optional explicit closer participates in finish error reporting.
    std::size_t stream(std::string name, std::shared_ptr<std::ostream> out, output_format format,
                       output_destination_options options = {}, std::function<void()> close = {});
    std::size_t standard_output(output_format format = output_format::terminal,
                                output_destination_options options = {});

    /// \brief Preflight aliases before opening any files, then emit the human/comment preamble once.
    output_report open();

    /// \brief Replay selected destinations, then subscribe them to future rows.
    template <presentation::DataTableValue... Ts>
    output_report attach(presentation::data_table<Ts...>& table, std::string name,
                         presentation::sink_replay replay = presentation::sink_replay::all,
                         std::vector<std::size_t> destinations = {})
    {
      return this->attach_impl(table, std::move(name), replay, std::move(destinations), std::nullopt);
    }

    /// \brief Export retained rows, closing only these subscriptions and leaving the source table appendable.
    /// \details No row copies are made. An already finished table replays its original summary.
    template <presentation::DataTableValue... Ts>
    output_report write_table(presentation::data_table<Ts...>& table, std::string name,
                              presentation::table_metadata summary = {})
    {
      if (table.retention_policy() != presentation::retention::all)
        throw std::logic_error("snapshot export requires retained history");
      if (table.finished() && !summary.empty() && summary != *table.summary())
        throw std::invalid_argument("finished table summary cannot be changed by a snapshot export");
      return this->attach_impl(table, std::move(name), presentation::sink_replay::all, {}, std::move(summary));
    }

    /// \brief Flush all active destinations without ending tables or writing summaries. Does not fsync.
    output_report flush();
    /// \brief Write the run summary, flush and close every destination, preserving original and cleanup failures.
    /// \details Repeated calls return/rethrow the recorded report without repeating any writes.
    output_report finish(metadata_document summary = {});
    [[nodiscard]] output_report report() const;

  private:
    template <presentation::DataTableValue... Ts>
    output_report attach_impl(presentation::data_table<Ts...>& table, std::string name,
                              presentation::sink_replay replay, std::vector<std::size_t> destinations,
                              std::optional<presentation::table_metadata> snapshot_summary)
    {
      if (finished_) throw std::logic_error("output session is finished");
      metadata_detail::dt::validate_identifier(name);
      if (replay == presentation::sink_replay::all && table.retention_policy() == presentation::retention::none &&
          table.size())
        throw std::logic_error("table history is unavailable; request future_only explicitly");
      auto selected = this->select(destinations);
      for (auto const& d : selected)
      {
        if (d->active_table || d->tables.contains(name))
          throw std::logic_error("table already attached or previous table active");
        if (!d->tables.empty() && d->format != output_format::terminal && d->format != output_format::named_json)
          throw std::logic_error("multiple tables require terminal or named_json output");
      }
      try
      {
        (void)this->open();
      }
      catch (output_error const&)
      {} // Still attempt healthy destinations.
      for (auto const& d : selected)
      {
        if (d->suppressed || !d->failures.empty()) continue;
        d->tables.insert(name);
        try
        {
          auto attachment =
              table.attach(output_detail::table_sink(d, name), {.required = d->options.required, .replay = replay});
          if (snapshot_summary && !table.finished()) (void)table.finish_sink(attachment.id, *snapshot_summary);
        }
        catch (presentation::data_delivery_error const&)
        {} // table has already recorded/disabled this sink.
      }
      return this->checked_report();
    }

    std::vector<std::shared_ptr<output_detail::destination>> select(std::vector<std::size_t> const& ids) const;
    void preflight() const;
    output_report checked_report() const;
    metadata_document initial_;
    bool quiet_ = false;
    bool finished_ = false;
    std::vector<std::shared_ptr<output_detail::destination>> outputs_;
};
} // namespace uni20
