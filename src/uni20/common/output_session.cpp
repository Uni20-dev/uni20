#include "output_session.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>
#include <sys/stat.h>
#include <unistd.h>

namespace uni20
{
namespace output_detail
{
void destination::check() const
{
  if (!failures.empty()) std::rethrow_exception(failures.front().exception);
  if (closed || !opened) throw std::logic_error("output destination is not open");
}
void destination::fail(output_operation operation, std::exception_ptr error)
{
  failures.push_back({id, name, options.required, operation, std::move(error)});
  active_table = false;
}
void destination::flush()
{
  stream->flush();
  presentation::data_table_detail::check_output(*stream);
}
void destination::comments(presentation::table_metadata const& fields)
{
  for (auto const& [key, value] : fields)
    *stream << "# " << metadata_line(key) << ": " << metadata_line(value) << '\n';
}
presentation::table_metadata destination::metadata_fields(metadata_document const& metadata) const
{
  auto keys = options.metadata_keys;
  // One destination policy covers snapshots with different field sets, including the final summary.
  std::erase_if(keys, [&](auto const& entry) { return !metadata.find(entry.first); });
  return metadata.strings(keys);
}
void destination::preamble(metadata_document const& metadata)
{
  initial = this->metadata_fields(metadata);
  if (format == output_format::terminal && options.preamble && !metadata.groups().empty())
    *stream << presentation::render_terminal(metadata_report(metadata), options.human_policy,
                                             options.human_policy.output_stream)
            << '\n';
  if (is_commented(format) && options.preamble) this->comments(initial);
  if (format == output_format::named_json) *stream << "{\"tables\":[";
  presentation::data_table_detail::check_output(*stream);
  if (format == output_format::terminal) this->flush();
}

table_sink::adapter table_sink::make_adapter(std::shared_ptr<destination> const& output)
{
  auto& d = *output;
  switch (d.format)
  {
    case output_format::csv:
    case output_format::tsv:
    case output_format::commented_csv:
    case output_format::commented_tsv:
      return presentation::data_table_detail::delimited_sink(
          *d.stream, d.format == output_format::csv || d.format == output_format::commented_csv ? ',' : '\t',
          {.projection = d.options.projection});
    case output_format::json:
    case output_format::named_json:
      return presentation::json_sink(*d.stream, {.columns = d.options.projection.columns});
    case output_format::terminal:
      return presentation::terminal_sink({.projection = d.options.projection,
                                          .show_metadata = false,
                                          .output =
                                              [output](display::event const& event) {
                                                auto policy = output->options.human_policy;
                                                if (event.preserve_layout) policy.wrap_width = std::nullopt;
                                                std::visit(
                                                    [&](auto const& content) {
                                                      *output->stream << presentation::render_terminal(
                                                          content, policy, policy.output_stream);
                                                    },
                                                    event.content);
                                                if (event.newline) *output->stream << '\n';
                                                presentation::data_table_detail::check_output(*output->stream);
                                              },
                                          .policy = d.options.human_policy});
  }
  throw std::invalid_argument("unknown output format");
}
table_sink::table_sink(std::shared_ptr<destination> output, std::string name)
    : output_(std::move(output)), name_(std::move(name)), sink_(make_adapter(output_))
{}
void table_sink::finish(presentation::table_metadata const& summary)
{
  this->attempt(output_operation::table_finish, [&] {
    std::visit([&](auto& sink) { sink.finish(summary); }, sink_);
    auto& d = *output_;
    if (d.format == output_format::named_json) *d.stream << '}';
    if (is_commented(d.format) && d.options.preamble) d.comments(summary);
    d.active_table = false;
  });
}
} // namespace output_detail

std::size_t output_session::file(std::filesystem::path path, output_format format, output_destination_options options)
{
  if (finished_) throw std::logic_error("output session is finished");
  if (path.empty()) throw std::invalid_argument("empty output path");
  auto id = outputs_.size();
  return this->register_destination(std::make_shared<output_detail::destination>(output_detail::destination{
      .id = id, .name = path.string(), .format = format, .options = std::move(options), .path = std::move(path)}));
}
std::size_t output_session::stream(std::string name, std::shared_ptr<std::ostream> out, output_format format,
                                   output_destination_options options, std::function<void()> close)
{
  if (finished_) throw std::logic_error("output session is finished");
  if (!out) throw std::invalid_argument("null output stream");
  auto id = outputs_.size();
  return this->register_destination(
      std::make_shared<output_detail::destination>(output_detail::destination{.id = id,
                                                                              .name = std::move(name),
                                                                              .format = format,
                                                                              .options = std::move(options),
                                                                              .stream = std::move(out),
                                                                              .close = std::move(close)}));
}
std::size_t output_session::standard_output(output_format format, output_destination_options options)
{
  if (finished_) throw std::logic_error("output session is finished");
  return this->register_destination(std::make_shared<output_detail::destination>(
      output_detail::destination{.id = outputs_.size(),
                                 .name = "stdout",
                                 .format = format,
                                 .options = std::move(options),
                                 .stream = std::shared_ptr<std::ostream>(&std::cout, [](auto*) {}),
                                 .suppressed = quiet_}));
}
std::size_t output_session::register_destination(std::shared_ptr<output_detail::destination> destination)
{
  auto id = destination->id;
  outputs_.push_back(std::move(destination));
  if (std::ranges::any_of(outputs_, [](auto const& d) { return d->opened; }))
  {
    // Once output has started, a rejected addition must not prevent existing streams from finishing.
    try
    {
      this->preflight();
    }
    catch (...)
    {
      outputs_.pop_back();
      throw;
    }
  }
  return id;
}
void output_session::preflight() const
{
  struct stat stdout_stat
  {};
  bool have_stdout = fstat(STDOUT_FILENO, &stdout_stat) == 0;
  for (std::size_t i = 0; i < outputs_.size(); ++i)
  {
    auto const& a = *outputs_[i];
    if (a.suppressed) continue;
    if (!a.path.empty())
    {
      struct stat file_stat
      {};
      if (have_stdout && stat(a.path.c_str(), &file_stat) == 0 && file_stat.st_dev == stdout_stat.st_dev &&
          file_stat.st_ino == stdout_stat.st_ino)
        throw std::invalid_argument("output path aliases stdout: " + a.name);
      if (std::filesystem::exists(a.path) && !std::filesystem::is_regular_file(a.path))
        throw std::invalid_argument("output path is not a regular file: " + a.name);
    }
    for (std::size_t j = 0; j < i; ++j)
    {
      auto const& b = *outputs_[j];
      if (b.suppressed) continue;
      bool alias = a.stream && b.stream && a.stream->rdbuf() == b.stream->rdbuf();
      if (!a.path.empty() && !b.path.empty())
      {
        std::error_code error;
        alias = alias || std::filesystem::weakly_canonical(a.path) == std::filesystem::weakly_canonical(b.path) ||
                std::filesystem::equivalent(a.path, b.path, error);
      }
      if (alias) throw std::invalid_argument("output destinations alias: " + a.name + " and " + b.name);
    }
  }
}

output_report output_session::open()
{
  if (finished_) throw std::logic_error("output session is finished");
  this->preflight();
  for (auto const& d : outputs_)
  {
    if (d->opened || d->suppressed || !d->failures.empty()) continue;
    try
    {
      if (!d->path.empty())
      {
        auto file = std::make_shared<std::ofstream>();
        file->exceptions(std::ios::badbit | std::ios::failbit);
        file->open(d->path, std::ios::out | (d->options.overwrite ? std::ios::trunc : std::ios::noreplace));
        d->stream = file;
        d->close = [file] { file->close(); };
      }
      d->opened = true;
    }
    catch (...)
    {
      d->fail(output_operation::open, std::current_exception());
      continue;
    }
    try
    {
      d->preamble(initial_);
    }
    catch (...)
    {
      d->fail(output_operation::begin, std::current_exception());
    }
  }
  return this->checked_report();
}

std::vector<std::shared_ptr<output_detail::destination>>
output_session::select(std::vector<std::size_t> const& ids) const
{
  if (ids.empty()) return outputs_;
  std::set<std::size_t> seen;
  std::vector<std::shared_ptr<output_detail::destination>> result;
  for (auto id : ids)
  {
    if (id >= outputs_.size() || !seen.insert(id).second) throw std::invalid_argument("invalid or repeated output id");
    result.push_back(outputs_[id]);
  }
  return result;
}

output_report output_session::report() const
{
  output_report result;
  for (auto const& d : outputs_)
    result.failures.insert(result.failures.end(), d->failures.begin(), d->failures.end());
  return result;
}
output_report output_session::checked_report() const
{
  auto result = this->report();
  if (std::ranges::any_of(result.failures, &output_failure::required)) throw output_error(std::move(result));
  return result;
}
output_report output_session::flush()
{
  if (finished_) return this->checked_report();
  try
  {
    (void)this->open();
  }
  catch (output_error const&)
  {}
  for (auto const& d : outputs_)
    if (d->opened && !d->suppressed && d->failures.empty()) try
      {
        d->flush();
      }
      catch (...)
      {
        d->fail(output_operation::flush, std::current_exception());
      }
  return this->checked_report();
}
output_report output_session::finish(metadata_document summary)
{
  if (finished_) return this->checked_report();
  try
  {
    (void)this->open();
  }
  catch (output_error const&)
  {}
  finished_ = true;
  for (auto const& d : outputs_)
  {
    if (d->suppressed || !d->opened) continue;
    if (d->failures.empty()) try
      {
        if (d->active_table) throw std::logic_error("finish the table before the output session");
        if (d->format == output_format::json && d->tables.empty())
          throw std::logic_error("JSON output requires one table");
        auto fields = d->metadata_fields(summary);
        if (d->format == output_format::named_json)
        {
          *d->stream << "],\"summary\":";
          presentation::data_table_detail::json_output out{*d->stream};
          presentation::data_table_detail::write_json_metadata(out, fields);
          *d->stream << ",\"status\":\"complete\"}";
        }
        if (d->format == output_format::terminal && !summary.groups().empty())
          *d->stream << presentation::render_terminal(metadata_report(summary), d->options.human_policy,
                                                      d->options.human_policy.output_stream)
                     << '\n';
        if (output_detail::is_commented(d->format) && d->options.preamble) d->comments(fields);
        presentation::data_table_detail::check_output(*d->stream);
      }
      catch (...)
      {
        d->fail(output_operation::finish, std::current_exception());
      }
    // A failed write can also fail flush/close. Preserve both instead of replacing the original error.
    try
    {
      d->flush();
    }
    catch (...)
    {
      d->fail(output_operation::flush, std::current_exception());
    }
    try
    {
      if (d->close) d->close();
    }
    catch (...)
    {
      d->fail(output_operation::close, std::current_exception());
    }
    d->closed = true;
  }
  return this->checked_report();
}
} // namespace uni20
