#include "run_context.hpp"
#include "terminal.hpp"
#include <algorithm>
#include <chrono>
#include <ctime>
#include <fmt/format.h>
#include <stdexcept>
#include <sys/resource.h>
#include <uni20/buildinfo.hpp>
#include <uni20/core/math.hpp>
#include <utility>

namespace uni20
{
namespace
{
std::optional<long double> difference(std::optional<long double> end, std::optional<long double> begin)
{
  if (!end || !begin || !uni20::isfinite(*end) || !uni20::isfinite(*begin) || *end < *begin) return std::nullopt;
  return *end - *begin;
}
} // namespace

run_clock_sample system_run_clock()
{
  auto wall = std::chrono::duration<long double>(std::chrono::steady_clock::now().time_since_epoch()).count();
  rusage usage{};
  std::optional<long double> cpu;
  if (getrusage(RUSAGE_SELF, &usage) == 0)
    cpu = static_cast<long double>(usage.ru_utime.tv_sec) + usage.ru_utime.tv_usec / 1000000.L +
          static_cast<long double>(usage.ru_stime.tv_sec) + usage.ru_stime.tv_usec / 1000000.L;
  return {wall, cpu};
}

std::string utc_now()
{
  auto now = std::chrono::system_clock::to_time_t(std::chrono::system_clock::now());
  std::tm utc{};
  if (!gmtime_r(&now, &utc)) throw std::runtime_error("cannot read UTC time");
  char result[32];
  if (!std::strftime(result, sizeof(result), "%Y-%m-%dT%H:%M:%SZ", &utc))
    throw std::runtime_error("cannot format UTC time");
  return result;
}

std::string metadata_line(std::string_view text)
{
  std::string result;
  for (unsigned char c : text)
    if (c < 32 || c == 127)
      result += fmt::format("\\x{:02x}", c);
    else
      result += static_cast<char>(c);
  return result;
}

std::string_view to_string(run_outcome value)
{
  switch (value)
  {
    case run_outcome::success:
      return "success";
    case run_outcome::partial:
      return "partial";
    case run_outcome::failed:
      return "failed";
    case run_outcome::cancelled:
      return "cancelled";
  }
  throw std::invalid_argument("invalid run outcome");
}

run_context::run_context(presentation::program_info program, run_context_options options)
    : program_(std::move(program)), options_(std::move(options)), start_(options_.clock())
{
  metadata_.group("run", "Run");
  metadata_.add("run", "program", program_.name, {.label = "Program"});
  metadata_.add("run", "version", program_.version, {.label = "Version"});
  metadata_.add("run", "revision", program_.revision.empty() ? "unknown" : program_.revision,
                {.label = "Application revision"});
  metadata_.add("run", "started_utc", options_.utc(), {.label = "Started (UTC)"});
  if (!options_.invocation.empty())
  {
    std::string command;
    for (auto const& token : options_.invocation)
    {
      if (!command.empty()) command += ' ';
      command += terminal::quote_shell(token);
    }
    metadata_.add("run", "invocation", std::move(command), {.label = "Invocation", .detail = true});
  }
  auto info = build_info::current();
  metadata_.group("build", "Build");
  metadata_.add("build", "uni20_revision", std::string(info.revision), {.label = "Uni20 revision", .detail = true});
  metadata_.add("build", "compiler", std::string(info.cxx_compiler_id) + " " + std::string(info.cxx_compiler_version),
                {.label = "Compiler", .detail = true});
  metadata_.add("build", "build_type", std::string(info.build_type), {.label = "Build type", .detail = true});
  metadata_.add("build", "platform", std::string(info.system_name) + " " + std::string(info.system_processor),
                {.label = "Platform", .detail = true});
}

metadata_document& run_context::metadata()
{
  if (final_) throw std::logic_error("run is finished");
  return metadata_;
}

void run_context::begin_computation()
{
  if (final_ || computation_start_) throw std::logic_error("run finished or computation scope already active");
  computation_start_ = options_.clock();
}

void run_context::end_computation()
{
  auto begin = std::exchange(computation_start_, std::nullopt);
  if (!begin) return;
  try
  {
    auto delta = difference(options_.clock().process_cpu_seconds, begin->process_cpu_seconds);
    if (computation_cpu_ && delta)
      *computation_cpu_ += *delta;
    else
      computation_cpu_.reset();
  }
  catch (...)
  {
    computation_cpu_.reset();
    throw;
  }
}

run_context::computation_scope::computation_scope(run_context& owner) : owner_(&owner) { owner_->begin_computation(); }
run_context::computation_scope::~computation_scope() noexcept
{
  try
  {
    this->finish();
  }
  catch (...)
  {}
}
void run_context::computation_scope::finish()
{
  if (auto* owner = std::exchange(owner_, nullptr)) owner->end_computation();
}

metadata_document const& run_context::finish(run_outcome outcome, metadata_document summary)
{
  if (final_) throw std::logic_error("run summary already finalized");
  if (computation_start_) throw std::logic_error("computation scope still active");
  auto end = options_.clock();
  summary.group("timing", "Timing and outcome");
  summary.add("timing", "outcome", std::string(to_string(outcome)), {.label = "Outcome"});
  summary.add("timing", "compute_cpu_seconds", computation_cpu_,
              {.label = "Computation CPU", .unit = "s", .display = {.numeric = {.precision = 6}}});
  summary.add("timing", "run_cpu_seconds", difference(end.process_cpu_seconds, start_.process_cpu_seconds),
              {.label = "Run CPU", .unit = "s", .display = {.numeric = {.precision = 6}}});
  summary.add("timing", "elapsed_seconds", difference(end.monotonic_seconds, start_.monotonic_seconds),
              {.label = "Elapsed", .unit = "s", .display = {.numeric = {.precision = 6}}});
  final_ = std::move(summary);
  return *final_;
}

presentation::report_builder metadata_report(metadata_document const& document, std::string title, bool detail)
{
  presentation::report_builder report(std::move(title));
  for (auto const& group : document.groups())
  {
    if (std::ranges::none_of(group.fields, [&](auto const& field) { return detail || !field.options.detail; }))
      continue;
    auto& table = report.table(metadata_line(group.label));
    table.column("Field").column("Value").preserve_tokens();
    for (auto const& field : group.fields)
    {
      if (field.options.detail && !detail) continue;
      auto value = field.value.text(field.options.display);
      if (!field.options.unit.empty()) value += " " + field.options.unit;
      table.row(metadata_line(field.options.label), metadata_line(value));
      if (detail && field.options.origin)
      {
        auto const& origin = *field.options.origin;
        table.row("Source",
                  metadata_line(std::string(to_string(origin.source)) + " " + origin.location + " " + origin.key));
      }
    }
  }
  return report;
}
} // namespace uni20
