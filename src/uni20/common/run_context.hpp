/**
 * \file run_context.hpp
 * \brief Owned run provenance, typed documents and explicitly scoped computation timing.
 */
#pragma once
#include "metadata.hpp"
#include "program.hpp"
#include <functional>
#include <optional>
#include <string>
#include <string_view>
#include <type_traits>
#include <utility>
#include <vector>

namespace uni20
{
/// \brief One clock sample; unavailable process CPU time is distinct from zero.
struct run_clock_sample
{
    long double monotonic_seconds;
    std::optional<long double> process_cpu_seconds = std::nullopt;
};

[[nodiscard]] run_clock_sample system_run_clock();
[[nodiscard]] std::string utc_now();
/// \brief Escape control bytes for a single human-readable metadata line.
[[nodiscard]] std::string metadata_line(std::string_view text);

enum class run_outcome
{
  success,
  partial,
  failed,
  cancelled
};
[[nodiscard]] std::string_view to_string(run_outcome value);

struct run_context_options
{
    std::function<run_clock_sample()> clock = system_run_clock;
    std::function<std::string()> utc = utc_now;
    /// \brief Empty omits invocation capture; supply owned argv tokens explicitly to opt in.
    std::vector<std::string> invocation = {};
};

/// \brief Parser-independent run state. Snapshots own their metadata and remain stable after edits.
/// \details Timing uses process CPU and monotonic elapsed time. Computation scopes are non-nested,
///          single-caller intervals; they must include asynchronous completion, not just submission.
class run_context {
  public:
    explicit run_context(presentation::program_info program, run_context_options options = {});
    run_context(run_context const&) = delete;
    run_context& operator=(run_context const&) = delete;

    [[nodiscard]] metadata_document& metadata();
    [[nodiscard]] metadata_document snapshot() const { return metadata_; }
    [[nodiscard]] std::vector<std::string> const& invocation() const { return options_.invocation; }
    [[nodiscard]] presentation::program_info const& program() const { return program_; }

    class computation_scope {
      public:
        explicit computation_scope(run_context& owner);
        computation_scope(computation_scope const&) = delete;
        computation_scope& operator=(computation_scope const&) = delete;
        ~computation_scope() noexcept;
        /// \brief End the interval explicitly, reporting clock errors to the caller.
        void finish();

      private:
        run_context* owner_;
    };
    [[nodiscard]] computation_scope computation() { return computation_scope(*this); }
    template <typename F> decltype(auto) measure(F&& function)
    {
      auto scope = this->computation();
      if constexpr (std::is_void_v<std::invoke_result_t<F>>)
      {
        std::invoke(std::forward<F>(function));
        scope.finish();
      }
      else
      {
        decltype(auto) result = std::invoke(std::forward<F>(function));
        scope.finish();
        return result;
      }
    }

    /// \brief Freeze the final summary, sampling before its rendering or output flush.
    /// \details The scientific outcome is independent of any subsequent output-session failure.
    [[nodiscard]] metadata_document const& finish(run_outcome outcome, metadata_document summary = {});
    [[nodiscard]] bool finished() const { return final_.has_value(); }

  private:
    void begin_computation();
    void end_computation();
    presentation::program_info program_;
    run_context_options options_;
    metadata_document metadata_;
    run_clock_sample start_;
    std::optional<run_clock_sample> computation_start_;
    std::optional<long double> computation_cpu_ = 0;
    std::optional<metadata_document> final_;
};

/// \brief Render ordered groups without using display strings as the machine metadata source.
[[nodiscard]] presentation::report_builder metadata_report(metadata_document const& document, std::string title = {},
                                                           bool detail = false);
} // namespace uni20
