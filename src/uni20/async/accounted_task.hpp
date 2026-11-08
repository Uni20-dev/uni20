#pragma once

/**
 * \file accounted_task.hpp
 * \brief Optional coroutine-local wall and thread CPU measurements.
 */
#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>

#include <chrono>
#include <concepts>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <time.h>
#include <utility>

namespace uni20::async
{
/// \brief Duration unit of all task-accounting measurements.
using AccountingDuration = std::chrono::nanoseconds;

/// \brief Select clocks for an explicitly accounted task.
enum class AccountingMode
{
  elapsed,
  active_wall,
  cpu_and_wall
};
/// \brief Coroutine outcome, independent of scientific acceptance.
enum class AccountingOutcome
{
  returned,
  failed,
  cancelled,
  interrupted
};

/// \brief Monotonic wall clock and optional current-thread CPU clock.
/// \details Clock failures return an absent sample. No process-wide CPU estimate is substituted.
struct SystemAccountingClock
{
    static constexpr bool has_cpu =
#ifdef CLOCK_THREAD_CPUTIME_ID
        true;
#else
        false;
#endif

    std::optional<AccountingDuration> wall() const noexcept
    {
      return std::chrono::duration_cast<AccountingDuration>(std::chrono::steady_clock::now().time_since_epoch());
    }
    std::optional<AccountingDuration> cpu() const noexcept
    {
#ifdef CLOCK_THREAD_CPUTIME_ID
      timespec value{};
      if (::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) == 0)
        return std::chrono::seconds(value.tv_sec) + AccountingDuration(value.tv_nsec);
#endif
      return std::nullopt;
    }
};

/// \brief Accounting selection and clock source, passed as the first coroutine argument.
/// \details Custom clocks provide noexcept wall()/cpu() optional durations and static has_cpu.
///          Clock state must remain valid until coroutine destruction.
template <class Clock = SystemAccountingClock> struct BasicAccountingOptions
{
    static_assert(
        requires(Clock& clock) {
          { clock.wall() } noexcept -> std::same_as<std::optional<AccountingDuration>>;
          { clock.cpu() } noexcept -> std::same_as<std::optional<AccountingDuration>>;
          { Clock::has_cpu } -> std::convertible_to<bool>;
        }, "Task accounting clocks must supply non-throwing optional wall and CPU samples");

    AccountingMode mode = AccountingMode::elapsed;
    Clock clock{};
};

/// \brief Sealed measurements retained independently of numerical outputs.
/// \details Missing measurements are absent, not zero. CPU and active-wall intervals exclude
///          suspended child work; elapsed includes waiting from promise construction to sealing.
struct AccountingReport
{
    /// \brief Promise construction to sealing, including queue and dependency waits.
    std::optional<AccountingDuration> elapsed{};
    /// \brief Wall duration summed over active intervals, including OS descheduling.
    std::optional<AccountingDuration> active_wall{};
    /// \brief Current-thread CPU duration summed within active intervals.
    std::optional<AccountingDuration> own_cpu{};
    /// \brief Number of entered intervals; ready awaits do not split an interval.
    std::uint64_t intervals = 0;
    /// \brief Coroutine execution began; a numerical kernel need not have been invoked.
    bool body_entered = false;
    AccountingOutcome outcome = AccountingOutcome::interrupted;
};

namespace detail
{
struct AccountingReportState
{
    Async<AccountingReport> output;
    WriteBuffer<AccountingReport> writer = output.write();

    void publish(AccountingReport const& report) noexcept
    {
      // This writer never registers as a numerical exception sink. Failed work
      // still publishes a value containing partial measurements.
      writer.emplace_assert(report);
      writer.release();
    }
};

template <class Clock> class AccountingPromise;
} // namespace detail

/// \brief Retained observation of sealed statistics, not a resource-lifetime join.
class AccountingReceipt {
  public:
    /// \brief Acquire an Async reader of the sealed report, usable with co_await.
    [[nodiscard]] ReadBuffer<AccountingReport> read() const { return state_->output.read(); }
    /// \brief Wait for statistics using the given scheduler; this does not join frame destruction.
    [[nodiscard]] AccountingReport wait(IAsyncScheduler& scheduler) const { return this->read().get_wait(scheduler); }

  private:
    explicit AccountingReceipt(std::shared_ptr<detail::AccountingReportState> state) : state_(std::move(state)) {}
    std::shared_ptr<detail::AccountingReportState> state_;
    template <class Clock> friend class detail::AccountingPromise;
};

/// \brief Move-only host coroutine with a separate accounting promise.
/// \details The first coroutine argument is BasicAccountingOptions<Clock>. Retrieve its
///          receipt before consuming the task. Ordinary AsyncTask has no accounting state.
///          CPU attribution requires cooperative awaits and single-threaded stage work;
///          synchronous nested scheduler waits and parallel work are not attributed.
template <class Clock = SystemAccountingClock> class BasicAccountedTask final : public BasicTask {
  public:
    using promise_type = detail::AccountingPromise<Clock>;
    BasicAccountedTask(BasicAccountedTask&&) noexcept = default;
    BasicAccountedTask& operator=(BasicAccountedTask&&) noexcept = default;
    /// \brief Retain the report before moving or awaiting this task.
    /// \pre This task still owns a nonempty coroutine handle.
    [[nodiscard]] AccountingReceipt receipt() const;

    /// \brief Set a diagnostic label while retaining this task's concrete type.
    BasicAccountedTask& debug_name(std::string const& label)
    {
      this->BasicTask::debug_name(label);
      return *this;
    }

  private:
    explicit BasicAccountedTask(TaskHandle handle) noexcept : BasicTask(handle, construction_key{}) {}
    friend detail::AccountingPromise<Clock>;
};

/// \brief Task options using the system wall and thread CPU clocks.
using AccountingOptions = BasicAccountingOptions<>;
/// \brief Host task with optional system-clock accounting.
using AccountedTask = BasicAccountedTask<>;

namespace detail
{
template <class Inner, class Clock> struct AccountingAwaiter
{
    Inner inner;
    AccountingPromise<Clock>* promise;
    bool stopped = false;

    bool await_ready() { return inner.await_ready(); }
    template <TaskPromise P> decltype(auto) await_suspend(std::coroutine_handle<P> handle)
    {
      stopped = true;
      promise->stop();
      try
      {
        return inner.await_suspend(handle);
      }
      catch (...)
      {
        // Factory num_awaiters() may throw before ownership is published.
        // The language bypasses await_resume() on this path.
        stopped = false;
        promise->start();
        throw;
      }
      // Successful delegation may already have destroyed this frame.
    }
    decltype(auto) await_resume()
    {
      if (stopped) promise->start();
      return inner.await_resume();
    }
};

template <class Clock> class AccountingPromise final : public TaskPromiseBase {
  public:
    static constexpr TaskDomain task_domain = TaskDomain::host;

    template <class... Args>
    AccountingPromise(BasicAccountingOptions<Clock> const& options, Args&... args)
        : TaskPromiseBase(options, args...), mode_(options.mode), clock_(options.clock),
          state_(std::make_shared<AccountingReportState>()), created_(clock_.wall())
    {
      if (mode_ != AccountingMode::elapsed) report_.active_wall = AccountingDuration::zero();
      if (mode_ == AccountingMode::cpu_and_wall && Clock::has_cpu) report_.own_cpu = AccountingDuration::zero();
    }

    [[nodiscard]] BasicAccountedTask<Clock> get_return_object() noexcept
    {
      return BasicAccountedTask<Clock>(
          this->initialize_task(std::coroutine_handle<AccountingPromise>::from_promise(*this)));
    }
    [[nodiscard]] AccountingReceipt receipt() const { return AccountingReceipt(state_); }

    void start() noexcept
    {
      active_ = true;
      report_.body_entered = true;
      ++report_.intervals;
      if (mode_ != AccountingMode::elapsed) wall_start_ = clock_.wall();
      if (mode_ == AccountingMode::cpu_and_wall)
      {
        thread_ = std::this_thread::get_id();
        cpu_start_ = clock_.cpu();
      }
    }
    void stop() noexcept
    {
      if (mode_ != AccountingMode::elapsed) this->accumulate(report_.active_wall, wall_start_, clock_.wall());
      if (mode_ == AccountingMode::cpu_and_wall)
      {
        if (thread_ != std::this_thread::get_id()) report_.own_cpu.reset();
        this->accumulate(report_.own_cpu, cpu_start_, clock_.cpu());
      }
      active_ = false;
    }
    auto initial_suspend() noexcept
    {
      struct Initial
      {
          decltype(std::declval<TaskPromiseBase&>().initial_suspend()) inner;
          AccountingPromise* promise;
          bool await_ready() noexcept { return inner.await_ready(); }
          void await_suspend(std::coroutine_handle<> handle) noexcept { inner.await_suspend(handle); }
          void await_resume() noexcept
          {
            inner.await_resume();
            promise->start();
          }
      };
      return Initial{TaskPromiseBase::initial_suspend(), this};
    }
    template <class A> auto await_transform(A&& value)
    {
      using Inner = decltype(TaskPromiseBase::await_transform(std::forward<A>(value)));
      return AccountingAwaiter<Inner, Clock>{TaskPromiseBase::await_transform(std::forward<A>(value)), this};
    }
    auto final_suspend() noexcept
    {
      this->stop();
      this->seal(this->is_cancel_on_resume() ? AccountingOutcome::cancelled
                 : failed_                   ? AccountingOutcome::failed
                                             : AccountingOutcome::returned);
      return TaskPromiseBase::final_suspend();
    }
    void unhandled_exception()
    {
      failed_ = true;
      TaskPromiseBase::unhandled_exception();
    }
    ~AccountingPromise() noexcept
    {
      if (!sealed_)
      {
        if (active_) this->stop();
        this->seal(this->is_cancel_on_resume() ? AccountingOutcome::cancelled : AccountingOutcome::interrupted);
      }
    }

  private:
    static void accumulate(std::optional<AccountingDuration>& sum, std::optional<AccountingDuration> begin,
                           std::optional<AccountingDuration> end) noexcept
    {
      if (sum && begin && end && *end >= *begin)
        *sum += *end - *begin;
      else
        sum.reset();
    }
    void seal(AccountingOutcome outcome) noexcept
    {
      sealed_ = true;
      report_.outcome = outcome;
      report_.elapsed = AccountingDuration::zero();
      this->accumulate(report_.elapsed, created_, clock_.wall());
      state_->publish(report_);
    }

    AccountingMode mode_;
    Clock clock_;
    std::shared_ptr<AccountingReportState> state_;
    AccountingReport report_{};
    std::optional<AccountingDuration> created_{};
    std::optional<AccountingDuration> wall_start_{};
    std::optional<AccountingDuration> cpu_start_{};
    std::thread::id thread_{};
    bool active_ = false;
    bool sealed_ = false;
    bool failed_ = false;
};

} // namespace detail

template <class Clock> AccountingReceipt BasicAccountedTask<Clock>::receipt() const
{
  // The constructor is restricted to the actual concrete promise.
  return static_cast<detail::AccountingPromise<Clock>&>(this->handle().promise()).receipt();
}

} // namespace uni20::async
