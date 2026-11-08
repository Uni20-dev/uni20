#pragma once

// Experimental implementation: deliberately outside the installed Uni20 headers.
#include <uni20/async/async.hpp>
#include <uni20/async/debug_scheduler.hpp>

#include <chrono>
#include <cstdint>
#include <memory>
#include <optional>
#include <thread>
#include <time.h>

namespace uni20::async::accounting_experiment
{
using Duration = std::chrono::nanoseconds;

enum class Mode
{
  elapsed,
  active_wall,
  cpu_and_wall
};
enum class Outcome
{
  returned,
  failed,
  cancelled,
  interrupted
};

struct SystemClock
{
    static constexpr bool has_cpu =
#ifdef CLOCK_THREAD_CPUTIME_ID
        true;
#else
        false;
#endif

    std::optional<Duration> wall() const noexcept
    {
      return std::chrono::duration_cast<Duration>(std::chrono::steady_clock::now().time_since_epoch());
    }
    std::optional<Duration> cpu() const noexcept
    {
#ifdef CLOCK_THREAD_CPUTIME_ID
      timespec value{};
      if (::clock_gettime(CLOCK_THREAD_CPUTIME_ID, &value) == 0)
        return std::chrono::seconds(value.tv_sec) + Duration(value.tv_nsec);
#endif
      return std::nullopt;
    }
};

template <class Clock = SystemClock> struct Options
{
    Mode mode = Mode::elapsed;
    Clock clock{};
};

struct Report
{
    // Elapsed begins in promise construction, not scheduler admission.
    std::optional<Duration> elapsed{};
    std::optional<Duration> active_wall{};
    std::optional<Duration> own_cpu{};
    std::uint64_t intervals = 0;
    bool body_entered = false;
    Outcome outcome = Outcome::interrupted;
};

struct ReportState
{
    Async<Report> output;
    WriteBuffer<Report> writer = output.write();

    void publish(Report const& report) noexcept
    {
      // This writer never registers as a numerical exception sink. Failed work
      // still publishes a value containing partial measurements.
      writer.emplace_assert(report);
      writer.release();
    }
};

/// \brief Retained observation of sealed statistics, not a resource-lifetime join.
class Receipt {
  public:
    explicit Receipt(std::shared_ptr<ReportState> state) : state_(std::move(state)) {}
    [[nodiscard]] ReadBuffer<Report> read() const { return state_->output.read(); }
    [[nodiscard]] Report wait(IAsyncScheduler& scheduler) const { return this->read().get_wait(scheduler); }

  private:
    std::shared_ptr<ReportState> state_;
};

template <class Clock> class Promise;

template <class Clock = SystemClock> class Task final : public BasicTask {
  public:
    using promise_type = Promise<Clock>;
    Task(Task&&) noexcept = default;
    Task& operator=(Task&&) noexcept = default;
    [[nodiscard]] Receipt receipt() const;

  private:
    explicit Task(TaskHandle handle) noexcept : BasicTask(handle, construction_key{}) {}
    friend Promise<Clock>;
};

template <class Inner, class Clock> struct Awaiter
{
    Inner inner;
    Promise<Clock>* promise;
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

template <class Clock> class Promise final : public TaskPromiseBase {
  public:
    static constexpr TaskDomain task_domain = TaskDomain::host;

    template <class... Args>
    Promise(Options<Clock> const& options, Args&... args)
        : TaskPromiseBase(options, args...), mode_(options.mode), clock_(options.clock),
          state_(std::make_shared<ReportState>()), created_(clock_.wall())
    {
      if (mode_ != Mode::elapsed) report_.active_wall = Duration::zero();
      if (mode_ == Mode::cpu_and_wall && Clock::has_cpu) report_.own_cpu = Duration::zero();
    }

    [[nodiscard]] Task<Clock> get_return_object() noexcept
    {
      return Task<Clock>(this->initialize_task(std::coroutine_handle<Promise>::from_promise(*this)));
    }
    [[nodiscard]] Receipt receipt() const { return Receipt(state_); }

    void start() noexcept
    {
      active_ = true;
      report_.body_entered = true;
      ++report_.intervals;
      if (mode_ != Mode::elapsed) wall_start_ = clock_.wall();
      if (mode_ == Mode::cpu_and_wall)
      {
        thread_ = std::this_thread::get_id();
        cpu_start_ = clock_.cpu();
      }
    }
    void stop() noexcept
    {
      if (mode_ != Mode::elapsed) this->accumulate(report_.active_wall, wall_start_, clock_.wall());
      if (mode_ == Mode::cpu_and_wall)
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
          Promise* promise;
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
      return Awaiter<Inner, Clock>{TaskPromiseBase::await_transform(std::forward<A>(value)), this};
    }
    auto final_suspend() noexcept
    {
      this->stop();
      this->seal(this->is_cancel_on_resume() ? Outcome::cancelled : failed_ ? Outcome::failed : Outcome::returned);
      return TaskPromiseBase::final_suspend();
    }
    void unhandled_exception()
    {
      failed_ = true;
      TaskPromiseBase::unhandled_exception();
    }
    ~Promise() noexcept
    {
      if (!sealed_)
      {
        if (active_) this->stop();
        this->seal(this->is_cancel_on_resume() ? Outcome::cancelled : Outcome::interrupted);
      }
    }

  private:
    static void accumulate(std::optional<Duration>& sum, std::optional<Duration> begin,
                           std::optional<Duration> end) noexcept
    {
      if (sum && begin && end && *end >= *begin)
        *sum += *end - *begin;
      else
        sum.reset();
    }
    void seal(Outcome outcome) noexcept
    {
      sealed_ = true;
      report_.outcome = outcome;
      report_.elapsed = Duration::zero();
      this->accumulate(report_.elapsed, created_, clock_.wall());
      state_->publish(report_);
    }

    Mode mode_;
    Clock clock_;
    std::shared_ptr<ReportState> state_;
    Report report_{};
    std::optional<Duration> created_{};
    std::optional<Duration> wall_start_{};
    std::optional<Duration> cpu_start_{};
    std::thread::id thread_{};
    bool active_ = false;
    bool sealed_ = false;
    bool failed_ = false;
};

template <class Clock> Receipt Task<Clock>::receipt() const
{
  // The constructor is restricted to the actual concrete promise.
  return static_cast<Promise<Clock>&>(this->handle().promise()).receipt();
}

} // namespace uni20::async::accounting_experiment
