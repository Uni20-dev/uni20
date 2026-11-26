/// \file debug_scheduler.hpp
/// \brief Coroutine scheduler for AsyncTask.

#pragma once

#include "async.hpp"
#include "cuda_task.hpp"
#include "scheduler.hpp"
#include "task_registry.hpp"
#include <algorithm>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <numeric>
#include <random>
#include <thread>
#include <uni20/common/display.hpp>
#include <utility>
#include <vector>

namespace uni20::async
{

namespace detail
{
inline void service_task_registry_debug_requests()
{
#if UNI20_DEBUG_ASYNC_TASKS
  TaskRegistry::service_debug_requests();
#endif
}

inline void dump_deadlock_graphviz_snapshot()
{
#if UNI20_DEBUG_ASYNC_TASKS
  auto const path = TaskRegistry::default_graphviz_dump_path();
  if (TaskRegistry::dump_graphviz_file_best_effort(path))
  {
    display::info("Async Graphviz DAG snapshot written to {}", path);
  }
#endif
}
} // namespace detail

/// \brief Runnable-batch ordering policy for `DebugScheduler`.
enum class DebugSchedulerOrder
{
  /// Execute each runnable batch in submission order.
  fifo,
  /// Execute each runnable batch in reverse submission order.
  reverse,
  /// Execute each runnable batch in seeded pseudo-random order.
  random,
};

/// \brief Configuration for deterministic debug scheduling.
struct DebugSchedulerOptions
{
    /// Runnable-batch ordering policy.
    DebugSchedulerOrder order = DebugSchedulerOrder::reverse;
    /// Initial state for the reproducible random policy.
    std::uint64_t random_seed = 0;
};

/// \brief How `DebugScheduler::run_all()` treats tasks that wait on another thread.
enum class ExternalWaitPolicy
{
  /// Abort if any task begins a wait on another thread during the call. The default.
  reject,
  /// Return normally; the caller accepts that such tasks may still be suspended.
  allow
};

/// \brief Single-threaded scheduler with configurable runnable-batch ordering.
/// \details Tasks run only on the thread that drives the scheduler. Other
///          threads, such as CUDA host callbacks, may reschedule tasks that
///          await an external wakeup registered through
///          `BasicTask::begin_external_wait`. Waits block for those wakeups
///          rather than reporting a deadlock; `run_all()` does not, and aborts
///          when a task began such a wait during the call unless told otherwise.
class DebugScheduler : public IAsyncScheduler {
  public:
    /// \brief Default-construct an empty scheduler.
    explicit DebugScheduler(DebugSchedulerOptions options = {}) : options_(options), random_engine_(options.random_seed)
    {}

    /// \brief Destructor.
    ~DebugScheduler()
    {
      TRACE_MODULE(ASYNC, "~DebugScheduler", Handles_.size(), external_waits_);
      // A pending external wakeup would later reschedule into a destroyed scheduler.
      if (std::uncaught_exceptions() == 0)
        CHECK(external_waits_ == 0, "DebugScheduler destroyed while tasks await external wakeups", external_waits_);

      if (std::uncaught_exceptions() > 0)
      {
        while (!Handles_.empty())
        {
          Handles_.back().abandon_leak();
          Handles_.pop_back();
        }
      }
    }

    /// \brief Reports whether at least one runnable task exists.
    /// \return `true` when the scheduler is not paused and has queued tasks.
    [[nodiscard]] bool can_run() const noexcept
    {
      std::lock_guard lock(mutex_);
      return !Blocked_ && !Handles_.empty();
    }

    /// \brief Run one batch of scheduled coroutines in the configured order.
    void run();

    /// \brief Run until no runnable tasks remain.
    /// \details A debugging aid. It returns while tasks are suspended outside
    ///          the scheduler, for example on a CUDA resource or I/O, so it
    ///          does not establish completion; wait on a result with
    ///          `get_wait()` or `wait_for()` for that.
    ///
    ///          Whether such a task has been woken by the time `run_all()`
    ///          returns depends on when another thread delivers the wakeup, so
    ///          a test that reads a result afterwards passes or fails by
    ///          timing. Any task that begins an external wait during the call
    ///          therefore makes it abort, whether or not the wakeup happened to
    ///          arrive in time. A task that was already waiting when the call
    ///          began is not counted. The check fires only on calls in which a
    ///          task actually suspends, so it turns a flake into a failure but
    ///          does not make a misused call fail every time.
    /// \param external_waits `ExternalWaitPolicy::allow` for a caller that
    ///        deliberately leaves such tasks suspended, for example to check
    ///        that a result is still pending while a resource is held.
    /// \note Failing with `PANIC` rather than an exception avoids unwinding a
    ///       test whose tasks still hold CUDA leases, which would end in an
    ///       unrelated resource-pool check that hides this message.
    void run_all(ExternalWaitPolicy external_waits = ExternalWaitPolicy::reject);

    /// \brief Block the scheduler from running.
    /// \note This turns run() and run_all() into no-operation;
    void pause() override { Blocked_ = true; }

    /// \brief Unblock the scheduler
    void resume() override { Blocked_ = false; }

    /// \brief Drives one runnable batch, or waits for an external wakeup, while waiting for readiness.
    /// \param is_ready Predicate that reports target readiness.
    void help_while_waiting(const WaitPredicate& is_ready) override
    {
      // Read before the readiness and queue checks below. A live snapshot on another thread can hold
      // back this wait's target without queuing anything here, and what it releases can make the
      // target ready, or wake a task, without queuing anything either (see
      // TaskRegistry::snapshots_in_progress()). So while a snapshot runs, an empty queue or a paused
      // scheduler is neither a deadlock nor a reason to park. A snapshot that has already ended
      // published its wakeups first, so the checks below see them.
      bool const snapshot_running = TaskRegistry::snapshots_in_progress();
      if (is_ready())
      {
        return;
      }

      if (Blocked_ && snapshot_running)
      {
        // Paused, so nothing runs here, but the snapshot may still release what the target needs.
        std::this_thread::yield();
        return;
      }

      bool deadlocked = Blocked_;
      if (!deadlocked)
      {
        std::unique_lock lock(mutex_);
        if (Handles_.empty())
        {
          if (snapshot_running)
          {
            // Let the snapshot finish, then return to the caller's readiness loop.
            lock.unlock();
            std::this_thread::yield();
            return;
          }
          if (external_waits_ == 0)
          {
            deadlocked = true;
          }
          else
          {
            this->wait_for_runnable_or_no_external_waits(lock);
            return;
          }
        }
      }
      if (deadlocked)
      {
        detail::dump_deadlock_graphviz_snapshot();
        TaskRegistry::dump();
        PANIC("**DEADLOCK** get_wait object is not available but there are no runnable tasks!");
      }
      this->run();
    }

    /// \brief Check whether the runnable queue is empty.
    /// \return `true` if the scheduler has no queued tasks. Tasks awaiting an
    ///         external wakeup are not queued; see `external_waits()`.
    [[nodiscard]] bool done() const noexcept
    {
      std::lock_guard lock(mutex_);
      return Handles_.empty();
    }

    /// \brief Return the number of tasks awaiting a wakeup from outside the scheduler.
    [[nodiscard]] std::size_t external_waits() const noexcept
    {
      std::lock_guard lock(mutex_);
      return external_waits_;
    }

    void begin_external_wait() noexcept override
    {
      std::lock_guard lock(mutex_);
      ++external_waits_;
      ++external_waits_begun_;
    }

    void complete_external_wait(BasicTask&& task) override
    {
      TRACE_MODULE(ASYNC, "Completing an external wait", &task, task.coroutine_handle());
      // Requeue, release the wait and notify in one critical section. The driving thread may run
      // the task to completion and destroy this scheduler as soon as the lock is released.
      std::lock_guard lock(mutex_);
      CHECK(external_waits_ > 0, "DebugScheduler external wait completed without a matching begin");
      if (task) Handles_.push_back(std::move(task));
      --external_waits_;
      wakeup_.notify_all();
    }

    /// \brief Return the scheduler's immutable ordering configuration.
    [[nodiscard]] DebugSchedulerOptions const& options() const noexcept { return options_; }

  private:
    /// \brief Enqueue a task for later run.
    /// \param task Admitted host task to bind to this scheduler.
    void schedule_host(BasicTask&& task) override
    {
      TRACE_MODULE(ASYNC, "Scheduling a task", &task, task.coroutine_handle());
      TaskRegistry::record_task_scheduled(task.coroutine_handle());
      if (task.set_scheduler(this))
      {
        this->enqueue_task(std::move(task));
      }
    }

    void execute_batch_impl(LightweightTaskBatch const& batch) override
    {
      std::vector<std::size_t> indices(batch.size());
      std::iota(indices.begin(), indices.end(), std::size_t{0});
      this->order_batch(indices);
      for (std::size_t const index : indices)
        batch(index);
    }

    bool accepts_route(TaskRoute route) const noexcept override
    {
      return route.domain == TaskDomain::host && !route.cuda_device;
    }

    bool can_direct_transfer(TaskHandle, TaskHandle) const noexcept override { return true; }

    // Internal resubmission
    void reschedule(BasicTask&& task) override
    {
      TRACE_MODULE(ASYNC, "Rescheduling a task", &task, task.coroutine_handle());
      // Assume sched_ is already set
      this->enqueue_task(std::move(task));
    }

    /// \brief Block until a task is runnable or no external wakeup is pending.
    void wait_for_runnable_or_no_external_waits(std::unique_lock<std::mutex>& lock)
    {
      wakeup_.wait(lock, [this] { return !Handles_.empty() || external_waits_ == 0; });
    }

    bool Blocked_ = false;

    // Guards Handles_ and external_waits_: external wakeups enqueue from other threads.
    mutable std::mutex mutex_;
    std::condition_variable wakeup_;
    std::size_t external_waits_ = 0;
    // Every begin_external_wait() ever made, so run_all() can tell whether the call began any.
    std::size_t external_waits_begun_ = 0;
    std::vector<BasicTask> Handles_;
    DebugSchedulerOptions options_;
    std::mt19937_64 random_engine_;

    template <class T> void order_batch(std::vector<T>& batch)
    {
      switch (options_.order)
      {
        case DebugSchedulerOrder::fifo:
          return;
        case DebugSchedulerOrder::reverse:
          std::reverse(batch.begin(), batch.end());
          return;
        case DebugSchedulerOrder::random:
          // Keep the permutation stable across standard-library implementations.
          for (std::size_t remaining = batch.size(); remaining > 1; --remaining)
          {
            auto const selected = static_cast<std::size_t>(random_engine_() % remaining);
            std::swap(batch[remaining - 1], batch[selected]);
          }
          return;
      }
      PANIC("invalid DebugScheduler ordering policy", static_cast<int>(options_.order));
    }

  protected:
    /// \brief Add one already-bound task to the runnable queue.
    /// \details Safe to call from any thread.
    void enqueue_task(BasicTask&& task)
    {
      std::lock_guard lock(mutex_);
      Handles_.push_back(std::move(task));
      wakeup_.notify_all();
    }

    /// \brief Resume one queued task in the execution context selected by this scheduler.
    virtual void resume_task(BasicTask& task) { task.resume(); }
};

namespace detail
{
/// \brief Process-wide default debug scheduler instance.
inline DebugScheduler DefaultScheduler;
/// \brief Global scheduler pointer used by free `schedule(...)` helpers.
inline IAsyncScheduler* global_scheduler = &DefaultScheduler;
} // namespace detail

/// \brief Sets the process-wide scheduler used by async helpers.
/// \param sched Scheduler pointer to install.
inline void set_global_scheduler(IAsyncScheduler* sched) { detail::global_scheduler = sched; }

/// \brief Returns the currently-installed process-wide scheduler.
/// \return Pointer to the active scheduler.
inline IAsyncScheduler* get_global_scheduler() { return detail::global_scheduler; }

/// \brief Restores the process-wide scheduler to the default debug scheduler.
inline void reset_global_scheduler() { detail::global_scheduler = &detail::DefaultScheduler; }

// ScopedScheduler is useful for testing; set the scheduler for the lifetime of a block
/// \brief RAII helper that temporarily overrides the global scheduler.
class ScopedScheduler {
  public:
    /// \brief Install a temporary global scheduler until destruction.
    /// \param sched Scheduler pointer to activate.
    explicit ScopedScheduler(IAsyncScheduler* sched)
    {
      old_ = get_global_scheduler();
      set_global_scheduler(sched);
    }
    /// \brief Restore the previous global scheduler.
    ~ScopedScheduler() { set_global_scheduler(old_); }

  private:
    IAsyncScheduler* old_;
};

/// \brief Schedule a task on the currently configured global scheduler.
/// \param task Task to schedule.
template <HostTask Task> void schedule(Task&& task) { get_global_scheduler()->schedule(std::move(task)); }

/// \brief Execute a synchronous lightweight batch on the global scheduler.
/// \tparam Function Callable accepting one `std::size_t` index.
/// \param size Number of work items.
/// \param function Work-item callable borrowed until the call returns.
template <class Function>
  requires std::invocable<Function&, std::size_t>
void execute_batch(std::size_t size, Function&& function)
{
  get_global_scheduler()->execute_batch(size, std::forward<Function>(function));
}

/// \brief Schedule a CUDA task on the currently configured global scheduler.
/// \details The installed scheduler must implement CUDA task admission.
/// \param task CUDA task to schedule without establishing device affinity.
inline void schedule(CudaTask&& task)
{
  auto* scheduler = dynamic_cast<ICudaScheduler*>(get_global_scheduler());
  CHECK(scheduler != nullptr, "the global scheduler does not accept CUDA tasks");
  scheduler->schedule(std::move(task));
}

/// \brief Schedule a CUDA task with initial device affinity.
/// \details Initial admission to the correct device avoids a later scheduler
///          migration when the task first selects that device.
/// \param task CUDA task to schedule.
/// \param device CUDA runtime device ordinal.
inline void schedule(CudaTask&& task, int device)
{
  auto* scheduler = dynamic_cast<ICudaScheduler*>(get_global_scheduler());
  CHECK(scheduler != nullptr, "the global scheduler does not accept CUDA tasks");
  scheduler->schedule(std::move(task), device);
}

/// \brief Wait for a reader context using the global scheduler.
/// \tparam T Stored value type.
template <typename T> void EpochContextReader<T>::wait() const
{
  auto* sched = get_global_scheduler();
  if (!this->ready())
  {
    CHECK(sched);
    auto epoch = epoch_;
    sched->wait_for(IAsyncScheduler::WaitRequest{
        .is_ready = [epoch] { return epoch->reader_ready(); },
        .notify_when_ready =
            [epoch](IAsyncScheduler::WaitWakeup notify) { epoch->reader_notify_when_ready(std::move(notify)); },
    });
  }
}

/// \brief Wait for a reader context using an explicit scheduler.
/// \tparam T Stored value type.
/// \param sched Scheduler used to drive progress.
template <typename T> void EpochContextReader<T>::wait(IAsyncScheduler& sched) const
{
  if (!this->ready())
  {
    auto epoch = epoch_;
    sched.wait_for(IAsyncScheduler::WaitRequest{
        .is_ready = [epoch] { return epoch->reader_ready(); },
        .notify_when_ready =
            [epoch](IAsyncScheduler::WaitWakeup notify) { epoch->reader_notify_when_ready(std::move(notify)); },
    });
  }
}

/// \brief Wait for a reader context using the global scheduler.
/// \tparam T Stored value type.
/// \return Reference to the ready value.
template <typename T> T const& EpochContextReader<T>::get_wait() const
{
  this->wait();
  return this->data();
}

/// \brief Wait for a reader context using an explicit scheduler.
/// \tparam T Stored value type.
/// \param sched Scheduler used to drive progress.
/// \return Reference to the ready value.
template <typename T> T const& EpochContextReader<T>::get_wait(IAsyncScheduler& sched) const
{
  this->wait(sched);
  return this->data();
}

/// \brief Wait for a writer context and move out the stored value.
/// \tparam T Stored value type.
/// \return Moved value from writer storage.
template <typename T> T&& EpochContextWriter<T>::move_from_wait()
{
  auto* sched = get_global_scheduler();
  if (!this->ready())
  {
    CHECK(sched);
    auto epoch = this->epoch_context_shared();
    sched->wait_for(IAsyncScheduler::WaitRequest{
        .is_ready = [this] { return this->ready(); },
        .notify_when_ready =
            [epoch](IAsyncScheduler::WaitWakeup notify) { epoch->writer_notify_when_ready(std::move(notify)); },
    });
  }
  this->rethrow_producer_failure();
  return std::move(this->data());
}

//-----------------------------------------------------------------------------
// Inline definitions
//-----------------------------------------------------------------------------

inline void DebugScheduler::run()
{
  DEBUG_TRACE_MODULE(ASYNC, "DebugScheduler::run");
  detail::service_task_registry_debug_requests();
  if (Blocked_)
  {
    DEBUG_TRACE_MODULE(ASYNC, "run() on a blocked DebugQueue: doing nothing");
    return;
  }
  std::vector<BasicTask> H;
  {
    std::lock_guard lock(mutex_);
    TRACE_MODULE(ASYNC, "Got some coroutines to resume", Handles_.size());
    std::swap(H, Handles_);
  }
  this->order_batch(H);
  for (auto&& h : H)
  {
    TRACE_MODULE(ASYNC, "resuming coroutine...", &h, h.coroutine_handle());
    this->resume_task(h);
    CHECK(!h);
    TRACE_MODULE(ASYNC, "here", &h);
  }
  detail::service_task_registry_debug_requests();
}

inline void DebugScheduler::run_all(ExternalWaitPolicy external_waits)
{
  DEBUG_TRACE_MODULE(ASYNC, "DebugScheduler::run_all");
  detail::service_task_registry_debug_requests();
  if (Blocked_)
  {
    DEBUG_TRACE_MODULE(ASYNC, "run() on a blocked DebugQueue: doing nothing");
    return;
  }
  std::size_t begun_before;
  {
    std::lock_guard lock(mutex_);
    begun_before = external_waits_begun_;
  }
  while (!done())
  {
    run();
  }
  detail::service_task_registry_debug_requests();
  std::size_t begun_after;
  {
    std::lock_guard lock(mutex_);
    begun_after = external_waits_begun_;
  }
  if (external_waits == ExternalWaitPolicy::reject && begun_after != begun_before)
  {
    PANIC("DebugScheduler::run_all() ran tasks that began waits on another thread, so whether they have resumed "
          "when it returns depends on timing. Wait for the result with get_wait() or wait_for(), or pass "
          "ExternalWaitPolicy::allow if the tasks are meant to stay suspended",
          begun_after - begun_before);
  }
}

} // namespace uni20::async
