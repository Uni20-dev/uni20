#pragma once
/// \file tbb_scheduler.hpp
/// \brief Scheduler implementation using oneAPI oneTBB task_arena + task_group.

#include "async_errors.hpp"
#include "scheduler.hpp"
#include "task_registry.hpp"
#include "tbb_task_submission.hpp"
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <memory>
#include <mutex>
#include <oneapi/tbb/concurrent_queue.h>
#include <oneapi/tbb/parallel_for.h>
#include <oneapi/tbb/task.h>
#include <oneapi/tbb/task_arena.h>
#include <oneapi/tbb/task_group.h>
#include <optional>
#include <uni20/config.hpp>
#include <utility>
#include <vector>

namespace uni20::async
{

namespace detail
{
inline void service_tbb_task_registry_debug_requests()
{
#if UNI20_DEBUG_ASYNC_TASKS
  TaskRegistry::service_debug_requests();
#endif
}
} // namespace detail

/// \brief Runtime policy for synchronous lightweight batches executed by a TBB scheduler.
struct TbbSchedulerBatchOptions
{
    /// Maximum number of concurrently executing batch runners; null uses ordinary `parallel_for` partitioning.
    std::optional<std::size_t> maximum_concurrency{std::nullopt};
};

/// \brief Scheduler backend that uses Intel oneTBB to resume coroutines.
///
/// Tasks scheduled on this scheduler are submitted to a task_group inside a
/// TBB task_arena. Resumption occurs on a worker thread or an application
/// thread participating in the arena.
///
/// \note Each coroutine records a non-owning scheduler route. A suspended task
///       may replace that route while externally owned; subsequent activations
///       are submitted only to the replacement scheduler.
/// \note Blocking waits require readiness notifications. There is no internal
///       deadlock detector or wait timeout; use a debugger or TaskRegistry
///       diagnostics to investigate stalled dependencies.
///
class TbbNumaScheduler;

class TbbScheduler : public IAsyncScheduler {
    class SuspendedWait;

  public:
    /// \brief Construct a TBB scheduler with a given arena concurrency.
    /// \param max_concurrency Maximum arena participation, including application threads.
    ///                        Use task_arena::automatic for the oneTBB default.
    /// \param batch_options Lightweight-batch concurrency policy.
    /// \note Explicit concurrency one shares its sole slot between application and worker threads;
    ///       reserving that slot would let oneTBB add a second participant for enqueued work.
    explicit TbbScheduler(int max_concurrency = oneapi::tbb::task_arena::automatic,
                          TbbSchedulerBatchOptions batch_options = {})
        : arena_(max_concurrency, /*reserved_for_application_threads=*/max_concurrency == 1 ? 0 : 1,
                 oneapi::tbb::task_arena::priority::normal, oneapi::tbb::task_arena::leave_policy::fast),
          batch_options_(std::move(batch_options)), paused_(false)
    {
      CHECK(!batch_options_.maximum_concurrency || *batch_options_.maximum_concurrency > 0);
    }

    /// \brief Construct a TBB scheduler constrained to a specific NUMA node.
    /// \param constraints Binding constraints applied to the underlying arena.
    /// \param batch_options Lightweight-batch concurrency policy.
    explicit TbbScheduler(oneapi::tbb::task_arena::constraints constraints, TbbSchedulerBatchOptions batch_options = {})
        : arena_(constraints, /*reserved_for_application_threads=*/constraints.max_concurrency == 1 ? 0 : 1,
                 oneapi::tbb::task_arena::priority::normal, oneapi::tbb::task_arena::leave_policy::fast),
          batch_options_(std::move(batch_options)), paused_(false)
    {
      CHECK(!batch_options_.maximum_concurrency || *batch_options_.maximum_concurrency > 0);
    }

    ~TbbScheduler() noexcept override
    {
      // ensure all tasks finish before destruction
      this->wait_for_submitted_tasks();
    }

    /// \brief Schedule a coroutine for initial execution.
    void schedule(AsyncTask&& t) override
    {
      TaskRegistry::record_task_scheduled(t.coroutine_handle());
      if (t.set_scheduler(this)) this->enqueue_task(std::move(t));
    }

    /// \brief Block until all tasks scheduled on this scheduler are complete.
    ///
    /// \note This guarantees quiescence with respect to tasks that were
    ///       scheduled on this TbbScheduler. Tasks blocked on external
    ///       events (I/O, MPI, etc.) may still be logically alive and
    ///       will resume later if rescheduled.
    void run_all()
    {
      this->resume();
      detail::service_tbb_task_registry_debug_requests();
      this->wait_for_submitted_tasks();
      detail::service_tbb_task_registry_debug_requests();
    }

    /// \brief Pause the scheduler. Don't execute scheduled tasks, but instead add them to a queue.
    void pause() override
    {
      std::scoped_lock lock(pause_mutex_);
      paused_.store(true, std::memory_order_release);
    }

    /// \brief Unpause the scheduler, and execute any tasks that have been queued.
    void resume() override
    {
      std::vector<TaskHandle> drained;
      {
        std::scoped_lock lock(pause_mutex_);
        paused_.store(false, std::memory_order_release);
        TaskHandle h;
        while (queue_.try_pop(h))
        {
          drained.push_back(h);
        }
      }

      for (auto h : drained)
      {
        TRACE_MODULE(ASYNC, "scheduling coroutine", h.coroutine());
        this->dispatch_handle(h);
      }
    }

    void help_while_waiting(const WaitPredicate& is_ready) override { this->wait_for(is_ready); }

    /// \brief Accept an already-satisfied predicate; pending TBB waits require a WaitRequest notification.
    void wait_for(const WaitPredicate& is_ready) override { this->wait_for(WaitRequest{.is_ready = is_ready}); }

    /// \brief Wait for targeted readiness while allowing the arena to execute other work.
    /// \param request Readiness predicate and race-safe wakeup registration.
    /// \throws async_state_error if a pending request has no notification registration.
    void wait_for(WaitRequest const& request) override
    {
      if (request.is_ready()) return;
      if (!request.notify_when_ready)
        throw async_state_error("TbbScheduler pending wait requires a readiness notification");

      arena_.execute([&] { this->wait_in_arena(request); });
    }

  private:
    void execute_batch_impl(LightweightTaskBatch const& batch) override
    {
      if (batch.size() == 0) return;

      arena_.execute([&] {
        // Serial batches must occupy the same arena as coroutine activations.
        // Running this loop outside execute() could overlap an enqueued task.
        if (arena_.max_concurrency() == 1)
        {
          for (std::size_t index = 0; index < batch.size(); ++index)
            batch(index);
          return;
        }

        if (!batch_options_.maximum_concurrency)
        {
          oneapi::tbb::parallel_for(std::size_t{0}, batch.size(), [&](std::size_t index) { batch(index); });
          return;
        }

        std::size_t const runner_count =
            std::min(*batch_options_.maximum_concurrency, static_cast<std::size_t>(arena_.max_concurrency()));
        if (runner_count >= batch.size())
        {
          oneapi::tbb::parallel_for(std::size_t{0}, batch.size(), [&](std::size_t index) { batch(index); });
          return;
        }

        std::atomic<std::size_t> next_index{0};
        oneapi::tbb::parallel_for(
            std::size_t{0}, runner_count,
            [&](std::size_t) {
              while (true)
              {
                std::size_t const index = next_index.fetch_add(1, std::memory_order_relaxed);
                if (index >= batch.size()) return;
                batch(index);
              }
            },
            oneapi::tbb::simple_partitioner{});
      });
    }

    bool accepts_route(TaskRoute route) const noexcept override
    {
      return route.domain == TaskDomain::host && !route.cuda_device;
    }

    bool can_direct_transfer(TaskHandle, TaskHandle) const noexcept override { return true; }

    /// \brief Reschedule a previously suspended coroutine.
    void reschedule(BasicTask&& t) override { this->enqueue_task(std::move(t)); }

    class SuspendedWait {
      public:
        void install(oneapi::tbb::task::suspend_point point) noexcept
        {
          oneapi::tbb::task::suspend_point point_to_resume = nullptr;
          {
            std::scoped_lock lock(mutex_);
            point_ = point;
            if (signaled_ && !resume_issued_)
            {
              resume_issued_ = true;
              point_to_resume = point_;
            }
          }
          if (point_to_resume) oneapi::tbb::task::resume(point_to_resume);
        }

        void signal() noexcept
        {
          oneapi::tbb::task::suspend_point point_to_resume = nullptr;
          {
            std::scoped_lock lock(mutex_);
            signaled_ = true;
            if (point_ && !resume_issued_)
            {
              resume_issued_ = true;
              point_to_resume = point_;
            }
          }
          if (point_to_resume) oneapi::tbb::task::resume(point_to_resume);
        }

      private:
        std::mutex mutex_;
        oneapi::tbb::task::suspend_point point_{nullptr};
        bool signaled_{false};
        bool resume_issued_{false};
    };

    void wait_in_arena(WaitRequest const& request)
    {
      while (!request.is_ready())
      {
        detail::service_tbb_task_registry_debug_requests();
        auto wait = std::make_shared<SuspendedWait>();
        request.notify_when_ready([weak_wait = std::weak_ptr<SuspendedWait>(wait)]() noexcept {
          if (auto active = weak_wait.lock()) active->signal();
        });

        // Registration must also notify if readiness raced with registration.
        // Recheck before suspending: a writer predicate may acquire its gate here.
        if (request.is_ready()) return;
        oneapi::tbb::task::suspend([wait](oneapi::tbb::task::suspend_point point) { wait->install(point); });
        // Notifications need not reserve a resource. If another writer acquired
        // it first, recheck and register a fresh notification on the next pass.
      }
    }

  protected:
    /// \brief Wait for every activation registered with the shared task group.
    void wait_for_submitted_tasks()
    {
      arena_.execute([&] { tg_.wait(); });
    }

    /// \brief Enqueue one task through the scheduler's domain-routing hook.
    void enqueue_task(BasicTask&& t)
    {
      TRACE_MODULE(ASYNC, "TBB scheduler enqueuing task", t.coroutine_handle());
      if (auto h = t.release_handle())
      {
        bool paused = paused_.load(std::memory_order_acquire);
        if (paused)
        {
          queue_.push(h);
        }
        else
        {
          {
            std::scoped_lock lock(pause_mutex_);
            if (paused_.load(std::memory_order_acquire))
            {
              queue_.push(h);
              return;
            }
          }

          this->dispatch_handle(h);
        }
      }
    }

    /// \brief Route one runnable task activation to its execution arena.
    virtual void dispatch_handle(TaskHandle handle)
    {
      this->dispatch_handle_in_arena(arena_, handle, {.scheduler = "TbbScheduler"}, [] {});
    }

    /// \brief Submit one activation to an arena through the shared task group.
    /// \tparam Prepare callable that establishes or verifies the arena execution context.
    /// \param arena Arena that will execute the activation.
    /// \param handle Promise-neutral task identity.
    /// \param context Diagnostic context for admission failure.
    /// \param prepare Callable invoked immediately before coroutine resumption.
    template <typename Prepare>
    void dispatch_handle_in_arena(oneapi::tbb::task_arena& arena, TaskHandle handle,
                                  detail::TbbTaskAdmissionContext context, Prepare&& prepare)
    {
      detail::enqueue_tbb_task(
          arena, tg_,
          [handle, prepare = std::forward<Prepare>(prepare)] {
            prepare();
            TRACE_MODULE(ASYNC, "resuming coroutine", handle.coroutine());
            TaskPromiseBase::resume_and_track(handle);
            detail::service_tbb_task_registry_debug_requests();
          },
          context);
    }

  private:
    oneapi::tbb::task_arena arena_;
    oneapi::tbb::task_group tg_;
    TbbSchedulerBatchOptions batch_options_;
    std::atomic<bool> paused_;
    std::mutex pause_mutex_;
    oneapi::tbb::concurrent_queue<TaskHandle> queue_;
    // FIXME: the concurrent_queue is overkill here, since we don't need to preserve order of tasks

    friend class TbbNumaScheduler;
};

} // namespace uni20::async
