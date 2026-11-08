# Task accounting measurement tools

Enable `UNI20_BUILD_BENCH=ON` to build the paired measurement executables.
Each suffix `0` excludes `<uni20/async/accounted_task.hpp>`; suffix `1` includes
it. Ordinary tasks use the same source in both builds. The header-presence macro
only affects these executables and never changes library layouts or behavior.

- `uni20_accounting_bench_{0,1}` measures ordinary and selected accounted tasks
  with DebugScheduler and one/four-participant TBB scheduling. CSV includes a
  checksum. Mode `-1` is ordinary; `0`, `1`, `2` select elapsed, active wall and
  CPU-plus-wall accounting. Workloads cover empty tasks, repeated direct
  self-transfers and a small integer recurrence. Submission and reporting cost
  are included in throughput timings.
- `uni20_accounting_allocations_{0,1}` counts ordinary allocations during task
  creation, after warmup. Replacement `new` is confined to these executables;
  it does not perturb the throughput benchmark. The first allocation is the
  observed frame, not a portable frame-size guarantee.
- `uni20_accounting_codegen_{0,1}` compiles the same ordinary coroutine to object
  files with and without the accounting definitions. Compare executable code
  sections using identical optimization/debug settings; debug information or
  source-path differences need not be identical.

Run timing comparisons on a quiet machine, separately from builds and sanitizer
runs, and retain individual measurements. These tools measure overhead; no
performance threshold is encoded as a test. For API and lifetime contracts see
[task accounting](../../docs/async/task_accounting.md).
