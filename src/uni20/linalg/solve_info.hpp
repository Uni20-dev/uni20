#pragma once

#include <uni20/core/scalar_concepts.hpp>

#include <cstddef>
#include <optional>

namespace uni20::linalg
{

/// \brief Numerical outcome of a square linear solve, independent of backend dispatch.
enum class SolveStatus
{
  /// \brief A finite solution was produced, or the problem was an empty no-op.
  success,
  /// \brief An exactly zero elimination pivot was found.
  singular,
  /// \brief A nonzero pivot failed the requested relative threshold.
  small_pivot,
  /// \brief An input coefficient or RHS component was nonfinite; both workspaces are preserved.
  nonfinite_input,
  /// \brief A factor, solution, or required magnitude became nonfinite during arithmetic.
  nonfinite_result
};

/// \brief Diagnostics written by a completed solve attempt, including numerical failure.
/// \details Success provides no residual bound or conditioning guarantee. Except
///          for nonfinite input, a failed attempt may have modified either
///          workspace; only success makes the RHS workspace a solution.
///          Zero-order systems and zero-column RHS matrices succeed without
///          inspecting values. These diagnostics do not represent dispatch declines.
struct SolveInfo
{
    SolveStatus status = SolveStatus::success;
    /// \brief Zero-based elimination column, present only for singular or small-pivot outcomes.
    /// \details This is not an original row index. Pivot choices and failure
    ///          classification can differ between backends.
    std::optional<std::size_t> pivot = {};

    /// \brief Report whether the attempt produced a finite solution or completed an empty no-op.
    [[nodiscard]] bool succeeded() const noexcept { return this->status == SolveStatus::success; }
};

/// \brief Native-real-precision pivot policy for a square linear solve.
/// \details A nonzero pivot is rejected when its magnitude divided by the
///          maximum entry magnitude of the original coefficient matrix is at
///          most `relative_pivot_tolerance`. This is not a condition estimate.
///          Zero (the default) rejects only exact zero pivots. The tolerance
///          must be finite and nonnegative. A positive threshold is application
///          policy: it can reject accurately solvable systems whose components
///          have different scales. For nonempty problems, nonfinite inputs/results
///          are reported regardless of tolerance, including magnitude overflow.
template <uni20::Real Real> struct SolveOptions
{
    /// \brief Reject nonzero pivots at or below this fraction of the original maximum entry magnitude.
    Real relative_pivot_tolerance = Real{};
};

} // namespace uni20::linalg
