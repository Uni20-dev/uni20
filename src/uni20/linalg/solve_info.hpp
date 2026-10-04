#pragma once

#include <uni20/core/runtime_precision.hpp>
#include <uni20/core/scalar_concepts.hpp>
#if UNI20_ENABLE_MPFR
#include <uni20/core/mpreal.hpp>
#endif

#include <cstddef>
#include <optional>

namespace uni20::linalg
{

/// \brief Numerical outcome of a square solve, LU factorization or log-determinant.
enum class SolveStatus
{
  /// \brief Finite output was produced, or the problem was an empty no-op.
  success,
  /// \brief An exactly zero elimination pivot was found.
  singular,
  /// \brief A nonzero pivot failed the requested relative threshold.
  small_pivot,
  /// \brief An input coefficient or RHS component was nonfinite; both workspaces are preserved.
  nonfinite_input,
  /// \brief A factor, solution, reduction, or required magnitude became nonfinite.
  nonfinite_result
};

/// \brief Diagnostics written by a completed numerical attempt, independent of dispatch.
/// \details Success provides no residual bound or conditioning guarantee. Except
///          for nonfinite input, a failed attempt may have modified either
///          workspace in a destructive one-shot solve; only success makes the RHS
///          workspace a solution. An owning LuFactorization stays unchanged on RHS failure.
///          For solves, zero-order systems and zero-column RHS matrices succeed without
///          inspecting values. LU always examines a nonempty matrix, even without
///          a RHS; success provides reusable factors. A singular log-determinant
///          has the conventional zero value despite its nonsuccess status.
///          These diagnostics do not represent dispatch declines.
struct SolveInfo
{
    SolveStatus status = SolveStatus::success;
    /// \brief Zero-based elimination column, present only for singular or small-pivot outcomes.
    /// \details This is not an original row index. Pivot choices and failure
    ///          classification can differ between backends.
    std::optional<std::size_t> pivot = {};

    /// \brief Report successful completion of the numerical operation.
    [[nodiscard]] bool succeeded() const noexcept { return this->status == SolveStatus::success; }
};

/// \brief Real-scalar pivot policy for a square linear solve.
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
