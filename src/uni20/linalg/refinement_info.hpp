#pragma once

#include <uni20/linalg/solve_info.hpp>

namespace uni20::linalg
{
/// \brief Scalar families and explicit precisions for three-precision LU refinement.
/// \details Each fixed scalar requires its native precision. Runtime scalars
///          require finite precisions. All three scalars must be real, or all
///          complex. Precision ordering is not enforced and convergence is not
///          guaranteed by any choice of bit counts.
template <RealOrComplex Factor, RealOrComplex Working = Factor, RealOrComplex Residual = Working>
  requires(Complex<Factor> == Complex<Working> && Complex<Working> == Complex<Residual>)
struct LuRefinementPrecision
{
    Precision factor;
    Precision working;
    Precision residual;

    /// \brief Select the native precisions of three fixed scalar families.
    static constexpr LuRefinementPrecision native()
      requires(!has_runtime_precision_v<Factor> && !has_runtime_precision_v<Working> &&
               !has_runtime_precision_v<Residual>)
    {
      return {native_precision<Factor>(), native_precision<Working>(), native_precision<Residual>()};
    }
};

/// \brief Residual and backward-error estimates for one RHS, at residual precision.
/// \details Complex entries use mathematical magnitude. These are evaluated
///          diagnostics, not outward-rounded bounds or forward-error estimates.
template <Real R> struct SolveError
{
    R residual_infinity_norm;
    R normwise_backward_error;
    R componentwise_backward_error;
};

/// \brief Refinement termination, distinct from the outcome of an individual LU solve.
enum class RefinementStatus
{
  converged,
  stagnated,
  iteration_limit,
  factorization_failed,
  correction_failed,
  nonfinite_input,
  nonfinite_result
};

/// \brief Outer-loop outcome and, on a factor/solve failure, the provider's numerical diagnostics.
struct RefinementInfo
{
    RefinementStatus status = RefinementStatus::converged;
    std::optional<SolveInfo> solve_info = {};
    bool succeeded() const noexcept { return this->status == RefinementStatus::converged; }
};

/// \brief Stopping policy; tolerance is expressed in the residual real scalar.
template <Real R> struct RefinementOptions
{
    /// \brief Componentwise backward-error target; absent means working-precision epsilon.
    std::optional<R> backward_error_tolerance = {};
    /// \brief Maximum correction steps per RHS, excluding the initial solve.
    std::size_t max_iterations = 10;
};

/// \brief Diagnostics of the published iterate for one RHS.
template <Real R> struct RefinementColumn
{
    SolveError<R> error;
    std::size_t corrections = 0;
    bool converged = false;
    /// \brief A computed correction could not change any component at working precision.
    bool stagnated = false;
};
} // namespace uni20::linalg
