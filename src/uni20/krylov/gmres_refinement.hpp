#pragma once

/**
 * \file gmres_refinement.hpp
 * \brief Matrix-free residual refinement with independently typed workspaces.
 */

#include <uni20/krylov/gmres.hpp>
#include <uni20/krylov/vector_space.hpp>

namespace uni20::krylov
{
/// \brief Bind precision roles and application operations for GMRES refinement.
/// \details Spaces carry working/update, inner-GMRES and residual precision.
///          Operators must represent the same original problem, evaluated at
///          their declared arithmetic precisions. The fixed preconditioner acts
///          in the inner space; its setup/storage/application precision belongs
///          to its implementation and can differ from the inner arithmetic.
///          Transfer is called as `transfer(output, input, destination_precision)`
///          and must preserve vector-space metadata while converting values.
///          It must support working-to-residual, residual-to-inner,
///          inner-to-residual, residual-to-working, and residual-to-residual.
///          All callbacks are borrowed; no hidden thread-local precision is set.
///          Spaces own their prototype descriptors; any underlying payload and
///          allocation contexts must remain valid through the solve.
template <class WorkingSpace, class InnerSpace, class ResidualSpace, class InnerOp, class ResidualOp,
          class Preconditioner, class Transfer>
struct GmresRefinementContext
{
    WorkingSpace working;
    InnerSpace inner;
    ResidualSpace residual;
    InnerOp& inner_operator;
    ResidualOp& residual_operator;
    Preconditioner& preconditioner;
    Transfer& transfer;
};

/// \brief Outer residual targets and independently configured inner solves.
template <Real InnerReal, Real ResidualReal> struct GmresRefinementOptions
{
    /// \brief Absent means working-precision epsilon, converted to residual precision.
    std::optional<ResidualReal> relative_tolerance = {};
    ResidualReal absolute_tolerance = ResidualReal{};
    std::size_t max_corrections = 10;
    GmresOptions<InnerReal> inner = {};
    GmresOperatorEstimates<ResidualReal> operator_estimates = {};
};

/// \brief Outer termination, distinct from convergence of an inner correction.
enum class GmresRefinementStatus
{
  converged,
  iteration_limit,
  stagnated,
  inner_failed,
  nonfinite_input,
  nonfinite_result
};

/// \brief Refinement diagnostics for the last finite published working iterate.
template <Real I, Real R> struct GmresRefinementInfo
{
    GmresRefinementStatus status = GmresRefinementStatus::iteration_limit;
    std::size_t corrections = 0;
    std::size_t inner_iterations = 0;
    std::size_t residual_operator_applications = 0;
    std::optional<GmresResidual<R>> residual = {};
    std::optional<GmresErrorEstimates<R>> errors = {};
    std::optional<GmresInfo<I>> last_inner = {};
    bool succeeded() const noexcept { return this->status == GmresRefinementStatus::converged; }
};

/// \brief Refine a supplied iterate using matrix-free GMRES correction solves.
/// \details Evaluates `r=b-A*x` in the residual space, normalizes r before
///          converting to the inner space, solves for a normalized correction,
///          rescales it in residual precision and updates x in working precision.
///          Inner convergence alone never implies outer convergence. An inner
///          failure stops refinement without applying its partial correction.
///          Candidate iterates are published only after successful evaluation
///          of their true residual. No automatic precision escalation occurs.
/// \pre Spaces represent the same mathematical vector space. Transfer operations
///      preserve its symmetry and layout semantics; there is no dense fallback.
/// \throws std::invalid_argument For incompatible dimensions or invalid options.
///         Callback exceptions propagate.
template <class WSpace, class ISpace, class RSpace, class IOp, class ROp, class P, class Transfer>
auto gmres_refine_with_info(
    GmresRefinementContext<WSpace, ISpace, RSpace, IOp, ROp, P, Transfer>& context, typename WSpace::vector_type& x,
    typename RSpace::vector_type const& b,
    GmresRefinementOptions<typename ISpace::real_type, typename RSpace::real_type> const& options = {})
{
  using W = typename WSpace::scalar_type;
  using I = typename ISpace::scalar_type;
  using R = typename RSpace::scalar_type;
  using WR = make_real_t<W>;
  using IR = make_real_t<I>;
  using RR = make_real_t<R>;
  static_assert(Complex<W> == Complex<I> && Complex<I> == Complex<R>,
                "GMRES refinement spaces must be all real or all complex");
  auto& ws = context.working;
  auto& is = context.inner;
  auto& rs = context.residual;
  ws.validate();
  is.validate();
  rs.validate();
  (void)gmres_detail::tolerances(options.inner, is.precision);
  (void)gmres_detail::operator_estimates_at_precision(options.inner.operator_estimates, is.precision);
  auto estimates = gmres_detail::operator_estimates_at_precision(options.operator_estimates, rs.precision);
  auto n = ws.ops.problem_dimension();
  if (is.ops.problem_dimension() != n || rs.ops.problem_dimension() != n || ws.ops.vector_dimension(x) != n ||
      rs.ops.vector_dimension(b) != n)
    throw std::invalid_argument("GMRES refinement spaces have incompatible dimensions");
  if ((options.relative_tolerance &&
       (!uni20::isfinite(*options.relative_tolerance) || *options.relative_tolerance < 0)) ||
      !uni20::isfinite(options.absolute_tolerance) || options.absolute_tolerance < 0)
    throw std::invalid_argument("GMRES refinement requires finite nonnegative tolerances");
  RR relative = options.relative_tolerance
                    ? scalar_at_precision<RR>(*options.relative_tolerance, rs.precision)
                    : gmres_detail::convert_scalar<RR>(numeric_limits<WR>::epsilon(ws.precision), rs.precision);
  RR absolute = scalar_at_precision<RR>(options.absolute_tolerance, rs.precision);
  if (!uni20::isfinite(relative) || !uni20::isfinite(absolute))
    throw std::invalid_argument("GMRES refinement tolerance is not representable at residual precision");
  auto rone = scalar_at_precision<R>(1, rs.precision);
  auto wone = scalar_at_precision<W>(1, ws.precision);
  GmresRefinementInfo<IR, RR> info;
  auto fail = [&](GmresRefinementStatus status) {
    info.status = status;
    return info;
  };
  auto rnorm = [&](auto const& v) { return gmres_detail::norm<R>(rs.ops, v, rs.precision); };
  auto wnorm = [&](auto const& v) { return gmres_detail::norm<W>(ws.ops, v, ws.precision); };
  if (!wnorm(x) || !rnorm(b)) return fail(GmresRefinementStatus::nonfinite_input);

  auto rhs = rs.ops.allocate_like(rs.prototype);
  context.transfer(rhs, b, rs.precision);
  auto bnorm = rnorm(rhs);
  if (!bnorm) return fail(GmresRefinementStatus::nonfinite_result);
  auto residual = rs.ops.allocate_like(rs.prototype);
  auto rx = rs.ops.allocate_like(rs.prototype);
  auto applied = rs.ops.allocate_like(rs.prototype);
  std::optional<RR> candidate_norm;
  auto evaluate = [&](auto const& candidate) {
    context.transfer(rx, candidate, rs.precision);
    candidate_norm = rnorm(rx);
    if (!candidate_norm) return std::optional<RR>{};
    ++info.residual_operator_applications;
    context.residual_operator.matvec(applied, rx);
    rs.ops.copy(residual, rhs);
    rs.ops.axpy(residual, -rone, applied);
    return rnorm(residual);
  };
  auto beta = evaluate(x);
  if (!beta) return fail(GmresRefinementStatus::nonfinite_result);
  auto diagnose = [&](RR const& residual_norm) {
    info.residual = GmresResidual<RR>{residual_norm, *bnorm, *candidate_norm};
    info.errors = gmres_detail::error_estimates(*info.residual, estimates);
  };
  diagnose(*beta);
  auto reached = [&](RR const& value) { return gmres_detail::reached(value, *bnorm, relative, absolute); };
  if (reached(*beta)) return fail(GmresRefinementStatus::converged);
  if (options.max_corrections == 0) return info;

  auto ib = is.ops.allocate_like(is.prototype);
  auto correction = is.ops.allocate_like(is.prototype);
  auto update = ws.ops.allocate_like(ws.prototype);
  auto candidate = ws.ops.allocate_like(ws.prototype);
  auto difference = ws.ops.allocate_like(ws.prototype);
  while (info.corrections < options.max_corrections)
  {
    // Normalization happens before narrowing: a small residual can still
    // generate an ordinary-size inner RHS even outside the inner exponent range.
    gmres_detail::normalize<R>(rs.ops, residual, *beta, rs.precision);
    context.transfer(ib, residual, is.precision);
    if (!gmres_detail::norm<I>(is.ops, ib, is.precision)) return fail(GmresRefinementStatus::nonfinite_result);
    is.ops.set_zero(correction);
    ++info.corrections;
    info.last_inner = gmres_with_info<I>(is.ops, context.inner_operator, context.preconditioner, correction, ib,
                                         options.inner, is.precision);
    info.inner_iterations += info.last_inner->iterations;
    if (!info.last_inner->succeeded()) return fail(GmresRefinementStatus::inner_failed);
    context.transfer(rx, correction, rs.precision);
    rs.ops.scal(rx, scalar_at_precision<R>(*beta, rs.precision));
    if (!rnorm(rx)) return fail(GmresRefinementStatus::nonfinite_result);
    context.transfer(update, rx, ws.precision);
    if (!wnorm(update)) return fail(GmresRefinementStatus::nonfinite_result);
    ws.ops.copy(candidate, x);
    ws.ops.axpy(candidate, wone, update);
    if (!wnorm(candidate)) return fail(GmresRefinementStatus::nonfinite_result);
    ws.ops.copy(difference, candidate);
    ws.ops.axpy(difference, -wone, x);
    auto change = wnorm(difference);
    if (!change) return fail(GmresRefinementStatus::nonfinite_result);
    beta = evaluate(candidate);
    if (!beta) return fail(GmresRefinementStatus::nonfinite_result);
    ws.ops.copy(x, candidate);
    diagnose(*beta);
    if (reached(*beta)) return fail(GmresRefinementStatus::converged);
    if (*change == 0) return fail(GmresRefinementStatus::stagnated);
  }
  return info;
}
} // namespace uni20::krylov
