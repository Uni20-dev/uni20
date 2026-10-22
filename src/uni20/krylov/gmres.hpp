#pragma once

/**
 * \file gmres.hpp
 * \brief Restarted right-preconditioned GMRES over opaque vector operations.
 */

#include <uni20/core/runtime_precision.hpp>
#include <uni20/krylov/gmres_diagnostics.hpp>
#include <uni20/krylov/matrix_free.hpp>

namespace uni20::krylov
{
/// \brief Numerical termination of a matrix-free linear solve.
enum class GmresStatus
{
  converged,
  iteration_limit,
  stagnated,
  breakdown,
  nonfinite_input,
  nonfinite_result
};

/// \brief Residual stopping policy and storage/iteration limits.
template <Real R> struct GmresOptions
{
    /// \brief Absent means sqrt(epsilon) at the GMRES working precision.
    std::optional<R> relative_tolerance = {};
    R absolute_tolerance = R{};
    std::size_t restart = 30;
    std::size_t max_iterations = 1000;
    GmresOperatorEstimates<R> operator_estimates = {};
};

/// \brief Recoverable solve outcome; counters include attempted operator calls.
/// \details A residual is present only after successful evaluation of an
///          iterate. Later numerical failures leave that iterate and diagnostic
///          intact. Iterations count attempted Arnoldi steps; cycles count
///          started restart cycles, excluding the initial residual check.
template <Real R> struct GmresInfo
{
    GmresStatus status = GmresStatus::iteration_limit;
    std::size_t iterations = 0;
    std::size_t cycles = 0;
    std::size_t operator_applications = 0;
    std::size_t preconditioner_applications = 0;
    std::optional<GmresResidual<R>> residual = {};
    std::optional<GmresErrorEstimates<R>> errors = {};
    bool succeeded() const noexcept { return this->status == GmresStatus::converged; }
};

/// \brief Fixed linear inverse-preconditioner application, `output <- M^{-1} input`.
/// \details Input and output are distinct vectors in the same vector space.
///          The operation must remain fixed during a solve. Variable or nonlinear
///          preconditioners require a flexible GMRES implementation.
template <class P, class V>
concept KrylovPreconditioner = requires(P p, V& output, V const& input) {
  { p.apply(output, input) } -> std::same_as<void>;
};

/// \brief Identity preconditioner using the supplied vector copy operation.
template <class Ops> struct IdentityPreconditioner
{
    Ops& ops;
    template <class V> void apply(V& output, V const& input) { ops.copy(output, input); }
};

namespace gmres_detail
{
template <Real R> struct Tolerances
{
    R relative;
    R absolute;
};

template <Real R> Tolerances<R> tolerances(GmresOptions<R> const& options, Precision precision)
{
  if (options.restart == 0 ||
      (options.relative_tolerance &&
       (!uni20::isfinite(*options.relative_tolerance) || *options.relative_tolerance < 0)) ||
      !uni20::isfinite(options.absolute_tolerance) || options.absolute_tolerance < 0)
    throw std::invalid_argument("GMRES requires a positive restart and finite nonnegative tolerances");
  R relative = options.relative_tolerance ? scalar_at_precision<R>(*options.relative_tolerance, precision)
                                          : math::sqrt(numeric_limits<R>::epsilon(precision));
  R absolute = scalar_at_precision<R>(options.absolute_tolerance, precision);
  if (!uni20::isfinite(relative) || !uni20::isfinite(absolute))
    throw std::invalid_argument("GMRES tolerance is not representable at working precision");
  return {relative, absolute};
}

template <RealOrComplex S> struct Rotation
{
    make_real_t<S> c;
    S s;
    S r;
};

// [c s; -conj(s) c] maps [f g] to [r 0]. Neither zero component
// requires division by zero, and hypot avoids squaring unscaled magnitudes.
template <RealOrComplex S> Rotation<S> rotation(S const& f, S const& g, Precision p)
{
  using R = make_real_t<S>;
  auto zero = scalar_at_precision<S>(0, p);
  auto rzero = scalar_at_precision<R>(0, p);
  auto rone = scalar_at_precision<R>(1, p);
  if (g == zero) return {rone, zero, f};
  R af = math::abs(f), ag = math::abs(g);
  if (f == zero) return {rzero, uni20::conj(g) / ag, scalar_at_precision<S>(ag, p)};
  R magnitude = math::hypot(af, ag);
  S phase = f / af;
  return {af / magnitude, phase * (uni20::conj(g) / magnitude), phase * magnitude};
}

template <class S> void rotate(Rotation<S> const& rotation, S& x, S& y)
{
  S previous = x;
  x = rotation.c * previous + rotation.s * y;
  y = -uni20::conj(rotation.s) * previous + rotation.c * y;
}

template <RealOrComplex S, class Ops, class V> std::optional<make_real_t<S>> norm(Ops& ops, V const& v, Precision p)
{
  using R = make_real_t<S>;
  if constexpr (KrylovNorm<Ops, V, S>)
  {
    R value = scalar_at_precision<R>(ops.norm(v), p);
    if (!uni20::isfinite(value) || value < 0) return {};
    return value;
  }
  else
  {
    S square = scalar_at_precision<S>(ops.inner_product(v, v), p);
    if (!uni20::isfinite(square) || math::real(square) < 0) return {};
    return math::sqrt(scalar_at_precision<R>(math::real(square), p));
  }
}

template <RealOrComplex S, class Ops, class V> void normalize(Ops& ops, V& v, make_real_t<S> const& length, Precision p)
{
  using R = make_real_t<S>;
  auto one = scalar_at_precision<R>(1, p);
  R inverse = one / length;
  if (uni20::isfinite(inverse) && inverse != 0)
    ops.scal(v, scalar_at_precision<S>(inverse, p));
  else
  {
    // A finite vector norm can have an unrepresentable reciprocal. Two
    // square-root factors avoid overflowing the scale before applying it.
    S factor = scalar_at_precision<S>(one / math::sqrt(length), p);
    ops.scal(v, factor);
    ops.scal(v, factor);
  }
}

template <Real R> bool reached(R const& residual, R const& rhs, R const& relative, R const& absolute)
{
  // Compare ratios instead of forming a potentially overflowing rtol*||b||.
  return residual <= absolute || (relative > 0 && rhs > 0 && residual / rhs <= relative);
}
} // namespace gmres_detail

/// \brief Solve `A*x=b` with restarted GMRES and a fixed right preconditioner.
/// \details Preserves the supplied initial guess. Acceptance uses a recomputed
///          residual, `||b-A*x|| <= max(atol, rtol*||b||)`. Modified Gram-Schmidt
///          uses selective second-pass orthogonalization. Only the Arnoldi basis
///          is stored; a combined basis correction is preconditioned once per
///          cycle. Callbacks own backend selection, memory placement and any
///          higher-precision operator/preconditioner arithmetic. Scalar returns
///          and Hessenberg arithmetic use \p precision; vector operations must
///          implement that working precision without discarding vector metadata.
/// \pre Vector copies allocate independent values. Norms/inner products must
///      expose nonfinite data as nonfinite results. The operator and
///      preconditioner are fixed linear maps throughout the call.
/// \throws std::invalid_argument For incompatible dimensions, invalid tolerances
///         or a zero restart length. Callback exceptions propagate.
/// \return Numerical status and diagnostics of the last published iterate.
template <RealOrComplex S, class Ops, class Op, class P, class V>
  requires KrylovVectorOps<Ops, V, S> && KrylovOperator<Op, V> && KrylovPreconditioner<P, V>
GmresInfo<make_real_t<S>> gmres_with_info(Ops& ops, Op& op, P& preconditioner, V& x, V const& b,
                                          GmresOptions<make_real_t<S>> const& options, Precision precision)
{
  using R = make_real_t<S>;
  require_precision<S>(precision);
  (void)precision.bit_count();
  std::size_t n = ops.problem_dimension();
  if (ops.vector_dimension(x) != n || ops.vector_dimension(b) != n)
    throw std::invalid_argument("GMRES vectors do not belong to the operator space");
  auto [relative, absolute] = gmres_detail::tolerances(options, precision);
  auto estimates = gmres_detail::operator_estimates_at_precision(options.operator_estimates, precision);
  auto zero = scalar_at_precision<S>(0, precision);
  auto one = scalar_at_precision<S>(1, precision);
  auto epsilon = numeric_limits<R>::epsilon(precision);
  auto dgks = math::sqrt(scalar_at_precision<R>(2, precision));
  GmresInfo<R> info;
  auto norm = [&](V const& v) { return gmres_detail::norm<S>(ops, v, precision); };
  auto fail = [&](GmresStatus status) {
    info.status = status;
    return info;
  };
  auto bnorm = norm(b);
  auto xnorm = norm(x);
  if (!bnorm || !xnorm) return fail(GmresStatus::nonfinite_input);
  auto diagnose = [&](R const& residual_norm, R const& solution_norm) {
    info.residual = GmresResidual<R>{residual_norm, *bnorm, solution_norm};
    info.errors = gmres_detail::error_estimates(*info.residual, estimates);
  };

  // Preserve b even if x aliases it, and publish x only after diagnosis.
  auto rhs = ops.allocate_like(b);
  ops.copy(rhs, b);
  auto residual = ops.allocate_like(b);
  auto work = ops.allocate_like(b);
  auto evaluate = [&](V const& candidate) {
    ++info.operator_applications;
    op.matvec(work, candidate);
    ops.copy(residual, rhs);
    ops.axpy(residual, -one, work);
    return norm(residual);
  };
  auto beta = evaluate(x);
  if (!beta) return fail(GmresStatus::nonfinite_result);
  diagnose(*beta, *xnorm);
  auto reached = [&](R const& value) { return gmres_detail::reached(value, *bnorm, relative, absolute); };
  if (reached(*beta)) return fail(GmresStatus::converged);
  if (options.max_iterations == 0) return info;

  auto z = ops.allocate_like(b);
  auto delta = ops.allocate_like(b);
  auto candidate = ops.allocate_like(b);
  std::size_t m = std::min({options.restart, options.max_iterations, n});
  std::vector<S> h((m + 1) * m, zero), g(m + 1, zero);
  auto H = [&](std::size_t row, std::size_t col) -> S& { return h[col * (m + 1) + row]; };
  while (info.iterations < options.max_iterations)
  {
    ++info.cycles;
    std::vector<V> basis;
    basis.reserve(m + 1);
    basis.push_back(ops.allocate_like(b));
    ops.copy(basis.back(), residual);
    gmres_detail::normalize<S>(ops, basis.back(), *beta, precision);
    std::fill(h.begin(), h.end(), zero);
    std::fill(g.begin(), g.end(), zero);
    g[0] = scalar_at_precision<S>(*beta, precision);
    std::vector<gmres_detail::Rotation<S>> rotations;
    rotations.reserve(m);
    std::size_t columns = 0;
    bool breakdown = false;
    for (std::size_t j = 0; j < m && info.iterations < options.max_iterations; ++j)
    {
      ++info.iterations;
      ++info.preconditioner_applications;
      preconditioner.apply(z, basis[j]);
      if (!norm(z)) return fail(GmresStatus::nonfinite_result);
      ++info.operator_applications;
      op.matvec(work, z);
      auto before = norm(work);
      if (!before) return fail(GmresStatus::nonfinite_result);
      auto orthogonalize = [&] {
        for (std::size_t k = 0; k <= j; ++k)
        {
          S dot = scalar_at_precision<S>(ops.inner_product(basis[k], work), precision);
          if (!uni20::isfinite(dot)) return false;
          H(k, j) += dot;
          ops.axpy(work, -dot, basis[k]);
        }
        return true;
      };
      if (!orthogonalize()) return fail(GmresStatus::nonfinite_result);
      auto after = norm(work);
      if (!after) return fail(GmresStatus::nonfinite_result);
      if (*after < *before / dgks)
      {
        if (!orthogonalize()) return fail(GmresStatus::nonfinite_result);
        after = norm(work);
        if (!after) return fail(GmresStatus::nonfinite_result);
      }
      H(j + 1, j) = scalar_at_precision<S>(*after, precision);
      breakdown = *after == 0 || (*before > 0 && *after / *before <= epsilon);
      if (!breakdown && j + 1 < m && info.iterations < options.max_iterations)
      {
        basis.push_back(ops.allocate_like(b));
        ops.copy(basis.back(), work);
        gmres_detail::normalize<S>(ops, basis.back(), *after, precision);
      }
      for (std::size_t k = 0; k < j; ++k)
        gmres_detail::rotate(rotations[k], H(k, j), H(k + 1, j));
      auto rotation = gmres_detail::rotation(H(j, j), H(j + 1, j), precision);
      if (!uni20::isfinite(rotation.c) || !uni20::isfinite(rotation.s) || !uni20::isfinite(rotation.r))
        return fail(GmresStatus::nonfinite_result);
      if (rotation.r == zero)
      {
        breakdown = true;
        break; // Keep the preceding full-rank projected prefix, if any.
      }
      H(j, j) = rotation.r;
      H(j + 1, j) = zero;
      rotations.push_back(rotation);
      gmres_detail::rotate(rotation, g[j], g[j + 1]);
      columns = j + 1;
      if (breakdown || reached(math::abs(g[j + 1]))) break;
    }
    if (columns == 0) return fail(GmresStatus::breakdown);
    std::vector<S> y(g.begin(), g.begin() + columns);
    for (std::size_t row = columns; row-- > 0;)
    {
      for (std::size_t col = row + 1; col < columns; ++col)
        y[row] -= H(row, col) * y[col];
      y[row] /= H(row, row);
      if (!uni20::isfinite(y[row])) return fail(GmresStatus::nonfinite_result);
    }
    ops.set_zero(delta);
    for (std::size_t j = 0; j < columns; ++j)
      ops.axpy(delta, y[j], basis[j]);
    ++info.preconditioner_applications;
    preconditioner.apply(z, delta);
    if (!norm(z)) return fail(GmresStatus::nonfinite_result);
    ops.copy(candidate, x);
    ops.axpy(candidate, one, z);
    auto candidate_norm = norm(candidate);
    if (!candidate_norm) return fail(GmresStatus::nonfinite_result);
    ops.copy(delta, candidate);
    ops.axpy(delta, -one, x);
    auto change = norm(delta);
    if (!change) return fail(GmresStatus::nonfinite_result);
    beta = evaluate(candidate);
    if (!beta) return fail(GmresStatus::nonfinite_result);
    ops.copy(x, candidate);
    diagnose(*beta, *candidate_norm);
    if (reached(*beta)) return fail(GmresStatus::converged);
    if (breakdown) return fail(GmresStatus::breakdown);
    if (*change == 0) return fail(GmresStatus::stagnated);
  }
  return info;
}

/// \brief GMRES without preconditioning, at explicit scalar precision.
template <RealOrComplex S, class Ops, class Op, class V>
  requires KrylovVectorOps<Ops, V, S> && KrylovOperator<Op, V>
auto gmres_with_info(Ops& ops, Op& op, V& x, V const& b, GmresOptions<make_real_t<S>> const& options, Precision p)
{
  IdentityPreconditioner<Ops> identity{ops};
  return gmres_with_info<S>(ops, op, identity, x, b, options, p);
}

/// \brief Fixed-scalar GMRES using the scalar's native precision.
template <RealOrComplex S, class Ops, class Op, class P, class V>
  requires(!has_runtime_precision_v<S>) && KrylovVectorOps<Ops, V, S> && KrylovOperator<Op, V> &&
          KrylovPreconditioner<P, V>
auto gmres_with_info(Ops& ops, Op& op, P& preconditioner, V& x, V const& b,
                     GmresOptions<make_real_t<S>> const& options = {})
{
  return gmres_with_info<S>(ops, op, preconditioner, x, b, options, native_precision<S>());
}

/// \brief Fixed-scalar GMRES without preconditioning.
template <RealOrComplex S, class Ops, class Op, class V>
  requires(!has_runtime_precision_v<S>) && KrylovVectorOps<Ops, V, S> && KrylovOperator<Op, V>
auto gmres_with_info(Ops& ops, Op& op, V& x, V const& b, GmresOptions<make_real_t<S>> const& options = {})
{
  return gmres_with_info<S>(ops, op, x, b, options, native_precision<S>());
}
} // namespace uni20::krylov
