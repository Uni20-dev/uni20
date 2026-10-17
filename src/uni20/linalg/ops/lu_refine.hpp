#pragma once

/**
 * \file lu_refine.hpp
 * \brief Synchronous preserving LU refinement with three explicit precision roles.
 */

#include <uni20/linalg/ops/lu.hpp>
#include <uni20/linalg/ops/solve_diagnostics.hpp>

namespace uni20::linalg
{
/// \brief Refinement outcome, retaining the last finite, diagnosed iterate on later failure.
/// \details A solution is present only once the initial solve and its residual
///          evaluation succeed. Columns always describe that published iterate.
///          Convergence is a backward-error stopping result, not a forward-error
///          guarantee or an application acceptance decision.
template <RealOrComplex W, RealOrComplex R> struct LuRefinementResult
{
    RefinementInfo info = {};
    std::optional<DenseMatrix<W>> solution = {};
    std::vector<RefinementColumn<make_real_t<R>>> columns = {};
};

namespace refinement_detail
{
template <class F, class W, class R>
auto validate(LuRefinementPrecision<F, W, R> p, RefinementOptions<make_real_t<R>> const& options)
{
  require_precision<F>(p.factor);
  require_precision<W>(p.working);
  require_precision<R>(p.residual);
  (void)p.factor.bit_count();
  (void)p.working.bit_count();
  (void)p.residual.bit_count();
  using RR = make_real_t<R>;
  if (options.backward_error_tolerance)
    ERROR_IF(!uni20::isfinite(*options.backward_error_tolerance) || *options.backward_error_tolerance < 0,
             "refinement requires a finite nonnegative backward-error tolerance");
  RR tolerance = options.backward_error_tolerance
                     ? convert<RR>(*options.backward_error_tolerance, p.residual)
                     : convert<RR>(numeric_limits<make_real_t<W>>::epsilon(p.working), p.residual);
  ERROR_IF(!uni20::isfinite(tolerance), "refinement tolerance is not representable at residual precision");
  return tolerance;
}

inline RefinementInfo conversion_failure(SolveInfo const& info)
{
  return {.status = info.status == SolveStatus::nonfinite_input ? RefinementStatus::nonfinite_input
                                                                : RefinementStatus::nonfinite_result};
}

template <class W, class R, class A, class B>
LuRefinementResult<W, R> empty_result(A const& a, B const& b, Precision working, Precision residual)
{
  LuRefinementResult<W, R> result;
  result.solution.emplace(a.extent(1), b.extent(1), working);
  auto zero = scalar_at_precision<make_real_t<R>>(0, residual);
  result.columns.assign(b.extent(1), {.error = {zero, zero, zero}, .converged = true});
  return result;
}

template <RealOrComplex S> auto component_magnitude(S const& value)
{
  if constexpr (Complex<S>)
    return std::max(math::abs(value.real()), math::abs(value.imag()));
  else
    return math::abs(value);
}

// Transfer the bounded significand and combine exponents before applying the
// scale. Neither scaling in From nor converting to To first is safe when their
// exponent ranges differ, even if the final value is representable.
template <RealOrComplex To, RealOrComplex From>
  requires(Complex<To> == Complex<From>)
To convert_scaled(From const& value, std::int64_t exponent, Precision p)
{
  if (exponent == 0) return convert<To>(value, p);
  if constexpr (Complex<To>)
    return To(convert_scaled<make_real_t<To>>(value.real(), exponent, p),
              convert_scaled<make_real_t<To>>(value.imag(), exponent, p));
  else
  {
    auto parts = math::frexp(value);
    return math::ldexp(convert<To>(parts.fraction, p), parts.exponent + exponent);
  }
}

// Normalize toward the factors' scale with headroom for the smallest pivot.
// Normalizing every RHS to unity can overflow a triangular solve even for tiny,
// well-conditioned matrices. Binary shifts avoid rounding the scale itself.
template <class Backend, class F, class W, class R>
std::optional<DenseMatrix<W>> correction(Backend& backend, LuFactorization<F> const& factor, DenseMatrix<R> const& rhs,
                                         LuRefinementPrecision<F, W, R> p, std::vector<bool> const& active,
                                         std::int64_t target_exponent, SolveInfo& info)
{
  using RR = make_real_t<R>;
  auto zero = scalar_at_precision<RR>(0, p.residual);
  std::size_t n = rhs.extent(0), count = rhs.extent(1);
  DenseMatrix<F> work(n, count, p.factor);
  std::vector<std::int64_t> shifts(count, 0);
  for (std::size_t j = 0; j < count; ++j)
  {
    if (!active[j]) continue;
    RR magnitude = zero;
    for (std::size_t i = 0; i < n; ++i)
      magnitude = std::max(magnitude, component_magnitude(R(rhs[i, j])));
    if (magnitude == 0) continue;
    shifts[j] = target_exponent - math::frexp(magnitude).exponent;
    for (std::size_t i = 0; i < n; ++i)
      work[i, j] = convert_scaled<F>(R(rhs[i, j]), shifts[j], p.factor);
  }
  info = lu_solve_inplace_with_info(backend, factor, work);
  if (!info.succeeded()) return {};
  DenseMatrix<W> result(n, count, p.working);
  for (std::size_t j = 0; j < count; ++j)
  {
    if (!active[j]) continue;
    for (std::size_t i = 0; i < n; ++i)
    {
      R scaled = convert_scaled<R>(F(work[i, j]), -shifts[j], p.residual);
      W value = convert<W>(scaled, p.working);
      if (!uni20::isfinite(value))
      {
        info.status = SolveStatus::nonfinite_result;
        return {};
      }
      result[i, j] = std::move(value);
    }
  }
  return result;
}

template <class Backend, class F, class W, class R>
LuRefinementResult<W, R> run(Backend& backend, LuFactorization<F> const& factor, DenseMatrix<R> const& a,
                             DenseMatrix<R> const& b, LuRefinementPrecision<F, W, R> p,
                             RefinementOptions<make_real_t<R>> const& options, make_real_t<R> const& tolerance)
{
  std::size_t n = a.extent(0), count = b.extent(1);
  LuRefinementResult<W, R> result;
  SolveInfo solve_info;
  std::vector<bool> active(count, true);
  auto upper_scale = scalar_at_precision<make_real_t<F>>(0, p.factor);
  auto pivot_scale = component_magnitude(F(factor.packed()[0, 0]));
  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t i = 0; i <= j; ++i)
    {
      auto magnitude = component_magnitude(F(factor.packed()[i, j]));
      upper_scale = std::max(upper_scale, magnitude);
      if (i == j) pivot_scale = std::min(pivot_scale, magnitude);
    }
  auto target_exponent = std::min(std::int64_t{0}, math::frexp(upper_scale).exponent);
  if constexpr (numeric_limits<make_real_t<F>>::is_bounded)
  {
    // Leave significand-sized exponent headroom when dividing by the smallest
    // diagonal. This limits inflation for disparate pivots without sacrificing
    // normalization of tiny RHS values against uniformly large factors.
    using FR = make_real_t<F>;
    auto pivot_exponent = math::frexp(pivot_scale).exponent;
    std::int64_t headroom = numeric_limits<FR>::max_exponent - numeric_limits<FR>::digits(p.factor);
    target_exponent = std::min(target_exponent, pivot_exponent + headroom);
  }
  auto initial = correction(backend, factor, b, p, active, target_exponent, solve_info);
  if (!initial)
  {
    result.info = {.status = RefinementStatus::correction_failed, .solve_info = solve_info};
    return result;
  }
  auto rx = pack<R>(*initial, p.residual, solve_info);
  if (!rx)
  {
    result.info = conversion_failure(solve_info);
    return result;
  }
  auto evaluated = evaluate(a, b, *rx, p.residual);
  if (!evaluated.info.succeeded())
  {
    result.info = conversion_failure(evaluated.info);
    return result;
  }
  result.solution.emplace(std::move(*initial));
  for (auto const& error : evaluated.columns)
    result.columns.push_back({.error = error, .converged = error.componentwise_backward_error <= tolerance});

  for (std::size_t step = 0;; ++step)
  {
    bool all_converged = true, any_active = false;
    for (std::size_t j = 0; j < count; ++j)
    {
      all_converged = all_converged && result.columns[j].converged;
      active[j] = !result.columns[j].converged && !result.columns[j].stagnated;
      any_active = any_active || active[j];
    }
    if (all_converged) return result;
    if (!any_active || step == options.max_iterations)
    {
      result.info.status = !any_active ? RefinementStatus::stagnated : RefinementStatus::iteration_limit;
      return result;
    }
    auto delta = correction(backend, factor, *evaluated.residual, p, active, target_exponent, solve_info);
    if (!delta)
    {
      result.info = {.status = RefinementStatus::correction_failed, .solve_info = solve_info};
      return result;
    }
    // Publish only after every new value and the recomputed diagnostics pass.
    // This owns a finite W iterate already at p.working, so copying it at the
    // same precision cannot return a numerical conversion failure.
    auto next = pack<W>(*result.solution, p.working, solve_info);
    std::vector<bool> changed(count, false);
    for (std::size_t j = 0; j < count; ++j)
      if (active[j])
        for (std::size_t i = 0; i < n; ++i)
        {
          W value = (*next)[i, j] + (*delta)[i, j];
          if (!uni20::isfinite(value))
          {
            result.info.status = RefinementStatus::nonfinite_result;
            return result;
          }
          changed[j] = changed[j] || value != (*next)[i, j];
          (*next)[i, j] = std::move(value);
        }
    auto next_r = pack<R>(*next, p.residual, solve_info);
    if (!next_r)
    {
      result.info = conversion_failure(solve_info);
      return result;
    }
    auto next_evaluated = evaluate(a, b, *next_r, p.residual);
    if (!next_evaluated.info.succeeded())
    {
      result.info = conversion_failure(next_evaluated.info);
      return result;
    }
    result.solution.emplace(std::move(*next));
    evaluated = std::move(next_evaluated);
    for (std::size_t j = 0; j < count; ++j)
    {
      auto& column = result.columns[j];
      column.error = evaluated.columns[j];
      column.converged = column.error.componentwise_backward_error <= tolerance;
      if (active[j])
      {
        ++column.corrections;
        column.stagnated = !column.converged && !changed[j];
      }
    }
  }
}

template <class A, class B> void require_shape(A const& a, B const& b)
{
  ERROR_IF(a.extent(0) != a.extent(1) || a.extent(0) != b.extent(0),
           "LU refinement requires square A and matching RHS rows");
}
} // namespace refinement_detail

/// \brief Refine a preserving solve using existing LU factors and the original A and B.
/// \details Factors must describe A rounded to the requested factor scalar and
///          precision. This correspondence is a caller precondition. The factor
///          precision must match p.factor. A and B are evaluated at p.residual;
///          the owning result uses W at p.working. Each RHS freezes once its
///          componentwise backward error meets the target. A correction that
///          cannot change the iterate signals stagnation. Failed later steps
///          preserve the last finite iterate and its recomputed diagnostics.
///          Empty systems/RHS sets succeed without inspecting numerical values.
///          All work is synchronous and host-resident; backend selects the LU
///          solve, while residual evaluation and updates use scalar arithmetic.
template <KernelBackendSelector Backend, RealOrComplex F, RealOrComplex W, RealOrComplex R,
          refinement_detail::HostMatrix A, refinement_detail::HostMatrix B>
  requires(Complex<F> == Complex<tensor_element_t<A>> && Complex<F> == Complex<tensor_element_t<B>>)
[[nodiscard]] auto lu_refine_with_info(Backend&& backend, LuFactorization<F> const& factor, A const& a, B const& b,
                                       LuRefinementPrecision<F, W, R> p,
                                       RefinementOptions<make_real_t<R>> const& options = {})
{
  auto tolerance = refinement_detail::validate(p, options);
  refinement_detail::require_shape(a, b);
  ERROR_IF(factor.order() != std::size_t(a.extent(0)) || factor.precision() != p.factor,
           "refinement factors must match the matrix order and factor precision");
  if (a.extent(0) == 0 || b.extent(1) == 0) return refinement_detail::empty_result<W, R>(a, b, p.working, p.residual);
  LuRefinementResult<W, R> result;
  SolveInfo info;
  auto av = refinement_detail::pack<R>(a, p.residual, info);
  auto bv = av ? refinement_detail::pack<R>(b, p.residual, info) : std::nullopt;
  if (!av || !bv)
  {
    result.info = refinement_detail::conversion_failure(info);
    return result;
  }
  return refinement_detail::run(backend, factor, *av, *bv, p, options, tolerance);
}

/// \brief Factor at p.factor, then solve and refine with independently selected working/residual arithmetic.
/// \details Preserves A and B. Factor conversion is directly from the original
///          logical A, independently of residual conversion. Factorization and
///          correction failures report their SolveInfo; no automatic retry or
///          precision escalation occurs. Diagnostic norm capture is disabled by
///          default for the temporary LU. Invalid shapes, precision arguments,
///          options and unavailable providers are ordinary errors.
template <KernelBackendSelector Backend, RealOrComplex F, RealOrComplex W, RealOrComplex R,
          refinement_detail::HostMatrix A, refinement_detail::HostMatrix B>
  requires(Complex<F> == Complex<tensor_element_t<A>> && Complex<F> == Complex<tensor_element_t<B>>)
[[nodiscard]] auto lu_refine_with_info(Backend&& backend, A const& a, B const& b, LuRefinementPrecision<F, W, R> p,
                                       RefinementOptions<make_real_t<R>> const& options = {},
                                       LuOptions<make_real_t<F>> const& factor_options = {.retain_norms = false})
{
  auto tolerance = refinement_detail::validate(p, options);
  refinement_detail::require_shape(a, b);
  detail::require_solve_options(SolveOptions<make_real_t<F>>{factor_options.relative_pivot_tolerance});
  if (a.extent(0) == 0 || b.extent(1) == 0) return refinement_detail::empty_result<W, R>(a, b, p.working, p.residual);
  LuRefinementResult<W, R> result;
  SolveInfo info;
  auto av = refinement_detail::pack<R>(a, p.residual, info);
  auto bv = av ? refinement_detail::pack<R>(b, p.residual, info) : std::nullopt;
  auto af = bv ? refinement_detail::pack<F>(a, p.factor, info) : std::nullopt;
  if (!av || !bv || !af)
  {
    result.info = refinement_detail::conversion_failure(info);
    return result;
  }
  auto factor = lu_factor_with_info(backend, *af, p.factor, factor_options);
  if (!factor.info.succeeded())
  {
    result.info = {.status = RefinementStatus::factorization_failed, .solve_info = factor.info};
    return result;
  }
  return refinement_detail::run(backend, *factor.factor, *av, *bv, p, options, tolerance);
}

/// \brief Refine with the ordinary host backend selection for the factor scalar.
template <RealOrComplex F, RealOrComplex W, RealOrComplex R, refinement_detail::HostMatrix A,
          refinement_detail::HostMatrix B>
  requires(Complex<F> == Complex<tensor_element_t<A>> && Complex<F> == Complex<tensor_element_t<B>>)
[[nodiscard]] auto lu_refine_with_info(A const& a, B const& b, LuRefinementPrecision<F, W, R> p,
                                       RefinementOptions<make_real_t<R>> const& options = {},
                                       LuOptions<make_real_t<F>> const& factor_options = {.retain_norms = false})
{
  return lu_refine_with_info(select_backend_for<DenseMatrix<F>>(lu_factor_op{}), a, b, p, options, factor_options);
}

/// \brief Refine from reusable factors with ordinary host solve dispatch.
template <RealOrComplex F, RealOrComplex W, RealOrComplex R, refinement_detail::HostMatrix A,
          refinement_detail::HostMatrix B>
  requires(Complex<F> == Complex<tensor_element_t<A>> && Complex<F> == Complex<tensor_element_t<B>>)
[[nodiscard]] auto lu_refine_with_info(LuFactorization<F> const& factor, A const& a, B const& b,
                                       LuRefinementPrecision<F, W, R> p,
                                       RefinementOptions<make_real_t<R>> const& options = {})
{
  return lu_refine_with_info(select_backend_for<DenseMatrix<F>>(lu_solve_op{}), factor, a, b, p, options);
}

/// \brief Return the refined solution only on convergence; other outcomes use the terminal error policy.
template <class... Args>
  requires requires(Args&&... args) { lu_refine_with_info(std::forward<Args>(args)...); }
[[nodiscard]] auto lu_refine(Args&&... args)
{
  auto result = lu_refine_with_info(std::forward<Args>(args)...);
  ERROR_IF(!result.info.succeeded(), "LU refinement failed", static_cast<int>(result.info.status));
  return std::move(*result.solution);
}
} // namespace uni20::linalg
