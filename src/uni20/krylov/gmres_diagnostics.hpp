#pragma once

/**
 * \file gmres_diagnostics.hpp
 * \brief Optional Euclidean-norm error diagnostics for matrix-free solves.
 */

#include <uni20/core/math.hpp>
#include <uni20/core/runtime_precision.hpp>

#include <algorithm>
#include <cstdint>
#include <optional>
#include <stdexcept>

namespace uni20::krylov
{
/// \brief Provenance of a supplied operator quantity, not of a computed error.
enum class GmresEstimateKind
{
  estimate,
  upper_bound
};

/// \brief An application-supplied estimate or known upper bound.
template <Real R> struct GmresOperatorEstimate
{
    R value;
    GmresEstimateKind kind = GmresEstimateKind::estimate;
};

/// \brief Optional properties of the original operator in the vector space's 2-norm.
/// \details These describe the operator actually solved, including any gauge
///          constraint, rather than a preconditioner or projected Hessenberg matrix.
///          Supplying them incurs no additional operator applications. Inverse
///          norms and condition numbers assume an invertible operator.
template <Real R> struct GmresOperatorEstimates
{
    std::optional<GmresOperatorEstimate<R>> norm = {};
    std::optional<GmresOperatorEstimate<R>> inverse_norm = {};
    std::optional<GmresOperatorEstimate<R>> condition_number = {};
};

/// \brief Recomputed Euclidean norms for the published iterate.
template <Real R> struct GmresResidual
{
    R norm;
    R rhs_norm;
    R solution_norm;
};

/// \brief Error estimates evaluated from the residual and supplied operator data.
/// \details All outputs are estimates: even a known operator bound does not bound
///          rounding error in the computed residual. An overflowing diagnostic is
///          infinity and does not invalidate the solve. Missing information leaves
///          a field absent. Supplied provenance is retained at diagnostic precision.
template <Real R> struct GmresErrorEstimates
{
    GmresOperatorEstimates<R> operator_estimates = {};
    /// \brief ||r|| / (||A|| ||x_hat|| + ||b||), requiring an operator norm.
    std::optional<R> normwise_backward_error = {};
    /// \brief ||A^{-1}|| ||r||, requiring an inverse norm.
    std::optional<R> absolute_solution_error = {};
    /// \brief kappa(A) ||r|| / ||b||, relative to the exact solution norm.
    /// \details Requires a condition number or both norms; absent for zero RHS.
    ///          An explicitly supplied condition number takes precedence.
    std::optional<R> relative_solution_error = {};
};

namespace gmres_detail
{
template <Real R>
GmresOperatorEstimates<R> operator_estimates_at_precision(GmresOperatorEstimates<R> const& input, Precision p)
{
  auto convert = [&](auto const& entry, int minimum, bool strict) -> std::optional<GmresOperatorEstimate<R>> {
    if (!entry) return {};
    if (!uni20::isfinite(entry->value) || (strict ? entry->value <= minimum : entry->value < minimum))
      throw std::invalid_argument("GMRES requires finite norms >= 0, inverse norms > 0, and condition numbers >= 1");
    R value = scalar_at_precision<R>(entry->value, p);
    if (entry->kind == GmresEstimateKind::upper_bound && value < entry->value) value = math::next_up(value);
    if (!uni20::isfinite(value) || (strict ? value <= minimum : value < minimum) || (entry->value > 0 && value == 0))
      throw std::invalid_argument("GMRES operator estimate is not representable at diagnostic precision");
    return GmresOperatorEstimate<R>{value, entry->kind};
  };
  return {convert(input.norm, 0, false), convert(input.inverse_norm, 0, true),
          convert(input.condition_number, 1, false)};
}

// Keep products and denominators scaled until the final diagnostic is formed.
// For example, kappa*r/b can be representable even when r/b underflows.
// Wide exponents also accommodate sums of MPFR exponents without signed overflow.
template <Real R> struct DiagnosticMagnitude
{
    R fraction;
    __int128_t exponent;
};

template <Real R> DiagnosticMagnitude<R> magnitude(R const& value)
{
  auto parts = math::frexp(value);
  return {parts.fraction, parts.exponent};
}

template <Real R> R diagnostic_scale(R const& fraction, __int128_t exponent)
{
  auto shift =
      std::clamp<__int128_t>(exponent, numeric_limits<std::int64_t>::min(), numeric_limits<std::int64_t>::max());
  return math::ldexp(fraction, static_cast<std::int64_t>(shift));
}

template <Real R> DiagnosticMagnitude<R> multiply(DiagnosticMagnitude<R> const& a, DiagnosticMagnitude<R> const& b)
{
  return {a.fraction * b.fraction, a.exponent + b.exponent};
}

template <Real R> DiagnosticMagnitude<R> add(DiagnosticMagnitude<R> const& a, DiagnosticMagnitude<R> const& b)
{
  if (a.fraction == 0) return b;
  if (b.fraction == 0) return a;
  auto exponent = std::max(a.exponent, b.exponent);
  return {diagnostic_scale(a.fraction, a.exponent - exponent) + diagnostic_scale(b.fraction, b.exponent - exponent),
          exponent};
}

template <Real R> R ratio(DiagnosticMagnitude<R> const& a, DiagnosticMagnitude<R> const& b)
{
  if (a.fraction == 0) return a.fraction; // Define the zero-over-zero diagnostic as zero.
  return diagnostic_scale(R(a.fraction / b.fraction), a.exponent - b.exponent);
}

template <Real R>
std::optional<GmresErrorEstimates<R>> error_estimates(GmresResidual<R> const& residual,
                                                      GmresOperatorEstimates<R> const& properties)
{
  if (!properties.norm && !properties.inverse_norm && !properties.condition_number) return {};
  GmresErrorEstimates<R> result{.operator_estimates = properties};
  auto r = magnitude(residual.norm), b = magnitude(residual.rhs_norm);
  if (properties.norm)
    result.normwise_backward_error =
        ratio(r, add(multiply(magnitude(properties.norm->value), magnitude(residual.solution_norm)), b));
  if (properties.inverse_norm)
  {
    auto error = multiply(magnitude(properties.inverse_norm->value), r);
    result.absolute_solution_error = diagnostic_scale(error.fraction, error.exponent);
  }
  if (residual.rhs_norm > 0)
  {
    if (properties.condition_number)
      result.relative_solution_error = ratio(multiply(magnitude(properties.condition_number->value), r), b);
    else if (properties.norm && properties.inverse_norm)
      result.relative_solution_error =
          ratio(multiply(multiply(magnitude(properties.norm->value), magnitude(properties.inverse_norm->value)), r), b);
  }
  return result;
}
} // namespace gmres_detail
} // namespace uni20::krylov
