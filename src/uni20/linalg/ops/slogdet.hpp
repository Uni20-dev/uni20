#pragma once

/**
 * \file slogdet.hpp
 * \brief Signed or complex-phase logarithmic determinants from reusable LU.
 */

#include "lu.hpp"

namespace uni20::linalg
{
/// \brief Determinant represented without multiplying diagonal magnitudes.
/// \details Phase is a real sign or complex unit phase. A zero determinant has
///          phase zero and log_absolute negative infinity, not a complex-log branch.
template <RealOrComplex S> struct LogDeterminant
{
    S phase;
    make_real_t<S> log_absolute;
};

/// \brief Logarithmic determinant with numerical diagnostics.
/// \details Singular has the conventional zero value; unresolved or nonfinite
///          outcomes have no value. Success does not certify conditioning.
template <RealOrComplex S> struct LogDeterminantResult
{
    SolveInfo info;
    std::optional<LogDeterminant<S>> value;
};

namespace detail
{
template <RealOrComplex S, class... P> LogDeterminant<S> zero_determinant(P... precision)
{
  using std::log;
  auto zero = lu_scalar<S>(0, precision...);
  return {.phase = zero, .log_absolute = log(uni20::real(zero))};
}

template <RealOrComplex S, class... P>
LogDeterminantResult<S> reduce_log_determinant(LuFactorization<S> const& factor, P... precision)
{
  using std::abs;
  using std::log;
  auto phase = lu_scalar<S>(1, precision...);
  auto sum = uni20::real(lu_scalar<S>(0, precision...));
  auto correction = sum;
  auto const& a = factor.packed();
  for (std::size_t k = 0; k < factor.order(); ++k)
  {
    auto diagonal = a[k, k];
    auto magnitude = lu_magnitude(diagonal);
    auto term = log(magnitude.scale) + log(magnitude.unit);
    // Neumaier reduction retains small contributions when logarithms cancel.
    auto next = sum + term;
    if (abs(sum) >= abs(term))
      correction += (sum - next) + term;
    else
      correction += (term - next) + sum;
    sum = std::move(next);
    if constexpr (Complex<S>)
    {
      phase *= (diagonal / magnitude.scale) / magnitude.unit;
      // Avoid accumulated drift of the product away from unit modulus.
      auto phase_magnitude = lu_magnitude(phase);
      phase = (phase / phase_magnitude.scale) / phase_magnitude.unit;
    }
    else if (diagonal < 0)
      phase = -phase;
    if (factor.pivots()[k] != k) phase = -phase;
    if (!uni20::isfinite(sum) || !uni20::isfinite(correction) || !uni20::isfinite(phase))
      return {.info = {.status = SolveStatus::nonfinite_result}, .value = {}};
  }
  sum += correction;
  if (!uni20::isfinite(sum)) return {.info = {.status = SolveStatus::nonfinite_result}, .value = {}};
  return {.info = {}, .value = LogDeterminant<S>{std::move(phase), std::move(sum)}};
}
} // namespace detail

/// \brief Reduce completed factors in O(n), without refactoring or changing them.
/// \details Uses factor precision and scaled complex magnitudes. A reduction
///          failure reports nonfinite_result and leaves the factors usable.
template <RealOrComplex S> [[nodiscard]] LogDeterminantResult<S> slogdet_with_info(LuFactorization<S> const& factor)
{
#if UNI20_ENABLE_MPFR
  if constexpr (has_runtime_precision_v<S>)
    return detail::reduce_log_determinant(factor, factor.precision());
  else
#endif
    return detail::reduce_log_determinant(factor);
}

namespace detail
{
template <KernelBackendSelector Backend, RankedTensorView<2> A, class... P>
auto slogdet_matrix_impl(Backend&& backend, A const& a, SolveOptions<make_real_t<tensor_element_t<A>>> const& options,
                         P... precision)
{
  using S = tensor_element_t<A>;
  auto attempt = lu_factor_impl(std::forward<Backend>(backend), a, options, precision...);
  if (attempt.factor) return slogdet_with_info(*attempt.factor);
  LogDeterminantResult<S> result{.info = attempt.info, .value = {}};
  if (attempt.info.status == SolveStatus::singular) result.value = zero_determinant<S>(precision...);
  return result;
}
} // namespace detail

#if UNI20_ENABLE_MPFR
/// \brief Factor once at explicit finite precision and return a recoverable log-determinant.
template <KernelBackendSelector Backend, RankedTensorView<2> A>
  requires RealOrComplex<tensor_element_t<A>> && has_runtime_precision_v<tensor_element_t<A>>
[[nodiscard]] auto slogdet_with_info(Backend&& backend, A const& a, Precision p,
                                     SolveOptions<mpreal> const& options = {})
{
  return detail::slogdet_matrix_impl(std::forward<Backend>(backend), a, options, p);
}
#endif

/// \brief Factor once at native or inferred finite precision and reduce its diagonal.
/// \details Empty input gives phase one and log zero. Singular gives phase zero,
///          log negative infinity and a singular diagnostic; threshold rejection
///          or nonfinite arithmetic gives no determinant value.
template <KernelBackendSelector Backend, RankedTensorView<2> A>
  requires RealOrComplex<tensor_element_t<A>>
[[nodiscard]] auto slogdet_with_info(Backend&& backend, A const& a,
                                     SolveOptions<make_real_t<tensor_element_t<A>>> const& options = {})
{
#if UNI20_ENABLE_MPFR
  if constexpr (has_runtime_precision_v<tensor_element_t<A>>)
    return slogdet_with_info(std::forward<Backend>(backend), a, a.default_precision(), options);
  else
#endif
    return detail::slogdet_matrix_impl(std::forward<Backend>(backend), a, options);
}

/// \brief Factor and reduce with the default backends for the owned host workspace.
template <RankedTensorView<2> A, class... Options>
  requires RealOrComplex<tensor_element_t<A>>
[[nodiscard]] auto slogdet_with_info(A const& a, Options const&... options)
{
  auto backend = select_backend_for<DenseMatrix<tensor_element_t<A>>>(lu_factor_op{});
  return slogdet_with_info(backend, a, options...);
}

/// \brief Return the log-determinant, including conventional zero for singular matrices.
/// \details Unresolved/nonfinite outcomes use the terminal numerical-error policy.
template <class... Args>
  requires requires(Args&&... args) { slogdet_with_info(std::forward<Args>(args)...); }
[[nodiscard]] auto slogdet(Args&&... args)
{
  auto result = slogdet_with_info(std::forward<Args>(args)...);
  ERROR_IF(!result.value, "logarithmic determinant could not be resolved", static_cast<int>(result.info.status));
  return std::move(*result.value);
}
} // namespace uni20::linalg
