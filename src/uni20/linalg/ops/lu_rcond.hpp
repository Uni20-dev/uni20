#pragma once

#include <uni20/linalg/backends/lapack/lu_rcond.hpp>
#include <uni20/linalg/ops/lu.hpp>

namespace uni20::linalg
{
/// \brief Reciprocal condition estimate, present only on numerical success.
/// \details Estimates 1/(norm(A)*norm(inv(A))) in the stated norm, at factor
///          precision. It is not an error bound. Zero may indicate underflow
///          or an unusably ill-conditioned matrix; it is a successful estimate.
template <Real R> struct LuConditionResult
{
    SolveInfo info = {};
    MatrixNorm norm = MatrixNorm::One;
    std::optional<R> estimate = {};
};

/// \brief Estimate conditioning using an explicitly supplied original matrix norm.
/// \details Preserves factors on every outcome. The supplied norm must describe
///          A after conversion to factor precision, using mathematical complex
///          magnitudes. It converts to that precision in the backend. A nonfinite
///          norm or estimate reports nonfinite_result without invalidating LU.
///          Empty factors have reciprocal condition one.
template <KernelBackendSelector Backend, RealOrComplex S>
[[nodiscard]] auto lu_rcond_with_info(Backend&& backend, LuFactorization<S> const& factor, MatrixNorm norm,
                                      make_real_t<S> const& original_norm)
{
  (void)condition_detail::norm_code(norm);
  LuConditionResult<make_real_t<S>> result{.norm = norm};
  auto a = mdspec_of(factor.packed());
  make_real_t<S> value;
  detail::dispatch_at_precision<S>(std::forward<Backend>(backend), lu_rcond_op{norm}, factor.precision(), value, a,
                                   original_norm, result.info);
  if (result.info.succeeded()) result.estimate = std::move(value);
  return result;
}

/// \brief Estimate conditioning from retained original norms; missing capture is an ordinary error.
template <KernelBackendSelector Backend, RealOrComplex S>
[[nodiscard]] auto lu_rcond_with_info(Backend&& backend, LuFactorization<S> const& factor,
                                      MatrixNorm norm = MatrixNorm::One)
{
  (void)condition_detail::norm_code(norm);
  ERROR_IF(!factor.norms(), "LU norms were not retained; supply the original norm explicitly");
  return lu_rcond_with_info(std::forward<Backend>(backend), factor, norm,
                            norm == MatrixNorm::One ? factor.norms()->one : factor.norms()->infinity);
}

/// \brief Estimate with default dispatch and an explicit original norm.
template <RealOrComplex S>
[[nodiscard]] auto lu_rcond_with_info(LuFactorization<S> const& factor, MatrixNorm norm,
                                      make_real_t<S> const& original_norm)
{
  return lu_rcond_with_info(select_backend_for<DenseMatrix<S>>(lu_rcond_op{norm}), factor, norm, original_norm);
}

/// \brief Estimate with default dispatch and the norms retained during factorization.
template <RealOrComplex S>
[[nodiscard]] auto lu_rcond_with_info(LuFactorization<S> const& factor, MatrixNorm norm = MatrixNorm::One)
{
  return lu_rcond_with_info(select_backend_for<DenseMatrix<S>>(lu_rcond_op{norm}), factor, norm);
}

/// \brief Return the reciprocal condition estimate; numerical failure uses the terminal error policy.
template <class... Args>
  requires requires(Args&&... args) { lu_rcond_with_info(std::forward<Args>(args)...); }
[[nodiscard]] auto lu_rcond(Args&&... args)
{
  auto result = lu_rcond_with_info(std::forward<Args>(args)...);
  detail::require_lu_success(result.info);
  return std::move(*result.estimate);
}
} // namespace uni20::linalg
