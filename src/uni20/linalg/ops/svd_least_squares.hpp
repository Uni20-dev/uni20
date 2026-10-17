#pragma once

/**
 * \file svd_least_squares.hpp
 * \brief Precision-generic minimum-norm SVD least squares with recoverable rank diagnostics.
 */
#include <uni20/linalg/backends/lapack/svd_least_squares.hpp>
#include <uni20/linalg/precision.hpp>
#include <uni20/tensor/precision.hpp>

namespace uni20::linalg
{
/// \brief Preserving least-squares result; solution and residuals are published only on success.
template <RealOrComplex S> struct SvdLeastSquaresResult
{
    LeastSquaresInfo info = {};
    std::optional<DenseMatrix<S>> solution = {};
    /// \brief Recomputed Euclidean residual norm per RHS, in working precision; not an error bound.
    std::vector<make_real_t<S>> residual_norms = {};
    /// \brief Present exactly on success, including rank zero.
    std::optional<SvdLeastSquaresDiagnostics<make_real_t<S>>> diagnostics = {};
};

/// \brief Compute a minimum-norm solution after singular-value truncation for rectangular A.
/// \details Both inputs are preserved. A finite precision is required; fixed
///          scalars accept their native precision only. Runtime inputs and the
///          cutoff convert in the backend. Rank-deficient and underdetermined
///          inputs are supported. Rank is computed even without RHS columns;
///          zero unknowns return the norms of B, and zero rows give the zero solution.
/// \return Owning host solution and residuals on success, otherwise only failure diagnostics.
template <KernelBackendSelector Backend, RankedTensorView<2> A, RankedTensorView<2> B>
  requires RealOrComplex<tensor_element_t<A>> && std::same_as<tensor_element_t<A>, tensor_element_t<B>>
[[nodiscard]] auto
svd_least_squares_with_info(Backend&& backend, A const& a, B const& b, Precision p,
                            SvdLeastSquaresOptions<make_real_t<tensor_element_t<A>>> const& options = {})
{
  using S = tensor_element_t<A>;
  require_precision<S>(p);
  (void)p.bit_count();
  ERROR_IF(a.extent(0) != b.extent(0), "SVD least squares requires matching RHS rows");
  least_squares_detail::require_svd_options(options);
  SvdLeastSquaresResult<S> result;
  result.solution.emplace(uninitialized, a.extent(1), b.extent(1), p);
  result.residual_norms.resize(b.extent(1));
  result.diagnostics.emplace();
  auto ad = mdspec_of(a);
  auto bd = mdspec_of(b);
  auto xd = mdspec_of(*result.solution);
  detail::dispatch_at_precision<S>(std::forward<Backend>(backend), svd_least_squares_op{}, p, xd, ad, bd,
                                   std::span(result.residual_norms), result.info, *result.diagnostics, options);
  if (!result.info.succeeded())
  {
    result.solution.reset();
    result.residual_norms.clear();
    result.diagnostics.reset();
  }
  return result;
}

/// \brief Infer working precision from input defaults, requiring matching finite defaults when both exist.
template <KernelBackendSelector Backend, RankedTensorView<2> A, RankedTensorView<2> B>
  requires RealOrComplex<tensor_element_t<A>> && std::same_as<tensor_element_t<A>, tensor_element_t<B>>
[[nodiscard]] auto
svd_least_squares_with_info(Backend&& backend, A const& a, B const& b,
                            SvdLeastSquaresOptions<make_real_t<tensor_element_t<A>>> const& options = {})
{
  return svd_least_squares_with_info(std::forward<Backend>(backend), a, b, common_default_precision(a, b), options);
}

/// \brief Use the ordinary backend selector for the owning host result.
template <RankedTensorView<2> A, RankedTensorView<2> B>
  requires RealOrComplex<tensor_element_t<A>> && std::same_as<tensor_element_t<A>, tensor_element_t<B>>
[[nodiscard]] auto
svd_least_squares_with_info(A const& a, B const& b, Precision p,
                            SvdLeastSquaresOptions<make_real_t<tensor_element_t<A>>> const& options = {})
{
  return svd_least_squares_with_info(select_backend_for<DenseMatrix<tensor_element_t<A>>>(svd_least_squares_op{}), a, b,
                                     p, options);
}

/// \brief Infer precision and select the ordinary host-result backends.
template <RankedTensorView<2> A, RankedTensorView<2> B>
  requires RealOrComplex<tensor_element_t<A>> && std::same_as<tensor_element_t<A>, tensor_element_t<B>>
[[nodiscard]] auto
svd_least_squares_with_info(A const& a, B const& b,
                            SvdLeastSquaresOptions<make_real_t<tensor_element_t<A>>> const& options = {})
{
  return svd_least_squares_with_info(a, b, common_default_precision(a, b), options);
}

/// \brief Return a preserving truncated-SVD solution; numerical failure uses the ordinary terminal error policy.
template <class... Args>
  requires requires(Args&&... args) { svd_least_squares_with_info(std::forward<Args>(args)...); }
[[nodiscard]] auto svd_least_squares(Args&&... args)
{
  auto result = svd_least_squares_with_info(std::forward<Args>(args)...);
  ERROR_IF(!result.info.succeeded(), "SVD least squares failed", static_cast<int>(result.info.status));
  return std::move(*result.solution);
}
} // namespace uni20::linalg
