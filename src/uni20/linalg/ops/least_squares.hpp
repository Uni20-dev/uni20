#pragma once

/**
 * \file least_squares.hpp
 * \brief Precision-generic full-column-rank least squares with recoverable diagnostics.
 */
#include <uni20/linalg/backends/cpu/least_squares.hpp>
#include <uni20/linalg/backends/lapack/least_squares.hpp>
#include <uni20/tensor/precision.hpp>

namespace uni20::linalg
{
/// \brief Preserving least-squares result; solution and residuals are published only on success.
template <RealOrComplex S> struct LeastSquaresResult
{
    LeastSquaresInfo info = {};
    std::optional<DenseMatrix<S>> solution = {};
    /// \brief Recomputed Euclidean residual norm per RHS, in working precision; not an error bound.
    std::vector<make_real_t<S>> residual_norms = {};
};

/// \brief Minimize each column's Euclidean residual for m-by-n A with m >= n.
/// \details Both inputs are preserved. A finite precision is required; fixed
///          scalars accept their native precision only. Runtime inputs and the
///          tolerance convert in the backend. Zero RHS columns are a no-op;
///          zero unknowns return the norms of B. Unsupported underdetermined
///          shapes and invalid options are errors, not numerical outcomes.
/// \return Owning host solution and residuals on success, otherwise only failure diagnostics.
template <KernelBackendSelector Backend, RankedTensorView<2> A, RankedTensorView<2> B>
  requires RealOrComplex<tensor_element_t<A>> && std::same_as<tensor_element_t<A>, tensor_element_t<B>>
[[nodiscard]] auto least_squares_with_info(Backend&& backend, A const& a, B const& b, Precision p,
                                           LeastSquaresOptions<make_real_t<tensor_element_t<A>>> const& options = {})
{
  using S = tensor_element_t<A>;
  require_precision<S>(p);
  (void)p.bit_count();
  ERROR_IF(a.extent(0) < a.extent(1) || a.extent(0) != b.extent(0),
           "least squares requires m >= n and matching RHS rows");
  least_squares_detail::require_options(options);
  LeastSquaresResult<S> result;
  result.solution.emplace(uninitialized, a.extent(1), b.extent(1), p);
  result.residual_norms.resize(b.extent(1));
  auto ad = mdspec_of(a);
  auto bd = mdspec_of(b);
  auto xd = mdspec_of(*result.solution);
  detail::dispatch_at_precision<S>(std::forward<Backend>(backend), least_squares_op{}, p, xd, ad, bd,
                                   std::span(result.residual_norms), result.info, options);
  if (!result.info.succeeded())
  {
    result.solution.reset();
    result.residual_norms.clear();
  }
  return result;
}

/// \brief Infer working precision from input defaults, requiring matching finite defaults when both exist.
template <KernelBackendSelector Backend, RankedTensorView<2> A, RankedTensorView<2> B>
  requires RealOrComplex<tensor_element_t<A>> && std::same_as<tensor_element_t<A>, tensor_element_t<B>>
[[nodiscard]] auto least_squares_with_info(Backend&& backend, A const& a, B const& b,
                                           LeastSquaresOptions<make_real_t<tensor_element_t<A>>> const& options = {})
{
  return least_squares_with_info(std::forward<Backend>(backend), a, b, common_default_precision(a, b), options);
}

/// \brief Use the ordinary backend selector for the owning host result.
template <RankedTensorView<2> A, RankedTensorView<2> B>
  requires RealOrComplex<tensor_element_t<A>> && std::same_as<tensor_element_t<A>, tensor_element_t<B>>
[[nodiscard]] auto least_squares_with_info(A const& a, B const& b, Precision p,
                                           LeastSquaresOptions<make_real_t<tensor_element_t<A>>> const& options = {})
{
  return least_squares_with_info(select_backend_for<DenseMatrix<tensor_element_t<A>>>(least_squares_op{}), a, b, p,
                                 options);
}

/// \brief Infer precision and select the ordinary host-result backends.
template <RankedTensorView<2> A, RankedTensorView<2> B>
  requires RealOrComplex<tensor_element_t<A>> && std::same_as<tensor_element_t<A>, tensor_element_t<B>>
[[nodiscard]] auto least_squares_with_info(A const& a, B const& b,
                                           LeastSquaresOptions<make_real_t<tensor_element_t<A>>> const& options = {})
{
  return least_squares_with_info(a, b, common_default_precision(a, b), options);
}

/// \brief Return a preserving least-squares solution; numerical failure uses the ordinary terminal error policy.
template <class... Args>
  requires requires(Args&&... args) { least_squares_with_info(std::forward<Args>(args)...); }
[[nodiscard]] auto least_squares(Args&&... args)
{
  auto result = least_squares_with_info(std::forward<Args>(args)...);
  ERROR_IF(!result.info.succeeded(), "least squares failed", static_cast<int>(result.info.status));
  return std::move(*result.solution);
}
} // namespace uni20::linalg
