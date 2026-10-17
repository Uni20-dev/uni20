#pragma once

/**
 * \file lu.hpp
 * \brief Owning reusable host LU factors and solves through operation dispatch.
 */

#include <uni20/linalg/backends/cpu/lu.hpp>
#include <uni20/linalg/backends/lapack/lu.hpp>
#if UNI20_HAS_MPLAPACK_FLOAT80
#include <uni20/linalg/backends/lapack/lu_binary80.hpp>
#endif
#include <optional>
#include <span>
#include <uni20/linalg/precision.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/precision.hpp>
#include <uni20/tensor/tensor.hpp>
#include <vector>

namespace uni20::linalg
{
namespace detail
{
struct LuFactorAccess;
}

/// \brief Immutable owning packed factors satisfying P A = L U.
/// \details Created only by successful factorization. P applies the stored row
///          swaps in elimination order; the unit diagonal of L is implicit.
///          Copies own their data. A moved-from object supports destruction and
///          reassignment; other operations require an intact factorization.
///          Factor storage cannot overlap a solve's writable RHS.
template <RealOrComplex S> class LuFactorization {
  public:
    using value_type = S;
    /// \brief Matrix order, including zero for a successfully factored empty matrix.
    std::size_t order() const noexcept { return std::size_t(factors_.extent(0)); }
    /// \brief Read-only packed L/U matrix, valid while this object retains its storage.
    DenseMatrix<S> const& packed() const& noexcept { return factors_; }
    /// \brief Zero-based swap sequence; at step k exchange rows k and pivots[k].
    std::span<std::size_t const> pivots() const& noexcept { return pivots_; }
    /// \brief Finite working precision used for these factors and subsequent solves.
    Precision precision() const { return precision_of(factors_); }

  private:
    friend struct detail::LuFactorAccess;
    LuFactorization(DenseMatrix<S>&& factors, std::vector<std::size_t>&& pivots)
        : factors_(std::move(factors)), pivots_(std::move(pivots))
    {}
    DenseMatrix<S> factors_;
    std::vector<std::size_t> pivots_;
};

/// \brief Completed factorization attempt; factor is present exactly on success.
template <RealOrComplex S> struct LuFactorizationResult
{
    SolveInfo info;
    std::optional<LuFactorization<S>> factor;
};

namespace detail
{
struct LuFactorAccess
{
    template <class S> static auto make(DenseMatrix<S>&& factors, std::vector<std::size_t>&& pivots)
    {
      return LuFactorization<S>(std::move(factors), std::move(pivots));
    }
};

template <KernelBackendSelector Backend, RankedTensorView<2> A>
auto lu_factor_impl(Backend&& backend, A const& a, SolveOptions<make_real_t<tensor_element_t<A>>> const& options,
                    Precision p)
{
  using S = tensor_element_t<A>;
  ERROR_IF(a.extent(0) != a.extent(1), "LU requires a square matrix");
  detail::require_solve_options(options);
  require_precision<S>(p);
  (void)p.bit_count();
  auto work = make_tensor<ColumnMajor>(a);
  std::vector<std::size_t> pivots(std::size_t(a.extent(0)));
  LuFactorizationResult<S> result{.info = {}, .factor = {}};
  auto descriptor = mdspec_of(work);
  dispatch_at_precision<S>(std::forward<Backend>(backend), lu_factor_op{}, p, descriptor, std::span(pivots),
                           result.info, options);
  if (result.info.succeeded())
  {
    uni20::detail::record_output_precision(work, p);
    result.factor = LuFactorAccess::make(std::move(work), std::move(pivots));
  }
  return result;
}
inline void require_lu_success(SolveInfo const& info)
{
  ERROR_IF(!info.succeeded(), "LU operation failed", static_cast<int>(info.status));
}
} // namespace detail

/// \brief Factor at explicit finite precision, preserving input values and defaults.
/// \details Conversion occurs in the backend. Exact inputs require this finite
///          override. Numerical failures return diagnostics with no factor.
template <KernelBackendSelector Backend, RankedTensorView<2> A>
  requires RealOrComplex<tensor_element_t<A>>
[[nodiscard]] auto lu_factor_with_info(Backend&& backend, A const& a, Precision p,
                                       SolveOptions<make_real_t<tensor_element_t<A>>> const& options = {})
{
  return detail::lu_factor_impl(std::forward<Backend>(backend), a, options, p);
}

/// \brief Factor at native precision or the input tensor's finite construction default.
/// \details Preserves the logical input, including accessor transformations, in
///          owned column-major host work. Shape, options and dispatch errors are
///          ordinary errors; numerical outcomes are returned in SolveInfo.
template <KernelBackendSelector Backend, RankedTensorView<2> A>
  requires RealOrComplex<tensor_element_t<A>>
[[nodiscard]] auto lu_factor_with_info(Backend&& backend, A const& a,
                                       SolveOptions<make_real_t<tensor_element_t<A>>> const& options = {})
{
  return detail::lu_factor_impl(std::forward<Backend>(backend), a, options, precision_of(a));
}

/// \brief Factor with the default backend for the materialized host workspace.
template <RankedTensorView<2> A, class... Options>
  requires RealOrComplex<tensor_element_t<A>>
[[nodiscard]] auto lu_factor_with_info(A const& a, Options const&... options)
{
  auto backend = select_backend_for<DenseMatrix<tensor_element_t<A>>>(lu_factor_op{});
  return lu_factor_with_info(backend, a, options...);
}

/// \brief Return completed owning factors; numerical failure uses the terminal error policy.
template <class... Args>
  requires requires(Args&&... args) { lu_factor_with_info(std::forward<Args>(args)...); }
[[nodiscard]] auto lu_factor(Args&&... args)
{
  auto result = lu_factor_with_info(std::forward<Args>(args)...);
  detail::require_lu_success(result.info);
  return std::move(*result.factor);
}

/// \brief Solve in an existing RHS workspace, preserving the factors on every outcome.
/// \details Finite RHS values convert to the factors' precision, regardless of
///          their construction default. Nonfinite input preserves the RHS; later
///          numerical failure may change it. Empty solves do not inspect values.
///          A successful nonempty owning RHS records factor precision; views do not.
/// \pre RHS storage does not overlap factor storage.
template <KernelBackendSelector Backend, RealOrComplex S, MutableRankedTensorView<2> B>
  requires std::same_as<S, tensor_element_t<B>>
[[nodiscard]] SolveInfo lu_solve_inplace_with_info(Backend&& backend, LuFactorization<S> const& factor, B&& b)
{
  ERROR_IF(factor.order() != std::size_t(b.extent(0)), "LU solve RHS row count mismatch");
  auto ad = mdspec_of(factor.packed());
  auto bd = mdspec_of(b);
  SolveInfo info;
  auto p = factor.precision();
  detail::dispatch_at_precision<S>(std::forward<Backend>(backend), lu_solve_op{}, p, ad, factor.pivots(), bd, info);
  if (info.succeeded() && factor.order() != 0 && b.extent(1) != 0) uni20::detail::record_output_precision(b, p);
  return info;
}

/// \brief Solve in place using the factors' and RHS storage-selected backends.
template <RealOrComplex S, MutableRankedTensorView<2> B>
  requires std::same_as<S, tensor_element_t<B>>
[[nodiscard]] SolveInfo lu_solve_inplace_with_info(LuFactorization<S> const& factor, B&& b)
{
  auto backend = select_backend(lu_solve_op{}, factor.packed(), b);
  return lu_solve_inplace_with_info(backend, factor, std::forward<B>(b));
}

/// \brief Strict solve in an existing RHS workspace; numerical failure is terminal.
template <class... Args>
  requires requires(Args&&... args) { lu_solve_inplace_with_info(std::forward<Args>(args)...); }
void lu_solve_inplace(Args&&... args)
{
  detail::require_lu_success(lu_solve_inplace_with_info(std::forward<Args>(args)...));
}

/// \brief Preserve the RHS and return an owning column-major solution at factor precision.
template <KernelBackendSelector Backend, RealOrComplex S, RankedTensorView<2> B>
  requires std::same_as<S, tensor_element_t<B>>
[[nodiscard]] auto lu_solve(Backend&& backend, LuFactorization<S> const& factor, B const& b)
{
  auto result = make_tensor<ColumnMajor>(b);
  lu_solve_inplace(std::forward<Backend>(backend), factor, result);
  uni20::detail::record_output_precision(result, factor.precision());
  return result;
}

/// \brief Preserve the RHS and solve with default host backends.
template <RealOrComplex S, RankedTensorView<2> B>
  requires std::same_as<S, tensor_element_t<B>>
[[nodiscard]] auto lu_solve(LuFactorization<S> const& factor, B const& b)
{
  auto backend = select_backend_for<DenseMatrix<S>, DenseMatrix<S>>(lu_solve_op{});
  return lu_solve(backend, factor, b);
}
} // namespace uni20::linalg
