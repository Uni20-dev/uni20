#pragma once

#include <span>
#include <uni20/core/math.hpp>
#include <uni20/linalg/backends/packed_matrix.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/least_squares_info.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/access.hpp>

namespace uni20::linalg::least_squares_detail
{
template <class S, class... P>
concept PrecisionArgs = (has_runtime_precision_v<S> && sizeof...(P) == 1 && (std::same_as<P, Precision> && ...)) ||
                        (!has_runtime_precision_v<S> && sizeof...(P) == 0);
template <class S> Precision working_precision() { return native_precision<S>(); }
template <class S> Precision working_precision(Precision p) { return p; }

template <class X, class A, class B>
concept Operands =
    MutableRankedMdspecLike<X, 2> && HostWritableMdspec<X> && RankedMdspecLike<A, 2> && HostReadableMdspec<A> &&
    RankedMdspecLike<B, 2> && HostReadableMdspec<B> && RealOrComplex<typename A::value_type> &&
    std::same_as<typename X::value_type, typename A::value_type> &&
    std::same_as<typename B::value_type, typename A::value_type>;

template <Real R> void require_options(LeastSquaresOptions<R> const& options)
{
  ERROR_IF(!uni20::isfinite(options.relative_diagonal_tolerance) || options.relative_diagonal_tolerance < 0,
           "least squares requires a finite nonnegative diagonal tolerance");
}
template <class Matrix> bool finite(Matrix const& a)
{
  for (std::size_t j = 0; j < std::size_t(a.extent(1)); ++j)
    for (std::size_t i = 0; i < std::size_t(a.extent(0)); ++i)
      if (!uni20::isfinite(typename Matrix::value_type(a[i, j]))) return false;
  return true;
}

template <class Matrix, Real R>
bool check_diagonal(Matrix const& a, R const& scale, R const& tolerance, LeastSquaresInfo& info)
{
  for (std::size_t k = 0; k < std::size_t(a.extent(1)); ++k)
  {
    R diagonal = math::abs(a[k, k]);
    if (!uni20::isfinite(diagonal))
      info.status = LeastSquaresStatus::nonfinite_result;
    else if (diagonal == 0)
      info = {.status = LeastSquaresStatus::zero_diagonal, .column = k};
    else if (tolerance != 0 && diagonal / scale <= tolerance)
      info = {.status = LeastSquaresStatus::small_diagonal, .column = k};
    if (!info.succeeded()) return false;
  }
  return true;
}

// Recompute against the original converted inputs, independently of factorization.
template <class S>
bool residual_norms(DenseMatrix<S> const& av, DenseMatrix<S> const& bv, DenseMatrix<S> const& solution,
                    std::vector<make_real_t<S>>& norms, Precision p)
{
  using R = make_real_t<S>;
  std::size_t m = av.extent(0), n = av.extent(1), nrhs = bv.extent(1);
  auto zero = scalar_at_precision<R>(0, p);
  norms.assign(nrhs, zero);
  // Reuse one RHS-sized scratch vector while accumulating along contiguous columns.
  std::vector<S> error(m);
  for (std::size_t j = 0; j < nrhs; ++j)
  {
    for (std::size_t i = 0; i < m; ++i)
      error[i] = -bv[i, j];
    for (std::size_t k = 0; k < n; ++k)
    {
      S const& value = solution[k, j];
      for (std::size_t i = 0; i < m; ++i)
        error[i] += av[i, k] * value;
    }
    for (std::size_t i = 0; i < m; ++i)
    {
      norms[j] = math::hypot(norms[j], math::abs(error[i]));
      if (!uni20::isfinite(norms[j]))
      {
        return false;
      }
    }
  }
  return true;
}

// Both implementations transform owned A and B into an upper triangular system.
// Conversion, checks, back substitution and residual reporting are shared.
template <class X, class A, class B, class Factor>
KernelAttempt run(X& x, A& a, B& b, std::span<make_real_t<typename A::value_type>> residuals, LeastSquaresInfo& info,
                  LeastSquaresOptions<make_real_t<typename A::value_type>> const& options, Precision p, Factor factor)
{
  using S = typename A::value_type;
  using R = make_real_t<S>;
  require_precision<S>(p);
  (void)p.bit_count();
  require_options(options);
  std::size_t m = a.extent(0), n = a.extent(1), nrhs = b.extent(1);
  ERROR_IF(m < n || std::size_t(b.extent(0)) != m || std::size_t(x.extent(0)) != n ||
               std::size_t(x.extent(1)) != nrhs || residuals.size() != nrhs,
           "incompatible full-rank least-squares shapes");
  info = {};
  if (nrhs == 0) return KernelAttempt::success;
  auto aa = acquire_host_read_access_sync(a);
  auto ba = acquire_host_read_access_sync(b);
  auto as = aa.mdspan();
  auto bs = ba.mdspan();
  if (!finite(as) || !finite(bs))
  {
    info.status = LeastSquaresStatus::nonfinite_input;
    return KernelAttempt::success;
  }
  auto av = packed_detail::pack(as, p);
  auto bv = packed_detail::pack(bs, p);
  auto tolerance = scalar_at_precision<R>(options.relative_diagonal_tolerance, p);
  if (!finite(av.mdspan()) || !finite(bv.mdspan()) || !uni20::isfinite(tolerance))
  {
    info.status = LeastSquaresStatus::nonfinite_result;
    return KernelAttempt::success;
  }
  auto zero = scalar_at_precision<R>(0, p);
  R scale = zero;
  for (std::size_t j = 0; j < n; ++j)
    for (std::size_t i = 0; i < m; ++i)
      scale = std::max(scale, R(math::abs(av[i, j])));
  if (!uni20::isfinite(scale))
  {
    info.status = LeastSquaresStatus::nonfinite_result;
    return KernelAttempt::success;
  }
  auto work = av;
  auto rhs = bv;
  if (n != 0)
  {
    factor(work, rhs, p);
    if (!finite(work.mdspan()) || !finite(rhs.mdspan()))
    {
      info.status = LeastSquaresStatus::nonfinite_result;
      return KernelAttempt::success;
    }
    if (!check_diagonal(work, scale, tolerance, info)) return KernelAttempt::success;
  }
  DenseMatrix<S> solution(uninitialized, n, nrhs, p);
  // Packing gives both backends column-major workspaces, irrespective of input layout.
  for (std::size_t j = 0; j < nrhs; ++j)
  {
    for (std::size_t i = 0; i < n; ++i)
      solution[i, j] = rhs[i, j];
    for (std::size_t k = n; k-- > 0;)
    {
      solution[k, j] /= work[k, k];
      S const& value = solution[k, j];
      for (std::size_t i = 0; i < k; ++i)
        solution[i, j] -= work[i, k] * value;
    }
  }
  if (!finite(solution.mdspan()))
  {
    info.status = LeastSquaresStatus::nonfinite_result;
    return KernelAttempt::success;
  }
  std::vector<R> norms;
  if (!residual_norms(av, bv, solution, norms, p))
  {
    info.status = LeastSquaresStatus::nonfinite_result;
    return KernelAttempt::success;
  }
  auto xa = acquire_host_write_access_sync(x);
  auto xs = xa.mdspan();
  packed_detail::unpack(solution, xs);
  std::copy_n(norms.begin(), nrhs, residuals.begin());
  return KernelAttempt::success;
}
} // namespace uni20::linalg::least_squares_detail
