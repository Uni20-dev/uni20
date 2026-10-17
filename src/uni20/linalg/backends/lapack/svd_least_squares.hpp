#pragma once

#include "common.hpp"
#include <uni20/backend/blas/blas_int.hpp>
#include <uni20/backend/lapack/lapack.hpp>
#include <uni20/linalg/backends/least_squares_common.hpp>
#include <uni20/linalg/svd_least_squares_info.hpp>
#if UNI20_HAS_MPLAPACK_MPREAL
#include <uni20/backend/mplapack/mpfr.hpp>
#endif
#if UNI20_HAS_MPLAPACK_FLOAT80
#include <uni20/backend/mplapack/binary80.hpp>
#endif

namespace uni20::linalg
{
namespace least_squares_detail
{
template <class S>
concept PackedGelss =
#if UNI20_HAS_MPLAPACK_MPREAL || UNI20_HAS_MPLAPACK_FLOAT80
    requires(std::size_t n, std::span<S> a, std::span<make_real_t<S>> values, make_real_t<S> const& cutoff,
             Precision p) { mplapack::gelss(n, n, n, a, a, values, cutoff, n, p); };
#else
    false;
#endif

template <Real R> void require_svd_options(SvdLeastSquaresOptions<R> const& options)
{
  ERROR_IF(options.relative_cutoff && (!uni20::isfinite(*options.relative_cutoff) || *options.relative_cutoff < 0),
           "SVD least squares requires a finite nonnegative relative cutoff");
}

template <LapackScalar S>
std::size_t native_gelss(DenseMatrix<S>& a, DenseMatrix<S>& b, std::vector<make_real_t<S>>& singular,
                         make_real_t<S> const& cutoff, std::size_t& rank)
{
  auto m = uni20::blas::try_blas_int(a.extent(0)), n = uni20::blas::try_blas_int(a.extent(1));
  auto nrhs = uni20::blas::try_blas_int(b.extent(1)), ldb = uni20::blas::try_blas_int(b.extent(0));
  S dummy{}, query{};
  S* rhs = nrhs ? b.storage().data() : &dummy;
  blas_int provider_rank = 0;
  std::vector<make_real_t<S>> rwork;
  if constexpr (Complex<S>) rwork.resize(*lapack_detail::try_size_product(5, singular.size()));
  auto call = [&](S* work, blas_int length) {
    if constexpr (Real<S>)
      return lapack::unchecked::gelss(m, n, nrhs, a.storage().data(), m, rhs, ldb, singular.data(), cutoff,
                                      provider_rank, work, length);
    else
      return lapack::unchecked::gelss(m, n, nrhs, a.storage().data(), m, rhs, ldb, singular.data(), cutoff,
                                      provider_rank, work, length, rwork.data());
  };
  auto code = call(&query, -1);
  lapack::detail::check_invalid_argument("gelss", code);
  if (code > 0) return std::size_t(code);
  auto length = lapack_detail::workspace_size(query);
  std::vector<S> work(std::size_t(length), S{});
  code = call(work.data(), length);
  lapack::detail::check_invalid_argument("gelss", code);
  if (!code) rank = std::size_t(provider_rank);
  return std::size_t(code);
}
} // namespace least_squares_detail

/// \brief Accept configured real/complex GELSS providers; runtime scalars require explicit precision.
template <class X, class A, class B, class... P>
  requires least_squares_detail::Operands<X, A, B> &&
           least_squares_detail::PrecisionArgs<typename A::value_type, P...> &&
           (LapackScalar<typename A::value_type> || least_squares_detail::PackedGelss<typename A::value_type>)
consteval auto kernel_accepts_types(LapackBackend, svd_least_squares_op const&, X&, A&, B&,
                                    std::span<make_real_t<typename A::value_type>>, LeastSquaresInfo&,
                                    SvdLeastSquaresDiagnostics<make_real_t<typename A::value_type>>&,
                                    SvdLeastSquaresOptions<make_real_t<typename A::value_type>> const&, P...)
{
  return kernel_types_maybe;
}

/// \brief Solve through GELSS; numerical failure publishes only status, preserving fixed outputs.
template <class X, class A, class B, class... P>
  requires least_squares_detail::Operands<X, A, B> &&
           least_squares_detail::PrecisionArgs<typename A::value_type, P...> &&
           (LapackScalar<typename A::value_type> || least_squares_detail::PackedGelss<typename A::value_type>)
KernelAttempt try_kernel(LapackBackend, svd_least_squares_op const&, X& x, A& a, B& b,
                         std::span<make_real_t<typename A::value_type>> residuals, LeastSquaresInfo& info,
                         SvdLeastSquaresDiagnostics<make_real_t<typename A::value_type>>& diagnostics,
                         SvdLeastSquaresOptions<make_real_t<typename A::value_type>> const& options, P... precision)
{
  using S = typename A::value_type;
  using R = make_real_t<S>;
  auto p = least_squares_detail::working_precision<S>(precision...);
  require_precision<S>(p);
  (void)p.bit_count();
  least_squares_detail::require_svd_options(options);
  std::size_t m = a.extent(0), n = a.extent(1), nrhs = b.extent(1), k = std::min(m, n);
  ERROR_IF(std::size_t(b.extent(0)) != m || std::size_t(x.extent(0)) != n || std::size_t(x.extent(1)) != nrhs ||
               residuals.size() != nrhs,
           "incompatible SVD least-squares shapes");
  for (auto extent : {m, n, nrhs})
    if (!uni20::blas::is_valid_blas_int(uni20::blas::try_blas_int(extent))) return KernelAttempt::unsupported_shape;
  if (!lapack_detail::try_size_product(5, k)) return KernelAttempt::unsupported_shape;
  info = {};
  auto aa = acquire_host_read_access_sync(a);
  auto ba = acquire_host_read_access_sync(b);
  auto as = aa.mdspan();
  auto bs = ba.mdspan();
  if (!least_squares_detail::finite(as) || !least_squares_detail::finite(bs))
  {
    info.status = LeastSquaresStatus::nonfinite_input;
    return KernelAttempt::success;
  }
  auto av = packed_detail::pack(as, p), bv = packed_detail::pack(bs, p);
  R cutoff =
      options.relative_cutoff ? scalar_at_precision<R>(*options.relative_cutoff, p) : numeric_limits<R>::epsilon(p);
  if (!least_squares_detail::finite(av.mdspan()) || !least_squares_detail::finite(bv.mdspan()) ||
      !uni20::isfinite(cutoff))
  {
    info.status = LeastSquaresStatus::nonfinite_result;
    return KernelAttempt::success;
  }
  DenseMatrix<S> solution(n, nrhs, p);
  std::vector<R> singular(k, scalar_at_precision<R>(0, p));
  std::size_t rank = 0;
  if (k)
  {
    auto work = av;
    DenseMatrix<S> rhs(std::max(m, n), nrhs, p);
    for (std::size_t j = 0; j < nrhs; ++j)
      for (std::size_t i = 0; i < m; ++i)
        rhs[i, j] = bv[i, j];
    std::size_t code = 0;
    if constexpr (LapackScalar<S>) code = least_squares_detail::native_gelss(work, rhs, singular, cutoff, rank);
#if UNI20_HAS_MPLAPACK_MPREAL || UNI20_HAS_MPLAPACK_FLOAT80
    else
      code = mplapack::gelss(m, n, nrhs, packed_detail::values(work), packed_detail::values(rhs), std::span(singular),
                             cutoff, rank, p);
#endif
    if (code)
    {
      info = {.status = LeastSquaresStatus::nonconvergence, .unconverged = code};
      return KernelAttempt::success;
    }
    for (std::size_t j = 0; j < nrhs; ++j)
      for (std::size_t i = 0; i < n; ++i)
        solution[i, j] = rhs[i, j];
  }
  std::vector<R> norms;
  if (!least_squares_detail::finite(solution.mdspan()) ||
      !std::all_of(singular.begin(), singular.end(), [](R const& s) { return uni20::isfinite(s); }) ||
      !least_squares_detail::residual_norms(av, bv, solution, norms, p))
  {
    info.status = LeastSquaresStatus::nonfinite_result;
    return KernelAttempt::success;
  }
  auto xa = acquire_host_write_access_sync(x);
  auto xs = xa.mdspan();
  packed_detail::unpack(solution, xs);
  std::copy_n(norms.begin(), nrhs, residuals.begin());
  diagnostics = {.rank = rank,
                 .relative_cutoff = std::move(cutoff),
                 .singular_values = options.return_singular_values ? std::move(singular) : std::vector<R>{}};
  return KernelAttempt::success;
}
} // namespace uni20::linalg
