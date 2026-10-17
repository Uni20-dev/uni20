#pragma once

#include <uni20/linalg/backends/lapack/qr.hpp>
#include <uni20/linalg/backends/least_squares_common.hpp>
#include <uni20/linalg/precision.hpp>

namespace uni20::linalg
{
namespace least_squares_detail
{
template <class S, class... P> consteval auto qr_acceptance()
{
  using D = decltype(mdspec_of(std::declval<DenseMatrix<S>&>()));
  return probe_dispatch_kernel_types<LapackBackend, qr_op, D&, D&, D&, P...>();
}

template <class Matrix> void provider_qr(Matrix& a, Matrix& b, Precision p)
{
  using S = typename Matrix::value_type;
  std::size_t m = a.extent(0), n = a.extent(1);
  DenseMatrix<S> q(uninitialized, m, n, p), r(uninitialized, n, n, p);
  auto ad = mdspec_of(a), qd = mdspec_of(q), rd = mdspec_of(r);
  detail::dispatch_at_precision<S>(LapackBackend{}, qr_op{}, p, qd, rd, ad);
  auto original_rhs = b;
  for (std::size_t j = 0; j < std::size_t(b.extent(1)); ++j)
    for (std::size_t k = 0; k < n; ++k)
    {
      S value = scalar_at_precision<S>(0, p);
      for (std::size_t i = 0; i < m; ++i)
        value += uni20::conj(q[i, k]) * original_rhs[i, j];
      b[k, j] = value;
    }
  // Only the n-by-n triangular system is used after this point.
  a = std::move(r);
}
} // namespace least_squares_detail

/// \brief Accept precisely those scalar families supported by the configured LAPACK QR kernel.
template <class X, class A, class B, class... P>
  requires least_squares_detail::Operands<X, A, B> &&
           least_squares_detail::PrecisionArgs<typename A::value_type, P...> &&
           (least_squares_detail::qr_acceptance<typename A::value_type, P...>() != KernelTypeAcceptance::no)
consteval auto kernel_accepts_types(LapackBackend, least_squares_op const&, X&, A&, B&,
                                    std::span<make_real_t<typename A::value_type>>, LeastSquaresInfo&,
                                    LeastSquaresOptions<make_real_t<typename A::value_type>> const&, P...)
{
  return kernel_types_maybe;
}

/// \brief Solve using configured LAPACK/MPLAPACK reduced QR and scalar-generic triangular substitution.
template <class X, class A, class B, class... P>
  requires least_squares_detail::Operands<X, A, B> &&
           least_squares_detail::PrecisionArgs<typename A::value_type, P...> &&
           (least_squares_detail::qr_acceptance<typename A::value_type, P...>() != KernelTypeAcceptance::no)
KernelAttempt try_kernel(LapackBackend, least_squares_op const&, X& x, A& a, B& b,
                         std::span<make_real_t<typename A::value_type>> residuals, LeastSquaresInfo& info,
                         LeastSquaresOptions<make_real_t<typename A::value_type>> const& options, P... precision)
{
  if (!uni20::blas::is_valid_blas_int(uni20::blas::try_blas_int(a.extent(0))) ||
      !uni20::blas::is_valid_blas_int(uni20::blas::try_blas_int(a.extent(1))))
    return KernelAttempt::unsupported_shape;
  return least_squares_detail::run(
      x, a, b, residuals, info, options, least_squares_detail::working_precision<typename A::value_type>(precision...),
      [](auto& work, auto& rhs, Precision p) { least_squares_detail::provider_qr(work, rhs, p); });
}
} // namespace uni20::linalg
