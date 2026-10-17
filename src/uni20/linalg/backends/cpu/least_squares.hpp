#pragma once

#include <uni20/linalg/backends/least_squares_common.hpp>

namespace uni20::linalg
{
namespace least_squares_detail
{
// Complex Givens rotation [c s; -conj(s) c], with real c.
template <class Matrix> void givens(Matrix& a, Matrix& b, Precision p)
{
  using S = typename Matrix::value_type;
  using R = make_real_t<S>;
  auto zero = scalar_at_precision<S>(0, p);
  auto one = scalar_at_precision<S>(1, p);
  std::size_t m = a.extent(0), n = a.extent(1), nrhs = b.extent(1);
  for (std::size_t k = 0; k < n; ++k)
    for (std::size_t i = k + 1; i < m; ++i)
    {
      S f = a[k, k], g = a[i, k];
      if (g == zero) continue;
      R af = math::abs(f), ag = math::abs(g), r = math::hypot(af, ag);
      S phase = af == 0 ? one : f / af;
      R c = af / r;
      S s = phase * (uni20::conj(g) / r);
      a[k, k] = phase * r;
      a[i, k] = zero;
      for (std::size_t j = k + 1; j < n; ++j)
      {
        S x = a[k, j], y = a[i, j];
        a[k, j] = c * x + s * y;
        a[i, j] = -uni20::conj(s) * x + c * y;
      }
      for (std::size_t j = 0; j < nrhs; ++j)
      {
        S x = b[k, j], y = b[i, j];
        b[k, j] = c * x + s * y;
        b[i, j] = -uni20::conj(s) * x + c * y;
      }
    }
}
} // namespace least_squares_detail

/// \brief Accept matching host matrices for finite-precision reference QR least squares.
template <class X, class A, class B, class... P>
  requires least_squares_detail::Operands<X, A, B> && least_squares_detail::PrecisionArgs<typename A::value_type, P...>
consteval auto kernel_accepts_types(CpuReferenceBackend, least_squares_op const&, X&, A&, B&,
                                    std::span<make_real_t<typename A::value_type>>, LeastSquaresInfo&,
                                    LeastSquaresOptions<make_real_t<typename A::value_type>> const&, P...)
{
  return kernel_types_yes;
}

/// \brief Solve by Givens QR; numerical failure completes dispatch without publishing output.
template <class X, class A, class B, class... P>
  requires least_squares_detail::Operands<X, A, B> && least_squares_detail::PrecisionArgs<typename A::value_type, P...>
KernelAttempt try_kernel(CpuReferenceBackend, least_squares_op const&, X& x, A& a, B& b,
                         std::span<make_real_t<typename A::value_type>> residuals, LeastSquaresInfo& info,
                         LeastSquaresOptions<make_real_t<typename A::value_type>> const& options, P... precision)
{
  return least_squares_detail::run(
      x, a, b, residuals, info, options, least_squares_detail::working_precision<typename A::value_type>(precision...),
      [](auto& work, auto& rhs, Precision p) { least_squares_detail::givens(work, rhs, p); });
}
} // namespace uni20::linalg
