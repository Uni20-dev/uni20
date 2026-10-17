#pragma once

#include "common.hpp"
#include <uni20/backend/blas/blas_int.hpp>
#include <uni20/backend/lapack/lapack.hpp>
#include <uni20/linalg/backends/packed_matrix.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/linalg/solve_info.hpp>
#if UNI20_HAS_MPLAPACK_MPREAL
#include <uni20/backend/mplapack/mpfr.hpp>
#endif
#if UNI20_HAS_MPLAPACK_FLOAT80
#include <uni20/backend/mplapack/binary80.hpp>
#endif

namespace uni20::linalg
{
namespace condition_detail
{
template <class S>
concept PackedGecon =
#if UNI20_HAS_MPLAPACK_MPREAL || UNI20_HAS_MPLAPACK_FLOAT80
    requires(std::size_t n, std::span<S const> a, make_real_t<S> const& norm, Precision p) {
      mplapack::gecon(n, a, norm, '1', p);
    };
#else
    false;
#endif

inline char norm_code(MatrixNorm norm)
{
  ERROR_IF(norm != MatrixNorm::One && norm != MatrixNorm::Infinity,
           "LU condition estimation requires the one- or infinity-norm");
  return norm == MatrixNorm::One ? '1' : 'I';
}
template <class S, class... P>
concept PrecisionArgs = (has_runtime_precision_v<S> && sizeof...(P) == 1 && (std::same_as<P, Precision> && ...)) ||
                        (!has_runtime_precision_v<S> && sizeof...(P) == 0);
template <class S> Precision working_precision() { return native_precision<S>(); }
template <class S> Precision working_precision(Precision p) { return p; }

template <LapackScalar S> make_real_t<S> native_gecon(DenseMatrix<S> const& a, make_real_t<S> norm, char code)
{
  auto n = uni20::blas::try_blas_int(a.extent(0));
  make_real_t<S> result{};
  std::vector<S> work(*lapack_detail::try_size_product(Complex<S> ? 2 : 4, std::size_t(n)));
  blas_int info;
  if constexpr (Real<S>)
  {
    std::vector<blas_int> iwork(std::size_t(n), blas_int{});
    info = lapack::unchecked::gecon(code, n, a.storage().data(), n, norm, result, work.data(), iwork.data());
  }
  else
  {
    std::vector<make_real_t<S>> rwork(*lapack_detail::try_size_product(2, std::size_t(n)));
    info = lapack::unchecked::gecon(code, n, a.storage().data(), n, norm, result, work.data(), rwork.data());
  }
  lapack::detail::check_invalid_argument("gecon", info);
  // Newer providers use positive INFO for an unusable condition estimate.
  return info > 0 ? numeric_limits<make_real_t<S>>::quiet_NaN() : result;
}
} // namespace condition_detail

/// \brief Accept configured GECON providers for read-only successful LU factors.
template <RankedMdspecLike<2> A, class... P>
  requires HostReadableMdspec<A> && condition_detail::PrecisionArgs<typename A::value_type, P...> &&
           (LapackScalar<typename A::value_type> || condition_detail::PackedGecon<typename A::value_type>)
consteval auto kernel_accepts_types(LapackBackend, lu_rcond_op const&, make_real_t<typename A::value_type>&, A&,
                                    make_real_t<typename A::value_type> const&, SolveInfo&, P...)
{
  return kernel_types_maybe;
}

/// \brief Estimate reciprocal conditioning without modifying the factors.
/// \pre A contains successful square LU factors at the supplied precision; the
///      original norm uses the same precision and the norm selected by the operation.
template <RankedMdspecLike<2> A, class... P>
  requires HostReadableMdspec<A> && condition_detail::PrecisionArgs<typename A::value_type, P...> &&
           (LapackScalar<typename A::value_type> || condition_detail::PackedGecon<typename A::value_type>)
KernelAttempt try_kernel(LapackBackend, lu_rcond_op const& op, make_real_t<typename A::value_type>& output, A& a,
                         make_real_t<typename A::value_type> const& original_norm, SolveInfo& info, P... precision)
{
  using S = typename A::value_type;
  using R = make_real_t<S>;
  auto p = condition_detail::working_precision<S>(precision...);
  require_precision<S>(p);
  (void)p.bit_count();
  auto code = condition_detail::norm_code(op.norm);
  ERROR_IF(a.extent(0) != a.extent(1), "LU condition estimation requires square factors");
  ERROR_IF(original_norm < 0, "original matrix norm must be nonnegative");
  auto n = std::size_t(a.extent(0));
  if (!uni20::blas::is_valid_blas_int(uni20::blas::try_blas_int(n)) || !lapack_detail::try_size_product(4, n))
    return KernelAttempt::unsupported_shape;
  info = {};
  if (!n)
  {
    output = scalar_at_precision<R>(1, p);
    return KernelAttempt::success;
  }
  R norm = scalar_at_precision<R>(original_norm, p);
  if (!uni20::isfinite(norm))
  {
    info.status = SolveStatus::nonfinite_result;
    return KernelAttempt::success;
  }
  auto access = acquire_host_read_access_sync(a);
  auto work = packed_detail::pack(access.mdspan(), p);
  R result;
  if constexpr (LapackScalar<S>) result = condition_detail::native_gecon(work, norm, code);
#if UNI20_HAS_MPLAPACK_MPREAL || UNI20_HAS_MPLAPACK_FLOAT80
  else
    result = mplapack::gecon(n, packed_detail::values(std::as_const(work)), norm, code, p);
#endif
  if (!uni20::isfinite(result))
    info.status = SolveStatus::nonfinite_result;
  else
    output = std::move(result);
  return KernelAttempt::success;
}
} // namespace uni20::linalg
