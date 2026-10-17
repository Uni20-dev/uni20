#pragma once

/**
 * \file decompositions_packed.hpp
 * \ingroup linalg
 * \brief Logical host adapters for value-converting dense decompositions.
 */

#include "common.hpp"
#include "packed_decomposition_capabilities.hpp"
#include <algorithm>
#include <span>
#include <uni20/linalg/backends/packed_matrix.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <utility>
#include <vector>

namespace uni20::linalg
{
namespace packed_detail
{
template <class A>
concept DecompositionMatrix =
    MutableRankedMdspecLike<A, 2> && HostWritableMdspec<A> && RealOrComplex<typename A::value_type>;
template <class V, class A>
concept DecompositionValues = MutableRankedMdspecLike<V, 1> && HostWritableMdspec<V> &&
                              std::same_as<typename V::value_type, make_real_t<typename A::value_type>>;
template <class A, class B>
concept SameMatrixScalar =
    DecompositionMatrix<A> && DecompositionMatrix<B> && std::same_as<typename A::value_type, typename B::value_type>;

template <class A> void require_matrix(A const& a, std::size_t rows, std::size_t cols)
{
  CHECK_EQUAL(std::size_t(a.extent(0)), rows);
  CHECK_EQUAL(std::size_t(a.extent(1)), cols);
}
inline char vector_job(SvdVectorExtent extent)
{
  switch (extent)
  {
    case SvdVectorExtent::Full:
      return 'A';
    case SvdVectorExtent::Reduced:
      return 'S';
  }
  PANIC("invalid SVD extent");
}

template <class V, class A> auto compute_svd(V& v, A& a, char jobu, char jobvt, Precision p)
{
  (void)p.bit_count();
  using S = typename A::value_type;
  std::size_t m = a.extent(0), n = a.extent(1), k = std::min(m, n);
  CHECK_EQUAL(std::size_t(v.extent(0)), k);
  auto work = pack(a, p);
  DenseMatrix<S> u(uninitialized, m, jobu == 'A' ? m : jobu == 'S' ? k : 0, p);
  DenseMatrix<S> vh(uninitialized, jobvt == 'A' ? n : jobvt == 'S' ? k : 0, n, p);
  std::vector<make_real_t<typename A::value_type>> w(k);
  mplapack::svd(m, n, values(work), w, values(u), values(vh), jobu, jobvt, p);
  unpack(work, a);
  for (std::size_t i = 0; i < k; ++i)
    v[i] = w[i];
  return std::pair(std::move(u), std::move(vh));
}
} // namespace packed_detail

/// \brief Accept real or complex host matrices for reduced QR at explicit precision.
template <packed_detail::DecompositionMatrix A, class X, class Y, class... P>
  requires packed_detail::SameMatrixScalar<A, X> && packed_detail::SameMatrixScalar<A, Y> &&
           packed_detail::QrScalar<typename A::value_type> &&
           packed_detail::KernelPrecision<typename A::value_type, P...>
consteval auto kernel_accepts_types(LapackBackend, qr_op const&, X&, Y&, A&, P... precision)
{
  return kernel_types_yes;
}

/// \brief Pack logical elements and compute reduced QR through MPLAPACK.
template <packed_detail::DecompositionMatrix A, class X, class Y, class... P>
  requires packed_detail::SameMatrixScalar<A, X> && packed_detail::SameMatrixScalar<A, Y> &&
           packed_detail::QrScalar<typename A::value_type> &&
           packed_detail::KernelPrecision<typename A::value_type, P...>
KernelAttempt try_kernel(LapackBackend, qr_op const&, X& q, Y& r, A& a, P... precision)
{
  auto const p = packed_detail::working_precision<typename A::value_type>(precision...);
  (void)p.bit_count();
  return lapack_detail::with_host_write_mdspans(
      [&](auto& xs, auto& ys, auto& as) {
        std::size_t m = as.extent(0), n = as.extent(1), k = std::min(m, n);
        packed_detail::require_matrix(xs, m, k);
        packed_detail::require_matrix(ys, k, n);
        auto av = packed_detail::pack(as, p);
        auto xv = packed_detail::pack(xs, p, false);
        auto yv = packed_detail::pack(ys, p, false);
        mplapack::qr(m, n, packed_detail::values(av), packed_detail::values(xv), packed_detail::values(yv), p);
        packed_detail::unpack(av, as);
        packed_detail::unpack(xv, xs);
        packed_detail::unpack(yv, ys);
        return KernelAttempt::success;
      },
      q, r, a);
}

/// \brief Accept real or complex host matrices for reduced LQ at explicit precision.
template <packed_detail::DecompositionMatrix A, class X, class Y, class... P>
  requires packed_detail::SameMatrixScalar<A, X> && packed_detail::SameMatrixScalar<A, Y> &&
           packed_detail::LqScalar<typename A::value_type> &&
           packed_detail::KernelPrecision<typename A::value_type, P...>
consteval auto kernel_accepts_types(LapackBackend, lq_op const&, X&, Y&, A&, P... precision)
{
  return kernel_types_yes;
}

/// \brief Pack logical elements and compute reduced LQ through MPLAPACK.
template <packed_detail::DecompositionMatrix A, class X, class Y, class... P>
  requires packed_detail::SameMatrixScalar<A, X> && packed_detail::SameMatrixScalar<A, Y> &&
           packed_detail::LqScalar<typename A::value_type> &&
           packed_detail::KernelPrecision<typename A::value_type, P...>
KernelAttempt try_kernel(LapackBackend, lq_op const&, X& l, Y& q, A& a, P... precision)
{
  auto const p = packed_detail::working_precision<typename A::value_type>(precision...);
  (void)p.bit_count();
  return lapack_detail::with_host_write_mdspans(
      [&](auto& xs, auto& ys, auto& as) {
        std::size_t m = as.extent(0), n = as.extent(1), k = std::min(m, n);
        packed_detail::require_matrix(xs, m, k);
        packed_detail::require_matrix(ys, k, n);
        auto av = packed_detail::pack(as, p);
        auto xv = packed_detail::pack(xs, p, false);
        auto yv = packed_detail::pack(ys, p, false);
        mplapack::lq(m, n, packed_detail::values(av), packed_detail::values(yv), packed_detail::values(xv), p);
        packed_detail::unpack(av, as);
        packed_detail::unpack(xv, xs);
        packed_detail::unpack(yv, ys);
        return KernelAttempt::success;
      },
      l, q, a);
}

/// \brief Accept matching real eigenvalues and real or complex host matrix storage.
template <class V, packed_detail::DecompositionMatrix A, class... P>
  requires packed_detail::EighScalar<typename A::value_type> &&
           packed_detail::KernelPrecision<typename A::value_type, P...> && packed_detail::DecompositionValues<V, A>
consteval auto kernel_accepts_types(LapackBackend, self_adjoint_eigh_op const&, V&, A&, P... precision)
{
  return kernel_types_yes;
}

/// \brief Read only the selected triangle and return values and optional vectors at p.
template <class V, packed_detail::DecompositionMatrix A, class... P>
  requires packed_detail::EighScalar<typename A::value_type> &&
           packed_detail::KernelPrecision<typename A::value_type, P...> && packed_detail::DecompositionValues<V, A>
KernelAttempt try_kernel(LapackBackend, self_adjoint_eigh_op const& op, V& v, A& a, P... precision)
{
  auto const p = packed_detail::working_precision<typename A::value_type>(precision...);
  (void)p.bit_count();
  return lapack_detail::with_host_write_mdspans(
      [&](auto& vs, auto& as) {
        std::size_t n = as.extent(0);
        packed_detail::require_matrix(as, n, n);
        CHECK_EQUAL(std::size_t(vs.extent(0)), n);
        CHECK(op.triangle == MatrixTriangle::Upper || op.triangle == MatrixTriangle::Lower);
        using S = typename A::value_type;
        DenseMatrix<S> work(n, n, p);
        for (std::size_t j = 0; j < n; ++j)
          for (std::size_t i = 0; i < n; ++i)
            if (op.triangle == MatrixTriangle::Upper ? i <= j : i >= j)
              work[i, j] = scalar_at_precision<S>(S(as[i, j]), p);
        std::vector<make_real_t<typename A::value_type>> w(n);
        mplapack::eigh(n, packed_detail::values(work), w, op.compute_vectors,
                       op.triangle == MatrixTriangle::Upper ? 'U' : 'L', p);
        packed_detail::unpack(work, as);
        for (std::size_t i = 0; i < n; ++i)
          vs[i] = w[i];
        return KernelAttempt::success;
      },
      v, a);
}

/// \brief Accept real or complex input and actual complex eigenvalue/vector outputs.
template <packed_detail::DecompositionMatrix A, packed_detail::DecompositionMatrix R, class... P>
  requires std::same_as<typename R::value_type, complex<make_real_t<typename A::value_type>>> &&
           packed_detail::EigenScalar<typename A::value_type> &&
           packed_detail::KernelPrecision<typename A::value_type, P...>
consteval auto kernel_accepts_types(LapackBackend, nonsymmetric_eigen_op const&, A&, std::span<typename R::value_type>,
                                    R&, P... precision)
{
  return kernel_types_yes;
}

/// \brief Unpack real conjugate pairs into complex columns; omit unread vector output when disabled.
template <packed_detail::DecompositionMatrix A, packed_detail::DecompositionMatrix R, class... P>
  requires std::same_as<typename R::value_type, complex<make_real_t<typename A::value_type>>> &&
           packed_detail::EigenScalar<typename A::value_type> &&
           packed_detail::KernelPrecision<typename A::value_type, P...>
KernelAttempt try_kernel(LapackBackend, nonsymmetric_eigen_op const& op, A& a, std::span<typename R::value_type> v,
                         R& r, P... precision)
{
  auto const p = packed_detail::working_precision<typename A::value_type>(precision...);
  (void)p.bit_count();
  return lapack_detail::with_host_write_mdspans(
      [&](auto& as, auto& rs) {
        std::size_t n = as.extent(0);
        packed_detail::require_matrix(as, n, n);
        CHECK_EQUAL(v.size(), n);
        if (op.compute_right_vectors) packed_detail::require_matrix(rs, n, n);
        auto work = packed_detail::pack(as, p);
        DenseMatrix<typename R::value_type> vectors(uninitialized, op.compute_right_vectors ? n : 0, n, p);
        mplapack::eigen(n, packed_detail::values(work), v, packed_detail::values(vectors), op.compute_right_vectors, p);
        packed_detail::unpack(work, as);
        if (op.compute_right_vectors) packed_detail::unpack(vectors, rs);
        return KernelAttempt::success;
      },
      a, r);
}

/// \brief Accept explicit-precision SVD with the requested output structure.
template <class V, packed_detail::DecompositionMatrix A, class... P>
  requires packed_detail::SvdScalar<typename A::value_type> &&
           packed_detail::KernelPrecision<typename A::value_type, P...> && packed_detail::DecompositionValues<V, A>
consteval auto kernel_accepts_types(LapackBackend, singular_values_op const& op, V& v, A& a, P... precision)
{
  return kernel_types_yes;
}

/// \brief Compute the requested SVD factors through logical host packing.
template <class V, packed_detail::DecompositionMatrix A, class... P>
  requires packed_detail::SvdScalar<typename A::value_type> &&
           packed_detail::KernelPrecision<typename A::value_type, P...> && packed_detail::DecompositionValues<V, A>
KernelAttempt try_kernel(LapackBackend, singular_values_op const& op, V& v, A& a, P... precision)
{
  auto const p = packed_detail::working_precision<typename A::value_type>(precision...);
  return lapack_detail::with_host_write_mdspans(
      [&](auto& vs, auto& as) {
        (void)op;
        [[maybe_unused]] auto factors = packed_detail::compute_svd(vs, as, 'N', 'N', p);

        return KernelAttempt::success;
      },
      v, a);
}

/// \brief Accept explicit-precision SVD with the requested output structure.
template <class V, packed_detail::DecompositionMatrix A, class... P>
  requires packed_detail::SvdScalar<typename A::value_type> &&
           packed_detail::KernelPrecision<typename A::value_type, P...> && packed_detail::DecompositionValues<V, A>
consteval auto kernel_accepts_types(LapackBackend, svd_left_op const& op, V& v, A& a, P... precision)
{
  return kernel_types_yes;
}

/// \brief Compute the requested SVD factors through logical host packing.
template <class V, packed_detail::DecompositionMatrix A, class... P>
  requires packed_detail::SvdScalar<typename A::value_type> &&
           packed_detail::KernelPrecision<typename A::value_type, P...> && packed_detail::DecompositionValues<V, A>
KernelAttempt try_kernel(LapackBackend, svd_left_op const& op, V& v, A& a, P... precision)
{
  auto const p = packed_detail::working_precision<typename A::value_type>(precision...);
  return lapack_detail::with_host_write_mdspans(
      [&](auto& vs, auto& as) {
        CHECK(op.overwrite_input && op.left == SvdVectorExtent::Reduced);
        [[maybe_unused]] auto factors = packed_detail::compute_svd(vs, as, 'O', 'N', p);

        return KernelAttempt::success;
      },
      v, a);
}

/// \brief Accept explicit-precision SVD with the requested output structure.
template <class V, packed_detail::DecompositionMatrix A, class... P>
  requires packed_detail::SvdScalar<typename A::value_type> &&
           packed_detail::KernelPrecision<typename A::value_type, P...> && packed_detail::DecompositionValues<V, A>
consteval auto kernel_accepts_types(LapackBackend, svd_right_op const& op, V& v, A& a, P... precision)
{
  return kernel_types_yes;
}

/// \brief Compute the requested SVD factors through logical host packing.
template <class V, packed_detail::DecompositionMatrix A, class... P>
  requires packed_detail::SvdScalar<typename A::value_type> &&
           packed_detail::KernelPrecision<typename A::value_type, P...> && packed_detail::DecompositionValues<V, A>
KernelAttempt try_kernel(LapackBackend, svd_right_op const& op, V& v, A& a, P... precision)
{
  auto const p = packed_detail::working_precision<typename A::value_type>(precision...);
  return lapack_detail::with_host_write_mdspans(
      [&](auto& vs, auto& as) {
        CHECK(op.overwrite_input && op.right == SvdVectorExtent::Reduced);
        [[maybe_unused]] auto factors = packed_detail::compute_svd(vs, as, 'N', 'O', p);

        return KernelAttempt::success;
      },
      v, a);
}

/// \brief Accept explicit-precision SVD with the requested output structure.
template <class V, packed_detail::DecompositionMatrix A, class L, class... P>
  requires packed_detail::SameMatrixScalar<A, L> && packed_detail::SvdScalar<typename A::value_type> &&
           packed_detail::KernelPrecision<typename A::value_type, P...> && packed_detail::DecompositionValues<V, A>
consteval auto kernel_accepts_types(LapackBackend, svd_left_op const& op, V& v, L& l, A& a, P... precision)
{
  return kernel_types_yes;
}

/// \brief Compute the requested SVD factors through logical host packing.
template <class V, packed_detail::DecompositionMatrix A, class L, class... P>
  requires packed_detail::SameMatrixScalar<A, L> && packed_detail::SvdScalar<typename A::value_type> &&
           packed_detail::KernelPrecision<typename A::value_type, P...> && packed_detail::DecompositionValues<V, A>
KernelAttempt try_kernel(LapackBackend, svd_left_op const& op, V& v, L& l, A& a, P... precision)
{
  auto const p = packed_detail::working_precision<typename A::value_type>(precision...);
  return lapack_detail::with_host_write_mdspans(
      [&](auto& vs, auto& ls, auto& as) {
        std::size_t m = as.extent(0), n = as.extent(1), k = std::min(m, n);
        CHECK(!op.overwrite_input);
        packed_detail::require_matrix(ls, m, op.left == SvdVectorExtent::Full ? m : k);
        [[maybe_unused]] auto factors = packed_detail::compute_svd(vs, as, packed_detail::vector_job(op.left), 'N', p);
        packed_detail::unpack(factors.first, ls);
        return KernelAttempt::success;
      },
      v, l, a);
}

/// \brief Accept explicit-precision SVD with the requested output structure.
template <class V, packed_detail::DecompositionMatrix A, class R, class... P>
  requires packed_detail::SameMatrixScalar<A, R> && packed_detail::SvdScalar<typename A::value_type> &&
           packed_detail::KernelPrecision<typename A::value_type, P...> && packed_detail::DecompositionValues<V, A>
consteval auto kernel_accepts_types(LapackBackend, svd_right_op const& op, V& v, R& r, A& a, P... precision)
{
  return kernel_types_yes;
}

/// \brief Compute the requested SVD factors through logical host packing.
template <class V, packed_detail::DecompositionMatrix A, class R, class... P>
  requires packed_detail::SameMatrixScalar<A, R> && packed_detail::SvdScalar<typename A::value_type> &&
           packed_detail::KernelPrecision<typename A::value_type, P...> && packed_detail::DecompositionValues<V, A>
KernelAttempt try_kernel(LapackBackend, svd_right_op const& op, V& v, R& r, A& a, P... precision)
{
  auto const p = packed_detail::working_precision<typename A::value_type>(precision...);
  return lapack_detail::with_host_write_mdspans(
      [&](auto& vs, auto& rs, auto& as) {
        std::size_t m = as.extent(0), n = as.extent(1), k = std::min(m, n);
        CHECK(!op.overwrite_input);
        packed_detail::require_matrix(rs, op.right == SvdVectorExtent::Full ? n : k, n);
        [[maybe_unused]] auto factors = packed_detail::compute_svd(vs, as, 'N', packed_detail::vector_job(op.right), p);
        packed_detail::unpack(factors.second, rs);
        return KernelAttempt::success;
      },
      v, r, a);
}

/// \brief Accept explicit-precision SVD with the requested output structure.
template <class V, packed_detail::DecompositionMatrix A, class L, class R, class... P>
  requires packed_detail::SameMatrixScalar<A, L> && packed_detail::SameMatrixScalar<A, R> &&
           packed_detail::SvdScalar<typename A::value_type> &&
           packed_detail::KernelPrecision<typename A::value_type, P...> && packed_detail::DecompositionValues<V, A>
consteval auto kernel_accepts_types(LapackBackend, svd_op const& op, V& v, L& l, R& r, A& a, P... precision)
{
  return kernel_types_yes;
}

/// \brief Compute the requested SVD factors through logical host packing.
template <class V, packed_detail::DecompositionMatrix A, class L, class R, class... P>
  requires packed_detail::SameMatrixScalar<A, L> && packed_detail::SameMatrixScalar<A, R> &&
           packed_detail::SvdScalar<typename A::value_type> &&
           packed_detail::KernelPrecision<typename A::value_type, P...> && packed_detail::DecompositionValues<V, A>
KernelAttempt try_kernel(LapackBackend, svd_op const& op, V& v, L& l, R& r, A& a, P... precision)
{
  auto const p = packed_detail::working_precision<typename A::value_type>(precision...);
  return lapack_detail::with_host_write_mdspans(
      [&](auto& vs, auto& ls, auto& rs, auto& as) {
        std::size_t m = as.extent(0), n = as.extent(1), k = std::min(m, n);
        CHECK(op.overwrite == SvdOverwrite::None);
        packed_detail::require_matrix(ls, m, op.left == SvdVectorExtent::Full ? m : k);
        packed_detail::require_matrix(rs, op.right == SvdVectorExtent::Full ? n : k, n);
        [[maybe_unused]] auto factors = packed_detail::compute_svd(vs, as, packed_detail::vector_job(op.left),
                                                                   packed_detail::vector_job(op.right), p);
        packed_detail::unpack(factors.first, ls);
        packed_detail::unpack(factors.second, rs);
        return KernelAttempt::success;
      },
      v, l, r, a);
}

/// \brief Accept explicit-precision SVD with the requested output structure.
template <class V, packed_detail::DecompositionMatrix A, class F, class... P>
  requires packed_detail::SameMatrixScalar<A, F> && packed_detail::SvdScalar<typename A::value_type> &&
           packed_detail::KernelPrecision<typename A::value_type, P...> && packed_detail::DecompositionValues<V, A>
consteval auto kernel_accepts_types(LapackBackend, svd_op const& op, V& v, F& f, A& a, P... precision)
{
  return kernel_types_yes;
}

/// \brief Compute the requested SVD factors through logical host packing.
template <class V, packed_detail::DecompositionMatrix A, class F, class... P>
  requires packed_detail::SameMatrixScalar<A, F> && packed_detail::SvdScalar<typename A::value_type> &&
           packed_detail::KernelPrecision<typename A::value_type, P...> && packed_detail::DecompositionValues<V, A>
KernelAttempt try_kernel(LapackBackend, svd_op const& op, V& v, F& f, A& a, P... precision)
{
  auto const p = packed_detail::working_precision<typename A::value_type>(precision...);
  return lapack_detail::with_host_write_mdspans(
      [&](auto& vs, auto& fs, auto& as) {
        std::size_t m = as.extent(0), n = as.extent(1), k = std::min(m, n);
        CHECK(op.overwrite == SvdOverwrite::Left || op.overwrite == SvdOverwrite::Right);
        if (op.overwrite == SvdOverwrite::Left)
        {
          CHECK(op.left == SvdVectorExtent::Reduced);
          packed_detail::require_matrix(fs, op.right == SvdVectorExtent::Full ? n : k, n);
        }
        else
        {
          CHECK(op.right == SvdVectorExtent::Reduced);
          packed_detail::require_matrix(fs, m, op.left == SvdVectorExtent::Full ? m : k);
        }
        [[maybe_unused]] auto factors = packed_detail::compute_svd(
            vs, as, op.overwrite == SvdOverwrite::Left ? 'O' : packed_detail::vector_job(op.left),
            op.overwrite == SvdOverwrite::Right ? 'O' : packed_detail::vector_job(op.right), p);
        if (op.overwrite == SvdOverwrite::Left)
          packed_detail::unpack(factors.second, fs);
        else
          packed_detail::unpack(factors.first, fs);
        return KernelAttempt::success;
      },
      v, f, a);
}
} // namespace uni20::linalg
