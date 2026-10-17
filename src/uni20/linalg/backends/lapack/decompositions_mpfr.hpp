#pragma once

/**
 * \file decompositions_mpfr.hpp
 * \ingroup linalg
 * \brief Logical host adapters for finite-precision MPFR/MPC dense decompositions.
 */

#include "common.hpp"
#include <algorithm>
#include <span>
#include <utility>
#include <vector>
#include <uni20/backend/mplapack/mpfr.hpp>
#include <uni20/linalg/backends/mpfr_common.hpp>
#include <uni20/linalg/operation_tags.hpp>

namespace uni20::linalg
{
namespace mpfr_detail
{
template <class A>
concept DecompositionMatrix =
    MutableRankedMdspecLike<A, 2> && HostWritableMdspec<A> && RuntimeScalar<typename A::value_type>;
template <class V>
concept DecompositionValues =
    MutableRankedMdspecLike<V, 1> && HostWritableMdspec<V> && std::same_as<typename V::value_type, mpreal>;
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
  std::vector<mpreal> w(k);
  mplapack::svd(m, n, values(work), w, values(u), values(vh), jobu, jobvt, p);
  unpack(work, a);
  for (std::size_t i = 0; i < k; ++i)
    v[i] = w[i];
  return std::pair(std::move(u), std::move(vh));
}
} // namespace mpfr_detail

/// \brief Accept real or complex host matrices for reduced QR at explicit precision.
template <mpfr_detail::DecompositionMatrix A, class X, class Y>
  requires mpfr_detail::SameMatrixScalar<A, X> && mpfr_detail::SameMatrixScalar<A, Y>
consteval auto kernel_accepts_types(LapackBackend, qr_op const&, X&, Y&, A&, Precision)
{
  return kernel_types_yes;
}

/// \brief Pack logical elements and compute reduced QR through MPLAPACK.
template <mpfr_detail::DecompositionMatrix A, class X, class Y>
  requires mpfr_detail::SameMatrixScalar<A, X> && mpfr_detail::SameMatrixScalar<A, Y>
KernelAttempt try_kernel(LapackBackend, qr_op const&, X& q, Y& r, A& a, Precision p)
{
  (void)p.bit_count();
  return lapack_detail::with_host_write_mdspans(
      [&](auto& xs, auto& ys, auto& as) {
        std::size_t m = as.extent(0), n = as.extent(1), k = std::min(m, n);
        mpfr_detail::require_matrix(xs, m, k);
        mpfr_detail::require_matrix(ys, k, n);
        auto av = mpfr_detail::pack(as, p);
        auto xv = mpfr_detail::pack(xs, p, false);
        auto yv = mpfr_detail::pack(ys, p, false);
        mplapack::qr(m, n, mpfr_detail::values(av), mpfr_detail::values(xv), mpfr_detail::values(yv), p);
        mpfr_detail::unpack(av, as);
        mpfr_detail::unpack(xv, xs);
        mpfr_detail::unpack(yv, ys);
        return KernelAttempt::success;
      },
      q, r, a);
}

/// \brief Accept real or complex host matrices for reduced LQ at explicit precision.
template <mpfr_detail::DecompositionMatrix A, class X, class Y>
  requires mpfr_detail::SameMatrixScalar<A, X> && mpfr_detail::SameMatrixScalar<A, Y>
consteval auto kernel_accepts_types(LapackBackend, lq_op const&, X&, Y&, A&, Precision)
{
  return kernel_types_yes;
}

/// \brief Pack logical elements and compute reduced LQ through MPLAPACK.
template <mpfr_detail::DecompositionMatrix A, class X, class Y>
  requires mpfr_detail::SameMatrixScalar<A, X> && mpfr_detail::SameMatrixScalar<A, Y>
KernelAttempt try_kernel(LapackBackend, lq_op const&, X& l, Y& q, A& a, Precision p)
{
  (void)p.bit_count();
  return lapack_detail::with_host_write_mdspans(
      [&](auto& xs, auto& ys, auto& as) {
        std::size_t m = as.extent(0), n = as.extent(1), k = std::min(m, n);
        mpfr_detail::require_matrix(xs, m, k);
        mpfr_detail::require_matrix(ys, k, n);
        auto av = mpfr_detail::pack(as, p);
        auto xv = mpfr_detail::pack(xs, p, false);
        auto yv = mpfr_detail::pack(ys, p, false);
        mplapack::lq(m, n, mpfr_detail::values(av), mpfr_detail::values(yv), mpfr_detail::values(xv), p);
        mpfr_detail::unpack(av, as);
        mpfr_detail::unpack(xv, xs);
        mpfr_detail::unpack(yv, ys);
        return KernelAttempt::success;
      },
      l, q, a);
}

/// \brief Accept matching real eigenvalues and real or complex host matrix storage.
template <mpfr_detail::DecompositionValues V, mpfr_detail::DecompositionMatrix A>
consteval auto kernel_accepts_types(LapackBackend, self_adjoint_eigh_op const&, V&, A&, Precision)
{
  return kernel_types_yes;
}

/// \brief Read only the selected triangle and return values and optional vectors at p.
template <mpfr_detail::DecompositionValues V, mpfr_detail::DecompositionMatrix A>
KernelAttempt try_kernel(LapackBackend, self_adjoint_eigh_op const& op, V& v, A& a, Precision p)
{
  (void)p.bit_count();
  return lapack_detail::with_host_write_mdspans(
      [&](auto& vs, auto& as) {
        std::size_t n = as.extent(0);
        mpfr_detail::require_matrix(as, n, n);
        CHECK_EQUAL(std::size_t(vs.extent(0)), n);
        CHECK(op.triangle == MatrixTriangle::Upper || op.triangle == MatrixTriangle::Lower);
        using S = typename A::value_type;
        DenseMatrix<S> work(n, n, p);
        for (std::size_t j = 0; j < n; ++j)
          for (std::size_t i = 0; i < n; ++i)
            if (op.triangle == MatrixTriangle::Upper ? i <= j : i >= j) work[i, j] = S(as[i, j]).at(p);
        std::vector<mpreal> w(n);
        mplapack::eigh(n, mpfr_detail::values(work), w, op.compute_vectors,
                       op.triangle == MatrixTriangle::Upper ? 'U' : 'L', p);
        mpfr_detail::unpack(work, as);
        for (std::size_t i = 0; i < n; ++i)
          vs[i] = w[i];
        return KernelAttempt::success;
      },
      v, a);
}

/// \brief Accept real or complex input and actual complex eigenvalue/vector outputs.
template <mpfr_detail::DecompositionMatrix A, mpfr_detail::DecompositionMatrix R>
  requires std::same_as<typename R::value_type, mpcomplex>
consteval auto kernel_accepts_types(LapackBackend, nonsymmetric_eigen_op const&, A&, std::span<mpcomplex>, R&,
                                    Precision)
{
  return kernel_types_yes;
}

/// \brief Unpack real conjugate pairs into complex columns; omit unread vector output when disabled.
template <mpfr_detail::DecompositionMatrix A, mpfr_detail::DecompositionMatrix R>
  requires std::same_as<typename R::value_type, mpcomplex>
KernelAttempt try_kernel(LapackBackend, nonsymmetric_eigen_op const& op, A& a, std::span<mpcomplex> v, R& r,
                         Precision p)
{
  (void)p.bit_count();
  return lapack_detail::with_host_write_mdspans(
      [&](auto& as, auto& rs) {
        std::size_t n = as.extent(0);
        mpfr_detail::require_matrix(as, n, n);
        CHECK_EQUAL(v.size(), n);
        if (op.compute_right_vectors) mpfr_detail::require_matrix(rs, n, n);
        auto work = mpfr_detail::pack(as, p);
        DenseMatrix<mpcomplex> vectors(uninitialized, op.compute_right_vectors ? n : 0, n, p);
        mplapack::eigen(n, mpfr_detail::values(work), v, mpfr_detail::values(vectors), op.compute_right_vectors, p);
        mpfr_detail::unpack(work, as);
        if (op.compute_right_vectors) mpfr_detail::unpack(vectors, rs);
        return KernelAttempt::success;
      },
      a, r);
}

/// \brief Accept explicit-precision SVD with the requested output structure.
template <mpfr_detail::DecompositionValues V, mpfr_detail::DecompositionMatrix A>
consteval auto kernel_accepts_types(LapackBackend, singular_values_op const& op, V& v, A& a, Precision p)
{
  return kernel_types_yes;
}

/// \brief Compute the requested SVD factors through logical host packing.
template <mpfr_detail::DecompositionValues V, mpfr_detail::DecompositionMatrix A>
KernelAttempt try_kernel(LapackBackend, singular_values_op const& op, V& v, A& a, Precision p)
{
  return lapack_detail::with_host_write_mdspans(
      [&](auto& vs, auto& as) {
        (void)op;
        [[maybe_unused]] auto factors = mpfr_detail::compute_svd(vs, as, 'N', 'N', p);

        return KernelAttempt::success;
      },
      v, a);
}

/// \brief Accept explicit-precision SVD with the requested output structure.
template <mpfr_detail::DecompositionValues V, mpfr_detail::DecompositionMatrix A>
consteval auto kernel_accepts_types(LapackBackend, svd_left_op const& op, V& v, A& a, Precision p)
{
  return kernel_types_yes;
}

/// \brief Compute the requested SVD factors through logical host packing.
template <mpfr_detail::DecompositionValues V, mpfr_detail::DecompositionMatrix A>
KernelAttempt try_kernel(LapackBackend, svd_left_op const& op, V& v, A& a, Precision p)
{
  return lapack_detail::with_host_write_mdspans(
      [&](auto& vs, auto& as) {
        CHECK(op.overwrite_input && op.left == SvdVectorExtent::Reduced);
        [[maybe_unused]] auto factors = mpfr_detail::compute_svd(vs, as, 'O', 'N', p);

        return KernelAttempt::success;
      },
      v, a);
}

/// \brief Accept explicit-precision SVD with the requested output structure.
template <mpfr_detail::DecompositionValues V, mpfr_detail::DecompositionMatrix A>
consteval auto kernel_accepts_types(LapackBackend, svd_right_op const& op, V& v, A& a, Precision p)
{
  return kernel_types_yes;
}

/// \brief Compute the requested SVD factors through logical host packing.
template <mpfr_detail::DecompositionValues V, mpfr_detail::DecompositionMatrix A>
KernelAttempt try_kernel(LapackBackend, svd_right_op const& op, V& v, A& a, Precision p)
{
  return lapack_detail::with_host_write_mdspans(
      [&](auto& vs, auto& as) {
        CHECK(op.overwrite_input && op.right == SvdVectorExtent::Reduced);
        [[maybe_unused]] auto factors = mpfr_detail::compute_svd(vs, as, 'N', 'O', p);

        return KernelAttempt::success;
      },
      v, a);
}

/// \brief Accept explicit-precision SVD with the requested output structure.
template <mpfr_detail::DecompositionValues V, mpfr_detail::DecompositionMatrix A, class L>
  requires mpfr_detail::SameMatrixScalar<A, L>
consteval auto kernel_accepts_types(LapackBackend, svd_left_op const& op, V& v, L& l, A& a, Precision p)
{
  return kernel_types_yes;
}

/// \brief Compute the requested SVD factors through logical host packing.
template <mpfr_detail::DecompositionValues V, mpfr_detail::DecompositionMatrix A, class L>
  requires mpfr_detail::SameMatrixScalar<A, L>
KernelAttempt try_kernel(LapackBackend, svd_left_op const& op, V& v, L& l, A& a, Precision p)
{
  return lapack_detail::with_host_write_mdspans(
      [&](auto& vs, auto& ls, auto& as) {
        std::size_t m = as.extent(0), n = as.extent(1), k = std::min(m, n);
        CHECK(!op.overwrite_input);
        mpfr_detail::require_matrix(ls, m, op.left == SvdVectorExtent::Full ? m : k);
        [[maybe_unused]] auto factors = mpfr_detail::compute_svd(vs, as, mpfr_detail::vector_job(op.left), 'N', p);
        mpfr_detail::unpack(factors.first, ls);
        return KernelAttempt::success;
      },
      v, l, a);
}

/// \brief Accept explicit-precision SVD with the requested output structure.
template <mpfr_detail::DecompositionValues V, mpfr_detail::DecompositionMatrix A, class R>
  requires mpfr_detail::SameMatrixScalar<A, R>
consteval auto kernel_accepts_types(LapackBackend, svd_right_op const& op, V& v, R& r, A& a, Precision p)
{
  return kernel_types_yes;
}

/// \brief Compute the requested SVD factors through logical host packing.
template <mpfr_detail::DecompositionValues V, mpfr_detail::DecompositionMatrix A, class R>
  requires mpfr_detail::SameMatrixScalar<A, R>
KernelAttempt try_kernel(LapackBackend, svd_right_op const& op, V& v, R& r, A& a, Precision p)
{
  return lapack_detail::with_host_write_mdspans(
      [&](auto& vs, auto& rs, auto& as) {
        std::size_t m = as.extent(0), n = as.extent(1), k = std::min(m, n);
        CHECK(!op.overwrite_input);
        mpfr_detail::require_matrix(rs, op.right == SvdVectorExtent::Full ? n : k, n);
        [[maybe_unused]] auto factors = mpfr_detail::compute_svd(vs, as, 'N', mpfr_detail::vector_job(op.right), p);
        mpfr_detail::unpack(factors.second, rs);
        return KernelAttempt::success;
      },
      v, r, a);
}

/// \brief Accept explicit-precision SVD with the requested output structure.
template <mpfr_detail::DecompositionValues V, mpfr_detail::DecompositionMatrix A, class L, class R>
  requires mpfr_detail::SameMatrixScalar<A, L> && mpfr_detail::SameMatrixScalar<A, R>
consteval auto kernel_accepts_types(LapackBackend, svd_op const& op, V& v, L& l, R& r, A& a, Precision p)
{
  return kernel_types_yes;
}

/// \brief Compute the requested SVD factors through logical host packing.
template <mpfr_detail::DecompositionValues V, mpfr_detail::DecompositionMatrix A, class L, class R>
  requires mpfr_detail::SameMatrixScalar<A, L> && mpfr_detail::SameMatrixScalar<A, R>
KernelAttempt try_kernel(LapackBackend, svd_op const& op, V& v, L& l, R& r, A& a, Precision p)
{
  return lapack_detail::with_host_write_mdspans(
      [&](auto& vs, auto& ls, auto& rs, auto& as) {
        std::size_t m = as.extent(0), n = as.extent(1), k = std::min(m, n);
        CHECK(op.overwrite == SvdOverwrite::None);
        mpfr_detail::require_matrix(ls, m, op.left == SvdVectorExtent::Full ? m : k);
        mpfr_detail::require_matrix(rs, op.right == SvdVectorExtent::Full ? n : k, n);
        [[maybe_unused]] auto factors =
            mpfr_detail::compute_svd(vs, as, mpfr_detail::vector_job(op.left), mpfr_detail::vector_job(op.right), p);
        mpfr_detail::unpack(factors.first, ls);
        mpfr_detail::unpack(factors.second, rs);
        return KernelAttempt::success;
      },
      v, l, r, a);
}

/// \brief Accept explicit-precision SVD with the requested output structure.
template <mpfr_detail::DecompositionValues V, mpfr_detail::DecompositionMatrix A, class F>
  requires mpfr_detail::SameMatrixScalar<A, F>
consteval auto kernel_accepts_types(LapackBackend, svd_op const& op, V& v, F& f, A& a, Precision p)
{
  return kernel_types_yes;
}

/// \brief Compute the requested SVD factors through logical host packing.
template <mpfr_detail::DecompositionValues V, mpfr_detail::DecompositionMatrix A, class F>
  requires mpfr_detail::SameMatrixScalar<A, F>
KernelAttempt try_kernel(LapackBackend, svd_op const& op, V& v, F& f, A& a, Precision p)
{
  return lapack_detail::with_host_write_mdspans(
      [&](auto& vs, auto& fs, auto& as) {
        std::size_t m = as.extent(0), n = as.extent(1), k = std::min(m, n);
        CHECK(op.overwrite == SvdOverwrite::Left || op.overwrite == SvdOverwrite::Right);
        if (op.overwrite == SvdOverwrite::Left)
        {
          CHECK(op.left == SvdVectorExtent::Reduced);
          mpfr_detail::require_matrix(fs, op.right == SvdVectorExtent::Full ? n : k, n);
        }
        else
        {
          CHECK(op.right == SvdVectorExtent::Reduced);
          mpfr_detail::require_matrix(fs, m, op.left == SvdVectorExtent::Full ? m : k);
        }
        [[maybe_unused]] auto factors = mpfr_detail::compute_svd(
            vs, as, op.overwrite == SvdOverwrite::Left ? 'O' : mpfr_detail::vector_job(op.left),
            op.overwrite == SvdOverwrite::Right ? 'O' : mpfr_detail::vector_job(op.right), p);
        if (op.overwrite == SvdOverwrite::Left)
          mpfr_detail::unpack(factors.second, fs);
        else
          mpfr_detail::unpack(factors.first, fs);
        return KernelAttempt::success;
      },
      v, f, a);
}
} // namespace uni20::linalg
