#pragma once

/**
 * \file mpfr.hpp
 * \brief Synchronous packed-array boundary to the serial MPFR/MPC provider.
 * \details Arrays use column-major order and may not overlap. Every call needs finite
 *          working precision, copies into owning provider objects, and restores the
 *          executing thread's precision context before returning or throwing.
 */

#include <cstddef>
#include <span>
#include <uni20/core/mpcomplex.hpp>
#include <vector>

namespace uni20::mplapack
{
/// \brief Multiply packed column-major Uni20 real arrays at explicit precision.
/// \details Provider objects and ambient precision are confined to the implementation.
///          The call never suspends or hands off work. Array sizes must match the dimensions.
void gemm(std::size_t m, std::size_t n, std::size_t k, mpreal const& alpha, std::span<mpreal const> a,
          std::span<mpreal const> b, mpreal const& beta, std::span<mpreal> c, Precision p);
/// \brief Complex counterpart of the packed real GEMM adapter.
void gemm(std::size_t m, std::size_t n, std::size_t k, mpcomplex const& alpha, std::span<mpcomplex const> a,
          std::span<mpcomplex const> b, mpcomplex const& beta, std::span<mpcomplex> c, Precision p);

/// \brief Factor a packed square matrix, returning zero or a one-based singular pivot.
/// \details Pivot vectors contain zero-based row indices and are reusable by GETRS
///          at the same precision. The coefficient array is overwritten with LU data.
std::size_t getrf(std::size_t n, std::span<mpreal> a, std::vector<std::size_t>& pivots, Precision p);
/// \brief Complex counterpart of the packed real LU factorization adapter.
std::size_t getrf(std::size_t n, std::span<mpcomplex> a, std::vector<std::size_t>& pivots, Precision p);
/// \brief Solve from packed LU factors and zero-based pivots, overwriting the RHS array.
void getrs(std::size_t n, std::size_t nrhs, std::span<mpreal const> a, std::span<std::size_t const> pivots,
           std::span<mpreal> b, Precision p);
/// \brief Complex counterpart of the packed real LU solve adapter.
void getrs(std::size_t n, std::size_t nrhs, std::span<mpcomplex const> a, std::span<std::size_t const> pivots,
           std::span<mpcomplex> b, Precision p);
/// \brief Overwrite m-by-n input work and return Q (m-by-k) and R (k-by-n), k=min(m,n).
void qr(std::size_t m, std::size_t n, std::span<mpreal> a, std::span<mpreal> q, std::span<mpreal> triangular,
        Precision p);

/// \brief Overwrite m-by-n input work and return Q (k-by-n) and L (m-by-k), k=min(m,n).
void lq(std::size_t m, std::size_t n, std::span<mpreal> a, std::span<mpreal> q, std::span<mpreal> triangular,
        Precision p);

/// \brief Compute ascending eigenvalues and optionally overwrite a with eigenvector columns.
/// \details The caller supplies an n-by-n matrix and n values. Triangle is 'U' or 'L'.
void eigh(std::size_t n, std::span<mpreal> a, std::span<mpreal> values, bool vectors, char triangle, Precision p);

/// \brief Compute descending singular values and requested U/Vh factors, destroying a.
/// \details Jobs are 'A' (full), 'S' (reduced), 'N' (omitted) or 'O' (overwrite a).
///          At most one factor may overwrite a. An omitted or overwritten factor
///          has an empty span; other spans must have their exact logical sizes.
void svd(std::size_t m, std::size_t n, std::span<mpreal> a, std::span<mpreal> values, std::span<mpreal> u,
         std::span<mpreal> vh, char jobu, char jobvt, Precision p);

/// \brief Compute unsorted complex eigenvalues and optional right eigenvectors, destroying a.
/// \details Real conjugate pairs are unpacked into complex vector columns. The vector
///          span is n-by-n when requested and empty otherwise.
void eigen(std::size_t n, std::span<mpreal> a, std::span<mpcomplex> values, std::span<mpcomplex> vectors,
           bool compute_vectors, Precision p);

/// \brief Complex QR counterpart, with unitary columns in Q.
void qr(std::size_t m, std::size_t n, std::span<mpcomplex> a, std::span<mpcomplex> q, std::span<mpcomplex> triangular,
        Precision p);

/// \brief Complex LQ counterpart, with unitary rows in Q.
void lq(std::size_t m, std::size_t n, std::span<mpcomplex> a, std::span<mpcomplex> q, std::span<mpcomplex> triangular,
        Precision p);

/// \brief Hermitian counterpart of the packed real self-adjoint eigensystem.
void eigh(std::size_t n, std::span<mpcomplex> a, std::span<mpreal> values, bool vectors, char triangle, Precision p);

/// \brief Complex SVD counterpart; Vh is the conjugate transpose of V.
void svd(std::size_t m, std::size_t n, std::span<mpcomplex> a, std::span<mpreal> values, std::span<mpcomplex> u,
         std::span<mpcomplex> vh, char jobu, char jobvt, Precision p);

/// \brief Complex counterpart of the packed general eigensystem.
void eigen(std::size_t n, std::span<mpcomplex> a, std::span<mpcomplex> values, std::span<mpcomplex> vectors,
           bool compute_vectors, Precision p);

/// \brief Packed GELSS; B has max(m,n) rows, values has min(m,n) entries.
/// \details Returns zero on success or a positive nonconvergence count. Rank and
///          outputs are published only on success. Requires nonzero m and n.
std::size_t gelss(std::size_t m, std::size_t n, std::size_t nrhs, std::span<mpreal> a, std::span<mpreal> b,
                  std::span<mpreal> values, mpreal const& cutoff, std::size_t& rank, Precision p);
/// \brief Estimate reciprocal condition from packed successful LU and the original matrix norm.
/// \details Norm is '1' or 'I'; returns one for an empty matrix. Factors are preserved.
mpreal gecon(std::size_t n, std::span<mpreal const> a, mpreal const& anorm, char norm, Precision p);
/// \brief Packed GELSS; B has max(m,n) rows, values has min(m,n) entries.
/// \details Returns zero on success or a positive nonconvergence count. Rank and
///          outputs are published only on success. Requires nonzero m and n.
std::size_t gelss(std::size_t m, std::size_t n, std::size_t nrhs, std::span<mpcomplex> a, std::span<mpcomplex> b,
                  std::span<mpreal> values, mpreal const& cutoff, std::size_t& rank, Precision p);
/// \brief Estimate reciprocal condition from packed successful LU and the original matrix norm.
/// \details Norm is '1' or 'I'; returns one for an empty matrix. Factors are preserved.
mpreal gecon(std::size_t n, std::span<mpcomplex const> a, mpreal const& anorm, char norm, Precision p);
} // namespace uni20::mplapack
