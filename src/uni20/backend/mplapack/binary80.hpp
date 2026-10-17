#pragma once

#include <span>
#include <uni20/core/runtime_precision.hpp>
#include <uni20/core/types.hpp>
#include <vector>

namespace uni20::mplapack
{
/// \brief Packed column-major binary80 GEMM; provider representations stay private.
/// \details Sizes must match the dimensions. Zero coefficients preserve BLAS
///          no-read semantics. Conversion preserves the native x87 value exactly.
void gemm(std::size_t m, std::size_t n, std::size_t k, float80 const& alpha, std::span<float80 const> a,
          std::span<float80 const> b, float80 const& beta, std::span<float80> c);
/// \brief Complex binary80 counterpart of the packed GEMM adapter.
void gemm(std::size_t m, std::size_t n, std::size_t k, complex160 const& alpha, std::span<complex160 const> a,
          std::span<complex160 const> b, complex160 const& beta, std::span<complex160> c);
/// \brief Factor a square binary80 matrix, returning zero or a one-based singular pivot.
/// \details Pivot output is a zero-based row-swap sequence; a is overwritten.
std::size_t getrf(std::size_t n, std::span<float80> a, std::vector<std::size_t>& pivots);
#if UNI20_HAS_MPLAPACK_FLOAT80_COMPLEX
/// \brief Complex binary80 counterpart of the packed LU adapter.
std::size_t getrf(std::size_t n, std::span<complex160> a, std::vector<std::size_t>& pivots);
#endif
/// \brief Solve from read-only binary80 LU factors and zero-based pivots.
void getrs(std::size_t n, std::size_t nrhs, std::span<float80 const> a, std::span<std::size_t const> pivots,
           std::span<float80> b);
#if UNI20_HAS_MPLAPACK_FLOAT80_COMPLEX
/// \brief Complex binary80 counterpart of the solve-from-factors adapter.
void getrs(std::size_t n, std::size_t nrhs, std::span<complex160 const> a, std::span<std::size_t const> pivots,
           std::span<complex160> b);
#endif
/// \brief Overwrite m-by-n input work and return Q (m-by-k) and R (k-by-n), k=min(m,n).
void qr(std::size_t m, std::size_t n, std::span<float80> a, std::span<float80> q, std::span<float80> triangular,
        Precision p);

/// \brief Overwrite m-by-n input work and return Q (k-by-n) and L (m-by-k), k=min(m,n).
void lq(std::size_t m, std::size_t n, std::span<float80> a, std::span<float80> q, std::span<float80> triangular,
        Precision p);

/// \brief Compute ascending eigenvalues and optionally overwrite a with eigenvector columns.
/// \details The caller supplies an n-by-n matrix and n values. Triangle is 'U' or 'L'.
void eigh(std::size_t n, std::span<float80> a, std::span<float80> values, bool vectors, char triangle, Precision p);

/// \brief Compute descending singular values and requested U/Vh factors, destroying a.
/// \details Jobs are 'A' (full), 'S' (reduced), 'N' (omitted) or 'O' (overwrite a).
///          At most one factor may overwrite a. An omitted or overwritten factor
///          has an empty span; other spans must have their exact logical sizes.
void svd(std::size_t m, std::size_t n, std::span<float80> a, std::span<float80> values, std::span<float80> u,
         std::span<float80> vh, char jobu, char jobvt, Precision p);

/// \brief Compute unsorted complex eigenvalues and optional right eigenvectors, destroying a.
/// \details Real conjugate pairs are unpacked into complex vector columns. The vector
///          span is n-by-n when requested and empty otherwise.
void eigen(std::size_t n, std::span<float80> a, std::span<complex160> values, std::span<complex160> vectors,
           bool compute_vectors, Precision p);

#if UNI20_HAS_MPLAPACK_FLOAT80_COMPLEX
/// \brief Complex QR counterpart, with unitary columns in Q.
void qr(std::size_t m, std::size_t n, std::span<complex160> a, std::span<complex160> q,
        std::span<complex160> triangular, Precision p);

/// \brief Complex LQ counterpart, with unitary rows in Q.
void lq(std::size_t m, std::size_t n, std::span<complex160> a, std::span<complex160> q,
        std::span<complex160> triangular, Precision p);

/// \brief Hermitian counterpart of the packed real self-adjoint eigensystem.
void eigh(std::size_t n, std::span<complex160> a, std::span<float80> values, bool vectors, char triangle, Precision p);

/// \brief Complex SVD counterpart; Vh is the conjugate transpose of V.
void svd(std::size_t m, std::size_t n, std::span<complex160> a, std::span<float80> values, std::span<complex160> u,
         std::span<complex160> vh, char jobu, char jobvt, Precision p);

/// \brief Complex counterpart of the packed general eigensystem.
void eigen(std::size_t n, std::span<complex160> a, std::span<complex160> values, std::span<complex160> vectors,
           bool compute_vectors, Precision p);
#endif

/// \brief Packed GELSS; B has max(m,n) rows, values has min(m,n) entries.
/// \details Returns zero on success or a positive nonconvergence count. Rank and
///          outputs are published only on success. Requires nonzero m and n.
std::size_t gelss(std::size_t m, std::size_t n, std::size_t nrhs, std::span<float80> a, std::span<float80> b,
                  std::span<float80> values, float80 const& cutoff, std::size_t& rank, Precision p);
/// \brief Estimate reciprocal condition from packed successful LU and the original matrix norm.
/// \details Norm is '1' or 'I'; returns one for an empty matrix. Factors are preserved.
float80 gecon(std::size_t n, std::span<float80 const> a, float80 const& anorm, char norm, Precision p);
#if UNI20_HAS_MPLAPACK_FLOAT80_COMPLEX
/// \brief Packed GELSS; B has max(m,n) rows, values has min(m,n) entries.
/// \details Returns zero on success or a positive nonconvergence count. Rank and
///          outputs are published only on success. Requires nonzero m and n.
std::size_t gelss(std::size_t m, std::size_t n, std::size_t nrhs, std::span<complex160> a, std::span<complex160> b,
                  std::span<float80> values, float80 const& cutoff, std::size_t& rank, Precision p);
/// \brief Estimate reciprocal condition from packed successful LU and the original matrix norm.
/// \details Norm is '1' or 'I'; returns one for an empty matrix. Factors are preserved.
float80 gecon(std::size_t n, std::span<complex160 const> a, float80 const& anorm, char norm, Precision p);
#endif
} // namespace uni20::mplapack
