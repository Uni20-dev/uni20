#pragma once

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
} // namespace uni20::mplapack
