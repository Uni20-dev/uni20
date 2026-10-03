#pragma once

#include <span>
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
/// \brief Complex binary80 counterpart of the packed LU adapter.
std::size_t getrf(std::size_t n, std::span<complex160> a, std::vector<std::size_t>& pivots);
/// \brief Solve from read-only binary80 LU factors and zero-based pivots.
void getrs(std::size_t n, std::size_t nrhs, std::span<float80 const> a, std::span<std::size_t const> pivots,
           std::span<float80> b);
/// \brief Complex binary80 counterpart of the solve-from-factors adapter.
void getrs(std::size_t n, std::size_t nrhs, std::span<complex160 const> a, std::span<std::size_t const> pivots,
           std::span<complex160> b);
} // namespace uni20::mplapack
