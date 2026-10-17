#pragma once

#include <uni20/linalg/solve_info.hpp>

namespace uni20::linalg
{
/// \brief LU pivot policy and optional capture of original matrix norms.
template <Real R> struct LuOptions
{
    /// \brief Same relative pivot rejection policy as SolveOptions.
    R relative_pivot_tolerance = R{};
    /// \brief Retain the original one- and infinity-norms for subsequent condition estimates.
    bool retain_norms = true;
};

/// \brief Original matrix norms after conversion to factor precision.
/// \details Complex entries use mathematical magnitude. An overflowing norm is
///          infinity; it does not invalidate an otherwise successful factorization.
template <Real R> struct LuNorms
{
    R one;
    R infinity;
};
} // namespace uni20::linalg
