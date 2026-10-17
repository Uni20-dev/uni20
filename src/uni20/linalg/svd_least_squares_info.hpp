#pragma once

#include <uni20/linalg/least_squares_info.hpp>
#include <vector>

namespace uni20::linalg
{
/// \brief Singular-value truncation policy for minimum-norm least squares.
template <Real R> struct SvdLeastSquaresOptions
{
    /// \brief Relative cutoff; absent selects epsilon at working precision.
    /// \details Must be finite and nonnegative. Zero disables relative truncation;
    ///          provider underflow safeguards still apply.
    std::optional<R> relative_cutoff = {};
    bool return_singular_values = false;
};

/// \brief Rank diagnostics for a successful truncated-SVD solution.
template <Real R> struct SvdLeastSquaresDiagnostics
{
    std::size_t rank = 0;
    R relative_cutoff = R{};
    /// \brief Descending singular values before truncation, empty unless requested.
    std::vector<R> singular_values = {};
};
} // namespace uni20::linalg
