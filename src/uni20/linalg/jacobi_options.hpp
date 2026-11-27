#pragma once

/**
 * \file jacobi_options.hpp
 * \ingroup linalg
 * \brief Controls and diagnostics for CUDA Jacobi decompositions.
 */

#include <optional>

namespace uni20::linalg
{

/// \brief Optional overrides for cuSOLVER's Jacobi stopping criteria.
/// \details Set values must be finite and positive. Unset values retain the provider defaults.
struct CudaJacobiOptions
{
    std::optional<double> tolerance{};
    std::optional<int> maximum_sweeps{};
};

/// \brief Stop status from one cuSOLVER Jacobi decomposition.
/// \details The residual is cuSOLVER's internal off-diagonal norm.
struct JacobiConvergence
{
    double residual = 0;
    int sweeps = 0;
    bool converged = true;
};

/// \brief Owning decomposition factors and the associated stopping information.
template <class Factors> struct CudaJacobiResult
{
    Factors factors;
    JacobiConvergence convergence;
};

} // namespace uni20::linalg
