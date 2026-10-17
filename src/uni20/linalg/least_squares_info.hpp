#pragma once

#include <uni20/core/runtime_precision.hpp>
#include <uni20/core/scalar_concepts.hpp>
#if UNI20_HAS_MPREAL
#include <uni20/core/mpreal.hpp>
#endif
#include <cstddef>
#include <optional>

namespace uni20::linalg
{
/// \brief Numerical outcome of QR or SVD least squares.
enum class LeastSquaresStatus
{
  success,
  zero_diagonal,
  small_diagonal,
  nonfinite_input,
  nonfinite_result,
  nonconvergence
};

/// \brief Completed numerical attempt; success does not certify rank or conditioning.
struct LeastSquaresInfo
{
    LeastSquaresStatus status = LeastSquaresStatus::success;
    /// \brief Zero-based QR column rejected by the diagonal policy, not a numerical rank.
    std::optional<std::size_t> column = {};
    /// \brief Provider nonconvergence count for an SVD attempt, when available.
    std::optional<std::size_t> unconverged = {};
    [[nodiscard]] bool succeeded() const noexcept { return this->status == LeastSquaresStatus::success; }
};

/// \brief Full-rank QR rejection policy, independent of any physical convergence criterion.
template <Real R> struct LeastSquaresOptions
{
    /// \brief Reject nonzero abs(R[k,k])/max(abs(A)) at or below this finite, nonnegative tolerance.
    /// \details Zero rejects only zero diagonals. This scaling-dependent heuristic
    ///          is not a condition estimate or a rank-revealing criterion.
    R relative_diagonal_tolerance = R{};
};
} // namespace uni20::linalg
