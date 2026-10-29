#pragma once

#include <uni20/linalg/backends/lu_common.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/mdspan/concepts.hpp>

namespace uni20::linalg::detail::cpu_reference
{
template <class MatrixMdspan>
void swap_rows(MatrixMdspan& matrix, std::size_t lhs, std::size_t rhs, std::size_t columns)
{
  if (lhs == rhs) return;
  using index_type = typename std::remove_cvref_t<MatrixMdspan>::index_type;
  for (std::size_t col = 0; col < columns; ++col)
  {
    auto const left_index = static_cast<index_type>(lhs);
    auto const right_index = static_cast<index_type>(rhs);
    auto const column = static_cast<index_type>(col);
    auto temporary = static_cast<typename std::remove_cvref_t<MatrixMdspan>::value_type>(matrix[left_index, column]);
    auto right_value = static_cast<typename std::remove_cvref_t<MatrixMdspan>::value_type>(matrix[right_index, column]);
    matrix[left_index, column] = right_value;
    matrix[right_index, column] = temporary;
  }
}

// Values already have the operation precision. Keep the arithmetic shared by
// native kernels, runtime-precision adapters and the one-shot runtime solve.
template <class Matrix>
KernelAttempt lu_factor(Matrix& s, std::span<std::size_t> pivots,
                        LuNorms<make_real_t<typename Matrix::value_type>>* norms, SolveInfo& info,
                        SolveOptions<make_real_t<typename Matrix::value_type>> const& options,
                        make_real_t<typename Matrix::value_type> const& zero)
{
  using S = typename Matrix::value_type;
  info = {};
  if (!detail::solve_matrix_is_finite(s))
  {
    info.status = SolveStatus::nonfinite_input;
    return KernelAttempt::success;
  }
  detail::capture_lu_norms(s, norms, zero);
  auto scale = detail::lu_input_scale(s, zero);
  auto n = pivots.size();
  // Public factors are column-major; direct kernel callers may supply other layouts.
  bool rows_inner = true;
  if constexpr (StridedMdspanLike<decltype(s)>) rows_inner = s.stride(0) <= s.stride(1);
  auto update_entry = [&](std::size_t i, std::size_t j, S const& multiplier, S const& upper) {
    S value = S(s[i, j]) - multiplier * upper;
    if (!uni20::isfinite(value))
    {
      info.status = SolveStatus::nonfinite_result;
      return false;
    }
    s[i, j] = value;
    return true;
  };
  for (std::size_t k = 0; k < n; ++k)
  {
    auto magnitude = detail::lu_magnitude(S(s[k, k]));
    std::size_t pivot = k;
    for (std::size_t i = k + 1; i < n; ++i)
    {
      auto candidate = detail::lu_magnitude(S(s[i, k]));
      if (detail::lu_magnitude_less(magnitude, candidate))
      {
        magnitude = std::move(candidate);
        pivot = i;
      }
    }
    if (magnitude.scale == 0)
      info = {.status = SolveStatus::singular, .pivot = k};
    else if (detail::lu_small_pivot(magnitude, scale, options.relative_pivot_tolerance))
      info = {.status = SolveStatus::small_pivot, .pivot = k};
    if (!info.succeeded()) return KernelAttempt::success;
    pivots[k] = pivot;
    // Previously computed columns of L must participate in the row swap.
    detail::cpu_reference::swap_rows(s, k, pivot, n);
    for (std::size_t i = k + 1; i < n; ++i)
    {
      S multiplier = S(s[i, k]) / S(s[k, k]);
      if (!uni20::isfinite(multiplier))
      {
        info.status = SolveStatus::nonfinite_result;
        return KernelAttempt::success;
      }
      s[i, k] = multiplier;
    }
    if (rows_inner)
    {
      for (std::size_t j = k + 1; j < n; ++j)
      {
        S upper = s[k, j];
        for (std::size_t i = k + 1; i < n; ++i)
          if (!update_entry(i, j, S(s[i, k]), upper)) return KernelAttempt::success;
      }
    }
    else
    {
      for (std::size_t i = k + 1; i < n; ++i)
      {
        S multiplier = s[i, k];
        for (std::size_t j = k + 1; j < n; ++j)
          if (!update_entry(i, j, multiplier, S(s[k, j]))) return KernelAttempt::success;
      }
    }
  }
  return KernelAttempt::success;
}

template <class Matrix, class Rhs>
KernelAttempt lu_solve(Matrix const& as, std::span<std::size_t const> pivots, Rhs& bs, SolveInfo& info)
{
  info = {};
  using S = typename Matrix::value_type;
  if (!detail::solve_matrix_is_finite(bs))
  {
    info.status = SolveStatus::nonfinite_input;
    return KernelAttempt::success;
  }
  auto n = pivots.size();
  auto count = std::size_t(bs.extent(1));
  for (std::size_t k = 0; k < n; ++k)
    detail::cpu_reference::swap_rows(bs, k, pivots[k], count);
  for (std::size_t col = 0; col < count; ++col)
  {
    for (std::size_t i = 0; i < n; ++i)
    {
      S value = bs[i, col];
      for (std::size_t k = 0; k < i; ++k)
        value -= S(as[i, k]) * S(bs[k, col]);
      if (!uni20::isfinite(value))
      {
        info.status = SolveStatus::nonfinite_result;
        return KernelAttempt::success;
      }
      bs[i, col] = value;
    }
    for (std::size_t i = n; i-- > 0;)
    {
      S value = bs[i, col];
      for (std::size_t k = i + 1; k < n; ++k)
        value -= S(as[i, k]) * S(bs[k, col]);
      value /= S(as[i, i]);
      if (!uni20::isfinite(value))
      {
        info.status = SolveStatus::nonfinite_result;
        return KernelAttempt::success;
      }
      bs[i, col] = value;
    }
  }
  return KernelAttempt::success;
}
} // namespace uni20::linalg::detail::cpu_reference
