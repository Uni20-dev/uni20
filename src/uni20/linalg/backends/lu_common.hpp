#pragma once

#include "linear_solve_common.hpp"
#include <algorithm>
#include <span>

namespace uni20::linalg::detail
{
// Mathematical magnitude = scale * unit, with unit in [1,sqrt(2)]. Never
// multiply these parts: the magnitude of finite complex components can overflow.
template <Real R> struct LuMagnitude
{
    R scale;
    R unit;
};

template <RealOrComplex S> auto lu_magnitude(S const& value)
{
  using R = make_real_t<S>;
  R scale = math::abs(uni20::real(value));
  R unit = scalar_like(scale, 1);
  if constexpr (Complex<S>)
  {
    R other = math::abs(uni20::imag(value));
    if (other > scale) std::swap(other, scale);
    if (scale != 0)
    {
      auto ratio = other / scale;
      unit = math::sqrt(unit + ratio * ratio);
    }
  }
  return LuMagnitude<R>{std::move(scale), std::move(unit)};
}

template <Real R> bool lu_magnitude_less(LuMagnitude<R> const& a, LuMagnitude<R> const& b)
{
  if (b.scale == 0) return false;
  if (a.scale <= b.scale) return (a.scale / b.scale) * a.unit < b.unit;
  return a.unit < (b.scale / a.scale) * b.unit;
}

template <Real R> bool lu_small_pivot(LuMagnitude<R> const& pivot, LuMagnitude<R> const& scale, R const& tolerance)
{
  if (tolerance == 0) return false;
  if (pivot.scale <= scale.scale) return (pivot.scale / scale.scale) * (pivot.unit / scale.unit) <= tolerance;
  // Compare the reciprocal ratio when the pivot grew beyond the input scale.
  // This avoids overflow in pivot.scale / scale.scale.
  return scalar_like(tolerance, 1) / tolerance <= (scale.scale / pivot.scale) * (scale.unit / pivot.unit);
}

template <class Matrix, Real R> auto lu_input_scale(Matrix const& a, R const& zero)
{
  LuMagnitude<R> scale{zero, scalar_like(zero, 1)};
  for (std::size_t j = 0; j < std::size_t(a.extent(1)); ++j)
    for (std::size_t i = 0; i < std::size_t(a.extent(0)); ++i)
    {
      auto next = lu_magnitude(static_cast<typename Matrix::value_type>(a[i, j]));
      if (lu_magnitude_less(scale, next)) scale = std::move(next);
    }
  return scale;
}

template <class Matrix, Real R>
bool check_lu_diagonal(Matrix const& a, LuMagnitude<R> const& scale, R const& tolerance, SolveInfo& info)
{
  for (std::size_t k = 0; k < std::size_t(a.extent(0)); ++k)
  {
    auto magnitude = lu_magnitude(static_cast<typename Matrix::value_type>(a[k, k]));
    if (magnitude.scale == 0)
      info = {.status = SolveStatus::singular, .pivot = k};
    else if (lu_small_pivot(magnitude, scale, tolerance))
      info = {.status = SolveStatus::small_pivot, .pivot = k};
    if (!info.succeeded()) return false;
  }
  return true;
}

// Direct dispatch is also callable without the owning-factor frontend. Shape
// and pivot metadata enter here; established factors need no numerical rescan.
template <class Matrix> void require_lu_shape(Matrix const& a, std::size_t pivot_count)
{
  ERROR_IF(a.extent(0) != a.extent(1), "LU requires a square matrix");
  ERROR_IF(std::size_t(a.extent(0)) != pivot_count, "LU pivot count mismatch");
}
template <class Matrix, class Rhs>
void require_lu_solve_shape(Matrix const& a, std::span<std::size_t const> pivots, Rhs const& b)
{
  require_lu_shape(a, pivots.size());
  ERROR_IF(a.extent(0) != b.extent(0), "LU solve RHS row count mismatch");
}
} // namespace uni20::linalg::detail
