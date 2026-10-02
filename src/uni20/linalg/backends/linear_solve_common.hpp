#pragma once

#include <uni20/common/trace.hpp>
#include <uni20/core/math.hpp>
#include <uni20/core/scalar_traits.hpp>
#include <uni20/linalg/solve_info.hpp>

#include <cmath>

namespace uni20::linalg::detail
{

template <class Matrix> bool solve_matrix_is_finite(Matrix const& matrix)
{
  using scalar_type = typename Matrix::value_type;
  for (typename Matrix::index_type row = 0; row < matrix.extent(0); ++row)
    for (typename Matrix::index_type col = 0; col < matrix.extent(1); ++col)
      if (!uni20::isfinite(static_cast<scalar_type>(matrix[row, col]))) return false;
  return true;
}

template <uni20::Real Real> void require_solve_options(SolveOptions<Real> const& options)
{
  ERROR_IF(!uni20::isfinite(options.relative_pivot_tolerance) || options.relative_pivot_tolerance < Real{},
           "solve requires a finite nonnegative relative pivot tolerance");
}

// Called only after a backend has committed to execution; a numerical failure
// writes diagnostics but must still return KernelAttempt::success to dispatch.
template <class Matrix, class Rhs, class Real>
bool prepare_linear_solve(Matrix const& coefficients, Rhs const& rhs, Real& scale, SolveInfo& info)
{
  info = {};
  if (!solve_matrix_is_finite(coefficients) || !solve_matrix_is_finite(rhs))
  {
    info.status = SolveStatus::nonfinite_input;
    return false;
  }
  using std::abs;
  using scalar_type = typename Matrix::value_type;
  for (typename Matrix::index_type row = 0; row < coefficients.extent(0); ++row)
    for (typename Matrix::index_type col = 0; col < coefficients.extent(1); ++col)
    {
      Real const magnitude = abs(static_cast<scalar_type>(coefficients[row, col]));
      if (!uni20::isfinite(magnitude))
      {
        info.status = SolveStatus::nonfinite_result;
        return false;
      }
      if (magnitude > scale) scale = magnitude;
    }
  return true;
}

template <class Real>
bool check_solve_pivot(Real magnitude, Real scale, SolveOptions<Real> const& options, std::size_t column,
                       SolveInfo& info)
{
  if (!uni20::isfinite(magnitude))
    info = {.status = SolveStatus::nonfinite_result};
  else if (magnitude == Real{})
    info = {.status = SolveStatus::singular, .pivot = column};
  else if (options.relative_pivot_tolerance > Real{} && magnitude / scale <= options.relative_pivot_tolerance)
    info = {.status = SolveStatus::small_pivot, .pivot = column};
  return info.succeeded();
}

} // namespace uni20::linalg::detail
