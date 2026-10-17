#pragma once

/**
 * \file solve_diagnostics.hpp
 * \brief Preserving host residual and backward-error evaluation at explicit precision.
 */

#include <algorithm>
#include <uni20/core/math.hpp>
#include <uni20/linalg/refinement_info.hpp>
#include <uni20/tensor/access.hpp>
#include <uni20/tensor/tensor.hpp>
#include <vector>

namespace uni20::linalg
{
/// \brief Recomputed residual and per-RHS diagnostics, published together only on success.
template <RealOrComplex S> struct SolveDiagnosticsResult
{
    SolveInfo info = {};
    std::optional<DenseMatrix<S>> residual = {};
    std::vector<SolveError<make_real_t<S>>> columns = {};
};

namespace refinement_detail
{
template <class T>
concept HostMatrix =
    RankedTensorView<T, 2> && HostReadableMdspec<tensor_mdspec_t<T const&>> && RealOrComplex<tensor_element_t<T>>;

template <RealOrComplex To, RealOrComplex From>
  requires(Complex<To> == Complex<From>)
To convert(From const& value, Precision p)
{
  if constexpr (Complex<To>)
  {
    using R = make_real_t<To>;
    return To(convert<R>(value.real(), p), convert<R>(value.imag(), p));
  }
  else if constexpr (has_runtime_precision_v<To>)
    return scalar_at_precision<To>(value, p);
  else
  {
    require_precision<To>(p);
    return static_cast<To>(value);
  }
}

// Conversion is an explicit algorithm workspace boundary. Observe logical
// accessor values while the host access lease is held, never raw storage.
template <RealOrComplex To, HostMatrix A>
  requires(Complex<To> == Complex<tensor_element_t<A>>)
std::optional<DenseMatrix<To>> pack(A const& input, Precision p, SolveInfo& info)
{
  using From = tensor_element_t<A>;
  auto descriptor = mdspec_of(input);
  auto access = acquire_host_read_access_sync(descriptor);
  auto values = access.mdspan();
  DenseMatrix<To> output(uninitialized, input.extent(0), input.extent(1), p);
  for (std::size_t j = 0; j < std::size_t(input.extent(1)); ++j)
    for (std::size_t i = 0; i < std::size_t(input.extent(0)); ++i)
    {
      From value(values[i, j]);
      if (!uni20::isfinite(value))
      {
        info.status = SolveStatus::nonfinite_input;
        return {};
      }
      auto converted = convert<To>(value, p);
      if (!uni20::isfinite(converted))
      {
        info.status = SolveStatus::nonfinite_result;
        return {};
      }
      output[i, j] = std::move(converted);
    }
  return output;
}

template <RealOrComplex S>
SolveDiagnosticsResult<S> evaluate(DenseMatrix<S> const& a, DenseMatrix<S> const& b, DenseMatrix<S> const& x,
                                   Precision p)
{
  using R = make_real_t<S>;
  auto const zero = scalar_at_precision<R>(0, p);
  auto const szero = scalar_at_precision<S>(0, p);
  std::size_t const n = a.extent(0), count = b.extent(1);
  SolveDiagnosticsResult<S> result;
  DenseMatrix<S> residual(n, count, p);
  std::vector<R> row_sums(n, zero);
  R a_norm = zero;
  for (std::size_t k = 0; k < n; ++k)
    for (std::size_t i = 0; i < n; ++i)
      row_sums[i] += math::abs(a[i, k]);
  for (auto const& sum : row_sums)
    a_norm = std::max(a_norm, sum);
  if (!uni20::isfinite(a_norm))
  {
    result.info.status = SolveStatus::nonfinite_result;
    return result;
  }
  for (std::size_t j = 0; j < count; ++j)
  {
    R x_norm = zero, b_norm = zero, r_norm = zero, componentwise = zero;
    std::vector<R> denominator(n, zero);
    for (std::size_t i = 0; i < n; ++i)
    {
      residual[i, j] = b[i, j];
      denominator[i] = math::abs(b[i, j]);
      b_norm = std::max(b_norm, denominator[i]);
      x_norm = std::max(x_norm, R(math::abs(x[i, j])));
    }
    // Column traversal follows the packed matrix layout. All products and
    // sums are at residual precision, including the converted iterate.
    for (std::size_t k = 0; k < n; ++k)
    {
      R x_magnitude = math::abs(x[k, j]);
      for (std::size_t i = 0; i < n; ++i)
      {
        residual[i, j] -= a[i, k] * x[k, j];
        denominator[i] += math::abs(a[i, k]) * x_magnitude;
      }
    }
    for (std::size_t i = 0; i < n; ++i)
    {
      R magnitude = math::abs(residual[i, j]);
      if (!uni20::isfinite(magnitude) || !uni20::isfinite(denominator[i]))
      {
        result.info.status = SolveStatus::nonfinite_result;
        return result;
      }
      r_norm = std::max(r_norm, magnitude);
      // A zero denominator means an identically zero row contribution, so its
      // residual is zero as well. Define that 0/0 contribution to be zero.
      R ratio = denominator[i] == 0 ? zero : R(magnitude / denominator[i]);
      if (denominator[i] == 0 && residual[i, j] != szero)
      {
        result.info.status = SolveStatus::nonfinite_result;
        return result;
      }
      componentwise = std::max(componentwise, ratio);
    }
    R denominator_norm = a_norm * x_norm + b_norm;
    if (!uni20::isfinite(denominator_norm))
    {
      result.info.status = SolveStatus::nonfinite_result;
      return result;
    }
    R normwise = denominator_norm == 0 ? zero : R(r_norm / denominator_norm);
    result.columns.push_back({r_norm, normwise, componentwise});
  }
  result.residual.emplace(std::move(residual));
  return result;
}
} // namespace refinement_detail

/// \brief Evaluate b - A*x and normwise/componentwise backward error in scalar S at p.
/// \details Requires square A and matching rank-two RHS/solution shapes. All
///          inputs are preserved and may have different scalar types of the same
///          real/complex kind. Logical host accessors are respected. Conversion,
///          products, sums and complex magnitudes use p. Nonfinite conversion or
///          arithmetic, including diagnostic denominator overflow, is recoverable
///          failure, with no published residual or columns. Zero RHS columns and
///          zero-order systems do not read numerical values. These diagnostics
///          are not certified bounds and can be optimistic near underflow.
template <RealOrComplex S, refinement_detail::HostMatrix A, refinement_detail::HostMatrix B,
          refinement_detail::HostMatrix X>
  requires(Complex<S> == Complex<tensor_element_t<A>> && Complex<S> == Complex<tensor_element_t<B>> &&
           Complex<S> == Complex<tensor_element_t<X>>)
[[nodiscard]] SolveDiagnosticsResult<S> solve_diagnostics_with_info(A const& a, B const& b, X const& x, Precision p)
{
  require_precision<S>(p);
  (void)p.bit_count();
  ERROR_IF(a.extent(0) != a.extent(1) || b.extent(0) != a.extent(0) || x.extent(0) != a.extent(1) ||
               x.extent(1) != b.extent(1),
           "solve diagnostics require square A and matching RHS/solution shapes");
  SolveDiagnosticsResult<S> result;
  if (a.extent(0) == 0 || b.extent(1) == 0)
  {
    auto zero = scalar_at_precision<make_real_t<S>>(0, p);
    result.residual.emplace(a.extent(0), b.extent(1), p);
    result.columns.assign(b.extent(1), {zero, zero, zero});
    return result;
  }
  auto av = refinement_detail::pack<S>(a, p, result.info);
  if (!av) return result;
  auto bv = refinement_detail::pack<S>(b, p, result.info);
  if (!bv) return result;
  auto xv = refinement_detail::pack<S>(x, p, result.info);
  if (!xv) return result;
  result = refinement_detail::evaluate(*av, *bv, *xv, p);
  if (!result.info.succeeded()) result.columns.clear();
  return result;
}
} // namespace uni20::linalg
