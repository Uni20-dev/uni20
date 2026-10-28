#pragma once

#include "math.hpp"
#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <stdexcept>

namespace uni20
{
/// \brief An abscissa and independently evaluated distances to its finite endpoints.
/// \details value may round to an endpoint while both distances remain positive.
///          Singular integrands should use the distances rather than subtracting
///          the rounded value from an endpoint. All three fields have working precision.
template <Real R> struct quadrature_point
{
    R value;
    R lower_distance;
    R upper_distance;
};

/// \brief Termination of finite-interval tanh-sinh integration.
enum class quadrature_status
{
  converged,
  evaluation_limit,
  refinement_limit,
  endpoint_limit,
  nonfinite_integrand,
  arithmetic_range
};

/// \brief Explicit precision, component tolerances and work limits for tanh-sinh quadrature.
/// \details Convergence uses max(absolute_tolerance[j], relative_tolerance*abs(value[j])).
///          endpoint_power controls the transformed cutoff -endpoint_power*log(epsilon)+4;
///          increasing it resolves stronger endpoint singularities at additional cost.
template <Real R, std::size_t N = 1> struct tanh_sinh_options
{
    static_assert(N > 0);
    explicit tanh_sinh_options(Precision p) : relative_tolerance(scalar_at_precision<R>(0, p))
    {
      if (p.is_exact()) throw std::invalid_argument("quadrature requires finite precision");
      relative_tolerance = numeric_limits<R>::epsilon(p) * scalar_at_precision<R>(64, p);
      absolute_tolerance.fill(scalar_at_precision<R>(0, p));
    }
    tanh_sinh_options()
      requires(!has_runtime_precision_v<R>)
        : tanh_sinh_options(native_precision<R>())
    {}

    R relative_tolerance;
    std::array<R, N> absolute_tolerance;
    std::size_t max_evaluations = 200000;
    std::size_t max_levels = 12;
    unsigned endpoint_power = 3;
};

/// \brief Last complete mesh, its diagnostic estimates and total work, including an interrupted mesh.
/// \details error is the maximum of mesh change, a roundoff floor and endpoint_indicator.
///          endpoint_indicator is the absolute transformed integrand at the outermost
///          sampled pair, not a rigorous omitted-tail bound. No interval certificate is
///          provided. error is infinity until two meshes complete; zero-width integrals
///          return exact typed zeros without evaluating the callback.
template <Real R, std::size_t N = 1> struct quadrature_result
{
    std::array<R, N> value;
    std::array<R, N> error;
    std::array<R, N> absolute_integral;
    std::array<R, N> endpoint_indicator;
    std::size_t evaluations = 0;
    std::size_t levels = 0;
    quadrature_status status = quadrature_status::refinement_limit;
};

namespace detail
{
template <class R> struct quadrature_sum
{
    R sum, correction;
    void add(R const& value)
    {
      R const adjusted = value - correction, next = sum + adjusted;
      correction = (next - sum) - adjusted;
      sum = next;
    }
};
} // namespace detail

/// \brief Integrate N real components over lo<=hi using successive tanh-sinh meshes.
/// \details f(quadrature_point<R> const&) returns std::array<R,N> at the bounds' finite
///          precision. Both bounds and every tolerance must have that same precision.
///          Two consecutive mesh agreements are required. Mesh agreement with a large
///          endpoint indicator returns endpoint_limit: increase endpoint_power or change
///          variables. Estimates cannot certify unresolved narrow features, integrand
///          error or arbitrary endpoint tails. Split interior singularities beforehand.
///          Callback exceptions propagate. Numerical range/budget failures retain the last
///          complete mesh. No global scalar settings or scheduler state are changed.
template <Real R, std::size_t N, class F>
quadrature_result<R, N> tanh_sinh_integrate(F const& f, R const& lo, R const& hi,
                                            tanh_sinh_options<R, N> const& options)
{
  auto const p = precision_of(lo);
  if (p.is_exact() || precision_of(hi) != p || precision_of(options.relative_tolerance) != p)
    throw std::invalid_argument("quadrature requires matching finite precision");
  auto number = [p](auto x) { return scalar_at_precision<R>(x, p); };
  R const zero = number(0), one = number(1), two = number(2);
  if (!isfinite(lo) || !isfinite(hi) || hi < lo || !isfinite(options.relative_tolerance) ||
      options.relative_tolerance < zero || options.endpoint_power < 2)
    throw std::invalid_argument("invalid quadrature interval, tolerance or endpoint power");
  for (auto const& tolerance : options.absolute_tolerance)
    if (precision_of(tolerance) != p || !isfinite(tolerance) || tolerance < zero ||
        (tolerance == zero && options.relative_tolerance == zero))
      throw std::invalid_argument("quadrature requires nonnegative tolerances, with a positive target per component");

  quadrature_result<R, N> out;
  // Infinity is only a diagnostic sentinel; no finite input is converted via double.
  R const infinity = number(numeric_limits<double>::infinity());
  out.value.fill(zero);
  out.error.fill(infinity);
  out.absolute_integral.fill(zero);
  out.endpoint_indicator.fill(infinity);
  if (lo == hi)
  {
    out.error.fill(zero);
    out.endpoint_indicator.fill(zero);
    out.status = quadrature_status::converged;
    return out;
  }
  R const width = hi - lo, eps = numeric_limits<R>::epsilon(p), pi_value = pi<R> * one;
  R const minimum = math::next_up(zero);
  R const limit = -number(options.endpoint_power) * math::log(eps) + number(4);
  if (!isfinite(width) || !isfinite(limit) || !(width / two > zero) || !(limit > zero))
  {
    out.status = quadrature_status::arithmetic_range;
    return out;
  }
  std::array<R, N> previous = out.value;
  unsigned stable = 0, endpoint_stable = 0;
  R h = one;
  for (std::size_t level = 0; level < options.max_levels; ++level)
  {
    std::array<detail::quadrature_sum<R>, N> sum, absolute;
    for (std::size_t j = 0; j < N; ++j)
      sum[j] = absolute[j] = {zero, zero};
    std::array<R, N> edge;
    std::array<std::size_t, N> nonzero_samples{};
    edge.fill(zero);
    auto sample = [&](R const& x, R const& lower, R const& upper, R const& unscaled_weight, bool endpoint) {
      if (out.evaluations == options.max_evaluations)
      {
        out.status = quadrature_status::evaluation_limit;
        return false;
      }
      R const weight = unscaled_weight * h;
      if (!(lower > zero) || !(upper > zero) || !(weight > zero) || !isfinite(weight))
      {
        out.status = quadrature_status::arithmetic_range;
        return false;
      }
      ++out.evaluations;
      std::array<R, N> const y = f(quadrature_point<R>{x, lower, upper});
      for (std::size_t j = 0; j < N; ++j)
      {
        if (precision_of(y[j]) != p) throw std::invalid_argument("quadrature integrand changed working precision");
        if (!isfinite(y[j]))
        {
          out.status = quadrature_status::nonfinite_integrand;
          return false;
        }
        R const term = weight * y[j];
        if (y[j] != zero) ++nonzero_samples[j];
        sum[j].add(term);
        absolute[j].add(math::abs(term));
        // Evaluate this diagnostic only at the final pair. At interior nodes
        // the unscaled product may overflow while the weighted term fits.
        if (endpoint) edge[j] += math::abs(unscaled_weight * y[j]);
        if (!isfinite(sum[j].sum) || !isfinite(absolute[j].sum) || !isfinite(edge[j]))
        {
          out.status = quadrature_status::arithmetic_range;
          return false;
        }
      }
      return true;
    };
    R const half = width / two;
    if (!sample(lo + half, half, half, width * (pi_value / number(4)), false)) return out;
    R t = h, argument = pi_value * math::sinh(t);
    while (argument <= limit)
    {
      R const next = t + h;
      if (!(next > t))
      {
        out.status = quadrature_status::arithmetic_range;
        return out;
      }
      R const next_argument = pi_value * math::sinh(next);
      bool const endpoint = next_argument > limit;
      R const z = math::exp(-argument), denominator = one + z;
      R const distance = width * (z / denominator), far = width / denominator;
      R const unscaled_weight = distance * (pi_value * math::cosh(t) / denominator);
      if (!sample(lo + distance, distance, far, unscaled_weight, endpoint) ||
          !sample(hi - distance, far, distance, unscaled_weight, endpoint))
        return out;
      t = next;
      argument = next_argument;
    }
    bool mesh_ok = level >= 2, endpoints_ok = true, tail_limited_mesh = level >= 2;
    for (std::size_t j = 0; j < N; ++j)
    {
      out.value[j] = sum[j].sum;
      out.absolute_integral[j] = absolute[j].sum;
      out.endpoint_indicator[j] = edge[j];
      // Relative roundoff alone can underflow to zero and falsely certify an
      // entirely subnormal integral. Include an absolute quantization allowance
      // for each nonzero component sample, even when its weighted term vanishes.
      R const roundoff = number(8) * (eps * out.absolute_integral[j] + minimum * number(nonzero_samples[j]));
      R const mesh_error = level == 0 ? infinity : std::max(math::abs(out.value[j] - previous[j]), roundoff);
      out.error[j] = std::max(mesh_error, edge[j]);
      R const target = std::max(options.absolute_tolerance[j], options.relative_tolerance * math::abs(out.value[j]));
      mesh_ok = mesh_ok && mesh_error <= target;
      tail_limited_mesh = tail_limited_mesh && mesh_error <= std::max(target, edge[j]);
      endpoints_ok = endpoints_ok && edge[j] <= target;
    }
    ++out.levels;
    stable = mesh_ok ? stable + 1 : 0;
    // A varying last abscissa can prevent mesh agreement even when the missing
    // endpoint tail dominates the discrepancy. Diagnose that limitation rather
    // than spending the refinement budget on the same insufficient cutoff.
    endpoint_stable = tail_limited_mesh && !endpoints_ok ? endpoint_stable + 1 : 0;
    if (stable == 2)
    {
      out.status = endpoints_ok ? quadrature_status::converged : quadrature_status::endpoint_limit;
      return out;
    }
    if (endpoint_stable == 2)
    {
      out.status = quadrature_status::endpoint_limit;
      return out;
    }
    previous = out.value;
    h /= two;
    if (!(h > zero))
    {
      out.status = quadrature_status::arithmetic_range;
      return out;
    }
  }
  return out;
}
} // namespace uni20
