# Finite-interval quadrature

`<uni20/core/finite_quadrature.hpp>` provides array-valued tanh-sinh integration
for native real scalars and explicit-precision `mpreal`. It needs scalar math
only, not BLAS or MPLAPACK. Independent integrations have no shared mutable
state and can be composed by the caller's Async tasks; mesh refinements within
an integration remain sequential.

## Calling the integrator

The callback receives the coordinate and both endpoint distances. It returns
an `std::array<R, N>` so related components can share an expensive evaluation.
For a scalar integral use `N=1`:

```cpp
#include <uni20/core/finite_quadrature.hpp>

uni20::tanh_sinh_options<double> options;
auto f = [](uni20::quadrature_point<double> const& x) {
    return std::array{1.0 / uni20::math::sqrt(x.upper_distance)};
};
auto result = uni20::tanh_sinh_integrate(f, 1.0, 2.0, options);
// Integral of (2-x)^(-1/2) is 2. Check status before using the estimate.
bool converged = result.status == uni20::quadrature_status::converged;
```

For MPFR, construct options, bounds, tolerances and callback values at the same
explicit precision. Exact or unset values do not supply a working precision:

```cpp
auto p = uni20::Precision::bits(256);
using R = uni20::mpreal; // requires UNI20_HAS_MPREAL
uni20::tanh_sinh_options<R> options(p);
auto one = R(1, p);
auto f = [one](uni20::quadrature_point<R> const& x) {
    return std::array<R, 1>{one / uni20::math::sqrt(x.upper_distance)};
};
auto result = uni20::tanh_sinh_integrate(f, one, R(2, p), options);
```

The implementation stays in `R`; it does not convert finite inputs through
binary64 or change the MPFR default precision or exponent range. Fixed types
require their native precision. A mismatch in bounds, tolerances or callback
results throws rather than silently converting. Callback exceptions propagate.

## Endpoint evaluation

`quadrature_point<R>` contains `value`, `lower_distance` and `upper_distance`.
Near an endpoint, `value` can round to that endpoint even though the directly
computed distance remains positive. Use the distance to evaluate a singular
factor. Computing `2-x.value` in the examples loses that information and can
produce an infinity. The integrator reports a nonfinite callback value; it
does not silently skip such samples.

The same problem and the endpoint-distance remedy are discussed in the
[Boost.Math tanh-sinh documentation](https://www.boost.org/doc/libs/latest/libs/math/doc/html/math_toolkit/double_exponential/de_tanh_sinh.html).
This interface exposes both distances explicitly. They need not agree bit for
bit with subtraction of the rounded coordinate from its endpoints.

## Tolerances and diagnostics

Every component uses the target
`max(absolute_tolerance[j], relative_tolerance*abs(value[j]))`. Defaults are
64 working epsilons relative and zero absolute. For cancellation to a small
or zero integral, supply an absolute tolerance. Each component must have a
positive relative or absolute target. Bounds must be finite with `lo<=hi`.
A zero-width interval returns typed zeros without invoking the callback.

The result retains the last **complete** mesh even if the next mesh exhausts
its budget or encounters a numerical failure. `evaluations` counts every
callback invocation, including failed or discarded-mesh samples; `levels`
counts complete meshes. No sample cache persists between meshes or calls.

| Field or status | Meaning |
| --- | --- |
| `value` | Component integral estimates |
| `absolute_integral` | Integral estimates of the absolute component values; useful for assessing cancellation |
| `endpoint_indicator` | Sum of absolute transformed integrand values at the outermost sampled pair |
| `error` | Maximum of the last mesh change, a roundoff floor and the endpoint indicator; infinity until two meshes complete |
| `converged` | Two consecutive mesh comparisons pass their targets and the latest endpoint indicator passes |
| `endpoint_limit` | The endpoint indicator exceeds a target and two mesh changes are no larger than the target or indicator; deepen the cutoff or change variables |
| `evaluation_limit` | The callback budget is exhausted; default 200,000 total invocations |
| `refinement_limit` | The complete-mesh budget is exhausted; default 12 meshes |
| `nonfinite_integrand` | A callback component is NaN or infinite |
| `arithmetic_range` | A required distance, weight, mesh step, interval width or accumulated value cannot be represented |

Zero budgets are allowed and return a failure status for a nonempty interval.
An unavailable estimate has zero value/absolute integral, infinite error and
endpoint indicator, and `levels=0`. Check the status and level count; those
zeros are not a computed integral.

The roundoff floor is `8*(epsilon*absolute_integral + minimum*nonzero_samples)`,
where `minimum` is the smallest positive value of `R` and `nonzero_samples`
counts nonzero callback components on that mesh. The absolute allowance prevents
underflow from turning the estimated error into zero for a subnormal integral.
An identically zero component retains a zero floor. This allowance is still a
diagnostic estimate, not a bound on the callback's own error.

## Meshes and endpoint truncation

The method uses the double-exponential transformation of
[Takahasi and Mori, *Publ. RIMS* **9**, 721–741 (1974)](https://doi.org/10.2977/PRIMS/1195192451).
For width `w=hi-lo` and positive transformed abscissa `t`, define
`z=exp(-pi*sinh(t))`. The paired endpoint distance is `w*z/(1+z)` and the
Jacobian is `w*pi*cosh(t)*z/(1+z)^2`. The central sample and successive paired
samples use step sizes 1, 1/2, 1/4, … with compensated component sums. The step
is included in each weight before accumulation.

`endpoint_power` (default 3, minimum 2) truncates sampling when
`pi*sinh(t)>-endpoint_power*log(epsilon)+4`. This corresponds roughly to an
endpoint distance proportional to `width*epsilon^endpoint_power`. The last
sample on each mesh lies somewhat farther from the endpoint. A stronger
algebraic singularity can require a larger power: a regression integrating
`x^(-3/4)` on `[0,1]` returns `endpoint_limit` at power 2 and reaches the exact
integral 4 at power 6. Increasing the mesh budget alone does not resolve an
insufficient endpoint cutoff.

**These are error estimates, not certified bounds.** The endpoint indicator is
a diagnostic sample, not a theorem about an arbitrary omitted tail. Mesh
agreement cannot prove that a narrow peak was sampled, nor account for error
inside the callback. Split known interior singularities and sharp features;
check cutoff sensitivity for unfamiliar endpoint behavior. Infinite-domain
tails, physical-kernel errors and outer discretization remain caller-owned.
Rescale intervals or integrands when intermediate range limits are reached.

## Validation

The shared numerical precision registry covers configured real binary32,
binary64, binary80, binary128 and MPFR cases. Complex-valued integration is
outside this interface; independent real and imaginary components can be
returned in the same array. Tests use exact polynomial/logarithmic moments,
translated and reflected inverse-square-root singularities, cancellation and
zero integrals. A 512-bit analytic `log(2)` oracle measures accuracy growth
across scalar families; additional MPFR tests run through 1,024 bits against a
1,536-bit reference and on interval scales `1e-7000` and `1e7000`.

Focused core tests cover endpoint-cutoff sensitivity, rounded-coordinate
failure, invalid/mixed/exact/unset precision, budget exhaustion, callback
exceptions and preservation of the last complete mesh. See
[numerical testing](../development/numerical_testing.md) for coverage-report
semantics and configuration requirements.
