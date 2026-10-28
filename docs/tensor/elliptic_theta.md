# Scaled rectangular theta constants

`<uni20/core/elliptic_theta.hpp>` supplies
`uni20::elliptic_theta_constant(kind, t)` for the positive real constants
$\theta_j(0\mid it)$, $j=2,3,4$, with finite $t>0$. The argument is the
imaginary lattice parameter, **not** the nome $q=\exp(-\pi t)$.

The result owns two real scalars, `scaled_value` and `log_scale`, representing

$$
\theta_j(0\mid it)=\mathrm{scaled\_value}\,\exp(\mathrm{log\_scale}).
$$

```cpp
#include <uni20/core/elliptic_theta.hpp>

auto constant = uni20::elliptic_theta_constant(4, 0.001);
// The constant itself underflows binary64, but both returned fields are finite.
auto log_constant = uni20::math::log(constant.scaled_value) + constant.log_scale;

// With UNI20_ENABLE_MPREAL=ON, no MPLAPACK kernels are needed:
auto p = uni20::Precision::bits(256);
auto wide = uni20::elliptic_theta_constant(4, uni20::mpreal("0.001", p));
```

## Precision and domain

Native float32, float64, native `long double` and configured float128 retain
their scalar type and use native arithmetic without an MPFR dependency.
MPFR inputs must have finite explicit precision. The algorithm uses 32 guard
bits internally and rounds each returned field to the input precision.
`elliptic_theta_constant(kind, t, p)` explicitly converts the input first;
native types require their native precision, whereas exact MPFR inputs may be
rounded to a finite `p`. No ambient precision or exponent limits are changed.

Invalid kind, nonpositive/nonfinite input, exact precision without a finite
override and a zero term budget are rejected. An unset MPFR value retains its
usual logic-error diagnostic. A logarithmic scale outside the scalar range is
an overflow error. The optional final `max_terms` argument, default 4096,
bounds the inspected nonconstant terms, including the first omitted term used
to test convergence. Exhausting it raises a runtime error instead of returning
an unfinished approximation.

This interface covers zero argument and rectangular lattices only. It does not
provide complex arguments, derivatives, complex lattice parameters or theta1.
It is a prerequisite for elliptic excitation bands; physical prefactors,
anisotropy conversions and branch selection remain in the application.

## Evaluation and accuracy

For $T\geq1$ the implementation sums the
[Fourier series](https://dlmf.nist.gov/20.2.i):

$$
\theta_2(0\mid iT)=2e^{-\pi T/4}\sum_{n=0}^{\infty}e^{-\pi Tn(n+1)},
$$

$$
\theta_3(0\mid iT)=1+2\sum_{n=1}^{\infty}e^{-\pi Tn^2},\qquad
\theta_4(0\mid iT)=1+2\sum_{n=1}^{\infty}(-1)^n e^{-\pi Tn^2}.
$$

For $t<1$, the
[modular transformation](https://dlmf.nist.gov/20.7.viii) maps to $T=1/t$:
theta2 and theta4 exchange roles, theta3 retains its kind, and all acquire a
factor $t^{-1/2}$. The exponential prefactor and this square-root factor are
stored in `log_scale`. The remaining series is well conditioned and accumulated
with compensation. In particular theta4 near $t=0$ is never obtained by
subtracting almost equal numbers.

Successive absolute terms in the reduced series have ratio at most
$e^{-3\pi}<1/2$. If the first omitted term is $a$, the remaining absolute tail
is less than $2a$. The stopping test uses $a\leq\epsilon/16$ at the working
precision. This is an analytic truncation bound in exact arithmetic, **not** an
interval certificate for floating-point evaluation. Native elementary-function
rounding, scale rounding and final reconstruction are separate errors.

The numerical acceptance test compares reconstructed values at a wider
precision with independent references, using relative error bounded by
$8\epsilon(1+|\mathrm{log\_scale}|)$. This states the tested error scale;
correct rounding and an all-input certified bound are not claimed. A large
logarithmic scale can amplify its absolute rounding error when exponentiated,
so extra precision may still be necessary for an accurately reconstructed
tiny gap. Differences between nearly equal theta constants can require more
precision too; the representation does not retain a correction below the
spacing of `scaled_value`.

The shared numerical cases cover each configured native family and MPFR at
128/256 bits. They use independent
[Jacobi products](https://dlmf.nist.gov/20.4.i), the theta fourth-power identity,
modular identities, and the Gamma-function value of theta3 at $t=1$.
Precision-growth tests distinguish every family from its preceding precision.
Core tests cover invalid inputs, budget failure, very low MPFR precisions,
extreme native ranges and MPFR inputs of order $10^{\pm7000}$.

## Follow-up scope

Nonzero-argument theta values and derivatives need their own scaled complex
representation, reduction limits and cancellation diagnostics. That extension
can share the conventions here, but it is not implied by constant support.
XXZ bulk-energy quadrature is a separate prerequisite for making the entire
Bethe XXZ dispersion frontend precision-generic.
