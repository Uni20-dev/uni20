# Scaled rectangular theta functions

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

The constant interface covers zero argument and rectangular lattices only.
Complex arguments and derivatives use the separate jet interface below; neither
interface accepts a general complex lattice parameter.
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

## Complex arguments and derivatives

`<uni20/core/elliptic_theta_jet.hpp>` supplies
`elliptic_theta_jet(kind, u, t)` for $f_j(u)=\theta_j(\pi u\mid it)$,
$j=1,2,3,4$. The complex argument is **u**, not $\pi u$. The first two
derivatives are with respect to **u**, so they include factors of $\pi$ and
$\pi^2$ relative to derivatives with respect to the conventional theta argument.

The owning `theta_jet_result<C>` contains `derivative[0..2]` and a real
`log_scale`, representing $f_j^{(r)}(u)=e^{\mathrm{log\_scale}}
\mathrm{derivative}[r]$. All three components have the same scale. The scale is
an evaluation device, not a function to differentiate when using these fields.

```cpp
#include <uni20/core/elliptic_theta_jet.hpp>

auto jet = uni20::elliptic_theta_jet(1, uni20::complex<double>(0.125, 0.0625), 0.5);
auto logarithmic_derivative = uni20::elliptic_theta_ratio(jet, 1, jet, 0);

// Complex MPFR/MPC uses the same explicit precision as the real lattice parameter.
auto p = uni20::Precision::bits(256);
using R = uni20::mpreal;
auto wide = uni20::elliptic_theta_jet(1, uni20::complex<R>(R("1/8", p), R("1/16", p)), R("1/2", p));
```

Inputs must be finite, with positive real `t` and matching finite precision.
The explicit overload `elliptic_theta_jet(kind, u, t, p)` rounds both inputs to
`p`, accepting exact MPC/MPFR values. Native types require their own precision.
Native arithmetic stays native; MPC evaluation uses 32 guard bits and rounds
each result field to `p`, without modifying ambient precision or exponent limits.

The evaluator reduces the argument to the fundamental rectangle, sums a
compensated Fourier series for `t>=1` or a Gaussian series for `t<1`, and restores
quasi-periods with differentiated phase factors. Gaussian derivatives use their
centered arguments to avoid cancellation of the curvature at small `t`.
Theta2 and theta3 use integer-centered Gaussian sums directly, retaining small
arguments that a real half-period translation would erase. Even functions use a
local Taylor expansion near zero to retain a tiny first derivative. Exact symmetry
zeros are restored explicitly. Reduction is limited by both the scalar's
significand and a signed integer period count; an unresolvable reduction raises
an overflow error. A scale or scaled derivative outside the scalar's range also
raises an overflow error. At extreme Fourier parameters, decay and growth are
combined as nonnegative factors before multiplication, avoiding spurious
`infinity - infinity` when the scaled jet is finite. Period contributions and the
reduced logarithmic scale are combined with fused multiply-add, and period
phases are reduced in cycles before multiplication by pi; exact quarter cycles
retain their exact phases.

Each reduced series uses a precision-dependent term count with a conservative
Gaussian-tail margin. The final `max_terms` argument (default 4096) bounds this
count; zero is invalid, and an insufficient budget raises a runtime error. This
is a truncation strategy, not a certified floating-point error bound. Cancellation
near zeros, elementary-function rounding, and reconstruction from a rounded
logarithmic scale can require additional precision. Correct rounding and uniform
relative accuracy of vanishing derivatives are not claimed.

`elliptic_theta_ratio(a, da, b, db)` forms a ratio of selected derivatives
without reconstructing the two functions. Orders must be 0 through 2, and the
selected components and scales must be finite with common precision. A zero
denominator raises a domain error; a zero numerator gives a precision-correct
zero. An unrepresentable nonzero ratio raises an overflow error. Small
denominators and differences of rounded scales still amplify errors. Finite
complex components may have an overflowing or subnormal norm; component-scaled
logarithmic norms and unit phases keep representable ratios evaluable.

The shared complex precision cases check conjugation, real and imaginary
quasi-periods, derivative transformations, symmetry zeros and scaled ratios.
Independent Jacobi products differentiated through logarithmic derivatives
provide 1280-bit references, including the original inputs' low bits. Error
comparisons distinguish each configured scalar family from the preceding
precision. Core tests add 80/160/256/333-bit convergence, tiny arguments,
finite ratios of underflowing functions, domain and term-budget failures.
Physical prefactors, anisotropy conversion and model branches remain in Bethe.
