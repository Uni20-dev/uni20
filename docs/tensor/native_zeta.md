# Native real Riemann zeta

`math::zeta(x)` evaluates real Riemann zeta at every configured native
precision, including binary128. Native execution uses Uni20 algorithms without
Boost, MPFR or mutable shared state. The former standard providers are replaced
after the accuracy and performance comparison below. MPFR semantics are unchanged.

## Domain and acceptance

Use the real continuation, with a pole at one, exact zeros at negative even
integers, zeta(0)=-1/2 and limit one at positive infinity. Match the verified
MPFR conventions: NaN and negative infinity give NaN, the pole gives positive
infinity, and exact negative even integers give positive zero. No complex
function or additional precision argument is added.

For f=zeta(x) at the exact represented input, let eps be spacing at one.
The empirical target is `32*eps*abs(f)` for nonnegative arguments.
For negative nonzero results, use
`32*eps*max(1,abs(log(A)))*abs(f) + denorm_min/2`, where
`A=2*(2*pi)^(x-1)*Gamma(1-x)*zeta(1-x)` is the positive reflection amplitude.
Its logarithmic factor accounts explicitly for native log-domain evaluation,
not uncertainty in the input. It is not a universal ULP or correct-rounding
guarantee. Exact input reduction should preserve relative accuracy close to
trivial zeros; zero values themselves require exact classification instead.

| Region / all four native formats | Measure | Intended criterion | Status |
| --- | --- | --- | --- |
| Nonnegative finite x, excluding one | Original-input absolute error | <=32*eps*abs(f) | Permanent tests and scans pass |
| Neighbors of the pole | Relative error at the represented input | <=32*eps; preserve sided sign | Permanent tests and scans pass |
| Negative ordinary x and neighbors of trivial zeros | Reflection-scaled relative error plus final subnormal allowance | Bound above; preserve nonzero sign unless underflow | Permanent tests and scans pass |
| Negative even integers | Exact value | Zero, without evaluating an overflowing amplitude | Permanent tests and scans pass |
| Negative overflow neighborhoods | Classification and finite accuracy separately | No premature intermediate overflow | Permanent tests and scans pass |
| Large positive x | Distance from limiting value | Return one only once the omitted tail is below half a spacing | Permanent tests and scans pass |
| Independent integer values and precision ladder | Bernoulli/pi identities | Document accuracy improvements with precision | Permanent tests pass |

Worst cases use MPFR512 references repeated at 1024 bits. Scans must
include adjacent native values around negative even integers, the pole and
range boundaries, plus ordinary grids and deterministic exponent samples.
Do not ask MPFR to evaluate enormous negative noninteger arguments when a
separate analytical range argument establishes native overflow.

## Algorithms and error control

For x>=1/2, use an Euler–Maclaurin tail after a finite Dirichlet sum. Handle
the pole explicitly and sum with compensation. For x>=p+1, p significand
bits, the integral bound on the omitted Dirichlet tail is below half a spacing
at one, so the rounded limit is one.

With terms through n=N-1 and M Bernoulli corrections, the absolute integral
remainder for positive s is at most
`abs(B_(2M)/(2M)!)*(s)_(2M-1)*N^(1-s-2M)`. On each unit interval [j,j+1],
bound the rising factorial at j+1 and the decreasing power at j. The generator
`scripts/generate-zeta-coefficients.py` uses exact rational arithmetic for
these bounds and prints 128-bit binary coefficient prefixes.

| Format | N | M | Absolute remainder / eps, upper bound |
| --- | ---: | ---: | ---: |
| float32 | 8 | 4 | 0.016667 |
| float64 | 16 | 8 | 0.001732 |
| binary80 | 16 | 10 | 0.004037 |
| binary128 | 32 | 16 | 0.000054 |

The bounds cover 0<s<=p+1 and control truncation only. Native coefficient,
power, summation and reflection errors need separate numerical validation.

For x<0, use reflection, preserving sin(pi*x/2) through the existing pi-scaled
reduction. Detect exact even integers before Gamma or exponential evaluation.
The reflection amplitude must be evaluated in scaled/log form, with its sine
factor included before deciding final range. Accumulate the Stirling logarithm
using product residuals and compensated addition. Preserve the accumulated
low part in the final exponential scaling; rounding the whole logarithm first
can move a finite binary128 result past overflow.
Keep the sine factor multiplicative: taking its logarithm loses relative
accuracy close to trivial zeros. Below the Stirling threshold, the finite
product `Gamma(t)*(2*pi)^(-t)*(t/pi)*zeta(1+t)`, t=-x, also avoids cancellation
between large logarithms and rounding the Gamma argument from t to 1+t.

For abs(x)<1/2, compute `delta*zeta(1+delta)`, delta=-x, directly from the
Euler–Maclaurin formula. The singular tail becomes `N^(-delta)` and the
remaining terms are multiplied by delta. Combine it with
`sinpi(x/2)/delta` in reflection. This avoids the severe near-zero cancellation
of direct Euler–Maclaurin evaluation, and rounding 1-x to one cannot introduce
a spurious pole. For abs(x)<eps/2 the first-order expansion
`-1/2-x*log(2*pi)/2` suffices; its omitted term is O(eps^2).

## Observed accuracy

The integrated native header was scanned against MPFR512 at the original
represented inputs. The scan includes a rational grid from about -443 to 443,
64 neighbors of joins and selected negative even integers through -4950,
deterministic exponent/significand samples and native exponent strides.
The native implementation remained independent of MPFR in this build.

| Format | Inputs | Largest fraction of acceptance bound | Worst input |
| --- | ---: | ---: | --- |
| float32 | 57143 | 0.673338 | -16.4594593048095703125 |
| float64 | 56679 | 0.864861 | -16.891891891891891219756871578283607959747314453125 |
| binary80 | 58889 | 0.640751 | -16.324324324324324324980706180099332414101809263229 |
| binary128 | 58821 | 0.341791 | 0.5000000000000000000000000000000012518544638517033 |

No accuracy or range-classification failures occurred. Worst references
agreed at 1024 bits. These observations do not prove a domain-wide bound.

A separate 10640-input scan per format concentrates deterministic random
mantissas in [-18,-14] and [1/2,1], plus join and pole neighbors. Its largest
bound fractions were .732773/.912335/.705631/.341791, with no failures and
1024-bit agreement. The worst inputs for the first three formats were
-16.84014129638671875, -16.784456336552974420328609994612634181976318359375
and -16.380167256238744729127110311139858822571113705635. Binary128's worst
input was the same half-integer neighbor as in the table.

The local prototype's 1513-input near-zero/pole scan per format passed the
target at MPFR512, with worst references agreeing at 1024 bits. Largest
fractions of the bound were .129/.150/.238/.342 for float32/64/80/128.
These are observed sample maxima, not global guarantees.

Independently bisected overflow boundaries in the first several overflowing
negative lobes were checked at 1536 neighboring native inputs per format.
The compensated logarithm and final scaling passed classification checks for
all four formats. A regression must retain the binary128 argument
`-2312.9484675583841235265933212349314204917536313162`, where rounding the
logarithm before exponentiation gave infinity for a finite result.

A separate 384-input scan of neighbors of -8, -16 and -32 passed after
preserving the multiplicative sine and finite Gamma product. Largest bound
fractions were .484/.671/.562/.110; worst references agree at 1024 bits.

The targeted MPFR-enabled run passed 77 tests; the MPFR-disabled run passed
12. The missing-standard-provider executable compiled and ran in both modes,
and the explanatory example ran without MPFR. These checks establish local
behavior and synthetic fallback wiring, not second-platform validation.

Permanent tests exercise all configured native types and MPFR128/256. They
include independently bisected native overflow boundaries, rational Bernoulli
values, pi identities and accuracy gains at zeta(2).

Full unfiltered integration validation also passed, with examples, CLI and
Python bindings enabled:

| Configuration | Passing CTest cases | Numerical tests | Runtime skips |
| --- | ---: | ---: | --- |
| GCC13 Debug, MPFR/MPC/MPLAPACK, native binary80/128 | 3336 | 623 | 1 NUMA topology test |
| GCC13 Release, MPFR disabled, native binary80/128 | 2523 | 262 | 1 NUMA topology test |
| Clang19 Debug, MPFR/MPC/MPLAPACK, native binary80, binary128 disabled | 2987 | 520 | 1 NUMA topology test |

The complete numerical coverage reports passed in all three configurations.
The NUMA skips reflect this host's topology and are separate from the passing
counts. Clang binary128, macOS and CUDA runtime were not exercised.

## Provider comparison

On GCC13/glibc2.39, a 7684-input scan per native standard format over [-35,100],
with pole/zero neighbors, exposed 2287/2293/2348 accuracy failures and
124/88/88 classification failures for float32/64/80 against the stated target.
The float provider returns infinity at -35 although zeta(-35) is about
3.80879311252454e11. Double zeta(-34) is about 1.00567e-4 instead of exact zero.
Inspection of GCC13's `tr1/riemann_zeta.tcc` explains these failures: reflection
forms sin(pi*x/2) without exact argument reduction and separately exponentiates
log-Gamma, which can overflow before the full result does. The ordinary-domain
summation also incurs cancellation. These are observations of the tested
provider, not claims about every standard-library version.

Representative 10000-call loops measured native costs of .003--.168 us for
float32, .003--.203 us for float64, .009--3.73 us for binary80 and .048--55.9 us
for binary128. The retained comparison provider took roughly 3.8--6.7,
39--44 and 478--647 us for the first three formats at -1.1, .25 and 10;
its large-positive shortcut is much faster. At -100, the new binary80 path was
about five times slower. Measurements during an unrelated build give orientation,
not timing guarantees. Accuracy, range, reentrancy and the ordinary-region
cost justify selecting Uni20 evaluation for all native formats. Native Beta's
separate retained-provider limitation is unchanged.

Sources: [DLMF reflection formulas](https://dlmf.nist.gov/25.4),
[Euler–Maclaurin representations](https://dlmf.nist.gov/25.11#iii), and
[integer values](https://dlmf.nist.gov/25.6). These are mathematical identities;
no implementation or coefficient table is copied from another library.
