# Native exponential and logarithmic extensions

This batch adds native float32/64/80/128 evaluation of `math::exp10`,
`exp2m1`, `exp10m1`, `log2p1` and `log10p1`. MPFR keeps its existing scalar
implementations and precision contracts. Native arguments retain their type;
there is no MPFR runtime fallback and no additional dependency.

The separately implemented [native compound](native_compound.md) function
retains the original increment when computing `(1+x)^n`; its guard-word
algorithm and acceptance criteria are independent of these unary functions.

Run `native_exponential_math_example` to compare stable small changes with
cancellation-prone compositions and to see native extended exponent ranges.

## Value and provider contracts

The exponential functions accept all real arguments. `exp10(0)` is one;
`exp2m1` and `exp10m1` preserve signed zero. At negative infinity they return
zero, -1 and -1 respectively; positive infinity maps to positive infinity.
The logarithmic functions accept x >= -1, return negative infinity at -1,
and preserve signed zero. Inputs below -1 yield NaN; positive infinity maps
to positive infinity. All five functions propagate NaN.

CMake probes each function and scalar type independently, requiring its exact
signature and linked symbol. The capability macros are
`UNI20_HAS_<EXP10|EXP2M1|EXP10M1|LOG2P1|LOG10P1>_<FLOAT|DOUBLE|LONG_DOUBLE|FLOAT128>`.
A missing routine uses the portable algorithm described below. A detected
routine uses its libc provider, with the decimal range scaling below. Native trailing `Precision` arguments and
conversion-only/integer arguments remain unsupported. There is no portable
`errno` or floating-point exception-flag promise.

## Portable algorithms

`exp10(x)` uses typed `pow(10,x)`, avoiding an intermediate rounded `x*log(10)`.
Near the ends of the native exponent range it evaluates `pow(10,x/2)` and
squares the result. This avoids premature range loss observed in a binary128
power provider near the largest finite result. Halving is exact in this region;
the intermediate square root of the result remains far from overflow/underflow.
The same range scaling surrounds a detected libc `exp10`: direct binary128
`exp10` returned a minimum subnormal where the wider reference rounded to zero
in the boundary-neighbor tests. Its reduced-argument evaluation is still supplied
by libc. The branch uses x outside `[min_exponent/4,max_exponent/4]`; it is not an
estimate of an exact decimal overflow boundary.

For |x| <= 1/2, `exp2m1` and `exp10m1` use native `expm1` with a typed ln(2)
or ln(10) factor. Outside that region, subtracting one from the corresponding
base exponential is sufficiently separated from cancellation. The decimal
portable path uses the portable `exp10`, independently of libc selection.

For |x| <= 1/4, logarithmic extensions use `log1p(x)` times a typed inverse
logarithm. Elsewhere, an error-free TwoSum represents `1+x = sum + low`.
Evaluate `log2(sum)` or `log10(sum)`, with a fused correction
`(low/sum)/ln(base)`. The omitted second-order term is O(epsilon^2) in this
region, where the output is bounded away from zero. This avoids rounding away
x near zero and reduces the effect of less accurate native `log1p` routines
at moderate arguments. Domain endpoints/nonfinite inputs are handled before
TwoSum, so the correction never subtracts infinities.

All constants come from typed `std::numbers` values; none widen a double
literal. Arithmetic uses ordinary nearest rounding without fast-math
reassociation. The TwoSum technique is described by Ogita, Rump and Oishi,
[Accurate Sum and Dot Product, SIAM SISC 26(6), 2005](https://www.tuhh.de/ti3/paper/rump/OgRuOi05.pdf).
Provider interfaces are documented in the
[GNU libc exponentials/logarithms reference](https://sourceware.org/glibc/manual/latest/html_node/Exponents-and-Logarithms.html).
The formulas here do not establish correct rounding for the composed functions.

## Numerical acceptance

These are the intended criteria, established before integrating the algorithms.
They apply under nearest rounding with gradual underflow, separately to each
configured scalar and the public/portable execution paths. The reference is
computed at the exact represented input.

| Region | Measure | Intended criterion | Guarantee level |
| --- | --- | --- | --- |
| Ordinary finite values and either side of algorithm joins | Absolute error in reference-output grid units | At most four units | Empirical target, not a proven global bound |
| Tiny inputs and subnormal outputs | Absolute error with grid spacing clamped to the minimum subnormal spacing | At most four units | Empirical target; no relative-error claim at zero |
| Exponential overflow/underflow neighborhoods | Value classification against rounded wider reference; finite neighbors also checked in grid units | Matching classification/sign, and at most four units for finite values | Tested boundary contract |
| Signed zeros, unit identities, infinities, NaNs and log-domain endpoints | Value/sign or classification | Exact | Value contract |
| Selected independent nonzero reference values | Error compared with the preceding precision | At least sixteen-fold accuracy improvement | Evidence on controlled examples |

A reference grid unit is `2^(e-p)`, where the nonzero reference has frexp
exponent e and the tested type has p significand bits. Clamp e to the minimum
normal exponent for native subnormal results. This definition stays finite at
the largest representable result. Record observed maxima and their inputs
separately; a test threshold is not an observed maximum.

Initial prototype evidence: GCC 13/glibc 2.39, 512-bit MPFR oracle, approximately
100,000 deterministic samples per native format (grid, joins, range boundaries,
fixed-seed significands and exponents). The maximum errors in grid units were:

| Type | Portable exp10 | exp2m1 | exp10m1 | log2p1 | log10p1 | Raw libc exp10 |
| --- | --- | --- | --- | --- | --- | --- |
| float32 | 1.731 | 1.998 | 2.344 | 1.497 | 2.189 | 0.502 |
| float64 | 0.934 | 2.006 | 3.168 | 1.373 | 2.000 | 0.507 |
| float80 | 0.959 | 3.101 | 2.798 | 1.743 | 1.679 | 1.357 |
| float128 | 1.751 | 2.712 | 2.285 | 1.977 | 1.890 | 1.796 |

These rounded-up diagnostic maxima describe the prototype scan, not a
certification of all inputs or the integrated library. The ordinary unscaled
binary128 power path overflowed prematurely at approximately
4932.075448958667902381898051166093642906902. Direct long-double
`log1p(x)/ln(10)` exceeded the target near
0.3229116324144490426137181493526639997071. Preserve these failures as
regressions. Integrated GCC 13 Debug validation passed all 93 core tests and 509 numerical
precision tests, including the complete coverage-report check. The integrated
precision suite records the following rounded-up maxima (inputs and full
values are XML properties); these use a smaller regression sample than the
prototype scan above.

Full unfiltered builds and CTest runs, including examples, CLI and Python bindings,
passed on the local Linux host:

| Configuration | Passed | Skipped |
| --- | ---: | ---: |
| GCC 13 Debug, MPFR/MPC and binary80/128, MPLAPACK enabled | 3,135 | 1 NUMA topology test |
| GCC 13 Release, MPFR disabled, binary80/128, MPLAPACK kernels disabled | 2,401 | 1 NUMA topology test |
| Clang 19 Debug, MPFR/MPC and binary80, binary128 disabled, MPLAPACK enabled | 2,825 | 1 NUMA topology test |

The complete numerical suites and coverage reports passed in each cell.
Clang's binary128-disabled cell is not evidence for native binary128; that
coverage comes from GCC. GPU/MPI execution and macOS were not exercised.

| Scalar / path | exp10 | exp2m1 | exp10m1 | log2p1 | log10p1 |
| --- | --- | --- | --- | --- | --- |
| RealFloat32 / public | 1.093 | 1.991 | 1.996 | 1.379 | 2.036 |
| RealFloat32 / portable | 1.093 | 1.991 | 1.996 | 1.379 | 2.036 |
| RealFloat64 / public | 0.927 | 1.994 | 2.848 | 1.241 | 1.776 |
| RealFloat64 / portable | 0.927 | 1.994 | 2.848 | 1.241 | 1.776 |
| RealFloat80 / public | 1.220 | 3.043 | 2.104 | 1.716 | 1.467 |
| RealFloat80 / portable | 0.959 | 3.043 | 2.104 | 1.716 | 1.467 |
| RealFloat128 / public | 2.480 | 2.591 | 1.807 | 1.721 | 1.635 |
| RealFloat128 / portable | 1.751 | 2.591 | 1.807 | 1.721 | 1.635 |
