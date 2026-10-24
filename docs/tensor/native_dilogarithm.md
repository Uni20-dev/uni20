# Native real dilogarithm

This batch adds `math::dilog_real(x)` for configured native float32/64/80/128.
It means the real part of Li2 on the whole real line, including x>1, matching
the existing MPFR scalar contract. It does not introduce a complex branch API.
Native arguments stay native; MPFR is only a test oracle. No existing native
provider is replaced. Integer, complex and conversion-only arguments remain
unavailable, as do trailing precision arguments for these fixed-type calls.

## Acceptance established before prototyping

Signed zeros are preserved, NaN gives NaN, both infinite arguments give
negative infinity, and every finite argument gives a finite result. At x=1
the value is pi squared over six. Assume nearest rounding and gradual underflow.

Let eps be spacing at one and f the reference at the exact represented input.
Use scale=abs(f) for abs(x)<=1/2, and max(1,abs(f)) otherwise. The intended
absolute error target is `16*eps*scale + denorm_min/2`. The absolute scale near
the nonzero real root avoids a meaningless relative bound; near x=0 the
relative target preserves small inputs. This is an empirical acceptance target,
not a universal correctly-rounded guarantee. Initial reference precision is
512 bits, with difficult cases repeated at 1024 bits.

| Region / all native formats | Measure | Intended criterion | Guarantee level | Observations |
| --- | --- | --- | --- | --- |
| abs(x)<=1/2, including subnormals | Absolute error divided by the bound above | <=1 | Empirical target | Wide scan and shared precision tests pass |
| Remaining real line and large exponents | Same bound, evaluated at original represented input | <=1 | Empirical target | Wide scan below passes |
| Region joins -1, 1/2, 1, 2 and nonzero-root neighborhood | Same bound; absolute scale near the root | <=1 | Empirical target | Neighbor scans pass |
| Signed zero, NaN, infinities, finite range | Value classification and zero sign | Stated value contract | API contract | Native tests pass |
| Independent pi/log identities and precision ladder | Wider exact-series/constants reference | At least sixteen-fold accuracy gain on controlled cases | Demonstrated examples | Shared precision tests pass |

## Algorithm and reference plan

Use the power series and connection formulas from
[DLMF 25.12](https://dlmf.nist.gov/25.12). Take real parts on x>1. Reduce to
`S(t)=sum(t^k/k^2)` on 0<=t<=1/2; use compensated summation. After N terms,
the positive tail is at most `t^(N+1)/((N+1)^2*(1-t))`. A cap of digits+4
terms makes truncation negligible relative to the target at every native
precision. Stop earlier if the current term is at most eps*sum/8: the remaining
tail is smaller than that term on this reduced interval. The truncation
estimate does not bound accumulated floating-point error; the acceptance target
also allows for series evaluation, logarithms and rounded reductions.
No external coefficient table or implementation is copied.

| Input region | Reduction |
| --- | --- |
| -1<=x<0 | t=x/(x-1); f=-S(t)-log1p(-x)^2/2 |
| 0<=x<=1/2 | f=S(x) |
| 1/2<x<1 | f=pi^2/6-log(x)*log1p(-x)-S(1-x) |
| 1<x<2 | t=(x-1)/x; f=pi^2/6+S(t)+log(x)*(log(x)/2-log(x-1)) |
| x>=2 | f=pi^2/3-log(x)^2/2-S(1/x) |
| x<-1 | Apply inversion f=-Li2(1/x)-pi^2/6-log(-x)^2/2, then the negative-input reduction above |

Logarithms and constants are evaluated in the result type; FMA combines
products and offsets. Handle x=1 separately. Inversion never forms x squared
or a product of huge arguments. Finite native arguments give logarithms whose
squares fit comfortably within the result format.

## Observed accuracy and cost

The source-header scan tests 32769 rational-grid arguments, 64 neighbors on
each side of every join and the nonzero root, every power-of-two exponent,
native extrema and 4000 deterministically generated mantissas/exponents with
both signs. The nonzero root is independently located with MPFR bisection.
References are evaluated at the original represented input; the maximum-error
case in each format agrees at 512 and 1024 bits.

| Format | Inputs | Largest observed fraction of acceptance bound | Worst input |
| --- | ---: | ---: | --- |
| float32 | 42349 | 0.355067 | 15.583751678466796875 |
| float64 | 45991 | 0.253190 | 16.05215646940822438182294717989861965179443359375 |
| binary80 | 107453 | 0.285339 | 16.12036108324974924797812203536295783124 |
| binary128 | 107551 | 0.308968 | 13.94583751253761283851554663991975957455 |

No sampled finite input produced a nonfinite result. These are observed sample
maxima, not a proof over the whole native domain. The shared registry tests
also include MPFR128/256, independent pi/log values at -1, 1/2, 1 and 2, and
400-term defining series at +/-1/4, whose tail is below 2^-800. The precision
ladder at 1/4 demonstrates increased accuracy without widening double inputs.
Native-only identity and exceptional-value tests do not require MPFR.

On the local GCC13 Linux host, 10000-call loops at representative inputs near
-100, -1/2, 1/4, 1/2, 1, 3/2 and 13 measured approximately 0.011--0.031 us
for float32, 0.016--0.069 us for float64, 0.060--0.92 us for binary80, and
2.8--11.8 us for binary128. Binary128 software arithmetic dominates its cost.
Measurements were taken while an unrelated build ran; these are orientation,
not benchmark guarantees or test thresholds. There is no existing selected
native dilogarithm provider to replace. The implementation has no allocation
or mutable shared state.

The local targeted MPFR-enabled suite passes 37 tests, including existing
scalar/MPFR special-function regressions. A separate MPFR-disabled build passes
12 tests, including the shared identities at all four native precisions; the
explanatory example also runs without MPFR.

Full unfiltered integration validation also passed, with examples, CLI and
Python bindings enabled:

| Configuration | Passing CTest cases | Runtime skips |
| --- | ---: | ---: |
| GCC13 Debug, MPFR/MPC/MPLAPACK, native128 | 3294 | 1 NUMA topology test |
| GCC13 Release, MPFR disabled, native128 | 2497 | 1 NUMA topology test |
| Clang19 Debug, MPFR/MPC/MPLAPACK, native80, native128 disabled | 2953 | 1 NUMA topology test |

Each configuration also passed the complete numerical precision executable and
coverage report. CUDA runtime and macOS were not exercised. The NUMA skips
reflect this host's topology and are separate from the passing counts.

The separate [native Ei implementation](native_expint.md) supplies the
exponential integral, including binary128.
