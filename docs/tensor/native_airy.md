# Native real Airy Ai investigation

This is the numerical design and acceptance checkpoint for native
`math::airy_ai(x)`. No native overload is integrated yet. The existing MPFR
scalar overload and its explicit-precision rules remain unchanged. Native
execution must use the argument's configured float32/64/80/128 arithmetic,
without Boost, MPFR evaluation, a new public numerical framework or a new
runtime dependency.

## Domain and acceptance before prototyping

Ai is real and finite for every finite real argument. Both signed zeros have
the same positive value Ai(0). NaN gives NaN; both infinite arguments give
positive zero. These exceptional conventions match the existing MPFR API and
were checked directly with MPFR 4.2.1.

Evaluate the reference at the exact represented input. Let eps be spacing at
one, d the smallest positive subnormal, and F=Ai(x). For x<=0 define the
positive scale `A(x) = (1+abs(x))^(-1/4)/sqrt(pi)`. This remains well-defined
at zero and follows the oscillation amplitude for large negative x.
For positive x define `zeta = 2*x^(3/2)/3` in the reference precision.

| Domain / all four native formats | Measure | Intended target | Guarantee level | Evidence |
| --- | --- | --- | --- | --- |
| Finite x<=0, including zeros and joins | abs(result-F)/A(x) | <=64*eps | Empirical accuracy target | Unmeasured |
| Finite x>0, ordinary and asymptotic regions | Absolute error | <=64*eps*max(1,zeta)*F + d/2 | Empirical target, exponent construction included | Unmeasured |
| Huge negative arguments | Same amplitude-scaled absolute measure | <=64*eps; bounded work, no overflowed phase | Empirical target and range requirement | Phase reduction investigation pending |
| Positive subnormal/zero boundary | Finite bound and classification separately | No premature underflow from intermediate exponentials | Empirical target and range requirement | Unmeasured |
| x=0 and exceptional arguments | Analytic value, classification and sign | Conventions above | API contract | MPFR domain probe only |
| Increased precision | Reference errors at selected nonzero values | Improvement beyond lower-precision results | Numerical evidence | Unmeasured |

A relative target is inappropriate near negative zeros. The negative-axis
absolute target deliberately does not grow with the oscillatory phase:
rounding x*sqrt(-x) before reduction is not an adequate implementation at
large x. An inaccurate phase is not excused as uncertainty in the input.
Correct rounding for every argument is not the objective. Record observed
maxima separately; a failed experiment does not relax this table.

## Candidate regions

The central expansion follows the two Maclaurin solutions of y''=x*y,
combined with Ai(0) and Ai'(0); see [DLMF 9.4](https://dlmf.nist.gov/9.4).
For positive x their subtraction can lose many bits. Investigate a bounded
native expansion for coefficients and accumulation, with enough guard words
for the chosen join. Native series evaluation with already rounded Ai(0)
and Ai'(0) is not sufficient. Constants need reproducible high-precision
prefixes and an independent check; do not copy a provider coefficient table.

Away from the central interval, use the positive decaying or negative
oscillatory asymptotic expansions and their real-variable remainder bounds
in [DLMF 9.7](https://dlmf.nist.gov/9.7). Generate coefficient ratios from
exact integers. Choose region joins and finite iteration caps from truncation
and cancellation estimates before accepting timing or sample accuracy.
For positive x, combine the decaying exponent and algebraic prefactor before
final rounding, retaining extra words near the underflow boundary if needed.

The negative asymptotic path needs accurate reduction of
`2*(-x)^(3/2)/3 + pi/4`. Extra native words suffice only while the required
phase bits fit their capacity. Even a correctly rounded native sqrt loses
essential fractional phase for sufficiently large x. A domain-wide solution
may need a narrowly scoped, bounded integer/limb reduction of the original
binary significand and exponent, with a sufficiently long pi prefix. This is
an implementation question to investigate, not an approved public big-number
API. Do not insert an arbitrary zero or use the fractional part of an already
inaccurate native phase when this path fails.

## First bounded experiments

1. Reproduce loss of phase with native sqrt/multiply at negative arguments
   whose exponents cross each format's significand size. Include adjacent
   representable inputs and arguments for which naive x*sqrt(-x) overflows.
2. Establish the amount of guard precision and reduction data required for
   the largest native exponent, and prototype that hard phase boundary before
   polishing the scalar facade.
3. Bound central-series cancellation at candidate joins, then compare native
   expansions against independent MPFR references through zeros and joins.
4. Test asymptotic truncation bounds and positive underflow independently of
   the central series. Measure costs for ordinary and hard inputs.

MPFR's own manual notes that `mpfr_ai` is intended for arguments of magnitude
roughly below 500; it is not an oracle for arbitrary huge inputs. Use it in
its practical range, repeat difficult references at higher precision, and
cross-check constants and identities. Outside that range, build an independent
high-precision asymptotic reference with explicit remainder control and enough
bits to resolve the phase before reduction. The oracle's precision may depend
on the input exponent even though the final requested value is native.
See the [MPFR Ai reference](https://www.mpfr.org/mpfr-current/mpfr.html#index-mpfr_005fai).

Only after the numerical investigation passes should the CPO, shared precision
registry, provider-absence tests and explanatory example be added. Full
compiler/configuration validation remains a publication gate. Airy derivatives,
Bi, complex Airy, tensor integration and CUDA are outside this batch.
