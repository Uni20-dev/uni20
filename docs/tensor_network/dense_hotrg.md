# Dense HOTRG example

The example in [`examples/models/`](../../examples/models/) implements real,
square-lattice HOTRG with a one-site unit cell. One round contracts in the y
direction and then the x direction, retaining at most `D` states on each fused
leg. The Gram matrix, truncated SVD, contraction geometry, and transfer-matrix
diagnostics are in `dense_hotrg.hpp`.

## Scales and lattice geometry

`vertex(model, q, K)` returns `scaled_tensor<R>`, representing
`exp(log_scale) * tensor`. Here `K = J/T > 0`. The ferromagnetic bond weights
have `exp(K)` removed before their Fourier transform, so they use decaying
exponentials. The vertex carries `log_scale = 2*K`. This avoids constructing
large Boltzmann factors even at low temperature. The final physical logarithm
must still fit the selected scalar's exponent range.

`flow(unit, D, rounds)` normalizes the input before forming the first Gram
matrix. Subsequent steps normalize their coarse tensors and retain the removed
scales as `log_norm_y` and `log_norm_x`. For round `k`, starting at 1, the
per-site log partition function is

```text
sum = initial_log_scale + sum_j<=k 4^(-j) * (2*log_norm_y[j] + log_norm_x[j])
lnz[k] = sum + 4^(-k) * log(torus_trace(T[k]))
```

The input normalization is included in `initial_log_scale`; `unit_lnz` is the
one-site periodic result. Input and coarse periodic traces must be positive.
`T0` holds the latest normalized tensor, and an optional observer receives the
record and that tensor after each round.

`log2_width[k]` stores the exact integer exponent of the represented square's
width. No integer or floating-point lattice width, volume, or product of
normalizations is constructed. There is no 62-round limit. The inverse volume
is updated by division by four in `R`; contributions eventually round away or
underflow according to that scalar's precision and exponent range. This does
not terminate the tensor flow. It also does not establish convergence: the
bond dimension and truncation errors still matter.

`scaling_dimensions(eigenvalues, log2_columns, log2_rows)` uses only the
difference of the geometry exponents. Equal exponents cancel even for very
large represented squares. Near-degenerate eigenvalues use `log1p` to preserve
their small gap. Only positive eigenvalues larger than 64 working-precision
epsilons times the leading eigenvalue are reported.

## Selecting arithmetic

All model parameters, tensor elements, decomposition results, normalization
logs, and physical diagnostics use the selected real type. Wall-clock timings
use `double`. Runtime precision is carried through tensor allocations and
passed explicitly to SVD and eigensystem operations. No process-global MPFR
precision setting is needed.

| CLI selection | Working precision | Required decomposition provider |
| --- | --- | --- |
| `fp32` | native float | native LAPACK |
| `fp64` (default) | native double | native LAPACK |
| `fp80` | configured binary80 scalar | MPLAPACK binary80 |
| `fp128` | configured binary128 scalar | MPLAPACK binary128 |
| `mpreal --bits=N` | N binary digits (default 128) | MPLAPACK MPFR |

Unavailable choices are rejected. `--bits` is valid only with `mpreal`.
`--gram-cut=svd|eigh|selected-eigh|jacobi-eigh|jacobi-svd` chooses how a step
takes the leading vectors of the gram. `svd` and `eigh` work at every precision;
`selected-eigh` needs the selected eigensolver, which covers `fp32`, `fp64` and
`fp128`, so `fp80` and `mpreal` reject it. The two Jacobi methods use cuSOLVER
and need `--execution=cuda`. `--benchmark-gram-cut` compares the methods (see
the [model examples](../../examples/models/)).
`--execution=cuda` runs the same step on CUDA tensors for `fp32` and `fp64`,
in a build with the CUDA, cuBLAS and cuSOLVER backends; the wider types are host
only. See the [model examples](../../examples/models/) for what the
device step does where a CUDA tensor has no counterpart of a host idiom.
Temperature is supplied as decimal `T/Tc` and parsed directly at the selected
precision. The critical coupling `log(1 + sqrt(2))/2` is evaluated at that same
precision; `--temperature=1` denotes criticality without a stored double `Tc`.

```bash
./build/examples/ising_hotrg_example --precision=fp128 --bond=8 --rounds=8
./build/examples/ising_hotrg_example --precision=mpreal --bits=256 --bond=4 --rounds=2
./build/examples/ising_hotrg_example --temperature=0.0001 --bond=4 --rounds=2
./build/examples/ising_hotrg_example --bond=1 --rounds=128
```

The contraction layouts lower to GEMM for every configured scalar. The
contraction API carries the operation precision into direct and looped GEMM
backends, so MPFR uses MPLAPACK for its contractions as well as its SVD and
eigensystems. Tests require these provider selections; reference-loop execution
cannot count as MPFR GEMM validation.

The Onsager diagnostic uses bounded exponentials and midpoint quadrature in
the selected type. `--onsager-points=N` controls its discretization independently
of arithmetic precision. Increasing `--bits` alone does not make the default
400000-point quadrature a high-precision reference. HOTRG bond truncation,
quadrature discretization, and arithmetic rounding are separate error sources.

## Presentation

The executable routes its setup, checks, timing summary, help and errors through
the presentation/display layer. Flow rows use a typed data table and its live
terminal sink. Numerical columns retain the selected scalar type and precision;
unavailable levels and references are optional values. The `log2(width)` column
stores the exponent, so a value of 1100 denotes a width of `2^1100`.

Output follows `UNI20_GLYPHS`, `UNI20_CHARSET`, `UNI20_COLOR`, `NO_COLOR`, and
terminal width (`COLUMNS`). Narrow terminals use the presentation layer's
vertical layout when needed, keeping numerical tokens intact. The live table
does not retain a second copy of the flow history.

## Exact strip reference and validation

`ising_strip_reference.hpp` contains 100-significant-digit decimal strings for
strip widths 2 through 256 at exact reduced temperatures 1 and 6/5. Values are
parsed directly into the tested type. A table is selected only when its rational
temperature rounds to the same value as the input; nearby temperatures are not
snapped to criticality. Runs above the table's precision retain the reference
decimals as exact values for presentation, displaying at most their original
100 significant digits without additional binary-rounding digits. Computed
HOTRG values continue to use the selected working precision.

The data come from the canonical fixture in
[Exact_Solution commit 1361a14](https://github.com/ianmccul/Exact_Solution/blob/1361a14aa978908bee8f82ac8f50d9ad9e2f466d/tests/data/ising_reference_high_precision.json),
submitted in [upstream PR #1](https://github.com/pcchen/Exact_Solution/pull/1).
That fixture was checked at 140 and 180 decimal working digits before rounding
to 100 significant digits. The generated header records its source SHA256.
Regenerate or check it with:

```bash
python3 scripts/generate_hotrg_reference.py /path/to/Exact_Solution/tests/data/ising_reference_high_precision.json
python3 scripts/generate_hotrg_reference.py /path/to/Exact_Solution/tests/data/ising_reference_high_precision.json --check
```

`tests/models/test_dense_hotrg.cpp` uses the shared numerical precision cases
for fp32, fp64, and each configured wide provider, including MPFR at 128 and
256 bits. It checks independently enumerated torus partition functions,
untruncated width-2 and width-4 strip spectra, low-temperature ground-state
limits, normalization before a potentially overflowing Gram matrix, and
128-round flows carrying increments smaller than double precision. Large equal
geometry exponents and tiny spectral gaps are also checked. The strip error
bound accounts for eigenvalue conditioning through `epsilon * exp(2*pi*X)`.
These comparisons validate up to 256 bits, below the fixture's 100-digit limit.

The test report records the actual contraction/decomposition dispatch and
marks unavailable scalar providers and inapplicable complex arithmetic.
`ising_hotrg_example --check` retains the original double-precision definition,
enumeration, truncation, and GEMM-dispatch checks. CTest also exercises CLI
precision selection, low temperatures, and 1100 rounds without constructing
the represented lattice size.
