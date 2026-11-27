# Model Examples

`spin_half_heisenberg_dmrg_example.cpp` constructs a U(1) spin-half Néel
product MPS and reduced-boundary Heisenberg MPO, then runs alternating two-site
DMRG sweeps. It prints one summary per directional traversal, elapsed time, and
the difference from a registered reference energy when one is available.

`--execution=cpu` is the default. A CUDA build with the CUDA, cuBLAS, and
cuSOLVER backends also supports `--execution=cuda`. The MPS, environments,
two-site centers, Krylov vectors, and SVD factors then remain on the selected
device; the MPO coefficients and compact singular-value selection data remain
on the host. CUDA DMRG currently supports real `float` or `double`, and this
example uses `double`.

The default four-site calculation can verify the exact open-chain energy

```text
-(3 + 2 sqrt(3)) / 4
```

with

```bash
spin_half_heisenberg_dmrg_example --check
```

The larger reference case is

```bash
spin_half_heisenberg_dmrg_example \
    --sites=20 --max-states=64 --max-sweeps=8 \
    --energy-tol=1e-10 --local-matvecs=4 \
    --scalar=real --precision=fp64 --block-threads=1 --check
```

`--local-matvecs=N` selects the fixed number of effective-Hamiltonian
applications requested for each local update. It defaults to four. This is a
local work budget, not a convergence tolerance; sweep-level environment and
energy convergence remain the controlling DMRG criteria.

`--energy-tol=VALUE` overrides the sweep-level energy tolerance. Its default is
zero, which selects `100 * numeric_limits<Real>::epsilon()` after applying the
chosen scalar precision.

`--scalar=complex` runs the same calculation with `uni20::complex<Real>`
storage and arithmetic. This is useful for controlled comparisons with
implementations that do not provide a real-scalar path.
`--precision=fp32|fp64` selects `Real` independently; the default is `fp64`.
Reference checks use precision-specific tolerances; the four-site fp32 real and
complex paths are registered CTests with a `1e-6` analytic-energy tolerance.

In an MPLAPACK-enabled build, `--precision=fp128` runs the same U(1)
BlockTensor DMRG path with `uni20::float128` or
`uni20::complex<uni20::float128>` storage. Precision selection does not change
the symmetry or block-sparse model.

`--block-threads=N` installs a `TbbScheduler` with concurrency `N`. Sparse
environments use `ParallelPackedSparseBlockStorage`; transient two-site centers
and their Krylov vectors use `ParallelPackedCompleteBlockStorage`. Independent
output blocks execute as synchronous lightweight batch items. Dense BLAS should
normally remain single-threaded when block-level concurrency is greater than
one.

For CUDA execution, `--cuda-submitters=N` independently limits the number of
lightweight batch runners that may submit device work. It defaults to one and
is bounded by `--block-threads`; `--cuda-streams=N` separately controls the
number of device streams available to those runners. The cuBLAS pool defaults
to one handle per stream and the cuSOLVER pool defaults to two handles, capped
by the stream count. `--cuda-cublas-handles=N` and
`--cuda-cusolver-handles=N` override those provider capacities. CPU execution
leaves lightweight batch concurrency uncapped within the scheduler arena.

On CPU, `--mps-storage=packed` retains the default serial packed storage for
MPS sites.
`--mps-storage=parallel-packed` enables scheduler batches while retaining one
packed site allocation. `--mps-storage=parallel-aligned-packed` pads each block
start to a 64-byte boundary within that allocation, while
`--mps-storage=parallel-separate` gives every dense site block an independently
aligned allocation. The environment and transient center policies are
unchanged, so these modes isolate MPS-site storage and execution effects.
CUDA execution currently selects parallel packed CUDA storage for the MPS,
environments, and transient centers independently of `--mps-storage`.

`--measurements=coarse` records and prints inclusive DMRG phase wall times.
`--measurements=detailed` additionally times every per-charge block-SVD item
and reports aggregate overlap and scheduler-tail fields. The default is
`--measurements=off`, which selects the compile-time disabled library path.
Detailed measurements intentionally perturb fine-grained work and should be
used for attribution, not final benchmark numbers.

`--steady-sweeps=N` selects the fixed-dimension benchmark path instead of the
energy-convergence path. Starting from the same rank-one Néel state, it runs
unmeasured alternating growth traversals until a completed traversal actually
reports `max-m` equal to `--max-states`. That traversal remains part of warm-up.
It then times `N` additional traversals and reports growth, steady, and combined
wall time separately. For spin one-half, the bond dimension can grow by at most
a factor of two per directional traversal, so `m=128`, `256`, and `512`
normally require seven, eight, and nine growth traversals respectively. The
example allows two additional growth traversals before reporting that the
requested dimension is not reachable for the selected fixture. `--max-sweeps`
is used only by the convergence path.

For example, the following commands compare the same optimized executable and
initial state after reaching `m=512`:

```bash
MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
spin_half_heisenberg_dmrg_example \
    --execution=cpu --sites=100 --max-states=512 --steady-sweeps=2 \
    --local-matvecs=4 --block-threads=8

MKL_NUM_THREADS=1 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
spin_half_heisenberg_dmrg_example \
    --execution=cuda --cuda-device=0 --cuda-streams=4 --cuda-submitters=1 \
    --sites=100 --max-states=512 --steady-sweeps=2 \
    --local-matvecs=4 --block-threads=8
```

The CUDA runtime, initial host-to-device MPS transfer, MPO construction, and
environment-cache construction occur before either reported timer. CUDA stream
work is synchronized at every traversal boundary so each wall time includes
completion of the submitted device work.

The first end-to-end scaling tables, local Matrix Product Toolkit orientation,
and exact current parallel boundary are recorded in
[DMRG Performance Baselines](../../docs/tensor_network/dmrg_performance_baselines.md).

Its comparison energy, `-8.682473334398985`, was calculated with Matrix Product
Toolkit `mp-dmrg-2site` from a Néel product state using `m=128`. The reference is
reproducible with:

```bash
spinchain-u1 -S 0.5 -o spinchain-u1.lattice
mp-construct --finite --repeat 10 -l spinchain-u1.lattice \
    -o spinchain-u1-l20-neel.psi '0.5:-0.5'
mp-dmrg-2site -w spinchain-u1-l20-neel.psi \
    -H spinchain-u1.lattice:H_J1 -m 128 -s 12 \
    --maxiter 40 --miniter 4 --maxtol 1e-12 --flush
```

The example remains entirely symmetry-aware. The reference energy is a terminal
assertion and does not feed a dense state back into the DMRG path.

## Dense HOTRG of the 2D Ising model

`ising_hotrg_example.cpp` coarse-grains the square-lattice Ising model with HOTRG
(Xie, Chen, Qin, Zhu, Yang and Xiang, Phys. Rev. B 86, 045139 (2012)) on dense
`Tensor<R, 4>` site tensors `T[ym, xm, xp, yp]`. The algorithm lives in the
example-local header `dense_hotrg.hpp`: one step along an axis joins two tensors
on that axis and squeezes the pair of legs on each side of the other axis back
to one leg with a single isometry, the leading eigenspace of the gram of one such
pair taken through `truncated_svd`. A round is a y step then an x step, so after
`k` rounds one coarse tensor spans `2^k x 2^k` sites.

```bash
./build/examples/ising_hotrg_example                 # T = Tc, bond 16, 8 rounds
./build/examples/ising_hotrg_example --bond=24 --rounds=8 --temperature=1.2
./build/examples/ising_hotrg_example --precision=fp128 --bond=8 --rounds=8
./build/examples/ising_hotrg_example --precision=mpreal --bits=256 --bond=4 --rounds=2
./build/examples/ising_hotrg_example --check
```

The default run prints, round by round, the free energy per site against
Onsager's and the two lowest scaling dimensions against the exactly solved strip
of the same width. The scaling dimensions are read from the coarse tensor's own
transfer matrix: joining its two horizontal legs gives `M = V^(2^k)` for a
periodic strip of `2^k` sites, so `X_i = ln(lambda_0 / lambda_i) / (2 pi)` with
no normalization bookkeeping.

`--check` verifies the original double-precision implementation and exits non-zero on any failure:

- the gram, the update network, and one step along both axes against the
  definition, by brute-force loops on a random tensor with unequal leg dimensions;
- untruncated flows against partition functions enumerated over spin
  configurations (Ising to the 4 x 4 torus, the 3- and 4-state clock model, the
  3-state Potts model);
- Onsager's free energy;
- every contraction of a round reaching GEMM, through the dispatch diagnostics;
- the transfer-matrix scaling dimensions against the exactly solved strip;
- two inputs at the edge: a cut that falls inside the null space of the gram, and
  a 128-round flow whose lattice width `2^rounds` would not fit in 64 bits.

The exact strip values are stored as 100-significant-digit decimal strings in
`ising_strip_reference.hpp`, parsed at the selected precision, for widths
2 to 256 at `T = Tc` and `T = 1.2 Tc`. They come from the Onsager/Kaufman closed
form as implemented in <https://github.com/pcchen/Exact_Solution>; the header
records their provenance; `scripts/generate_hotrg_reference.py` regenerates it
from the canonical JSON fixture. The comparison tolerance is per level rather than
a fixed epsilon: the transfer matrix holds `lambda_i / lambda_0 = exp(-2 pi X_i)`,
so the rounding of the leading eigenvalue limits `X_i` to about
`eps exp(2 pi X_i) / (2 pi)`.

`contract` does not pack an operand, so each contraction in the header is
written as a matrix product, with the surviving index group leading or trailing
and the summed group adjacent on both operands, which is the form that
dispatches to GEMM. Each regrouping between contractions is a `permute<>` view
of the dense tensor, copied into a column-major tensor allocated at the selected
precision.

The flow carries normalization logarithms and integer `log2_width` exponents;
it never forms the represented lattice size or products of removed scales.
The initial tensor is normalized before the first Gram matrix, and bond weights
factor out the growing exponential before evaluation. There is no 62-round cap.
Temperature is parsed as `T/Tc` and the critical coupling is computed at the
selected precision.

`uni20_models_tests` covers all configured arithmetic types, including MPFR at
128 and 256 bits. MPFR contractions and decompositions use MPLAPACK with explicit working
precision. The Onsager quadrature sample count remains a
separate accuracy control. See [the HOTRG numerical contract](../../docs/tensor_network/dense_hotrg.md)
for supported providers, reference regeneration, and precision coverage.

### The same step on CUDA tensors

In a build with the CUDA, cuBLAS and cuSOLVER backends, `--execution=cuda` runs
the flow on CUDA tensors of `float` or `double` (`--precision=fp32` or `fp64`),
on the device chosen by `--cuda-device=N`:

```bash
./build/examples/ising_hotrg_example --execution=cuda --bond=32 --rounds=5
./build/examples/ising_hotrg_example --execution=cuda --check
```

The step in `dense_hotrg.hpp` is written once: the gram, the update network, one
step and the flow are templates over the tensor type, and `host_place` and
`device_place` allocate in their place. The model tensor is built and
normalized on the host and transferred once, and the flow's observer receives a
host copy of each coarse tensor, because a CUDA tensor has no element access.
The contractions run in cuBLAS and the regroupings are device copies of
`permute<>` views. Where a CUDA tensor has no counterpart of a host idiom, the
header has an overload for it:

| Host idiom | CUDA tensor |
| --- | --- |
| the largest entry, by a loop over the storage | `maximum_host(t)` |
| scaling, by a loop over the storage | `transform_inplace(t, linalg::scale{f})` |
| the traces, by element access | `sum_host` of `diagonal_view<0, 3, 1, 2>(t)`, and of `diagonal_view<0, 2, 1, 3>` for the gram |
| `truncated_svd` of the gram | the full cuSOLVER SVD; the singular values are read on the host, and U is the left factor times the leading `k` columns of the identity, since a device matrix has no column view |

Each of these returns one host scalar or keeps its result on the device. With
`--execution=cuda`, `--check` adds group I: the gram and the update network
against the host entry by entry, one step along each axis, and two flows of the
critical Ising model. `--second-cuda-device=N` also enrolls device `N` and
checks that a step on a tensor held there keeps its results on that device.

### Cutting the gram

Each step keeps the leading `D` vectors of a symmetric positive semidefinite
gram of size `D^2 x D^2` at the plateau. `--gram-cut=M` chooses how:

| Method | Host | CUDA |
| --- | --- | --- |
| `svd` (default) | `truncated_svd` (LAPACK `gesvd`, or MPLAPACK) | the full cuSOLVER SVD, then the leading columns by a GEMM |
| `eigh` | `eigh` (LAPACK `syev`, or MPLAPACK) | cuSOLVER `syevd`, then the trailing columns by a GEMM |

Both keep the same subspace when the gap at the cut is not tiny; the flows agree
to rounding.

`--benchmark-gram-cut` compares the methods at fp64 on the selected place, for
the bonds in `--bonds=D,D,...`:

```bash
./build/examples/ising_hotrg_example --benchmark-gram-cut --bonds=8,16,24,32 --rounds=6
./build/examples/ising_hotrg_example --execution=cuda --benchmark-gram-cut --bonds=16,32,48,64 --rounds=5
```

It cuts the y-step gram of a 4-round critical flow with each method, keeps the
fastest of `--repeat=N` runs, and reports the error of the kept spectrum and of
the discarded weight, the distance `||U U^T - V V^T||_F` between the kept
subspaces, and the gap at the cut, all against a host `eigh` of the same gram.
It then runs a `--rounds` flow with each method and reports the final ln Z/N and
X_1 against the `svd` flow. More methods are tracked in
[issue 200](https://github.com/Uni20-dev/uni20/issues/200).

On a Quadro GV100 (CUDA 12.9), `eigh` cuts the bond-64 gram in 0.41 s against
2.17 s for `svd`, and a five-round bond-64 flow takes 9.3 s against 15.8 s. On
the host (Intel Alder Lake, OpenBLAS), the bond-32 cut takes 0.49 s against 1.97 s.
