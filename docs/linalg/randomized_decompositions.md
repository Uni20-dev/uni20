# Randomized dense decompositions

`uni20::linalg::randomized_truncated_svd(a, options, identity)` preserves a real
float or double matrix and returns owning `[u, s, vh, info]` on its host or CUDA
device. For an $m$ by $n$ input and target rank $k$, the shapes are $m$ by $k$,
$k$, and $k$ by $n$. Singular values are in descending order. The metadata
records the retained extent and actual sketch width. Use the exact
`truncated_svd` when a cutoff or a discarded-weight decision needs the full spectrum.

```cpp
auto [u, s, vh, info] = uni20::linalg::randomized_truncated_svd(
    a, {.rank = 12}, uni20::random::root(42));
```

The caller chooses an identity for each operation. Reusing the identity
replays the Gaussian sketch. Independent calls can use different children of
a common root. The first child (`child(identity, 0)`) supplies
the sketch. Float sketches round the standard normal draws to float.
Singular-vector signs and bases within repeated singular values depend on the solver.

The defaults are eight extra columns ($p=8$) and two power iterations ($q=2$).
The sketch width is $\ell=\min(k+p,m,n)$ for $k>0$. The rank must lie in
$[0,\min(m,n)]$ and both oversampling $p$ and the power count $q$ must be
nonnegative. Rank zero returns empty factors immediately. Increasing $p$ samples
more directions in each pass. Increasing $q$ improves separation for slowly
decaying spectra at the cost of two additional matrix products per iteration.

The range finder takes a reduced Householder QR of $A\Omega$, then alternates
QR of $A^TQ$ and $AZ$ on each power step. It factors the tall matrix $A^TQ$
and lifts the retained left vectors through $Q$. This also supports wide
inputs. On CUDA the sketch, QR workspaces, small SVD and factors reside on
the input device. CUDA requires cuBLAS and cuSOLVER.
