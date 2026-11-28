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

The range finder calls the dispatched Q-only operation `orthonormal_basis` on $A\Omega$, then alternates
QR of $A^TQ$ and $AZ$ on each power step. It factors the tall matrix $A^TQ$
and lifts the retained left vectors through $Q$. This also supports wide
inputs. On CUDA the sketch, QR workspaces, small SVD and factors reside on
the input device. CUDA requires cuBLAS and cuSOLVER.

## Positive semidefinite matrices

`randomized_nystrom(a, options, identity)` returns `[eigenvalues, eigenvectors, info]`.
Supply a square real symmetric positive semidefinite matrix with both triangles
populated. The result contains the largest $k$ approximate eigenvalues in
descending order and an orthonormal column for each value, in the same storage
domain as the input. The options, RNG identity and rank-zero convention are
shared with the randomized SVD.

After orthonormalizing the Gaussian sketch, each power iteration applies $A$
once and reorthogonalizes. With the resulting basis $Q$, the algorithm forms
$Y=AQ$ and the small projected matrix $W=Q^T Y$. The Nyström approximation is
$YW^\dagger Y^T$. If $W=VDV^T$, a thin SVD of $YV(D^\dagger)^{1/2}$ yields
its eigenvectors. Squaring the retained singular values gives its eigenvalues.
This requires $q+1$ applications of the input, compared with $2q+2$ for the
randomized SVD. The remaining work acts on $n$ by $\ell$ or $\ell$ by $\ell$
matrices. The CUDA path keeps the whitening factors and all decomposition work
on the device.

A positive projected eigenvalue is retained when its ratio to the largest
exceeds $\ell\,\epsilon$ at the working precision. All other projected modes
receive zero weight in the pseudoinverse. A zero or rank-deficient Gram matrix
therefore still returns the requested factor shapes. In a zero eigenspace,
the choice of orthonormal completion is arbitrary.
