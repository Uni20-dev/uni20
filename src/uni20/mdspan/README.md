# src/uni20/mdspan

This directory contains Uni20's native views and layouts and the structural
utilities used by dense kernels and layout-aware algorithms. Tensor producers
and CUDA acquisition use native mdspan types. The Kokkos implementation
supplies extents and supports interop.

## Contents

- `mdspan.hpp`: native borrowed view and checked import from the standard
  and Kokkos mdspans. It also includes the configured Kokkos implementation.
- `layouts.hpp`: native contiguous and nonnegative strided mappings.
- `layout_signed_stride.hpp`: affine mappings with signed strides and reachable offset bounds.
- `default_accessor.hpp`: pointer access with signed element offsets.
- `offset_range.hpp`: inclusive physical offset bounds for iteration plans.
- `mdspec.hpp`: mdspan-shaped mapping and accessor metadata paired with
  a data descriptor for later handle acquisition.
- `format.hpp`: `std::format` and `{fmt}` support for mdspan extents.
- `concepts.hpp`: concepts for mdspan-like objects and related structural
  properties.
- `conjugate_accessor.hpp`: read-only mdspan accessor adaptor and `conj(...)`
  view helper for lazy complex conjugation.
- `diagonal_accessor.hpp`: generalized-diagonal logical mdspans backed by an
  explicit rank-one strided component view.
- `generated_accessor.hpp` and `generated_layout.hpp`: read-only generated
  values with synthetic, non-strided logical offset mapping.
- `strides.hpp`: stride inspection and stride utility helpers.
- `permute.hpp`: zero-copy axis permutations of strided mdspans and mdspecs,
  preserving static extents and accessor semantics.
- `reverse.hpp`: zero-copy reversal of one axis in a native pointer-backed mdspan.
- `iteration_plan.hpp`: backend-neutral compact affine iteration planning over
  one or more strided mappings, including element counts and reachable offset
  ranges.
- `transform_view.hpp`: lazy read-only unary and variadic elementwise views.
- `zip_layout.hpp`: helpers for matching or combining view layouts.

## Notes

- Include `uni20/mdspan/mdspan.hpp` when code needs the configured mdspan
  implementation itself.
- This module should describe structure and layout, not ownership or backend
  dispatch policy. CPU execution of iteration plans belongs to the linalg CPU
  backend.
- `MdspanLike` is the complete readable mdspan protocol used by leaf kernels: its
  descriptor aliases must agree, it exposes rank and extent observers, and its
  rank-dimensional `operator[]` returns the declared `reference` type.
  `MutableMdspanLike` additionally proves assignment through that indexed result.
- `MdspecLike` is the broader structural protocol for mdspan metadata whose
  data handle is either immediately present or available through a data
  descriptor. `mdspec` is the standard unresolved materialization, but
  independent types may satisfy the concept directly. It preserves the actual
  mapping and accessor while intentionally exposing neither `data_handle()` nor
  element indexing.
- `MutableMdspecLike` refines eventual write capability through an
  assignable accessor reference. It does not add indexing to an unresolved
  descriptor.
- Immediate and descriptor-backed refinements use the same adjective order:
  mutable, ranked, strided, then the representation. For example,
  `MutableRankedStridedMdspanLike` requires an immediately usable mdspan, while
  `MutableRankedStridedMdspecLike` also accepts descriptor-backed
  metadata.
- `AffineMdspanLike` and `AffineMdspecLike` require an always-strided mapping
  and stride observers. `StridedMdspanLike` and `StridedMdspecLike` additionally
  require a statically zero base offset for consumers that need nonnegative
  strides. The logical origin of `layout_signed_stride` may sit inside the
  storage span, at `base_offset()` elements from its start.
- `DiagonalMdspecLike` refines `MdspecLike` without weakening its full logical
  rank-N interface. Off-diagonal access observes structural zero, while
  `diagonal_components(span)` exposes the physically stored rank-one values for
  specialized kernels. The component view preserves its signed stride, so a
  diagonal borrowed from another tensor need not be contiguous or forward
  ordered. The initial `make_diagonal_mdspan` factory accepts a default-accessor
  component mdspan and therefore produces a host-domain accessor. A future CUDA
  form must retain and propagate the CUDA component accessor rather than
  reclassifying a raw pointer-shaped handle.
- An extent-one axis has no observable stride. Stride coalescing treats it as
  an identity: it never blocks a merge, and the merged descriptor retains the
  non-singleton axis's stride. Generic iteration plans remove singleton axes
  before coalescing.
- A pointer `data_handle_type` is not enough to prove direct memory semantics.
  Backends that bypass `access(...)` must check for the default accessor
  or an explicitly lowerable accessor such as Uni20's `conjugated_accessor`.
- Read-only accessor policies declare const `element_type`, including calculated
  accessors that return values. Ordinary `MutableMdspanLike` accessors also prove
  indexed assignment. An opaque accessor without assignable element semantics
  is not mutable merely because its handle is pointer-shaped.
- `uni20::const_access(span, indices...)` performs read-only scalar access
  directly through a span's mapping and const-adapted accessor. Use it when a
  descriptor owner needs const element semantics without constructing a second
  mdspan for each access.
- A tensor descriptor's const `mdspan()` overload resolves a const-element view.
  Mutable tensor access is exposed only by the non-const overload.
- `uni20::conj(span)` is the user-facing lazy conjugation helper. Its accessor
  follows the C++26 `std::linalg::conjugated_accessor` direction while keeping
  Uni20's value-level `conj` behavior for real scalar types.
- `uni20::transform_view(function, spans...)` constructs a read-only expression
  descriptor. A function evaluated through a CUDA-accessible transform accessor
  must make its invoked call operator device-callable with
  `UNI20_HOST_DEVICE`; CUDA kernel compilation validates that contract. Eager
  overwrite and update use the dispatched tensor operations `assign_transform`
  and `transform_inplace`.
- Uni20-owned mapping, accessor, proxy-reference, generator, and transform
  execution surfaces are `UNI20_HOST_DEVICE`. This is independent of accessor
  domain registration: annotations make code compilable for a device, while
  `enable_accessor_in_domain` states where an acquired handle may be evaluated.
  Data descriptors and acquisition state remain backend-side control objects
  unless a leaf kernel explicitly defines a lowered descriptor ABI.
- A device-callable mapping or accessor is not automatically supported by every
  precompiled CUDA backend. Such a backend must carry a sufficient type-erased
  plan, provide an explicitly instantiated typed lowering, or cleanly decline.

## Related Documentation

- [Source tree map](../)
- [Mdspec contract](../../../docs/tensor/mdspec.md)
- [Tensor dispatch and view semantics](../../../docs/tensor/dispatch_and_view_semantics_draft.md)
- [BLAS/LAPACK mdspan wrappers](../../../docs/linalg/blas_lapack_wrappers.md)
- [Mdspan linear algebra dispatch](../../../docs/linalg/mdspan_dispatch.md)
