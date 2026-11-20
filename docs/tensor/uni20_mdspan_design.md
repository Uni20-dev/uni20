# Uni20 mdspan Design

Status: proposal, 2026-11-20. Nothing here is implemented yet.

Uni20 uses the Kokkos reference implementation of `std::mdspan`, included as
`stdex` through [`mdspan/mdspan.hpp`](../../src/uni20/mdspan/mdspan.hpp). This
note proposes Uni20's own `mdspan` class and layout mappings, keeping the
standard extents. The goals are a stride observer that exists at every rank,
well-defined signed offsets, and a reachable-offset range on every mapping.
Negative strides then become possible as an opt-in layout.

## Why Not the Standard Types

**Rank-zero strides.** [mdspan.layout.reqmts] note 5 says "It is not required
for `m.stride(r)` to be well-formed if `m.extents().rank()` is zero, even if
`m.is_always_strided()` is true." `layout_left` and `layout_right` constrain
`stride` with `extents_type::rank() > 0`; `layout_stride` does not. A
constrained-away member cannot be named, even in a loop over the axes that never
runs at rank zero. `std::mdspan::stride` is declared without a constraint but
forwards to the mapping, so a `requires` check reports it present when calling
it fails to compile. With the standard layouts, Uni20's strided concepts would
need a rank-zero escape and generic code a free stride helper; pull request #148
tried that and is superseded by this design.

**Offsets.** The standard accessor policy takes a `size_t` offset, and
`layout_stride` requires positive strides. `required_span_size`,
`is_exhaustive` and `is_unique` assume nonnegative strides. Uni20 accessors
already declare a signed `offset_type` (`span_offset_t` in
[`mdspan/concepts.hpp`](../../src/uni20/mdspan/concepts.hpp)), and Kokkos passes
the mapping's signed result to `access` unchanged, so negative offsets reach
those accessors today. That works by accident: it violates `layout_stride`'s
precondition, `stdex::default_accessor` takes `size_t`, and the span queries
give wrong answers.

**Control.** Owning the types lets every mapping and mdspan member be
`UNI20_HOST_DEVICE`, report precondition failures through Uni20 diagnostics,
and grow layouts the standard does not have.

### Non-goals

- `submdspan`. Uni20 does not use it.
- C++26 padded layouts.
- New extents. `std::extents` is adequate.

## Current Use

Counts are over `src`, `tests` and `examples` on `main` (`33439da6`):

| Name | Uses |
|---|---|
| `stdex::mdspan` | 256 |
| `stdex::layout_stride` | 139 |
| `stdex::layout_left` | 121 |
| `stdex::dextents` | 108 |
| `stdex::extents` | 54 |
| `stdex::layout_right` | 52 |
| `stdex::default_accessor` | 35 |
| `stdex::dynamic_extent` | 18 |

Fifteen source files name `stdex::mdspan`; those are where the type crosses
public APIs. `RowMajor` and `ColumnMajor` in
[`tensor/tensor.hpp`](../../src/uni20/tensor/tensor.hpp) alias the standard
layouts, and twelve sites compare a layout type with `stdex::layout_left`,
`layout_right` or `layout_stride`. Uni20 already defines its own layouts
(`StridedZipLayout`, `GeneralZipLayout`, `GeneratedLayout`), accessors
(`const_accessor_adaptor`, `conjugated_accessor`, `diagonal_accessor`,
`generated_accessor`, the transform accessors and the CUDA pointer accessors) and
the `uni20::mdspec` descriptor. The concepts in `mdspan/concepts.hpp` are
structural, so they accept any type with the mdspan interface.

## Types

All live in namespace `uni20`, in new headers under `src/uni20/mdspan/`.

| Type | Definition |
|---|---|
| `extents`, `dextents`, `dynamic_extent` | Aliases of the standard ones. |
| `layout_left`, `layout_right`, `layout_stride` | Uni20 mappings with the standard semantics for nonnegative strides, plus the rules below. |
| `layout_signed_stride` | A strided mapping whose strides may have either sign. |
| `default_accessor<T>` | `offset_type` is `std::ptrdiff_t`; `access(p, i)` is `p[i]`, `offset(p, i)` is `p + i`. |
| `mdspan<T, Extents, Layout, Accessor>` | Same interface as `std::mdspan`, with the offset rule below. |
| `offset_range` | Inclusive `[minimum, maximum]` of handle-relative offsets, empty when `minimum > maximum`; the existing `iteration_offset_range` moves here. |

`mdspan` defaults to `layout_right` and `default_accessor<T>`, like the standard,
so translated code keeps its meaning; Uni20 Tensors keep naming their layout
explicitly.

## Rules That Differ From the Standard

1. **`stride(r)` exists at every rank** in every strided Uni20 mapping and in
   `mdspan`, with precondition `r < rank()`. At rank zero no axis meets it, so a
   call fails the precondition, while a loop over the axes compiles and never
   calls it.
2. **Offsets have the accessor's type.** `mdspan` converts the mapping result to
   `span_offset_t<Accessor>`, which is the accessor's `offset_type` when it
   declares one and `size_t` otherwise. A standard accessor therefore still
   requires nonnegative offsets, and a Uni20 accessor receives signed ones.
3. **Every mapping reports `reachable_range()`**, the offsets that some valid
   index reaches. For nonnegative strides it is `[0, required_span_size() - 1]`,
   or empty when an extent is zero. Buffer validation compares the descriptor
   offset plus this range with the buffer, instead of `required_span_size()`.
   `required_span_size()` remains for nonnegative layouts, as the standard's
   layout-mapping requirements need it.
4. **Strided means strided.** `StridedMdspanLike` and `StridedMdspecLike` require
   `is_always_strided()` and a declared `stride(axis)`, with no rank-zero escape.
   A standard `layout_left` or `layout_right` mdspan of rank zero does not
   qualify; Uni20 Tensors and descriptors never produce one, and standard
   mdspans convert at the boundary (below).
5. **Negative strides are a layout, not a runtime state.** Only
   `layout_signed_stride` may have negative strides. An operation that cannot
   handle them rejects the layout by type at compile time, so a reversed view
   never reaches a backend that would decline it at run time and end the
   program. Operations opt in one at a time.
6. **Empty mappings constrain their strides less.** The standard requires
   `layout_stride` strides to be positive and to admit an axis order in which
   no two axes overlap, even when the mapping has no elements. Uni20's
   `layout_stride` requires that only of nonempty mappings; an empty mapping
   accepts any nonnegative strides, including zero. That is the range canonical
   layouts already produce for empty tensors (`layout_left` with extents
   `(0, 3, 4)` has strides `(1, 0, 0)`), so code that reads strides already
   decides emptiness from the extents, and negative values stay excluded. A
   permutation of an empty source then permutes the source's strides like any
   other, with no new strides to invent. This follows LWG 4603 and P3959R0,
   applied to every constructor, including conversion from a canonical layout.
7. **The nonzero extents have a representable product.** The standard requires
   only the number of elements to be representable, which says nothing about an
   empty shape such as `(0, 2^32, 2^32)`, whose `layout_right` strides overflow.
   Uni20's mappings require the product of the nonzero extents to be
   representable in `index_type`. No tensor with those extents could hold its
   elements otherwise, and canonical strides of every supported shape, empty or
   not, are representable. Requiring only each stride to be representable would
   make validity depend on the order of the extents: with 32-bit indices and
   `n = 2^16`, `layout_right` with extents `(n, 0, n)` has strides `(0, n, 1)`,
   but `(0, n, n)` needs `n * n`. Materializing a permuted view, as
   `Tensor(permute(view))` does, builds a canonical layout with the permuted
   extents, so a valid empty tensor could yield an invalid one. The product
   condition is exactly what keeps every canonical stride representable under
   every permutation of the extents: putting all the nonzero extents before a
   zero one in `layout_left` makes the zero axis's stride their full product.
   No LWG issue covers this for `layout_left` and `layout_right`; LWG 4545
   fixes the same overflow for the padded layouts.

### Alignment With the Standard

Uni20 adopts these library issue resolutions, so its layouts behave as the
standard is expected to:

| Issue | Status (2026-11) | Adopted rule |
|---|---|---|
| LWG 4603, P3959R0 | Ready | Rule 6: an empty mapping accepts nonnegative strides. P3959R0 also relaxes the converting constructor; 4603's proposed resolution does not yet. |
| LWG 4606 | Ready | For a nonempty `layout_stride`, every stride is nonnegative and representable, and axes of extent one impose no ordering constraint. Only the axes of extent greater than one need an order in which none overlaps. A strided slice such as every other column of a `(2, 5)` row-major matrix, extents `(2, 3)` with strides `(5, 2)`, is valid. |
| LWG 4266 | C++26 | An empty mapping is exhaustive; `is_always_exhaustive()` is true when the rank is zero or a static extent is zero. |
| LWG 4562 | Ready | `is_exhaustive()` is `required_span_size()` equal to the product of the extents, so axes of extent one do not affect it. |
| LWG 4272 | C++26 | Conversions between rank-zero mappings of different layouts are implicit whenever the extents convert implicitly. |
| LWG 4217 | C++26 | Not adopted: rule 1 provides `stride()` at every rank instead of letting rank-zero mappings omit it. |

### Signed-stride Semantics

`layout_signed_stride` keeps the handle at the element with all indices zero,
as NumPy does, so offsets below the handle are negative. `operator()` is
`sum(index[k] * stride[k])`. `reachable_range()` adds
`(extent[k] - 1) * stride[k]` to the maximum for positive strides and to the
minimum for negative ones. `is_unique()` and `is_exhaustive()` are evaluated on
the absolute strides, which describe the same set of offsets shifted by
`minimum`. `required_span_size()` is not provided; consumers use
`reachable_range()`.

The first operations to accept it are `reverse` views of one axis, host `copy`,
host elementwise transforms and host reductions. The backend-neutral iteration
plan already normalizes negative strides. CUDA plan lowering would rebase the
descriptor to `minimum` later. BLAS and LAPACK do not accept it: BLAS's
negative-increment convention is different, and LAPACK leading dimensions must be
positive.

### Rejected Alternative

Allowing negative strides in `layout_stride` with runtime checks would keep one
strided type, but every backend that cannot handle them would decline at run
time. Item 0.1 of the CUDA coverage work exists to remove exactly that failure
mode.

## Conversions and Interop

- **From the standard.** Uni20 mappings construct from the matching standard
  mapping: `layout_left` from `stdex::layout_left`, `layout_right` from
  `stdex::layout_right`, and `layout_stride` from any standard strided mapping.
  The conversions are implicit where the standard's own conversions are, and
  `uni20::mdspan` constructs from a `stdex::mdspan` with a convertible layout and
  accessor. `stdex::default_accessor<T>` converts to `default_accessor<T>`.
- **To the standard.** An explicit conversion for `layout_left`, `layout_right`
  and `layout_stride`, for external code. `layout_signed_stride` converts only
  through a checked call that requires every stride to be positive.
- **Concepts** stay structural. A `stdex::mdspan` still satisfies `MdspanLike`;
  it satisfies `StridedMdspanLike` at every rank except zero in the left and
  right layouts.
- **Device code.** Every member is `UNI20_HOST_DEVICE constexpr`.
  `tests/linalg/test_cuda_descriptor_compile.cu` gains Uni20 mdspan and mapping
  compositions.

## Migration

Each phase is one pull request.

1. **Types.** The mappings, `default_accessor`, `mdspan`, `offset_range`,
   conversions and `reachable_range()` on the Uni20 layouts, with conformance
   tests for the standard behaviors Uni20 relies on, rank zero, signed accessor
   offsets and device compilation. Nothing else changes.
2. **Producers.** `RowMajor` and `ColumnMajor` alias the Uni20 layouts; Tensor
   `mdspan()`, `mdspec` mapping types, storage policies and the CUDA accessors
   produce Uni20 types. The strided concepts require `stride()` at every rank. The
   twelve layout comparisons use the Uni20 layouts. Buffer range checks move to
   `reachable_range()`. Generic code calls the member `stride()`, which exists at
   every rank, so no free stride helper is needed. CUDA `ScalarTensor` transfers,
   copies and reductions start to compile here, with the rank-zero CUDA tests
   from #148. `permute` of an empty source permutes the source's strides
   (rule 6), so the stride construction for empty sources in
   `mdspan/permute.hpp` is removed. This phase touches every layer and carries
   most of the risk.
3. **Tests and examples.** `stdex::` names become Uni20 names, about 550
   references, mostly mechanical. A few interop tests keep `stdex::mdspan`
   inputs.
4. **Signed strides.** `layout_signed_stride`, a `reverse` view and the first
   opt-in operations listed above.

## Relation to #148

Pull request #148 kept the standard types, added a free `uni20::stride` that
exists at every rank, and gave the strided concepts a rank-zero escape. Uni20's
own layouts make both unnecessary, so that approach is superseded. Its rank-zero
CUDA tests move into phase 2.

Local CI once reported GCC 13.4 crashing in its IPA modref pass on that pull
request's rank-zero CUDA test on Fracton. Four repeat compiles with the same
compiler, flags and sources succeeded, so it is treated as transient unless it
recurs.

### Warnings in Public Headers

Decided 2026-11-20: consumers embed uni20 as `SYSTEM`. uni20 exports its include
directories without `SYSTEM`, so otherwise a consumer compiles these headers
under its own warning flags, and a warning disabled only in uni20's targets can
fail its `-Werror` build. GCC's `-Wtype-limits` reports `axis < rank` as always
false when a generic lambda inside a template loops to a rank of zero, which
rank-generic code meets often. uni20 builds with `-Wno-type-limits`, writes rank
loops in whatever form reads best, and documents
`FetchContent_Declare(... SYSTEM)` and `add_subdirectory(... SYSTEM)` in
[CMake Consumers](../development/cmake_consumers.md).

## Open Questions

1. The name of the signed layout: `layout_signed_stride`, or a policy parameter
   on `layout_stride`.
2. What a precondition failure does in device code, where Uni20 diagnostics are
   unavailable: trap, or nothing in release builds.
3. Whether `mdspan::size()` and the span queries keep the standard's `size_t`
   types.
4. NumPy interop: whether arrays with negative strides map to
   `layout_signed_stride` views, once Python bindings exist.
