# Uni20 mdspan and mdspec Design

Status: proposal, 2026-11-20. Nothing here is implemented yet.

Uni20 uses the Kokkos reference implementation of `std::mdspan`, included as
`stdex` through [`mdspan/mdspan.hpp`](../../src/uni20/mdspan/mdspan.hpp). This
note proposes Uni20's own `mdspan` class and layout mappings, keeping the
standard extents. The goals are a stride observer that exists at every rank,
well-defined signed offsets, and a reachable-offset range on every mapping.
Negative strides then become possible as an opt-in layout. `uni20::mdspec`,
which Uni20 already owns, follows the same specification
([below](#mdspec)).

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
`is_exhaustive` and `is_unique` assume nonnegative strides. Uni20 has the
mechanism for signed offsets: an accessor may declare its own `offset_type`
(`span_offset_t` in [`mdspan/concepts.hpp`](../../src/uni20/mdspan/concepts.hpp)),
and Kokkos passes the mapping's signed result to `access` unchanged. No in-tree
accessor uses it for a signed type yet. The CUDA pointer accessors,
`diagonal_accessor` and `generated_accessor` declare `std::size_t`, host
tensors use `stdex::default_accessor`, and the const, conjugating and transform
adaptors inherit their wrapped accessor's type; only test accessors declare
`std::ptrdiff_t`.

Negative strides already occur, without a consistent rule. The iteration plan
computes in `std::ptrdiff_t` and normalizes negative strides, and the diagonal
component view keeps a signed stride, so a diagonal borrowed from another
tensor may run backwards. That component view stores its stride in a
`stdex::layout_stride` mapping and reads through `stdex::default_accessor`, so a
negative offset reaches `access` as a wrapped `size_t` and works only because
the pointer arithmetic wraps back. That violates `layout_stride`'s precondition
and is undefined behavior by the letter of the language, and the span queries
give wrong answers for such a mapping. This design replaces the wrapping with
declared signed offsets.

**Control.** Owning the types lets every mapping and mdspan member be
`UNI20_HOST_DEVICE`, report precondition failures through Uni20 diagnostics,
and grow layouts the standard does not have.

### Non-goals

- `submdspan`. Uni20 does not use it.
- New extents. `std::extents` is adequate.

C++26's padded layouts, `layout_left_padded` and `layout_right_padded`, are not
part of this migration, but nothing here excludes them. They are strided
mappings, so rules 1 to 7 apply to them unchanged; LWG 4545 is their case of
the overflow that rule 7 covers. They say in the type what `layout_stride`
says only at run time, that the inner stride is one and only the leading
dimension varies, as BLAS and LAPACK matrices need. Uni20 can add them if that
proves useful over a general strided layout.

## Current Use

Counts are occurrences over `src`, `tests` and `examples` on `main`
(`33439da6`), from `git grep -ohP 'stdex::NAME\b(?!_)' main -- src tests examples | wc -l`:

| Name | Uses |
|---|---|
| `stdex::mdspan` | 254 |
| `stdex::layout_stride` | 135 |
| `stdex::layout_left` | 109 |
| `stdex::dextents` | 106 |
| `stdex::extents` | 53 |
| `stdex::layout_right` | 44 |
| `stdex::default_accessor` | 34 |
| `stdex::dynamic_extent` | 18 |

Fifteen files in `src` name `stdex::mdspan`; those are where the type crosses
public APIs. `RowMajor` and `ColumnMajor` in
[`tensor/tensor.hpp`](../../src/uni20/tensor/tensor.hpp) alias the standard
layouts, and 13 lines in `src` compare a layout type with `stdex::layout_left`,
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
| `mdspec<T, Extents, Layout, Accessor, Descriptor>` | Existing; specified to match `mdspan` ([below](#mdspec)). |
| `offset_range` | Inclusive `[minimum, maximum]` of handle-relative offsets, with `std::ptrdiff_t` bounds, empty when `minimum > maximum`; the existing `iteration_offset_range` moves here. |

`mdspan` defaults to `layout_right` and `default_accessor<T>`, like the standard,
so translated code keeps its meaning; Uni20 Tensors keep naming their layout
explicitly.

## Rules That Differ From the Standard

1. **`stride(r)` exists at every rank** in every strided Uni20 mapping, in
   `mdspan` and in `mdspec`, with precondition `r < rank()`. At rank zero no axis meets it, so a
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
   program. Operations opt in one at a time. The same holds for composition:
   an `mdspan` or `mdspec` with `layout_signed_stride` requires a signed
   `index_type` and an accessor whose `span_offset_t` is signed, so a signed
   layout paired with `stdex::default_accessor`, or any accessor that falls back
   to `size_t`, does not compile. Uni20's `index_type` is already
   `std::ptrdiff_t` ([`core/types.hpp`](../../src/uni20/core/types.hpp)), and
   Tensor extents use it.
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
   representable in `index_type`. That keeps every canonical stride
   representable under every permutation of the extents, so materializing a
   permuted view, as `Tensor(permute(view))` from #137 does, cannot turn a valid
   empty tensor into an invalid one. A condition on each stride would not be
   enough, because it depends on the order of the extents. Such shapes do not
   arise in practice, and Uni20 does not check for them: the rule is a
   precondition that Uni20 assumes of every mapping, including one converted
   from the standard. No LWG issue covers this for `layout_left` and
   `layout_right`; LWG 4545 fixes the same overflow for the padded layouts.

### Alignment With the Standard

Uni20 adopts these library issue resolutions, so its layouts behave as the
standard is expected to:

| Issue | Status (2026-11-18) | Adopted rule |
|---|---|---|
| LWG 4603, P3959R0 | Voting | Rule 6: an empty mapping accepts nonnegative strides. Both also drop the converting constructor's precondition that every stride of the source is positive. |
| LWG 4606 | Voting | For a nonempty `layout_stride`, every stride is nonnegative and representable, and axes of extent one impose no ordering constraint. Only the axes of extent greater than one need an order in which none overlaps. A strided slice such as every other column of a `(2, 5)` row-major matrix, extents `(2, 3)` with strides `(5, 2)`, is valid. |
| LWG 4266 | C++26 | An empty mapping is exhaustive; `is_always_exhaustive()` is true when the rank is zero or a static extent is zero. |
| LWG 4562 | Voting | `is_exhaustive()` is `required_span_size()` equal to the product of the extents, so axes of extent one do not affect it. |
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

Negative strides are useful in only a few cases, reversing the elements of an
axis being the main one, and then only over operations where they are easy or
cheap to support. Operations therefore accept `layout_signed_stride` as uses
require, not in advance. The likely first are a one-axis `reverse` view and
host `copy`. Host elementwise transforms and reductions cost little, because the
backend-neutral iteration plan already normalizes negative strides. CUDA plan
lowering would rebase the descriptor to `minimum` when a use needs it. BLAS and
LAPACK do not accept it: BLAS's negative-increment convention is different, and
LAPACK leading dimensions must be positive.

### Rejected Alternatives

Allowing negative strides in `layout_stride` with runtime checks would keep one
strided type, but every backend that cannot handle them would decline at run
time. Item 0.1 of the CUDA coverage work exists to remove exactly that failure
mode.

A sign policy parameter on `layout_stride` would also distinguish the cases by
type, but every existing use and comparison of `layout_stride` would then have
to say which policy it means. A separate layout leaves them meaning the
nonnegative one. Decided tentatively on 2026-11-20: the signed layout is
`layout_signed_stride`.

## Conversions and Interop

- **From the standard.** Uni20 mappings construct from the matching standard
  mapping: `layout_left` from `stdex::layout_left`, `layout_right` from
  `stdex::layout_right`, and `layout_stride` from any standard strided mapping.
  The conversions are implicit where the standard's own conversions are, and
  `uni20::mdspan` constructs from a `stdex::mdspan` with a convertible layout and
  accessor. `stdex::default_accessor<T>` converts to `default_accessor<T>`.
- **Interop is assumed to be sane.** A mapping that meets the standard's
  preconditions also meets Uni20's, except for shapes that break rule 7, which
  do not arise. Uni20 therefore takes a standard mapping as it is. A conversion
  may check Uni20's rules in debug builds; release builds do not.
- **To the standard.** An explicit conversion for `layout_left`, `layout_right`
  and `layout_stride`, for external code. The strides pass through unchanged. A
  zero stride that Uni20 allows, on an empty mapping or an axis of extent one,
  breaks C++23's requirement that every `layout_stride` stride be positive
  ([mdspan.layout.stride.cons]), just as the standard's own conversion of an
  empty `layout_left` mapping to `layout_stride` does until LWG 4603. Kokkos does
  not check that precondition, and Uni20 does not rewrite the strides to satisfy
  it. `layout_signed_stride` converts only through a checked call that requires
  every stride to be positive, since negative strides are that layout's
  ordinary state.
- **Concepts** stay structural. A `stdex::mdspan` still satisfies `MdspanLike`;
  it satisfies `StridedMdspanLike` at every rank except zero in the left and
  right layouts.
- **Device code.** Every member is `UNI20_HOST_DEVICE constexpr`.
  `tests/linalg/test_cuda_descriptor_compile.cu` gains Uni20 mdspan and mapping
  compositions.

## mdspec

`uni20::mdspec` is an mdspan whose data handle has not been acquired yet. It
stores a data descriptor in place of the handle, together with the mapping and
accessor that the eventual mdspan will use ([Mdspec](mdspec.md)). Uni20 already
owns it, so nothing needs replacing, but its specification should match
`uni20::mdspan` as closely as the missing handle allows. Everything above
applies to it:

- **One vocabulary.** mdspec's template parameters, type aliases and observers
  are `uni20::mdspan`'s. It lacks `data_handle()` and `operator[]`, and adds
  `data_descriptor_type` and `data_descriptor()`. It gains `size()` and
  `empty()`, the two mdspan observers it lacks today.
- **Strides at every rank (rules 1 and 4).** `mdspec::stride(r)` today exists
  only where the mapping's does, so a rank-zero `layout_left` mdspec has none.
  With Uni20 layouts it exists at every rank with the precondition
  `r < rank()`, and `StridedMdspecLike` and `StridedMdspanLike` keep one
  definition.
- **Offsets (rules 2 and 5).** The stored accessor fixes the offset type of the
  mdspan an mdspec resolves to, so the composition rule applies before any
  handle exists: an mdspec with `layout_signed_stride` and a `size_t`-offset
  accessor does not compile.
- **Reachable range (rule 3).** Acquisition checks the descriptor's offset plus
  `mapping().reachable_range()` against the buffer. Today
  `validate_cuda_descriptor_range` in
  [`tensor/cuda_access.hpp`](../../src/uni20/tensor/cuda_access.hpp) checks
  `element_offset()` plus `required_span_size()`.
- **Empty mappings and representable products (rules 6 and 7)** come with the
  mapping and need nothing separate.
- **Resolution.** Acquisition combines the leased handle with `mapping()` and
  `accessor()` into a `uni20::mdspan`, where `resolve_cuda_mdspan` now builds a
  `stdex::mdspan`, and the leases' `.mdspan()` return that type.
- **Concepts.** `MdspecLike` already accepts every `MdspanLike`. With one
  specification, a generic algorithm constrained on `StridedMdspecLike` sees
  the same strides, offsets and ranges for an immediate input and for a
  descriptor-backed one.

A conformance test checks, for each Uni20 layout, that `mdspec` and `mdspan`
expose the same observers with the same types and values.

## Migration

Each phase is one pull request.

1. **Types.** The mappings, `default_accessor`, `mdspan`, `offset_range`,
   conversions and `reachable_range()` on the Uni20 layouts; `mdspec` gains
   `size()` and `empty()`. Conformance tests cover the standard behaviors Uni20
   relies on, rank zero, signed accessor offsets, device compilation and the
   agreement of `mdspec` with `mdspan`. Nothing else changes.
2. **Producers.** `RowMajor` and `ColumnMajor` alias the Uni20 layouts; Tensor
   `mdspan()`, `mdspec` mapping types, storage policies and the CUDA accessors
   produce Uni20 types, and acquisition resolves an `mdspec` to a
   `uni20::mdspan`. The CUDA pointer accessors declare `std::ptrdiff_t` offsets.
   The strided concepts require `stride()` at every rank. The layout comparisons
   use the Uni20 layouts. Buffer range checks, for Tensors and for
   descriptor acquisition, move to `reachable_range()`. Generic code calls the
   member `stride()`, which exists at every rank, so no free stride helper is
   needed. CUDA `ScalarTensor` transfers, copies and reductions start to compile
   here, with the rank-zero CUDA tests from #148. `permute` of an empty source
   permutes the source's strides (rule 6), so the stride construction for empty
   sources in `mdspan/permute.hpp` is removed; that header comes with dense
   `permute` in #137, so this phase assumes #137 has merged. This phase touches
   every layer and carries most of the risk.
3. **Tests and examples.** `stdex::` names become Uni20 names, about 620
   references, mostly mechanical. A few interop tests keep `stdex::mdspan`
   inputs.
4. **Signed strides.** `layout_signed_stride`, a `reverse` view, and support in
   the operations its first use needs, host `copy` for example; other
   operations follow as uses require them. The diagonal component view moves to
   `layout_signed_stride`, ending its reliance on wrapped `size_t` offsets, and
   the remaining code that computes signed offsets through unsigned arithmetic
   is found and fixed here.

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

Decided 2026-11-20 and proposed in #149, which has not merged: consumers embed
uni20 as `SYSTEM`. uni20 exports its include directories without `SYSTEM`, so
otherwise a consumer compiles these headers under its own warning flags, and a
warning disabled only in uni20's targets can fail its `-Werror` build. GCC's
`-Wtype-limits` reports `axis < rank` as always false when a generic lambda
inside a template loops to a rank of zero, which rank-generic code meets often.
With #149, uni20 builds with `-Wno-type-limits`, writes rank loops in whatever
form reads best, and documents `FetchContent_Declare(... SYSTEM)` and
`add_subdirectory(... SYSTEM)` in
[CMake Consumers](../development/cmake_consumers.md).

## Open Questions

1. What a precondition failure does in device code, where Uni20 diagnostics are
   unavailable: trap, or nothing in release builds.
2. Whether `size()` and the span queries of `mdspan` and `mdspec` keep the
   standard's `size_t` types.
3. NumPy interop: whether arrays with negative strides map to
   `layout_signed_stride` views, once Python bindings exist.
