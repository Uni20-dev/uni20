# Uni20 mdspan and mdspec Design

Status: proposal, 2026-11-20. Nothing here is implemented yet.

Uni20 uses the Kokkos reference implementation of `std::mdspan`, included as
`stdex` through [`mdspan/mdspan.hpp`](../../src/uni20/mdspan/mdspan.hpp). This
note proposes Uni20's own `mdspan` class and layout mappings, keeping the
standard extents. The goals are a stride observer that exists at every rank,
well-defined signed offsets, and a reachable-offset range on every mapping.
Negative strides then become possible as an opt-in layout. `uni20::mdspec`,
which Uni20 already owns, follows the same specification, as the mdspec
section below describes.

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
mappings, so rules 1 to 7 apply to them unchanged, and converting one to a
narrower `index_type` needs the stride condition under rule 7, which LWG 4545
proposes for the standard's. They say in the type what `layout_stride`
says only at run time, that the inner stride is one and only the leading
dimension varies, as BLAS and LAPACK matrices need. Uni20 can add them if that
proves useful over a general strided layout.

## Current Use

Counts are occurrences over `src`, `tests` and `examples` on `main`
(`33439da6`), from this command for each name:

```sh
git grep -ohP 'stdex::NAME\b(?!_)' main -- src tests examples | wc -l
```

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
| `mdspec<T, Extents, Layout, Accessor, Descriptor>` | Existing; specified to match `mdspan`; see mdspec below. |
| `offset_range` | Inclusive `[minimum, maximum]` of handle-relative offsets, with `std::ptrdiff_t` bounds, empty when `minimum > maximum`; the existing `iteration_offset_range` moves here. |

`mdspan` defaults to `layout_right` and `default_accessor<T>`, like the standard,
so translated code keeps its meaning; Uni20 Tensors keep naming their layout
explicitly.

`mdspan` and `mdspec` take `index_type` and `size_type` from `extents_type`.
Their `size()` returns the product of the extents as `extents_type::size_type`.
For standard extents this is `std::make_unsigned_t<index_type>`.
The `layout_left`, `layout_right` and `layout_stride` mappings return
`index_type` from `required_span_size()`, which includes any gaps between
elements. A view of shape `(2, 2)` with strides `(3, 1)` therefore has
`size() == 4` and `mapping().required_span_size() == 5`.

## Rules That Differ From the Standard

1. **`stride(r)` exists at every rank** in every strided Uni20 mapping, in
   `mdspan` and in `mdspec`, with precondition `r < rank()`. At rank zero no axis meets it, so a
   call fails the precondition, while a loop over the axes compiles and never
   calls it.
2. **Offsets have the accessor's type.** `mdspan` converts the mapping result to
   `span_offset_t<Accessor>`, which is the accessor's `offset_type` when it
   declares one and `size_t` otherwise. A standard accessor therefore still
   requires nonnegative offsets. Every reachable offset must fit the paired
   accessor. View construction establishes this once when the types alone
   cannot guarantee it.
3. **Every mapping reports `reachable_range()`**, the offsets that some valid
   index reaches. For nonnegative strides it is `[0, required_span_size() - 1]`,
   or empty when an extent is zero. Buffer validation compares the descriptor
   offset plus this range with the buffer, instead of `required_span_size()`.
   `required_span_size()` remains for nonnegative layouts, as the standard's
   layout-mapping requirements need it. The rule covers a mapping that returns
   one offset, read by an accessor that uses it as an element offset into one
   buffer: the layouts in the Types table, with `default_accessor` or the CUDA
   pointer accessors. The zip layouts are outside it. `StridedZipLayout` and
   `GeneralZipLayout` return one offset per span, so they are Uni20 mappings
   rather than conforming layout mappings, and they should not claim to be:
   they provide neither `required_span_size()` nor `reachable_range()`. They
   work with the accessors and operations written for them, and anything that
   needs those queries, such as buffer validation or conversion to the
   standard, does not accept them. PR #152 removed the conformance claims and
   the zip span-size query. A translating accessor is outside the rule too: `diagonal_accessor`
   maps a logical offset to a stored component, so element `(2, 2)` of a 3×3
   diagonal is mapping offset 8 but component 2, and validating its buffer is
   left to the accessor.
4. **Strided means strided.** `StridedMdspanLike` and `StridedMdspecLike` require
   `is_always_strided()` and a declared `stride(axis)`, with no rank-zero escape.
   Explicit import of a standard rank-zero mapping preserves the scalar shape
   without querying a stride.
5. **Negative strides are a layout, not a runtime state.** Only
   `layout_signed_stride` may have negative strides. An operation that cannot
   handle them rejects the layout by type at compile time, so a reversed view
   never reaches a backend that would decline it at run time and end the
   program. Operations opt in one at a time. The strided concepts carry the
   distinction: `layout_signed_stride` declares that its strides may be
   negative, `StridedMdspanLike` and `StridedMdspecLike` reject a layout that
   does, and an operation that opts in is constrained on a separate concept
   that admits both. Without that declaration a structural strided concept
   would accept the signed layout, since it is always strided. The same holds for composition:
   an `mdspan` or `mdspec` with `layout_signed_stride` requires a signed
   `index_type` and an accessor whose `span_offset_t` is signed, so a signed
   layout paired with `stdex::default_accessor`, or any accessor that falls back
   to `size_t`, does not compile. Uni20's `index_type` is already
   `std::ptrdiff_t` ([`core/types.hpp`](../../src/uni20/core/types.hpp)), and
   Tensor extents use it.
6. **Empty mappings constrain their strides less.** The standard requires
   `layout_stride` strides to be positive and to admit an axis order in which
   no two axes overlap, even when the mapping has no elements. Uni20's
   `layout_stride` accepts any nonnegative strides on an empty mapping. On a
   nonempty mapping a singleton axis may also have stride zero because its
   only index is zero. The remaining axes retain the nonoverlap requirement.
   Nonnegative strides include the values canonical
   layouts already produce for empty tensors (`layout_left` with extents
   `(0, 3, 4)` has strides `(1, 0, 0)`), so code that reads strides already
   decides emptiness from the extents, and negative values stay excluded. A
   permutation of an empty source then permutes the source's strides like any
   other, with no new strides to invent. This follows LWG 4603, applied to
   every constructor, including conversion from a canonical layout. P3959R0, a
   related proposal, is narrower: it relaxes positivity but keeps the axis-order
   condition at every nonzero rank, so it rejects strides `(1, 1, 1)` for
   extents `(0, 3, 4)`, which rule 6 accepts.
7. **The nonzero extents have a representable product.** The standard requires
   only the number of elements to be representable, which says nothing about an
   empty shape such as `(0, 2^32, 2^32)`, whose `layout_right` strides overflow.
   Uni20's mappings require the product of the nonzero extents to be
   representable in `index_type`. That keeps every canonical stride
   representable under every permutation of the extents, so materializing a
   permuted view, as `Tensor(permute(view))` from #137 does, cannot turn a valid
   empty tensor into an invalid one. A condition on each stride would not be
   enough, because it depends on the order of the extents. Uni20 algorithms
   can assume this invariant once a mapping has been constructed. An explicit
   import checks it when the source contract leaves it open. No LWG issue covers this for `layout_left` and
   `layout_right`. Rule 7 bounds every canonical stride, but not a stride that
   a mapping stores: on an axis of extent zero or one, a `layout_stride` or
   padded stride adds nothing to the product or to `required_span_size()`.
   Converting such a mapping to a narrower `index_type` therefore also requires
   each stride to be representable in the destination, as LWG 4545 proposes for
   the padded layouts, where a padding of 256 with extents `(1, 0)` would
   otherwise narrow to zero in `uint8_t`.

### Alignment With the Standard

Uni20 adopts these library issue resolutions, so its layouts behave as the
standard is expected to:

| Issue | Status (2026-11-18) | Adopted rule |
|---|---|---|
| LWG 4603 | Voting | Rule 6: an empty mapping accepts nonnegative strides. It also drops the converting constructor's precondition that every stride of the source is positive. |
| LWG 4606 | Voting | For a nonempty `layout_stride`, every stride is nonnegative and representable, and axes of extent one impose no ordering constraint. Only the axes of extent greater than one need an order in which none overlaps. A strided slice such as every other column of a `(2, 5)` row-major matrix, extents `(2, 3)` with strides `(5, 2)`, is valid. |
| LWG 4266 | C++26 | An empty mapping is exhaustive; `is_always_exhaustive()` is true when the rank is zero or a static extent is zero. |
| LWG 4562 | Voting | `is_exhaustive()` is `required_span_size()` equal to the product of the extents, so axes of extent one do not affect it. |
| LWG 4272 | C++26 | Conversions between rank-zero mappings of different layouts are implicit whenever the extents convert implicitly. |
| LWG 4217 | C++26 | Not adopted: rule 1 provides `stride()` at every rank instead of letting rank-zero mappings omit it. |

### Signed-stride Semantics

`layout_signed_stride` keeps the handle at the element with all indices zero,
as NumPy does, so offsets below the handle are negative. `operator()` is
`sum(index[k] * stride[k])`. `reachable_range()` is empty when an extent is
zero. Otherwise it starts at `[0, 0]` and adds `(extent[k] - 1) * stride[k]` to
the maximum for positive strides and to the minimum for negative ones. `is_unique()` and `is_exhaustive()` are evaluated on
the absolute strides, which describe the same set of offsets shifted by
`minimum`. `required_span_size()` is not provided; consumers use
`reachable_range()`.

At first `layout_signed_stride` is view-only. A Tensor that owns storage keeps
a nonnegative layout, and a signed layout appears only in a view of storage it
does not own, so no allocation has to hold its logical origin away from its
start. `reverse` of an axis moves the handle to that axis's last element and
negates its stride, but only when the mapping is nonempty. An empty view keeps
its handle, which may not point into any storage, so moving it would be
undefined.

Negative strides are useful in only a few cases, reversing the elements of an
axis being the main one, and then only over operations where they are easy or
cheap to support. Operations therefore accept `layout_signed_stride` as uses
require, not in advance. The likely first are a one-axis `reverse` view and
host `copy`. Host elementwise transforms and reductions cost little, because the
backend-neutral iteration plan already normalizes negative strides. CUDA plan
lowering would rebase the descriptor to `minimum` when a use needs it. BLAS and
LAPACK do not accept it: BLAS's negative-increment convention is different, and
LAPACK leading dimensions must be positive.

Uni20's C++ API keeps temporaries explicit, within reason: an operation that
does not accept an operand's layout rejects it rather than copying it. Copies
that make a call succeed belong to layers built on Uni20. Python bindings would
copy a negatively strided NumPy array that an operation does not accept, as
NumPy itself copies before calling BLAS. A convenience C++ library, in the
style of MATLAB, could make the same trade for code where the answer matters
more than the cost. Rule 5 makes such a layer simple: acceptance is a
compile-time property, so the layer can test it with a concept and insert the
copy only where it is needed, without catching run-time declines.

### Rejected Alternatives

Allowing negative strides in `layout_stride` with runtime checks would keep one
strided type, but every backend that cannot handle them would decline at run
time, which rule 5 exists to prevent.

A sign policy parameter on `layout_stride` would also distinguish the cases by
type, but every existing use and comparison of `layout_stride` would then have
to say which policy it means. A separate layout leaves them meaning the
nonnegative one. The two layouts may still share one implementation templated
on the sign; what is rejected is a policy parameter in the public type. Decided
tentatively on 2026-11-20: the signed layout is
`layout_signed_stride`.

## Conversions and Interop

- **Supported views.** `MdspanLike` admits Uni20 views with mappings and
  accessors whose semantics Uni20 supports. An external mdspan enters a Uni20
  algorithm through explicit import. Initially import the standard left,
  right and strided layouts with the default accessor. Add other combinations
  when their offset and access semantics have a Uni20 implementation.
- **Import.** Use Uni20's usual `std::ptrdiff_t` indices and physical offsets.
  Smaller source index types are welcome. Before casting, check any extent,
  stored stride, or reachable offset whose representability the source contract
  does not guarantee. This includes unsigned-to-signed conversion of equal
  width and stored strides on singleton axes. A single element with stride 256
  needs one storage element, but retaining that stride in eight bits is
  impossible. Report a failed conversion at construction in release builds too.
- **Composition.** Each operand keeps its own physical offsets when views are
  combined. A factory that changes their representation must establish that
  both the strides and the reachable offsets fit. For example, strides 200
  and 100 each fit in eight bits, while offset 300 at `[1,1]` does not.
  Element access can rely on the established representation. CUDA kernels
  may select a narrower execution plan during backend lowering.
- **Export.** Explicit export preserves the mapping and accessor semantics.
  Check the destination layout's preconditions before constructing it. For
  example, C++23 `layout_stride` requires positive strides. A Uni20
  mapping with a zero stride on an empty or singleton axis reports an export
  failure for that destination. The same rule applies to negative strides.

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
  `mapping().reachable_range()` against the buffer, for the mappings and
  accessors rule 3 covers. Today
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

Each phase is one pull request. PR #152 already addressed the zip layouts'
conformance claims (rule 3).

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
   needed. Tests that assert a producer's exact layout type, in
   `test_cuda_copy.cpp` and `test_self_adjoint_eigh.cpp`, change with the
   producer. CUDA `ScalarTensor` transfers, copies and reductions start to compile
   here, with the rank-zero CUDA tests from #148. When #137's `permute`
   returns a Uni20 layout, permute the strides and extents together. Remove
   the empty-stride reconstruction and the blanket positive-stride check.
   Test an empty permutation and a nonempty transpose from shape `(1,3)`,
   strides `(0,1)`, to shape `(3,1)`, strides `(1,0)`. The latter visits
   the same three values. Exercise both mdspan and mdspec.
3. **Tests and examples.** `stdex::` names become Uni20 names, about 620
   references, mostly mechanical. Interop tests import standard mdspans
   explicitly and exercise export failure for an unrepresentable mapping.
   Migrate tests that need negative strides together with the signed layout.
4. **Signed strides.** `layout_signed_stride`, a `reverse` view, and support in
   the operations its first use needs, host `copy` for example; other
   operations follow as uses require them. The diagonal component view moves to
   `layout_signed_stride`, ending its reliance on wrapped `size_t` offsets, and
   the negative-stride test helpers move with it. The remaining code that
   computes signed offsets through unsigned arithmetic is found and fixed here,
   and composite producers, such as zip and transform views over a signed
   span, and accessor lowering are checked to keep the signed declaration
   visible to the concepts. Conformance tests include an empty mapping with
   mixed signs.

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
2. NumPy interop: whether the Python bindings pass negatively strided arrays as
   `layout_signed_stride` views to the operations that accept them, or always
   copy them at the boundary.
