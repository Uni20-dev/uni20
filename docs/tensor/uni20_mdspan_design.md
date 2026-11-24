# Uni20 mdspan and mdspec Design

Tensor producers and CUDA acquisition use Uni20 views. CPU copy and
elementwise transforms support native signed-stride views.
[`mdspan/extents.hpp`](../../src/uni20/mdspan/extents.hpp) supplies
`uni20::extents` and `uni20::dextents`. CMake selects the standard-library
types when the host compiler can use them. A CUDA build also requires a device
compilation of mixed static and dynamic extents before selecting them.
Otherwise the entire build uses Uni20's fallback, so C++ and CUDA translation
units agree on the types of tensors and descriptors.
Both choices use signed or unsigned integer index types, including signed and
unsigned char.

Uni20 owns its views and mappings to provide stride observers at every rank,
signed offsets, and storage-span queries. `uni20::mdspec` uses the same
mapping and accessor vocabulary before the data handle is acquired.

## Why Not the Standard Types

**Rank-zero strides.** [mdspan.layout.reqmts] note 5 says "It is not required
for `m.stride(r)` to be well-formed if `m.extents().rank()` is zero, even if
`m.is_always_strided()` is true." `layout_left` and `layout_right` constrain
`stride` with `extents_type::rank() > 0`; `layout_stride` does not. A
constrained-away member cannot be named, even in a loop over the axes that never
runs at rank zero. `std::mdspan::stride` is declared without a constraint but
forwards to the mapping, so a `requires` check reports it present, although calling
it fails to compile. With the standard layouts, Uni20's strided concepts would
need a rank-zero escape and generic code a free stride helper; pull request #148
tried that and is superseded by this design.

**Offsets.** The standard accessor policy takes a `size_t` offset, and
`layout_stride` requires positive strides. `required_span_size`,
`is_exhaustive` and `is_unique` assume nonnegative strides. Each Uni20 accessor
declares its own `offset_type`, and `mdspan` passes the mapped offset to `access`
in that type. Host tensors and CUDA pointer accessors use `std::ptrdiff_t`.
The diagonal and generated accessors retain their own offset types. Const,
conjugating and unary transform adaptors inherit the wrapped accessor's offset
type. Multi-operand transform accessors keep an offset type for each operand.

The iteration plan computes in `std::ptrdiff_t` and normalizes negative
strides. A diagonal can borrow values in reverse order through
`layout_signed_stride` and the signed default accessor. Forward component
views keep `layout_stride`.

**Control.** Owning the types lets every mapping and mdspan member be
`UNI20_HOST_DEVICE`, report precondition failures through Uni20 diagnostics,
and grow layouts the standard does not have.

### Non-goals

- `submdspan`. Uni20 does not use it.
- New extents semantics. Use standard extents when available and a fallback
  with the same interface on older toolchains. [Issue #171](https://github.com/Uni20-dev/uni20/issues/171)
  tracks selecting one implementation for the host and CUDA translation units.

C++26's `layout_left_padded` and `layout_right_padded` are deferred. They
encode a unit inner stride in the layout type, which may be useful for BLAS
and LAPACK.

## Use Before the Migration

Counts are occurrences over `src`, `tests` and `examples` on `main`
(`33439da6`), from this command for each name:

```sh
git grep -ohP 'stdex::NAME\b(?!_)' 33439da6 -- src tests examples | wc -l
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

Fifteen files in that revision of `src` name `stdex::mdspan`; those are where the type crosses
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
| `extents`, `dextents`, `dynamic_extent` | Standard extents semantics. Use `std::dynamic_extent` from `<span>` in either configuration. |
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
2. **Both views expose the accessor's offset type.** Native `mdspan` and
   `mdspec` declare `using offset_type = typename accessor_type::offset_type;`.
   Generic code names `typename S::offset_type`. The accessor declares the
   argument type for `access` and `offset`, and the mdspan passes the mapping
   result in that type. The default accessor uses `std::ptrdiff_t`. A transform
   that forwards the offset preserves its wrapped accessor's type, and a zip
   accessor can take a tuple of offsets. Pairing a mapping with an accessor
   establishes that every reachable offset fits whenever the types alone
   cannot guarantee it.
3. **Single-buffer mappings with element offsets describe their storage span.**
   `required_span_size()` is the number of elements in that span, including
   padding. `base_offset()` is the displacement from the start of that span to
   the data handle. A native affine mapping returns zero for the all-zero
   coordinate, so the handle addresses the logical origin. Both storage queries
   return zero for an empty mapping. The built-in nonnegative layouts declare
   `is_base_offset_always_zero()`. A legacy mapping
   exposing only `required_span_size()` describes a span starting at its handle.
   The rule covers a mapping that returns
   one offset, read by an accessor that uses it as an element offset into one
   buffer. This includes paired-axis diagonal views and accessors that forward
   the offset unchanged, such as conjugation over a CUDA pointer accessor.
   The zip layouts are outside it. `StridedZipLayout` and
   `GeneralZipLayout` return one offset per span, so they are Uni20 mappings
   rather than conforming layout mappings, and they should not claim to be:
   they provide neither `required_span_size()` nor `base_offset()`. They
   work with the accessors and operations written for them, and anything that
   needs those queries, such as buffer validation or conversion to the
   standard, does not accept them. PR #152 removed the conformance claims and
   the zip span-size query. A translating accessor is outside the rule too: `diagonal_accessor`
   maps a logical offset to a stored component, so element `(2, 2)` of a 3×3
   diagonal is mapping offset 8 but component 2, and validating its buffer is
   left to the accessor.
4. **Affine layouts expose a stride per axis.** `AffineMdspanLike` and
   `AffineMdspecLike` require `is_always_strided()` and a declared
   `stride(axis)` at every rank.
   Explicit import of a standard rank-zero mapping preserves the scalar shape
   without querying a stride.
5. **Signed strides have a dedicated layout.** `AffineMdspanLike` and
   `AffineMdspecLike` admit `layout_signed_stride`.
   `StridedMdspanLike` and `StridedMdspecLike` additionally require a statically
   zero base offset. For native affine layouts, that describes nonnegative
   element strides on the varying axes. Both layout families can have a signed
   `index_type`. An `mdspan` or `mdspec` with
   `layout_signed_stride` requires a signed
   `index_type` and a signed `offset_type`. An accessor taking `size_t`
   therefore cannot be paired with this layout. Uni20's `index_type` is already
   `std::ptrdiff_t` ([`core/types.hpp`](../../src/uni20/core/types.hpp)), and
   the default Tensor extents use it. Explicit Tensor extents can use another
   index type.
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

   A paired-axis diagonal of a source satisfying LWG 4606 also fits Uni20
   `layout_stride`. Order the surviving axes and pairs by their last varying
   source axis in the source's physical stride order. A pair's stride is the
   sum of its source strides. Its maximum offset contribution is at most the
   sum of the original two contributions, since its extent is their minimum.
   Each new stride therefore still exceeds the total contribution of the
   preceding groups. For example, the diagonal of axes 0 and 1 of a
   column-major `(2,2,3)` tensor has shape `(3,2)` and strides `(4,3)`.
   The second axis contributes at most 3, so a step of 4 on the first suffices.
   Use the native strided layout for these sources when migrating
   `diagonal_view`. Today's `detail::DiagonalLayout` also serves custom
   strided sources whose only guarantee is uniqueness. That guarantee alone
   does not establish the LWG ordering.
7. **The nonzero extents have a representable product.** The standard requires
   only the number of elements to be representable, which says nothing about an
   empty shape such as `(0, 2^32, 2^32)`, whose `layout_right` strides overflow.
   Uni20's mappings require the product of the nonzero extents to be
   representable in `index_type`. Native shape factories such as `zeros()`
   take this as an unchecked caller precondition, including in debug builds.
   The bound keeps canonical strides representable under any permutation,
   including when materializing `Tensor(permute(view))` as in #137. Uni20
   algorithms can assume this invariant. An explicit import checks it when the
   source contract leaves it open. No LWG issue covers this for `layout_left`
   and `layout_right`.

   An operation that derives a result shape establishes the same bound in the
   destination's `index_type` before constructing its mapping. For example,
   when a reduction writes into a narrower-index tensor, checking each retained
   extent individually is insufficient. Their nonzero product must also fit.
   A reduction retaining a subset of the axes in the same index type inherits
   the product bound from its input.

   Every stride a mapping exposes, stored or derived, must fit its
   `index_type`, including on empty and singleton axes. An import checks this
   when the source type admits strides outside the destination range.

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
`sum(index[k] * stride[k])`. The lowest element offset is the sum of
`(extent[k] - 1) * stride[k]` for axes with negative strides. Its magnitude is
`base_offset()`. The width from the lowest through the highest element offset,
inclusive, is `required_span_size()`. Both observers return `std::size_t`, so
the span may be wider than the mapping's signed index range. An empty mapping
returns zero for both queries, while a rank-zero mapping has base offset zero
and span size one. The constructor requires unique offsets, using increasing
absolute stride order. The mapping is exhaustive when the span size equals
the logical element count.

`layout_signed_stride` is view-only. An owning Tensor requires a mapping
whose storage span starts at the handle. `reverse<Axis>(span)` borrows from a native pointer-backed span
with a nonnegative layout and signed indices. For a nonempty view with more
than one element along that axis, it moves the handle to the last element
along the axis and negates the stride. Empty views and singleton axes
retain the original handle and strides.

Host copy and elementwise transforms use the affine iteration plan to
normalize negative strides. If any extent is zero, the plan returns before
normalizing strides. Precompiled CUDA kernels and BLAS and LAPACK lowering
require nonnegative layouts. BLAS's negative-increment convention and
LAPACK's positive leading dimensions each require separate consideration.

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

- **Foreign constructor.** `MdspanLike` admits Uni20 views with supported
  mapping and accessor semantics. To borrow a foreign view's storage, write
  `auto v = uni20::mdspan{foreign};`. This `explicit` constructor takes its
  argument by const reference. It requires an immediate view with the standard
  mdspan interface and a supported mapping and accessor, and excludes native
  `MdspanLike` types. A foreign type may declare its own `offset_type` and
  still qualify for import. Native views use their copy, move and converting
  constructors. A deferred `mdspec` goes through acquisition to obtain a handle.
- **Supported conversions.** Import standard
  `layout_left`, `layout_right` and `layout_stride` mappings paired with
  their default accessor. A deduction guide selects the corresponding Uni20
  layout and default accessor, preserves the element type (including const)
  and static extents, and uses Uni20's usual `std::ptrdiff_t` indices.
  Construction preserves the handle and strides. An explicitly named target
  must have compatible rank, extents and element qualifications. A pointer
  handle alone says nothing about a custom accessor's semantics.
- **Import checks.** Establish the destination layout's preconditions before
  narrowing. Check each extent, the product of the nonzero extents (rule 7),
  stored strides (including on singleton axes) and reachable offsets wherever
  the source contract leaves their representability open, including for
  same-width unsigned-to-signed conversions. For nonnegative layouts the
  required span size must also fit the destination `index_type`.
  For example, a 16-bit source with extent 2 and stride 127 reaches offsets
  0 and 127, but its required span size is 128. Import into an explicitly
  requested `int8_t` mapping therefore fails. These checks apply in release
  builds. The resulting native mapping carries that guarantee through
  subsequent Uni20 calls. Element access and kernel loops incur no import checks.
- **Composition.** Each operand keeps its own physical offsets when views are
  combined. A factory that changes their representation must establish that
  both the strides and the reachable offsets fit. For example, strides 200
  and 100 each fit in eight bits, while offset 300 at `[1,1]` does not.
  Element access can rely on the established representation. CUDA kernels
  may select a narrower execution plan during backend lowering.
- **Export.** Explicit export preserves the mapping and accessor semantics.
  Check the preconditions of the destination constructor. For example, the
  C++23 `layout_stride` constructor taking an array of strides requires
  positive strides and its prescribed axis order. A zero-stride empty mapping
  can still be exported if a standard construction, such as the default
  constructor, reproduces its extents and strides exactly. Reject the export
  if the destination cannot represent the mapping.

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
  `data_descriptor_type` and `data_descriptor()`. Its `size()` and `empty()`
  inspect the logical extents before acquisition.
- **Strides at every rank (rules 1 and 4).** `mdspec::stride(r)` today exists
  only where the mapping's does, so a rank-zero `layout_left` mdspec has none.
  With Uni20 layouts it exists at every rank with the precondition
  `r < rank()`, and `StridedMdspecLike` and `StridedMdspanLike` keep one
  definition.
- **Offsets (rules 2 and 5).** `S::offset_type` names the accessor offset on
  a deferred descriptor just as it does on an immediate view. Pairing the
  mapping and accessor establishes representability before a handle is
  acquired. For example,
  `layout_signed_stride` requires a signed accessor offset type in both
  `mdspec` and the resolved `mdspan`.
- **Storage span (rule 3).** Acquisition validates the descriptor's logical
  origin against the buffer. It subtracts `base_offset()` from the descriptor
  offset to locate the span and checks that `required_span_size()` elements
  fit from there. The acquired handle still addresses the descriptor's origin.
  An empty view permits a one-past origin.
- **Empty mappings and representable products (rules 6 and 7)** come with the
  mapping and need nothing separate.
- **Resolution.** Acquisition combines the leased handle, the mapping and the
  accessor for the acquired memory domain into a `uni20::mdspan`. For example,
  CUDA's `resolve_cuda_mdspan` lowers a conjugating accessor before
  constructing the view.
- **Concepts.** `MdspecLike` accepts a native `MdspanLike` as its immediate
  case. Passing a foreign mdspan to a descriptor-taking API goes through the
  same explicit import described above. Native immediate and deferred operands
  follow the same mapping rules, including rule 3's scope for range queries.

A conformance test checks, for each Uni20 layout, that `mdspec` and `mdspan`
expose the same observers with the same types and values.

## Migration

The steps below give a suggested order. Move producers and their callers
together as needed. [Issue #173](https://github.com/Uni20-dev/uni20/issues/173)
tracks the implementation. PR #152 already addressed the zip layouts'
conformance claims (rule 3).

1. **Types.** The mappings, `default_accessor`, `mdspan`, and conversions
   (including the explicit foreign constructor and deduction guide).
   Native layouts with element offsets expose `base_offset()` and
   `required_span_size()`. Both views expose
   `offset_type`, and `mdspec` gains `size()` and `empty()`. Conformance tests
   cover rank zero, signed accessor offsets, device compilation and agreement
   between the two views. Test
   direct construction from a supported foreign view, preservation of const
   and static extents, rejection of unsupported accessors, and native copying.
2. **Producers.** `RowMajor` and `ColumnMajor` alias the Uni20 layouts; Tensor
   `mdspan()`, `mdspec` mapping types, storage policies and the CUDA accessors
   produce Uni20 types, and acquisition resolves an `mdspec` to a
   `uni20::mdspan`. The CUDA pointer accessors declare `std::ptrdiff_t` offsets.
   Use the view's or accessor's nested offset type. Where zip composes
   mappings, use each component mapping's result type.
   The strided concepts require `stride()` at every rank. The layout comparisons
   use the Uni20 layouts. Tensors allocate the storage span at a zero base
   offset. Descriptor acquisition validates the span at its actual position
   within the buffer. Generic code calls the
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

   Register the new default accessor for host access, `Tensor::data()` and
   direct BLAS calls, including its const form. When BLAS probes an output
   before allocation or resizing, pass its prospective mapping and accessor
   as metadata. Construct the mdspan once the output has storage.
3. **Tests and examples.** Producers, tests and examples use Uni20 names.
   GCC 13 exercises fallback extents. The GCC 16 CI job exercises standard
   extents and explicit import, including rejection of unrepresentable views.
4. **Signed strides.** Add `layout_signed_stride` and `reverse`. Host copy
   and elementwise transforms use the affine iteration plan. Diagonal component
   views use the source component layout's zero-base-offset guarantee to
   choose between the nonnegative and signed layouts. Zip with a
   signed operand uses `GeneralZipLayout` to preserve each operand's mapping.
   The tests cover empty mappings with mixed strides and signed offsets through
   unary transform and zip accessors.

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

Since #149, consumers can embed Uni20 as `SYSTEM`. Uni20 exports its include
directories without `SYSTEM`, so
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
