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
tuple-valued mapping results for zip, and device-callable views.
`uni20::mdspec` uses the same mapping and accessor vocabulary before the
data handle is acquired.

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

**Mapping results.** The standard permits a custom layout with signed strides
and nonnegative scalar offsets. The built-in `std::layout_stride` only covers
forward strides. With standard extents, `std::mdspan` can use
`uni20::layout_signed_stride` with its default accessor. Uni20 also
supports zip mappings, which return a tuple with
one offset per operand. Indexing passes the mapping result directly to the
accessor. The view exposes its type as `offset_type`.

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
| `default_accessor<T>` | Accepts a `std::size_t` element offset. `access(p, i)` returns `p[i]` and `offset(p, i)` returns `p + i`. |
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
2. **Both views name the mapping result.** `mdspan::offset_type` and
   `mdspec::offset_type` denote the type returned by the stored mapping called
   with one `index_type` per axis. The accessor must accept every reachable
   result. A mapping may return a move-only value. Indexing forwards it to
   the accessor. When the accessor accepts a const reference, a caller can
   save the result in `auto n = mapping(indices...)` and reuse it with
   `accessor.access(handle, n)`. Accessors may omit the alias. An affine mapping
   returns an integer. A zip mapping returns one offset per operand and
   has no single storage span.
3. **Storage bounds begin at the handle.** For a mapping whose offsets
   address elements in one buffer, `required_span_size()` includes any gaps.
   An empty mapping has span size zero. With a translating accessor the
   physical buffer size is determined by the accessor. For example, offset
   eight in the logical mapping of a 3×3 compressed diagonal addresses its
   third stored component.
4. **Affine layouts expose a stride per axis.** `AffineMdspanLike` and
   `AffineMdspecLike` require integral mapping results, constant differences
   between neighbouring coordinates, and a stride observer at every rank.
   A nonempty affine view's logical origin is `mapping(0, …, 0)`. Forward
   layouts guarantee zero origin and nonnegative strides by construction.
   `StridedMdspanLike` and `StridedMdspecLike` select those layouts for
   operations that reconstruct offsets from strides alone. Other affine
   layouts retain their full mappings when composed in zip.
5. **Signed strides have a dedicated layout.** `layout_signed_stride` uses a
   signed `index_type` for strides, mapped offsets and the enclosing span
   size. The constructor checks that the full span fits. Select a wider index
   type for a wider span. The default tensor index type is `std::ptrdiff_t`.
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

The handle addresses the lowest reachable element of a nonempty signed-stride
view. The mapping stores the displacement of its logical origin from there.
For a coordinate `i` the offset is

```text
origin = sum((extent[k] - 1) * abs(stride[k]) for each negative stride)
offset = origin + sum(i[k] * stride[k])
```

For a 3×4 matrix with reversed columns, the strides are `(4, -1)` and the
mapping returns `3 + 4*i - j`. The span size is 12. With row pitch 6 the
same view spans 16 elements including the padding. Rank zero has origin zero
and span size one. An empty view has no logical origin to evaluate.

`reverse<Axis>(span)` borrows a native pointer-backed view with signed
indices. It preserves the storage handle and negates the selected stride
when that axis varies. Reversing twice restores the original offsets.
Empty and singleton axes retain their strides. Owning tensors and direct
BLAS/LAPACK projections use forward layouts. CPU elementwise iteration and
CUDA random fill can normalize signed strides. Dense random fill selects
the element position `origin_argument + mapping(indices...)` on both backends.

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
- **Strides at every rank (rules 1 and 4).** Native strided mappings provide
  `mdspec::stride(r)` at every rank with the precondition `r < rank()`.
  `StridedMdspecLike` and `StridedMdspanLike` use the same requirement.
- **Offsets (rules 2 and 5).** The mapping determines `S::offset_type` before
  acquisition. The resolved accessor accepts that mapping result.
- **Storage span (rule 3).** [CUDA acquisition](mdspec.md#cuda-vertical-slice)
  checks that `required_span_size()` elements fit starting at the descriptor's
  element offset for a storage-preserving accessor. A translating accessor
  can provide its own physical range. An empty view permits a one-past handle.
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

[Issue #173](https://github.com/Uni20-dev/uni20/issues/173) records the migration
from Kokkos to the native views. [Issue #219](https://github.com/Uni20-dev/uni20/issues/219)
revises the signed-stride representation and the accessor requirements.

## Relation to #148

Pull request #148 kept the standard types, added a free `uni20::stride` that
exists at every rank, and gave the strided concepts a rank-zero escape. Uni20's
own layouts make both unnecessary, so that approach is superseded.

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
