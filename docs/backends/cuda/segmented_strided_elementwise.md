# Segmented Strided Elementwise Execution

**Status:** active design note. Nothing described here is implemented. The
single-view strided elementwise plan and the whole-buffer partitioned fast path
that this note generalizes are implemented.

This note defines a dense primitive that applies one elementwise operation to a
list of strided view tuples of different shapes and ranks in a single execution
node and, on CUDA, a single kernel launch. Block-sparse operations lower to it
whenever their operands do not share one packed layout.

Related notes:

- [Raw Primitives and Symmetric Lowering](../../symmetry/raw_primitives_and_lowering.md)
  — why this belongs to the dense layer, not the symmetric layer.
- [Block-Sparse Tensors and Layout](../../symmetry/block_sparse_tensor.md) — blocks
  as strided views into shared buffers.
- [Block Coalescing and GEMM Grouping](../../symmetry/block_coalescing.md) — the
  pack step this primitive provides.
- [CUDA Buffer Completion Lowering](epoch_design_draft.md) — access and
  completion contract for the buffers a segmented node touches.
- [CUDA Kernel Dispatch and Device Scheduling](kernel_dispatch.md)

## Problem

The CUDA reference backend already has a good kernel for one strided view
tuple. `StridedElementwisePlan`
([`elementwise_plan.hpp`](../../../src/uni20/linalg/backends/cuda/elementwise_plan.hpp))
compacts the rank to at most eight, lowers to 32-bit indices and offsets when
they fit, is passed to the kernel by value, and maps one logical element to one
thread in a grid-stride loop with the innermost dimension first. The kernel
([`elementwise_kernel.cuh`](../../../src/uni20/linalg/backends/cuda/elementwise_kernel.cuh))
is arity-generic, so copy, add, axpy, scale and conjugation share one executor.

Block tensors do not get this benefit except in one case. The operations in
[`block_tensor_linear.hpp`](../../../src/uni20/symmetry/block_tensor_linear.hpp)
take the whole-buffer `cuda_partitioned_*` path only when
`same_packed_layout(output, input)` holds: identical stored keys, identical
block offsets and identical buffer size. Every other case loops over
`stored_block_count()` on the host and issues one `transform_inplace` per block,
which is one kernel launch plus one round of buffer-ledger traffic per block.
That fallback is reached by:

- assignment from a zero-copy `permute` or `repartition` view, whose blocks do
  not share the output's packed layout;
- `add`, `axpy` and assignment when the output stores a superset of the input
  keys;
- truncation, which keeps a leading range of each block;
- packing blocks into a contiguous operand for a coalesced or grouped GEMM, and
  scattering the result back;
- assembling each SVD sector matrix from its block fragments, one transposed
  copy per fragment, in
  [`block_tensor_svd.hpp`](../../../src/uni20/symmetry/block_tensor_svd.hpp).

Block sizes in DMRG are ragged. One tensor can hold 1×1 blocks next to
1000×1000 blocks, and the calibrated R/A/B/C data shows a launch-bound tail of
near-zero-flop blocks that does not shrink with bond dimension. One launch per
block puts the launch cost on the tail.

## The Primitive

The operation is

```text
for each segment s, for each logical index i in segment s:
  output_s[i] = op(inputs_s[i]...)           // overwrite
  output_s[i] = op(output_s[i], inputs_s[i]...)  // update
```

where each segment is a tuple of dense strided views with the same extents, and
different segments may have different extents and ranks. The primitive knows
nothing about quantum numbers, keys, or fusion rules. The symmetric layer turns
key matching, permutation, repartition, truncation and pack/scatter maps into a
segment list. This matches the rule in
[Raw Primitives and Symmetric Lowering](../../symmetry/raw_primitives_and_lowering.md):
a list of strided views with one elementwise operation is a useful operation on
raw tensor data in its own right.

The single-view plan is the one-segment case. The operation set does not
change; only the step from a logical index to per-operand offsets does.

## Plan Representation

```cpp
template <std::size_t OperandCount, std::size_t MaximumRank, class LogicalIndex, class Offset>
struct SegmentDescriptor
{
    LogicalIndex element_count = 0;
    std::uint8_t compact_rank = 0;
    std::uint8_t vector_width = 1;
    std::array<LogicalIndex, MaximumRank> extents{};
    std::array<std::array<Offset, OperandCount>, MaximumRank> strides{};
    std::array<Offset, OperandCount> base_offsets{};
};
```

A segmented plan holds:

| Field | Meaning |
|---|---|
| `segments[]` | one descriptor per view tuple |
| `segment_begin[]` | exclusive prefix sum of `element_count`; `segment_begin[S]` is the total |
| `chunk_first_segment[]` | optional, the first segment overlapping each fixed-size chunk |
| index kind | 32-bit when the total element count and every reachable offset fit, otherwise 64-bit |

Rules:

- **Per-segment compaction.** Each segment is compacted independently with the
  existing single-view lowering. A block with an extent-1 leg compacts to a
  lower rank than its neighbours. Empty segments are dropped at plan time.
- **Offsets, not pointers.** Every operand is one buffer, and descriptors carry
  element offsets into it. Packed block storage
  ([`block_tensor_storage.hpp`](../../../src/uni20/symmetry/block_tensor_storage.hpp))
  already places all blocks in one buffer, so its `offsets()` become
  `base_offsets` directly. Offsets keep the 32-bit lowering available and the
  descriptors small.
- **Index kind is per plan.** One plan uses one index kind for all segments, so
  the kernel is not branching on it per segment.
- **Vector width is per segment.** A segment may use vector loads and stores of
  width `w` when every operand has unit innermost stride, the innermost extent
  is a multiple of `w`, and every base offset is a multiple of `w`. Packed
  storage's `BlockAlignment` makes this the common case for block starts. The
  width is decided at plan time and recorded in the descriptor.
- **Padding is never touched.** Segments cover block payloads only. The
  zero-initialized alignment padding between packed blocks stays zero, so the
  segmented path does not need the `has_padding() || isfinite(factor)` guard
  that the whole-buffer path needs.
- **Positive strides only**, as in the single-view plan. Negative-stride views
  need descriptor rebasing and signed traversal and are out of scope here.

A descriptor at rank eight with two operands is about 220 bytes with 64-bit
indices and half that with 32-bit indices. Ten thousand blocks is about 2 MB of
descriptors, which is too large for kernel parameters, so descriptors live in a
device buffer. The scalar plan header (counts, chunk size, index kind) stays a
kernel parameter.

## CUDA Execution

### Load-balanced chunks

The scheduling unit is a fixed-size chunk of `K` logical elements, not a
segment. Thread block `c` owns logical elements `[c*K, (c+1)*K)` of the
concatenated segment space.

1. Find the first overlapping segment: read `chunk_first_segment[c]`, or
   binary-search `segment_begin` once per thread block.
2. Load that segment's descriptor into shared memory.
3. Threads stride over the part of the chunk that lies in this segment,
   decoding each local index with the segment's extents and strides exactly as
   the single-view plan does.
4. Synchronize, advance to the next segment, and repeat until the chunk ends.

A large block is split across many thread blocks and many tiny blocks share
one thread block, so load balance depends only on the total element count. This
is the standard load-balanced segmented pattern used for CSR SpMV and
moderngpu's `load_balance_search`. The search cost is paid once per thread
block, not once per element.

`chunk_first_segment[]` is cheap to build on the host when the plan is built and
removes the device-side search. Whether it beats a shared-memory binary search
is a measurement question.

Chunks that cross many tiny segments pay a shared-memory reload and a barrier
per segment. If the tail dominates, a warp-per-segment variant for segments
below a threshold is a tuning option, not a v1 requirement.

### Segment classes

Copying a block between layouts with different innermost legs is a transpose.
One-element-per-thread mapping makes only one side coalesced. Segments are
therefore classified at plan time:

| Class | Condition | Execution | Work unit |
|---|---|---|---|
| A | all operands share the innermost leg | element chunks, vectorized | elements |
| B | permuted, both innermost extents at least about 16 | shared-memory tiled transpose | tiles |
| C | permuted and small | element chunks, one side uncoalesced | elements |

Class B uses its own prefix sum over per-segment tile counts. Classes A and C
use the same kernel. A launch handles one class, so an operation costs at most
two or three launches regardless of block count; a single launch with tagged
work items is an alternative.

Class C blocks are small enough that their uncoalesced side mostly hits L2, so
the lost bandwidth is small in absolute terms. Whether class B is worth its own
kernel, and where the threshold is, must be decided by benchmarks on real DMRG
block-size distributions, not assumed.

### Index lists

Truncation keeps a leading range of each block, which is affine. It needs no
index tables. Arbitrary per-axis index selection, as in NumPy-style advanced
indexing, would extend a dimension of one operand from `coordinate * stride` to
`table[coordinate]`. This is deferred until a caller needs it. A dimension
backed by a table cannot be merged with its neighbours, so compaction must run
before tables are attached.

## Plans, Caching, and the Async Runtime

**A plan depends only on structure.** The segment list is a function of the
operands' stored keys, sector dimensions, packed offsets, and the view mapping
(permutation, repartition, truncation ranges). It does not depend on numerical
data. Within a DMRG sweep and across repeated R/A/B/C applications the
structure is stable, so a plan can be built once, its descriptors uploaded
once, and the plan reused. Stable kernel arguments also make the launches
suitable for CUDA graph capture.

**Descriptor upload is an ordinary async operation.** The descriptor buffer is
written by a host-to-device copy on the same stream and is an input of the
segmented node. Its lifetime is pinned by the completion token under the
two-clocks rule in
[Ordering and Backend Lowering](../../architecture/ordering_and_backend_lowering.md),
like any other operand.

**Structure changes build a new plan.** Bond-dimension growth or truncation
changes sector dimensions and therefore the plan. Building one is O(blocks)
host work plus one small copy.

**One node per operation.** A segmented operation is one DAG node with one
access per operand buffer. This matches the buffer-granularity hazard decision
in [Block Coalescing and GEMM Grouping](../../symmetry/block_coalescing.md) and
replaces the per-block ledger traffic of the current fallback, which may cost
as much as the launches. The existing precondition that distinct operands do
not destructively overlap covers aliasing.

**Multi-device.** The layout maps blocks to devices. Plans are built per device
and per operand buffer set. A segment whose operands live on different devices
is a transfer, handled outside the kernel.

**Output blocks with no input.** When an output stores keys the input lacks,
those output blocks are not segments of the binary operation. They are a
separate zero-input fill over their own segment list, issued before the main
node or as a second node on the same stream.

## Host Execution

The segment list is backend-neutral, like `multi_iteration_plan`, and is
lowered to the device descriptor format only for CUDA. The host executor can
run the same chunked schedule under oneTBB. That also fixes the host
load-balance problem of parallelizing over blocks when one block is much larger
than the rest.

## Rejected Alternatives

| Alternative | Why not |
|---|---|
| One launch per block (current fallback) | launch and ledger cost on every block, dominated by the tail |
| One thread per segment, looping over its elements | serial work on large blocks and strided, uncoalesced access across threads |
| Device-side child launches or `cudaMemcpyAsync` per segment from a kernel | bounded by the pending-launch limit; Cytnx shipped this and it silently dropped copies and hung (Cytnx-dev/Cytnx#1229) |
| Padding blocks to a common shape for a fixed-shape batched kernel | wastes bandwidth on ragged sizes; fixed-shape batching remains right for groups that are already same-shape |
| `cudaMemcpyBatchAsync` (CUDA 12.8 and later) | handles only contiguous byte ranges, not strided or permuted views or arithmetic; possibly useful for a few very large contiguous segments, where the element kernel already saturates bandwidth |

## Testing and Benchmarks

Correctness tests compare each segmented operation against a host reference that
loops over the segments and their mdspans directly. They should cover:

- one segment, which must agree with the single-view plan;
- empty segments, extent-1 legs, and segments of differing compact rank;
- one very large segment beside many tiny ones;
- chunks that start and end mid-segment, and segments smaller than one chunk;
- the 32-bit/64-bit index boundary;
- every segment class, including permutations that land in class C;
- packed storage with padding, checking that padding stays zero;
- overwrite and update operations, real and complex.

Benchmarks compare the segmented path with the per-block fallback on block-size
distributions taken from real DMRG runs, using the calibrated R/A/B/C data where
possible. Performance claims about class B, the chunk size, or the
warp-per-segment variant need those measurements.

## Proposed First Implementation

1. The backend-neutral segment list and its CUDA lowering, offset-only.
2. One kernel serving classes A and C, with the load-balanced chunk schedule.
3. Wire it into the per-block fallback loops for copy-from-view, `add`, `axpy`
   and assignment in `block_tensor_linear.hpp`.

Class B, index tables, plan caching across operations, and the host chunked
executor follow as separate changes.

## Open Questions

- **Separate storage on CUDA.** Offset-only descriptors require each operand
  to be one buffer. Should `SeparateSparseBlockStorageData` tensors decline on
  CUDA, or should descriptors also have a pointer form? Declining is simpler and
  matches the packed layout the CUDA path already prefers.
- **Class B timing.** Implement the tiled transpose class in the first change,
  or only after benchmarks show the class C fallback is a bottleneck?
- **Plan ownership.** Should cached plans live on the block tensor's structure
  object, or be owned by the lowering and prepared-contraction plan that issues
  the operation? The second fits prepared, repeated R/A/B/C contractions
  better.
- **Fill segments.** Is a separate zero-input fill node the right treatment of
  output-only keys, or should a segmented plan carry segments of different
  arities?
