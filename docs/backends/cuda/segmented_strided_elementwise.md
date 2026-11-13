# Segmented Strided Elementwise Execution

**Status:** active design note. Nothing described here is implemented. The
single-view strided elementwise plan and the allocation-wide partitioned paths
that this note generalizes are implemented.

This note defines a dense primitive that applies one elementwise operation to a
list of strided view tuples of different shapes and ranks in one CUDA
submission and, in the common case, one kernel launch. Block-sparse operations
lower to it whenever their operands do not share one packed layout.

Related notes:

- [Raw Primitives and Symmetric Lowering](../../symmetry/raw_primitives_and_lowering.md)
  — why this belongs to the dense layer, not the symmetric layer.
- [Block-Sparse Tensors and Layout](../../symmetry/block_sparse_tensor.md) — blocks
  as strided views into shared buffers.
- [Block Coalescing and GEMM Grouping](../../symmetry/block_coalescing.md) — the
  future pack and scatter steps this primitive serves.
- [CUDA Buffers](buffers.md#disjoint-logical-buffers) — one completion ledger per
  block of packed CUDA storage.
- [CUDA Buffer Completion Lowering](epoch_design_draft.md) — access and
  completion contract for the buffers a segmented submission touches.
- [CUDA Kernel Dispatch and Device Scheduling](kernel_dispatch.md)

## Problem

The CUDA reference backend has a good kernel for one strided view tuple.
`StridedElementwisePlan`
([`elementwise_plan.hpp`](../../../src/uni20/linalg/backends/cuda/elementwise_plan.hpp))
compacts the rank to at most eight, lowers to 32-bit indices and offsets when
they fit, is passed to the kernel by value, and maps one logical element to one
thread in a grid-stride loop with the innermost dimension first. The kernel
([`elementwise_kernel.cuh`](../../../src/uni20/linalg/backends/cuda/elementwise_kernel.cuh))
is arity-generic, so copy, add, axpy, scale and conjugation share one executor.

Block tensors use one launch for a whole tensor only on the allocation-wide
`cuda_partitioned_*` paths in
[`block_tensor_linear.hpp`](../../../src/uni20/symmetry/block_tensor_linear.hpp):
`set_zero` always, `scale` unless padding meets a non-finite factor, and
`assign_scale`, `copy`, `add_inplace` and `axpy` when
`same_packed_layout(output, input)` holds (identical stored keys, block offsets
and buffer size), with the same padding guard on scaling factors. Every other
case makes one CUDA submission per block: acquire a stream, acquire the block's
ledger, launch, record a completion. The current per-block callers are:

- Any operation with a zero-copy `permute` or `repartition` view as an
  operand. These are mapped views and expose no `storage()`, so they never
  qualify for the partitioned path.
- `assign_scale` (and so `copy`), `add_inplace` and `axpy` whenever the key
  sets or offsets differ, and every three-operand `add`, which has no
  partitioned path.
- Zero fill of output keys that the input lacks, and of output blocks with no
  contraction binding (`zero_unbound_pairwise_output_blocks` in
  [`block_tensor_contract.hpp`](../../../src/uni20/symmetry/block_tensor_contract.hpp)).
- CUDA block-SVD sector assembly, one transposed copy per stored fragment, in
  [`block_tensor_svd.hpp`](../../../src/uni20/symmetry/block_tensor_svd.hpp).
- CUDA block-SVD factor materialization in the same file, which launches one
  one-dimensional copy, or scaled copy when singular values are absorbed, per
  fragment and per selected singular vector. Selections are arbitrary sorted
  subsets of each sector.
- `materialize_block_tensor_copy` in
  [`two_site_split.hpp`](../../../src/uni20/tensor_network/two_site_split.hpp).

Pack and scatter for coalesced GEMMs
([Block Coalescing](../../symmetry/block_coalescing.md)) would be further
callers. They are not implemented.

Block sizes in DMRG are ragged. One tensor can hold 1×1 blocks next to
1000×1000 blocks, and the calibrated R/A/B/C data shows a launch-bound tail of
near-zero-flop blocks that does not shrink with bond dimension. One launch per
block puts the launch cost on the tail.

Ledger traffic is a second cost. Packed CUDA storage gives every block its own
`CudaBuffer` child with its own completion ledger, so independent blocks can be
scheduled on different streams. Acquiring a child installs one
`cudaStreamWaitEvent` per outstanding predecessor completion, and
`Stream::wait_on` does not deduplicate. After an allocation-wide operation
publishes one shared completion to all N children, the next allocation-wide
acquisition issues N waits on that same event. A segmented submission that
acquires children the same way would inherit this cost, so acquisition is part
of this design (see [Submission](#submission)).

## The Primitive

```text
for each segment s, for each logical index i in segment s:
  output_s[i] = op(inputs_s[i]...)               // overwrite
  output_s[i] = op(output_s[i], inputs_s[i]...)  // update
```

Each segment is a tuple of dense strided views with the same extents. Different
segments may have different extents and ranks. All segments of one plan have
the same operand count and the same operation. The primitive knows nothing
about quantum numbers, keys, or fusion rules. The symmetric layer pairs blocks
by key and hands over one view tuple per pair. This matches the rule in
[Raw Primitives and Symmetric Lowering](../../symmetry/raw_primitives_and_lowering.md):
a list of strided views with one elementwise operation is a useful operation on
raw tensor data in its own right.

The single-view plan is the one-segment case.

Preconditions, beyond those of the single-view plan:

- output regions of distinct segments are disjoint;
- no segment reads an element that another segment writes. An update segment
  reads its own output elements, as in the single-view plan.

Segments run concurrently within one launch, whereas the per-block path ordered
them through ledgers. Block-tensor lowering satisfies both conditions by
construction: distinct blocks of one packed allocation are disjoint, and the
linear operations already require that distinct output and input values do not
overlap.

## Building a Plan

**Segments come from block views, not from storage offsets.** For each segment
the builder runs the existing `make_multi_iteration_plan` over the operands'
block mappings, which compacts extent-1 and mergeable dimensions. The operand's
base offset is the block's buffer offset within its physical allocation plus
the data-descriptor element offset and the iteration plan's base offset. Owned
blocks, `permute` and `repartition` views, SVD sector sub-views and diagonal
components all lower the same way, with no layout-specific cases.

**Empty segments are dropped.**

**Adjacent segments fuse.** Two consecutive segments that are both rank 1 with
unit stride in every operand, where the second starts in every operand
immediately after the first ends, become one segment. Two tensors with the same
unpadded packed layout fuse to a single segment, which is the allocation-wide
partitioned path. Fusion never crosses padding, so padding is never touched and
the `has_padding() || isfinite(factor)` guard on the allocation-wide path has
no counterpart here.

**One plan per operation form.** Some block operations need different forms on
different blocks. `add(output, lhs, rhs)` copies lhs-only and rhs-only blocks
and zeros blocks present in neither input. `assign_scale` zeros output blocks
the input lacks. Each form gets its own segment list, executed as consecutive
launches in the same submission. A missing input is not emulated by loading
zero: `-0.0 + 0.0` is `+0.0`, and `factor * 0` is NaN for a non-finite factor,
whereas the current contract copies present blocks exactly and makes missing
blocks exact zeros.

**Selection is segmentation.** A sorted selection along one axis needs no index
table. Each maximal run of consecutive selected indices is an affine range of
that axis and becomes one segment. For block-SVD materialization, a per-sector prefix selection gives
one segment per fragment, and an arbitrary sorted selection gives one segment
per run rather than one launch per selected vector. Absorbing singular values is
a multiply by a stride-zero operand over the device singular values. That needs
a real-by-scalar multiply that the registered CUDA operations do not provide.

Index tables inside a segment (NumPy-style advanced indexing along an inner
axis) are deferred until a caller needs them. A dimension backed by a table
cannot merge with its neighbours, so compaction must run before tables are
attached.

## Plan Representation

```cpp
template <std::size_t OperandCount, class LogicalIndex, class Offset>
struct SegmentDescriptor
{
    LogicalIndex element_count = 0;
    std::uint8_t compact_rank = 0;
    std::array<std::uint64_t, OperandCount> base_offsets{};  // allocation-relative
    // followed by extents[plan_rank] and strides[plan_rank][OperandCount]
};
```

A plan holds:

| Field | Meaning |
|---|---|
| descriptors | one per segment, each row sized by the plan's maximum compact rank |
| `segment_begin[]` | 64-bit exclusive prefix sum of `element_count`; `segment_begin[S]` is the total |
| `chunk_first_segment[]` | `C + 1` entries: entry `c` is the segment containing element `c*K`, entry `C` the last segment |
| index kind | 32-bit unless some segment's element count or intra-segment offset span needs 64 bits |
| chunk size `K` | elements per thread block |

Rules:

- **Allocation-relative offsets, not pointers.** Every operand's segments lie in
  one physical allocation, whose start is a kernel parameter. Descriptors carry
  element offsets from that start. A plan therefore depends only on layout and
  can be reused with any buffers of the same layout, such as new Krylov vectors
  each iteration.
- **64-bit bases, narrow decode.** Base offsets are always 64-bit and are added
  once per element. The per-element decode (division by extents, multiplication
  by strides) uses the plan's index kind, which depends on the largest segment,
  not the allocation size. Segments that fit 32-bit indexing keep the 32-bit
  decode in allocations above 2^31 elements.
- **Rows sized by the plan, not the kernel maximum.** Block segments typically
  compact to rank 1–3. A rank-3, two-operand, 32-bit descriptor is about 60
  bytes. The worst case allowed by the current maximum rank of eight, three
  operands with 64-bit indices, is about 300 bytes.
- **Positive strides only**, as in the single-view plan. Negative-stride views
  need descriptor rebasing and signed traversal and are out of scope.
- **One-segment plans use the existing kernel.** After fusion, a plan with one
  segment lowers to `StridedElementwisePlan` and is passed by value, with no
  descriptor upload.

Descriptors for multi-segment plans live in a device buffer. Since CUDA 12.1,
kernels on Volta and later accept up to 32,764 bytes of parameters, enough for
a few hundred compact descriptors. Passing small plans inline instead of
uploading them is a tuning option to decide by measurement.

## CUDA Execution

### Load-balanced search

The scheduling unit is a fixed-size chunk of `K` logical elements of the
concatenated segment space, not a segment. Thread block `c` owns elements
`[c*K, (c+1)*K)`.

1. Read `s0 = chunk_first_segment[c]` and `s1 = chunk_first_segment[c + 1]`.
   The host computes this array while building the plan in O(S + C), so the
   device does no global search.
2. If `s0 == s1`, the chunk lies inside one segment, which is the common case
   for large blocks. Every thread decodes with that descriptor directly.
3. Otherwise the thread block stages `segment_begin[s0..s1]`, as 32-bit offsets
   relative to the chunk start, in shared memory and synchronizes once. Each
   thread finds the segment of each of its elements by binary search in shared
   memory. Empty segments were dropped, so at most `K` segments overlap a chunk
   and the staging is bounded.
4. With `T` threads per block, thread `t` handles chunk elements `t`, `t + T`,
   `t + 2T`, …, so neighbouring threads touch neighbouring elements of a large
   segment, as in the current grid-stride kernel.

A large block is split across many thread blocks and many tiny blocks share
one, so load balance depends only on the total element count. There is one
barrier per chunk however many segments it crosses. Walking the segments of a
chunk in sequence with a barrier per segment would cost a barrier per tiny
block. This is the load-balancing search of moderngpu's `transform_lbs`, the
same merge-path partitioning used for load-balanced CSR SpMV.

Descriptors are read through the read-only cache, and threads of a warp in one
segment read the same descriptor. For segments of a few elements, descriptor
and `segment_begin` reads are comparable to or larger than the data moved. That
is still far cheaper than a launch per block. CUB's `DeviceMemcpy::Batched`
handles the same regime with thread-, warp- and block-level size classes. The
same split is the tuning option here if measurement shows the tail is
descriptor-bound.

### Permuted segments

`make_multi_iteration_plan` orders dimensions by the first operand, which is the
output. A permuted copy therefore writes coalesced and reads strided, which
the L1 and L2 caches tolerate better than strided writes, especially for small
blocks. Segments with and without a permutation run in the same kernel.

Large permuted segments lose bandwidth on the strided side. A shared-memory
tiled transpose over its own tile list, issued as an extra launch in the same
submission, is the remedy. Whether it is needed, and above what extents, must
be decided by benchmarks on real DMRG block-size distributions.

Vector loads and stores are also deferred. Counting work in units of `w`
elements per segment keeps per-segment vector widths compatible with the chunk
schedule. Complex double elements are already 16-byte accesses.

## Submission

One segmented operation is one submission:

1. Acquire one stream on the output's device. Operands on another device
   decline with `incompatible_devices`, as the single-view path does.
2. Collect the distinct child buffers the segments touch. Take write access to
   every child written and read access to every child that is only read. A child
   that is both read and written, as in an in-place update, takes one write
   access, because `CudaBuffer` rejects a live read and write access together.
   Untouched children are not acquired, so unrelated blocks stay free for
   block-local work on other streams.
3. Install predecessor waits once per distinct completion. After a previous
   segmented or allocation-wide operation, every child carries the same
   completion, so this is one `cudaStreamWaitEvent` instead of one per block.
   This needs a ledger interface that collects predecessors separately from
   installing waits. `PartitionedReadAccess` and `PartitionedWriteAccess`
   should use it too, since they currently wait once per child.
4. Upload the descriptors for a multi-segment plan and launch once per
   operation form.
5. Record one completion through `AccessCompletion` and publish it to every
   acquired child.

On compute capability 9.0 and later, consecutive launches within one
submission can use same-stream programmatic dependent launch. The kernel's
setup reads only plan data, so it can overlap the previous launch. See
[CUDA Runtime](runtime.md#programmatic-dependent-launch).

The descriptor buffer and its host source are pinned by the completion under
the two-clocks rule in
[Ordering and Backend Lowering](../../architecture/ordering_and_backend_lowering.md),
like any other operand.

Scope of the first implementation:

- **Single-allocation operands.** Packed CUDA storage, `permute` and
  `repartition` views of it, and `CudaMatrix` or `CudaTensor` sector buffers
  qualify. Separate-block CUDA storage declines to the per-block path.
- **No `AsyncBlockTensorView` operands.** Block tensors whose blocks are
  separate `Async` values would need one node to join every block's dependency. That
  trades block-level pipelining for launch count, which is a separate decision.

## Plans and Caching

A plan depends only on structure: the operands' stored keys, sector dimensions,
packed offsets, and the view mapping. It does not depend on numerical data.
Within a DMRG sweep and across repeated R/A/B/C applications the structure is
stable, so a cached plan would save the O(S) host build and the descriptor
upload per call. Bond-dimension growth or truncation changes the structure and
requires a new plan.

The first implementation builds a plan per call and measures that cost before
adding a cache. A cached plan holds descriptors, `segment_begin` and the chunk
partition, keyed by layout. It holds no buffer pointers. Stable plans also keep
kernel arguments stable for CUDA graph capture.

## Host Execution

The segment list is backend-neutral, like `multi_iteration_plan`, and only its
descriptors are CUDA-specific. A host executor can run the same chunked
schedule under oneTBB. That also fixes the host load imbalance of parallelizing
over blocks when one block is much larger than the rest. It is also the
reference implementation for tests.

## Rejected Alternatives

| Alternative | Why not |
|---|---|
| One launch per block (current fallback) | launch and ledger cost on every block, dominated by the tail |
| One thread per segment, looping over its elements | serial work on large blocks and uncoalesced access across threads |
| Device-side child launches or `cudaMemcpyAsync` per segment from a kernel | bounded by the pending-launch limit; Cytnx shipped this and it silently dropped copies and hung ([Cytnx-dev/Cytnx#1229](https://github.com/Cytnx-dev/Cytnx/issues/1229)) |
| Padding blocks to a common shape for a fixed-shape batched kernel | wastes bandwidth on ragged sizes; fixed-shape batching remains right for groups that are already same-shape |
| `cudaMemcpyBatchAsync` and `cudaMemcpy3DBatchAsync` (CUDA 12.8 and later) | contiguous or pitched copies only: no permutation, no arithmetic, no inner strides |
| CUB `DeviceMemcpy::Batched` | contiguous byte copies only; its size classes are the model for tiny-segment tuning |
| CUDA graph capture of the per-block loop | removes host launch cost but still runs one small kernel per block, and the graph depends on structure, so it needs the same cache as a plan |
| Per-segment pointers instead of allocation-relative offsets | ties a plan to specific buffers, so it cannot be reused across values with the same layout |

## Testing and Benchmarks

Correctness tests compare each segmented operation against the host executor
and against a loop over the segments' mdspans. They should cover:

- one segment, which must agree with the single-view plan;
- empty segments, extent-1 legs, and segments of differing compact rank;
- fusion: an unpadded same-layout pair becomes one segment, and a padded pair
  leaves padding zero;
- one very large segment beside many tiny ones;
- chunks that start and end mid-segment, and a chunk crossing `K` one-element
  segments;
- index-kind selection on the host plan builder, including allocation-relative
  base offsets above 2^31 without allocating them;
- permuted segments;
- each operation form of `add` with partial key overlap, including negative
  zeros in lhs-only blocks, and `assign_scale` with a non-finite factor and
  missing input blocks;
- an in-place update whose output child is also an input, and children outside
  the segment list keeping their ledger state;
- block-SVD materialization with prefix and non-contiguous selections;
- overwrite and update operations, real and complex.

Benchmarks compare the segmented path with the per-block fallback on block-size
distributions from real DMRG runs, using the calibrated R/A/B/C data where
possible. An Nsight Systems trace should show one stream wait for a segmented
operation that follows an allocation-wide one. The tiled transpose, the chunk
size, size classes, inline descriptors and plan caching all need these
measurements.

## Proposed First Implementation

1. A backend-neutral plan builder: segments from block view tuples, empty-segment
   removal, fusion, index-kind selection, and the host executor.
2. Ledger acquisition over a set of children with one wait per distinct
   completion, also used by the partitioned access guards.
3. One CUDA load-balanced kernel, with descriptors uploaded per call.
4. Route the elementwise operations in `block_tensor_linear.hpp` and the
   contraction zero fill through the segmented path. The unpadded same-layout
   case is the allocation-wide path by construction. Retire the elementwise
   `cuda_partitioned_*` functions once benchmarks show the padded case, several
   segments and one upload, is no slower than the allocation-wide launch. The
   cuBLAS reductions stay.

Block-SVD assembly and materialization, `materialize_block_tensor_copy`, the
tiled transpose, plan caching and the host chunked executor follow as separate
changes.

## Open Questions

- **Separate storage on CUDA.** Allocation-relative offsets require each operand
  to lie in one allocation. The recommendation is to decline
  `SeparateSparseBlockStorageData` tensors on CUDA rather than add a pointer
  descriptor form, since the CUDA path already prefers packed storage.
- **Tiled transpose.** The recommendation is to add it only after benchmarks
  show strided reads of large permuted blocks are a bottleneck.
- **Plan ownership.** When caching is added, should plans live on the block
  tensor's structure object, or be owned by the lowering and prepared
  contraction that issues the operation? The second fits prepared, repeated
  R/A/B/C contractions better.
- **Operation forms in one launch.** A per-segment form tag dispatched inside
  one kernel would turn the two to four launches of a mixed `add` into one. Is
  that worth the extra kernel complexity once measured?
- **Async block tensors.** Should a later change join per-block `Async`
  dependencies into one segmented node, giving up block-level pipelining?
