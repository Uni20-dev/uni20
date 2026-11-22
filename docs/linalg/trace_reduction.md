# Trace As A Dense Reduction Primitive

`diagonal_view<A, B, ...>(tensor)` ties each pair of axes together without
copying elements. Each pair must have equal extents. Unpaired axes keep
their original order, followed by one independent index per pair in the
specified order.

```cpp
auto matrix_trace = uni20::sum_host(uni20::diagonal_view<0, 1>(matrix));
auto torus_trace = uni20::sum_host(uni20::diagonal_view<0, 3, 1, 2>(tensor));

// Trace two axes of a rank-three host tensor, keeping the third axis.
auto diagonal = uni20::diagonal_view<0, 1>(host_rank3);
auto partial_trace = uni20::sum(diagonal, 1);
```

The first two calls work on host tensors and on positive-strided CUDA tensors
with the default CUDA pointer accessor. The CUDA reduction reads the diagonal
elements in place and transfers only its result. The partial-trace example
uses the CPU axis-reduction backend.

A diagonal view borrows its tensor and retains the source accessor, placement,
and backend selector. A mutable source supports write-through access on its
diagonal. A const source produces a read-only view. This is a dense tensor
primitive. A block-sparse trace must apply the sector and leg rules before
reducing its dense blocks.

## Dense Trace Model

Given an input tensor with shape and element strides, tracing axes `a` and `b` requires:

```text
shape[a] == shape[b]
```

The diagonal of those two axes is represented as a synthetic axis `t` with extent:

```cpp
diagonal_extent = shape[a];
```

and stride:

```cpp
diagonal_stride = stride[a] + stride[b];
```

The surviving axes retain their input strides. The trace is then:

```text
out[survivors...] = sum_t input_view[survivors..., t]
```

where:

```text
input_view offset =
    base_offset
  + sum_i survivor_index[i] * survivor_stride[i]
  + t * diagonal_stride
```

So trace is a partial reduction over a strided view.

## Required Dense Primitives

This motivates two important dense tensor primitives:

- arbitrary or synthetic strided views;
- reduction over one or more axes.

The CPU axis-reduction backend preserves logical surviving-axis order and
respects the source accessor. The CUDA full-reduction backend lowers the
diagonal strides through the same iteration planner as other strided
reductions. Diagonal and survivor strides can interleave, so the view uses
a mapping that supports these strides directly.

## Layout Optimization

The public output axis order should normally be the input axis order with the traced axes removed. That logical order does not have to be the same as the physical traversal order used by the backend.

A dense trace implementation should build survivor records:

```cpp
struct SurvivorAxis {
  int logical_axis;
  index_type extent;
  index_type input_stride;
};
```

Then, for backend execution:

1. Drop extent-1 survivor axes from the kernel layout, while preserving them in output metadata.
2. Sort the remaining survivor axes by increasing `input_stride`.
3. Coalesce adjacent physical axes when:

   ```cpp
   next.input_stride == current.input_stride * current.extent
   ```

4. Use the coalesced layout for the reduction kernel.
5. Represent the returned tensor with the required logical output order, either directly or as a view/permutation where possible.

For non-coalescable survivor axes, increasing input stride is the natural default physical order. It gives the smallest-stride input dimension the fastest-varying output coordinate, improving locality without changing logical semantics.

## Backend Mapping

A custom kernel can implement this directly, but the primitive is also a natural match for backend libraries that support strided reductions.

For example, cuTENSOR has tensor descriptors with explicit strides and a generic partial reduction operation. A dense GPU trace can be expressed as a reduction over the synthetic diagonal axis whose stride is `stride[a] + stride[b]`. This is a better primitive than materializing an identity tensor or routing trace through a general contraction.

Backend selection can still choose a custom kernel for unsupported dtypes, small fixed cases, or builds without the relevant backend library. The design point is that the operation being selected is a strided reduction, not a special trace kernel.

## Symmetric Tensors

Trace at the level of a block-sparse quantum-number-aware tensor is a different operation. It may involve sector constraints, dual leg conventions, block matching, and representation data.

The symmetric trace should therefore lower into a set of dense trace/reduction operations over blocks after the relevant sector logic has been resolved. The dense primitive layer should not be burdened with quantum-number semantics.
