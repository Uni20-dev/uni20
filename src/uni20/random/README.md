# Random numbers

[`philox.hpp`](philox.hpp) provides `uni20::random::philox4x32_10(counter, key)`,
a stateless permutation callable on the host, on CUDA devices and in constant
expressions. Pass a `Philox4x32Block` and a `Philox4x32Key` in Random123 word
order. The returned block's `words[0]` through `words[3]` are the four output words.

[`transforms.hpp`](transforms.hpp) converts one block at a time. `Uniform<float>{}`
returns four unit uniforms, and `Uniform<double>{}` returns two. Pass endpoints
as `Uniform{low, high}` for a scaled interval. `Normal{mu, sigma}` returns two
doubles and `ComplexNormal{mu, sigma}` returns one circular complex double with
variance `sigma * sigma`. Pass a group as `transform({block})` and read `.values[i]`.
A fixed-width transform declares how many consecutive Philox blocks it takes
(`input_type::size`) and how many values it produces (`result_type::size`).
The current transforms all consume one block.

[`fill.hpp`](fill.hpp) fills dense tensors, default-accessor host mdspans,
and raw CUDA buffer descriptors. Offsets count **elements**. For a view
sharing a leaf, pass the allocation offset of its **lowest storage address**,
including for a reversed view. Omit it only for a leaf exclusive to that view.
Padding stays untouched. For example, if a subview starts five elements into
the allocation, the two calls below assign it the same addresses.

```cpp
auto leaf = uni20::random::child(uni20::random::root(123), 7);
uni20::random::fill(whole, leaf, uni20::random::Uniform<double>{});
uni20::random::fill(subview, leaf, uni20::random::Uniform<double>{}, 5);
```

[`value.hpp`](value.hpp) provides `value_at(leaf, offset, transform)` for a
single value. `value_group(leaf, group_index, transform)` evaluates a whole group.
With `b` blocks and `g` values per group, element `o` uses output `o % g`
of the group starting at Philox block `b * (o / g)`.
The provisional [identity mapping](identity.hpp) takes explicit child labels.

[`stream.hpp`](stream.hpp) provides `PhiloxStream{leaf}` for standard algorithms
such as `std::shuffle`. A call draws the next 32-bit word and `discard(n)` skips
`n` words in constant time. Copying a stream preserves its position, including
partially consumed blocks. A leaf contains 2⁶³ blocks of four words each.

See the [random number generation design](../../../docs/architecture/random_number_generation.md)
for how higher layers use these blocks.
