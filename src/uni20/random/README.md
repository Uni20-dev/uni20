# Random numbers

[`philox.hpp`](philox.hpp) provides `uni20::random::philox4x32_10(counter, key)`,
a stateless permutation callable on the host, on CUDA devices and in constant
expressions. Pass a `Philox4x32Block` and a `Philox4x32Key` in Random123 word
order. The returned block's `words[0]` through `words[3]` are the four output words.

[`transforms.hpp`](transforms.hpp) converts one block at a time. `Uniform<float>{}`
returns four unit uniforms, and `Uniform<double>{}` returns two. Pass endpoints
as `Uniform{low, high}` for a scaled interval. `Normal{mu, sigma}` returns two
doubles and `ComplexNormal{mu, sigma}` returns one circular complex double with
variance `sigma * sigma`. Apply a transform to the block and read `.values[i]`.

See the [random number generation design](../../../docs/architecture/random_number_generation.md)
for how higher layers use these blocks.
