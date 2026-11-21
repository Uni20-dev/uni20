#pragma once

/**
 * \file elementwise_copy.hpp
 * \ingroup linalg
 * \brief Launch boundary for CUDA accessor-respecting element copies.
 */

#include <uni20/core/compiler_attributes.hpp>
#include <uni20/core/scalar_traits.hpp>
#include <uni20/core/types.hpp>
#include <uni20/linalg/backends/cuda/elementwise_plan.hpp>

#include <cuda_runtime_api.h>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace uni20::linalg::detail::cuda_reference
{

enum class ElementwiseCopyTransform
{
  identity,
  conjugate
};

inline constexpr std::size_t elementwise_copy_maximum_rank = 8;
using ElementwiseCopyPlan32 = StridedElementwisePlan32<2, elementwise_copy_maximum_rank>;
using ElementwiseCopyPlan64 = StridedElementwisePlan64<2, elementwise_copy_maximum_rank>;
using LoweredElementwiseCopyPlan = LoweredStridedElementwisePlan<2, elementwise_copy_maximum_rank>;

/// \brief Largest element, in words, copied by a fixed-width kernel.
/// \details Wider elements use a kernel that loops over their words.
inline constexpr std::size_t elementwise_copy_maximum_words = 4;

/// \brief One element's object representation as unsigned words.
/// \details An identity copy moves bytes and never interprets them, so one
///          kernel per word size and count serves every trivially copyable
///          element type, and floating-point payloads such as signaling NaNs
///          are preserved.
template <class Word, std::size_t Count> struct ElementWords
{
    Word words[Count];
};

/// \brief Word layout that reproduces a `Scalar` element bit for bit.
/// \details Words are as wide as the element alignment, at most eight bytes,
///          so every word access is aligned. An element size is a multiple of
///          its alignment, so every trivially copyable element is supported:
///          up to `elementwise_copy_maximum_words` words through a fixed-width
///          kernel, wider elements through a loop over their words.
template <class Scalar> struct ElementwiseCopyWords
{
    static constexpr std::size_t word_size = alignof(Scalar) < 8 ? alignof(Scalar) : 8;
    using word_type =
        std::conditional_t<word_size == 1, std::uint8_t,
                           std::conditional_t<word_size == 2, std::uint16_t,
                                              std::conditional_t<word_size == 4, std::uint32_t, std::uint64_t>>>;
    static constexpr std::size_t count = sizeof(Scalar) / word_size;
    static constexpr bool supported = std::is_trivially_copyable_v<Scalar> && sizeof(Scalar) % word_size == 0;
    static constexpr bool fixed_width = count <= elementwise_copy_maximum_words;
    /// \brief Fixed-width element representation; meaningful only when `fixed_width`.
    using element_type = ElementWords<word_type, fixed_width ? count : 1>;
};

template <class Word, std::size_t Count>
void enqueue_elementwise_word_copy(ElementWords<Word, Count>* output, ElementWords<Word, Count> const* input,
                                   ElementwiseCopyPlan32 const& plan, cudaStream_t stream, int device);
template <class Word, std::size_t Count>
void enqueue_elementwise_word_copy(ElementWords<Word, Count>* output, ElementWords<Word, Count> const* input,
                                   ElementwiseCopyPlan64 const& plan, cudaStream_t stream, int device);

/// \brief Copy elements of `words_per_element` words each, for elements wider than the fixed-width kernels.
/// \details Plan offsets count elements; each element starts `words_per_element`
///          words after the previous one.
template <class Word>
void enqueue_elementwise_word_run_copy(Word* output, Word const* input, std::size_t words_per_element,
                                       ElementwiseCopyPlan32 const& plan, cudaStream_t stream, int device);
template <class Word>
void enqueue_elementwise_word_run_copy(Word* output, Word const* input, std::size_t words_per_element,
                                       ElementwiseCopyPlan64 const& plan, cudaStream_t stream, int device);

void enqueue_elementwise_conjugating_copy(uni20::cfloat* output, uni20::cfloat const* input,
                                          ElementwiseCopyPlan32 const& plan, cudaStream_t stream, int device);
void enqueue_elementwise_conjugating_copy(uni20::cfloat* output, uni20::cfloat const* input,
                                          ElementwiseCopyPlan64 const& plan, cudaStream_t stream, int device);
void enqueue_elementwise_conjugating_copy(uni20::cdouble* output, uni20::cdouble const* input,
                                          ElementwiseCopyPlan32 const& plan, cudaStream_t stream, int device);
void enqueue_elementwise_conjugating_copy(uni20::cdouble* output, uni20::cdouble const* input,
                                          ElementwiseCopyPlan64 const& plan, cudaStream_t stream, int device);

/// \brief Whether the elementwise kernel can copy `Scalar` without changing its value.
template <class Scalar> inline constexpr bool supports_elementwise_copy = ElementwiseCopyWords<Scalar>::supported;

/// \brief Whether the elementwise kernel can copy and conjugate `Scalar`.
/// \details Conjugating a real element is an identity copy.
template <class Scalar>
inline constexpr bool supports_elementwise_conjugating_copy =
    std::same_as<Scalar, uni20::cfloat> || std::same_as<Scalar, uni20::cdouble> ||
    (!uni20::is_complex_v<Scalar> && supports_elementwise_copy<Scalar>);

} // namespace uni20::linalg::detail::cuda_reference
