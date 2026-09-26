#pragma once

/**
 * \file matrix_exponential.hpp
 * \ingroup linalg
 * \brief Fixed-output dense matrix exponential operation.
 */

#include <uni20/linalg/backends/cpu/matrix_exponential.hpp>
#include <uni20/linalg/dispatch.hpp>
#include <uni20/linalg/operation_tags.hpp>
#include <uni20/tensor/concepts.hpp>

#include <utility>

namespace uni20::linalg
{

/// \brief Overwrite a fixed-size output with the matrix exponential `exp(time * input)`.
/// \details Both matrices must have the same square shape; output is not resized and
///          its old values are ignored. This is the matrix exponential, not entrywise
///          exponentiation. The CPU backend accepts real or complex time and promotes
///          a real input to a complex result for complex time; output elements must be
///          assignable from the computed result type.
/// \note The current CPU backend materializes the input before writing the output.
///       This implementation detail does not establish an aliasing contract for other
///       backends or Async overloads; Async input and output require distinct queues.
template <class BackendSelector, uni20::MutableRankedTensorView<2> OutputTensor, uni20::RankedTensorView<2> InputTensor,
          class TimeScalar>
void matrix_exponential(BackendSelector&& selector, OutputTensor&& output, InputTensor const& input, TimeScalar time)
{
  auto output_descriptor = uni20::mdspec_of(output);
  auto input_descriptor = uni20::mdspec_of(input);
  dispatch_kernel(std::forward<BackendSelector>(selector), matrix_exponential_op{}, output_descriptor, input_descriptor,
                  time);
}

/// \brief Apply the fixed-output `matrix_exponential` contract using tensor storage policy.
template <uni20::MutableRankedTensorView<2> OutputTensor, uni20::RankedTensorView<2> InputTensor, class TimeScalar>
void matrix_exponential(OutputTensor&& output, InputTensor const& input, TimeScalar time)
{
  auto selector = select_backend(matrix_exponential_op{}, output, input);
  matrix_exponential(selector, std::forward<OutputTensor>(output), input, time);
}

} // namespace uni20::linalg
