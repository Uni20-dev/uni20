#include "elementwise_arithmetic.hpp"
#include "elementwise_conjugate.hpp"
#include "elementwise_copy.hpp"
#include "elementwise_kernel.cuh"
#include "elementwise_scale.hpp"

#include <uni20/core/scalar_concepts.hpp>
#include <uni20/linalg/elementwise_functions.hpp>
#include <uni20/storage/cuda_accessor.hpp>

#include <concepts>
#include <cstddef>
#include <cstdint>
#include <type_traits>

namespace uni20::linalg::detail::cuda_reference
{
namespace
{

struct AssignElement
{
    template <class Value> [[nodiscard]] UNI20_HOST_DEVICE constexpr auto operator()(Value value) const
    {
      return value;
    }
};

template <class Scalar, class InputAccessor, class Plan>
void launch_elementwise_copy(Scalar* output, Scalar const* input, Plan const& plan, cudaStream_t stream, int device)
{
  using output_accessor = uni20::cuda::CudaPointerAccessor<Scalar>;
  launch_elementwise_kernel<false>(plan, AssignElement{}, ElementwiseOperand{output, output_accessor{}}, stream, device,
                                   "launch CUDA reference elementwise copy",
                                   ElementwiseOperand{input, InputAccessor{}});
}

template <class Scalar, class Plan>
void enqueue_elementwise_conjugating_copy_impl(Scalar* output, Scalar const* input, Plan const& plan,
                                               cudaStream_t stream, int device)
{
  using real_type = uni20::make_real_t<Scalar>;
  using conjugating_accessor = uni20::cuda::CudaConjugatingPointerAccessor<real_type>;
  launch_elementwise_copy<Scalar, conjugating_accessor>(output, input, plan, stream, device);
}

template <class Scalar, class Plan>
void enqueue_elementwise_conjugate_impl(Scalar* output, Plan const& plan, cudaStream_t stream, int device)
{
  using output_accessor = uni20::cuda::CudaPointerAccessor<Scalar>;
  launch_elementwise_kernel<true>(plan, uni20::cuda::CudaConjugate{}, ElementwiseOperand{output, output_accessor{}},
                                  stream, device, "launch CUDA reference elementwise conjugation");
}

template <class Scalar, class Operation, class Plan>
void enqueue_elementwise_unary_impl(Scalar* output, Scalar const* input, Operation operation, Plan const& plan,
                                    cudaStream_t stream, int device)
{
  using output_accessor = uni20::cuda::CudaPointerAccessor<Scalar>;
  using input_accessor = uni20::cuda::CudaPointerAccessor<Scalar const>;
  launch_elementwise_kernel<false>(plan, operation, ElementwiseOperand{output, output_accessor{}}, stream, device,
                                   "launch CUDA reference unary arithmetic",
                                   ElementwiseOperand{input, input_accessor{}});
}

template <class Value> [[nodiscard]] auto to_cuda_execution_value(Value value) { return value; }

template <class Real> [[nodiscard]] auto to_cuda_execution_value(uni20::detail::standard_complex<Real> const& value)
{
  return ::cuda::std::complex<Real>{value.real(), value.imag()};
}

template <class Scalar, class Plan>
void enqueue_elementwise_fill_impl(Scalar* output, Scalar value, Plan const& plan, cudaStream_t stream, int device)
{
  using output_accessor = uni20::cuda::CudaPointerAccessor<Scalar>;
  auto operation = uni20::linalg::constant{to_cuda_execution_value(value)};
  launch_elementwise_kernel<false>(plan, operation, ElementwiseOperand{output, output_accessor{}}, stream, device,
                                   "launch CUDA reference elementwise fill");
}

template <class Scalar, class Factor, class Plan>
void enqueue_elementwise_scale_impl(Scalar* output, Scalar const* input, Factor factor, Plan const& plan,
                                    cudaStream_t stream, int device)
{
  using output_accessor = uni20::cuda::CudaPointerAccessor<Scalar>;
  using input_accessor = uni20::cuda::CudaPointerAccessor<Scalar const>;
  auto operation = uni20::linalg::scale{to_cuda_execution_value(factor)};
  launch_elementwise_kernel<false>(plan, operation, ElementwiseOperand{output, output_accessor{}}, stream, device,
                                   "launch CUDA reference elementwise scaling",
                                   ElementwiseOperand{input, input_accessor{}});
}

template <class Scalar, class Operation, class Plan>
void enqueue_elementwise_binary_impl(Scalar* output, Scalar const* lhs, Scalar const* rhs, Operation operation,
                                     Plan const& plan, cudaStream_t stream, int device)
{
  using output_accessor = uni20::cuda::CudaPointerAccessor<Scalar>;
  using input_accessor = uni20::cuda::CudaPointerAccessor<Scalar const>;
  launch_elementwise_kernel<false>(plan, operation, ElementwiseOperand{output, output_accessor{}}, stream, device,
                                   "launch CUDA reference binary arithmetic", ElementwiseOperand{lhs, input_accessor{}},
                                   ElementwiseOperand{rhs, input_accessor{}});
}

template <class Scalar, class Factor, class Plan>
void enqueue_elementwise_inplace_scale_impl(Scalar* output, Factor factor, Plan const& plan, cudaStream_t stream,
                                            int device)
{
  using output_accessor = uni20::cuda::CudaPointerAccessor<Scalar>;
  auto operation = uni20::linalg::scale{to_cuda_execution_value(factor)};
  launch_elementwise_kernel<true>(plan, operation, ElementwiseOperand{output, output_accessor{}}, stream, device,
                                  "launch CUDA reference in-place scaling");
}

template <class Scalar, class Operation, class Plan>
void enqueue_elementwise_inplace_binary_impl(Scalar* output, Scalar const* input, Operation operation, Plan const& plan,
                                             cudaStream_t stream, int device)
{
  using output_accessor = uni20::cuda::CudaPointerAccessor<Scalar>;
  using input_accessor = uni20::cuda::CudaPointerAccessor<Scalar const>;
  launch_elementwise_kernel<true>(plan, operation, ElementwiseOperand{output, output_accessor{}}, stream, device,
                                  "launch CUDA reference in-place binary arithmetic",
                                  ElementwiseOperand{input, input_accessor{}});
}

template <class Scalar, class Factor, class Plan>
void enqueue_elementwise_add_scaled_impl(Scalar* output, Scalar const* input, Factor factor, Plan const& plan,
                                         cudaStream_t stream, int device)
{
  auto operation = uni20::linalg::add_scaled{to_cuda_execution_value(factor)};
  enqueue_elementwise_inplace_binary_impl(output, input, operation, plan, stream, device);
}

} // namespace

template <class Scalar>
void enqueue_elementwise_fill(Scalar* output, Scalar value, ElementwiseFillPlan32 const& plan, cudaStream_t stream,
                              int device)
{
  enqueue_elementwise_fill_impl(output, value, plan, stream, device);
}

template <class Scalar>
void enqueue_elementwise_fill(Scalar* output, Scalar value, ElementwiseFillPlan64 const& plan, cudaStream_t stream,
                              int device)
{
  enqueue_elementwise_fill_impl(output, value, plan, stream, device);
}

template <class Function, class Scalar>
void enqueue_elementwise_unary(Scalar* output, Scalar const* input, Function function,
                               ElementwiseUnaryPlan32 const& plan, cudaStream_t stream, int device)
{
  static_assert(RegisteredStatelessUnary<Function>);
  enqueue_elementwise_unary_impl(output, input, function, plan, stream, device);
}

template <class Function, class Scalar>
void enqueue_elementwise_unary(Scalar* output, Scalar const* input, Function function,
                               ElementwiseUnaryPlan64 const& plan, cudaStream_t stream, int device)
{
  static_assert(RegisteredStatelessUnary<Function>);
  enqueue_elementwise_unary_impl(output, input, function, plan, stream, device);
}

template <class Function, class Scalar>
void enqueue_elementwise_binary(Scalar* output, Scalar const* lhs, Scalar const* rhs, Function function,
                                ElementwiseBinaryPlan32 const& plan, cudaStream_t stream, int device)
{
  static_assert(RegisteredStatelessBinary<Function>);
  enqueue_elementwise_binary_impl(output, lhs, rhs, function, plan, stream, device);
}

template <class Scalar, class Factor>
void enqueue_elementwise_inplace_scale(Scalar* output, Factor factor, ElementwiseInplaceUnaryPlan32 const& plan,
                                       cudaStream_t stream, int device)
{
  enqueue_elementwise_inplace_scale_impl(output, factor, plan, stream, device);
}

template <class Scalar, class Factor>
void enqueue_elementwise_inplace_scale(Scalar* output, Factor factor, ElementwiseInplaceUnaryPlan64 const& plan,
                                       cudaStream_t stream, int device)
{
  enqueue_elementwise_inplace_scale_impl(output, factor, plan, stream, device);
}

template <class Function, class Scalar>
void enqueue_elementwise_inplace_binary(Scalar* output, Scalar const* input, Function function,
                                        ElementwiseInplaceBinaryPlan32 const& plan, cudaStream_t stream, int device)
{
  static_assert(RegisteredStatelessBinary<Function>);
  enqueue_elementwise_inplace_binary_impl(output, input, function, plan, stream, device);
}

template <class Function, class Scalar>
void enqueue_elementwise_inplace_binary(Scalar* output, Scalar const* input, Function function,
                                        ElementwiseInplaceBinaryPlan64 const& plan, cudaStream_t stream, int device)
{
  static_assert(RegisteredStatelessBinary<Function>);
  enqueue_elementwise_inplace_binary_impl(output, input, function, plan, stream, device);
}

template <class Scalar, class Factor>
void enqueue_elementwise_add_scaled(Scalar* output, Scalar const* input, Factor factor,
                                    ElementwiseInplaceBinaryPlan32 const& plan, cudaStream_t stream, int device)
{
  enqueue_elementwise_add_scaled_impl(output, input, factor, plan, stream, device);
}

template <class Scalar, class Factor>
void enqueue_elementwise_add_scaled(Scalar* output, Scalar const* input, Factor factor,
                                    ElementwiseInplaceBinaryPlan64 const& plan, cudaStream_t stream, int device)
{
  enqueue_elementwise_add_scaled_impl(output, input, factor, plan, stream, device);
}

template <class Function, class Scalar>
void enqueue_elementwise_binary(Scalar* output, Scalar const* lhs, Scalar const* rhs, Function function,
                                ElementwiseBinaryPlan64 const& plan, cudaStream_t stream, int device)
{
  static_assert(RegisteredStatelessBinary<Function>);
  enqueue_elementwise_binary_impl(output, lhs, rhs, function, plan, stream, device);
}

template <class Word, std::size_t Count>
void enqueue_elementwise_word_copy(ElementWords<Word, Count>* output, ElementWords<Word, Count> const* input,
                                   ElementwiseCopyPlan32 const& plan, cudaStream_t stream, int device)
{
  using element_type = ElementWords<Word, Count>;
  launch_elementwise_copy<element_type, uni20::cuda::CudaPointerAccessor<element_type const>>(output, input, plan,
                                                                                              stream, device);
}

template <class Word, std::size_t Count>
void enqueue_elementwise_word_copy(ElementWords<Word, Count>* output, ElementWords<Word, Count> const* input,
                                   ElementwiseCopyPlan64 const& plan, cudaStream_t stream, int device)
{
  using element_type = ElementWords<Word, Count>;
  launch_elementwise_copy<element_type, uni20::cuda::CudaPointerAccessor<element_type const>>(output, input, plan,
                                                                                              stream, device);
}

/// \brief Copy each element as a run of `words_per_element` words.
template <class Word, class Plan>
__global__ void elementwise_word_run_copy_kernel(Word* output, Word const* input, std::size_t words_per_element,
                                                 Plan plan)
{
  using logical_index_type = typename Plan::logical_index_type;
  auto index = static_cast<logical_index_type>(blockIdx.x * blockDim.x + threadIdx.x);
  auto const grid_stride = static_cast<logical_index_type>(blockDim.x * gridDim.x);
  auto const width = static_cast<std::ptrdiff_t>(words_per_element);
  while (index < plan.element_count)
  {
    auto const offsets = plan.offsets(index);
    Word* target = output + static_cast<std::ptrdiff_t>(offsets[0]) * width;
    Word const* source = input + static_cast<std::ptrdiff_t>(offsets[1]) * width;
    for (std::ptrdiff_t word = 0; word < width; ++word)
      target[word] = source[word];
    index += grid_stride;
  }
}

template <class Word, class Plan>
void launch_word_run_copy(Word* output, Word const* input, std::size_t words_per_element, Plan const& plan,
                          cudaStream_t stream, int device)
{
  if (plan.element_count == 0) return;
  constexpr unsigned int threads = 256;
  auto const required_blocks = (plan.element_count + threads - 1) / threads;
  constexpr std::size_t maximum_blocks = 65535;
  auto const blocks = static_cast<unsigned int>(required_blocks < maximum_blocks ? required_blocks : maximum_blocks);
  elementwise_word_run_copy_kernel<<<blocks, threads, 0, stream>>>(output, input, words_per_element, plan);
  uni20::cuda::check(cudaGetLastError(), "launch CUDA reference wide-element copy", device);
}

template <class Word>
void enqueue_elementwise_word_run_copy(Word* output, Word const* input, std::size_t words_per_element,
                                       ElementwiseCopyPlan32 const& plan, cudaStream_t stream, int device)
{
  launch_word_run_copy(output, input, words_per_element, plan, stream, device);
}

template <class Word>
void enqueue_elementwise_word_run_copy(Word* output, Word const* input, std::size_t words_per_element,
                                       ElementwiseCopyPlan64 const& plan, cudaStream_t stream, int device)
{
  launch_word_run_copy(output, input, words_per_element, plan, stream, device);
}

#define UNI20_INSTANTIATE_ELEMENTWISE_WORD_RUN_COPY(Word)                                                              \
  template void enqueue_elementwise_word_run_copy(Word*, Word const*, std::size_t, ElementwiseCopyPlan32 const&,       \
                                                  cudaStream_t, int);                                                  \
  template void enqueue_elementwise_word_run_copy(Word*, Word const*, std::size_t, ElementwiseCopyPlan64 const&,       \
                                                  cudaStream_t, int)

UNI20_INSTANTIATE_ELEMENTWISE_WORD_RUN_COPY(std::uint8_t);
UNI20_INSTANTIATE_ELEMENTWISE_WORD_RUN_COPY(std::uint16_t);
UNI20_INSTANTIATE_ELEMENTWISE_WORD_RUN_COPY(std::uint32_t);
UNI20_INSTANTIATE_ELEMENTWISE_WORD_RUN_COPY(std::uint64_t);

#undef UNI20_INSTANTIATE_ELEMENTWISE_WORD_RUN_COPY

#define UNI20_INSTANTIATE_ELEMENTWISE_WORD_COPY(Word, Count)                                                           \
  template void enqueue_elementwise_word_copy(ElementWords<Word, Count>*, ElementWords<Word, Count> const*,            \
                                              ElementwiseCopyPlan32 const&, cudaStream_t, int);                        \
  template void enqueue_elementwise_word_copy(ElementWords<Word, Count>*, ElementWords<Word, Count> const*,            \
                                              ElementwiseCopyPlan64 const&, cudaStream_t, int)

#define UNI20_INSTANTIATE_ELEMENTWISE_WORD_COPIES(Word)                                                                \
  UNI20_INSTANTIATE_ELEMENTWISE_WORD_COPY(Word, 1);                                                                    \
  UNI20_INSTANTIATE_ELEMENTWISE_WORD_COPY(Word, 2);                                                                    \
  UNI20_INSTANTIATE_ELEMENTWISE_WORD_COPY(Word, 3);                                                                    \
  UNI20_INSTANTIATE_ELEMENTWISE_WORD_COPY(Word, 4)

static_assert(elementwise_copy_maximum_words == 4, "instantiate one word copy per supported word count");
UNI20_INSTANTIATE_ELEMENTWISE_WORD_COPIES(std::uint8_t);
UNI20_INSTANTIATE_ELEMENTWISE_WORD_COPIES(std::uint16_t);
UNI20_INSTANTIATE_ELEMENTWISE_WORD_COPIES(std::uint32_t);
UNI20_INSTANTIATE_ELEMENTWISE_WORD_COPIES(std::uint64_t);

#undef UNI20_INSTANTIATE_ELEMENTWISE_WORD_COPIES
#undef UNI20_INSTANTIATE_ELEMENTWISE_WORD_COPY

#define UNI20_DEFINE_ELEMENTWISE_CONJUGATING_COPY(Scalar)                                                              \
  void enqueue_elementwise_conjugating_copy(Scalar* output, Scalar const* input, ElementwiseCopyPlan32 const& plan,    \
                                            cudaStream_t stream, int device)                                           \
  {                                                                                                                    \
    enqueue_elementwise_conjugating_copy_impl(output, input, plan, stream, device);                                    \
  }                                                                                                                    \
                                                                                                                       \
  void enqueue_elementwise_conjugating_copy(Scalar* output, Scalar const* input, ElementwiseCopyPlan64 const& plan,    \
                                            cudaStream_t stream, int device)                                           \
  {                                                                                                                    \
    enqueue_elementwise_conjugating_copy_impl(output, input, plan, stream, device);                                    \
  }

UNI20_DEFINE_ELEMENTWISE_CONJUGATING_COPY(uni20::cfloat)
UNI20_DEFINE_ELEMENTWISE_CONJUGATING_COPY(uni20::cdouble)

#undef UNI20_DEFINE_ELEMENTWISE_CONJUGATING_COPY

#define UNI20_DEFINE_ELEMENTWISE_CONJUGATE(Scalar)                                                                     \
  void enqueue_elementwise_conjugate(Scalar* output, ElementwiseConjugatePlan32 const& plan, cudaStream_t stream,      \
                                     int device)                                                                       \
  {                                                                                                                    \
    enqueue_elementwise_conjugate_impl(output, plan, stream, device);                                                  \
  }                                                                                                                    \
                                                                                                                       \
  void enqueue_elementwise_conjugate(Scalar* output, ElementwiseConjugatePlan64 const& plan, cudaStream_t stream,      \
                                     int device)                                                                       \
  {                                                                                                                    \
    enqueue_elementwise_conjugate_impl(output, plan, stream, device);                                                  \
  }

UNI20_DEFINE_ELEMENTWISE_CONJUGATE(uni20::cfloat)
UNI20_DEFINE_ELEMENTWISE_CONJUGATE(uni20::cdouble)

#undef UNI20_DEFINE_ELEMENTWISE_CONJUGATE

#define UNI20_DEFINE_ELEMENTWISE_SCALE(Scalar, Factor)                                                                 \
  void enqueue_elementwise_scale(Scalar* output, Scalar const* input, Factor factor,                                   \
                                 ElementwiseScalePlan32 const& plan, cudaStream_t stream, int device)                  \
  {                                                                                                                    \
    enqueue_elementwise_scale_impl(output, input, factor, plan, stream, device);                                       \
  }                                                                                                                    \
                                                                                                                       \
  void enqueue_elementwise_scale(Scalar* output, Scalar const* input, Factor factor,                                   \
                                 ElementwiseScalePlan64 const& plan, cudaStream_t stream, int device)                  \
  {                                                                                                                    \
    enqueue_elementwise_scale_impl(output, input, factor, plan, stream, device);                                       \
  }

UNI20_DEFINE_ELEMENTWISE_SCALE(float, float)
UNI20_DEFINE_ELEMENTWISE_SCALE(double, double)
UNI20_DEFINE_ELEMENTWISE_SCALE(uni20::cfloat, float)
UNI20_DEFINE_ELEMENTWISE_SCALE(uni20::cfloat, uni20::cfloat)
UNI20_DEFINE_ELEMENTWISE_SCALE(uni20::cdouble, double)
UNI20_DEFINE_ELEMENTWISE_SCALE(uni20::cdouble, uni20::cdouble)

#undef UNI20_DEFINE_ELEMENTWISE_SCALE

#define UNI20_INSTANTIATE_FILL(Scalar)                                                                                 \
  template void enqueue_elementwise_fill(Scalar*, Scalar, ElementwiseFillPlan32 const&, cudaStream_t, int);            \
  template void enqueue_elementwise_fill(Scalar*, Scalar, ElementwiseFillPlan64 const&, cudaStream_t, int)

UNI20_INSTANTIATE_FILL(float);
UNI20_INSTANTIATE_FILL(double);
UNI20_INSTANTIATE_FILL(uni20::cfloat);
UNI20_INSTANTIATE_FILL(uni20::cdouble);

#undef UNI20_INSTANTIATE_FILL

#define UNI20_INSTANTIATE_INPLACE_SCALE(Scalar, Factor)                                                                \
  template void enqueue_elementwise_inplace_scale(Scalar*, Factor, ElementwiseInplaceUnaryPlan32 const&, cudaStream_t, \
                                                  int);                                                                \
  template void enqueue_elementwise_inplace_scale(Scalar*, Factor, ElementwiseInplaceUnaryPlan64 const&, cudaStream_t, \
                                                  int)

UNI20_INSTANTIATE_INPLACE_SCALE(float, float);
UNI20_INSTANTIATE_INPLACE_SCALE(double, double);
UNI20_INSTANTIATE_INPLACE_SCALE(uni20::cfloat, float);
UNI20_INSTANTIATE_INPLACE_SCALE(uni20::cfloat, uni20::cfloat);
UNI20_INSTANTIATE_INPLACE_SCALE(uni20::cdouble, double);
UNI20_INSTANTIATE_INPLACE_SCALE(uni20::cdouble, uni20::cdouble);

#undef UNI20_INSTANTIATE_INPLACE_SCALE

#define UNI20_INSTANTIATE_ADD_SCALED(Scalar, Factor)                                                                   \
  template void enqueue_elementwise_add_scaled(Scalar*, Scalar const*, Factor, ElementwiseInplaceBinaryPlan32 const&,  \
                                               cudaStream_t, int);                                                     \
  template void enqueue_elementwise_add_scaled(Scalar*, Scalar const*, Factor, ElementwiseInplaceBinaryPlan64 const&,  \
                                               cudaStream_t, int)

UNI20_INSTANTIATE_ADD_SCALED(float, float);
UNI20_INSTANTIATE_ADD_SCALED(double, double);
UNI20_INSTANTIATE_ADD_SCALED(uni20::cfloat, float);
UNI20_INSTANTIATE_ADD_SCALED(uni20::cfloat, uni20::cfloat);
UNI20_INSTANTIATE_ADD_SCALED(uni20::cdouble, double);
UNI20_INSTANTIATE_ADD_SCALED(uni20::cdouble, uni20::cdouble);

#undef UNI20_INSTANTIATE_ADD_SCALED

#define UNI20_INSTANTIATE_UNARY(Function, Scalar)                                                                      \
  template void enqueue_elementwise_unary(Scalar*, Scalar const*, Function, ElementwiseUnaryPlan32 const&,             \
                                          cudaStream_t, int);                                                          \
  template void enqueue_elementwise_unary(Scalar*, Scalar const*, Function, ElementwiseUnaryPlan64 const&,             \
                                          cudaStream_t, int)

#define UNI20_INSTANTIATE_UNARY_SCALARS(Function)                                                                      \
  UNI20_INSTANTIATE_UNARY(Function, float);                                                                            \
  UNI20_INSTANTIATE_UNARY(Function, double);                                                                           \
  UNI20_INSTANTIATE_UNARY(Function, uni20::cfloat);                                                                    \
  UNI20_INSTANTIATE_UNARY(Function, uni20::cdouble)

UNI20_INSTANTIATE_UNARY_SCALARS(uni20::linalg::negate);
UNI20_INSTANTIATE_UNARY_SCALARS(uni20::linalg::square);
UNI20_INSTANTIATE_UNARY_SCALARS(uni20::linalg::reciprocal);

#undef UNI20_INSTANTIATE_UNARY_SCALARS
#undef UNI20_INSTANTIATE_UNARY

#define UNI20_INSTANTIATE_BINARY(Function, Scalar)                                                                     \
  template void enqueue_elementwise_binary(Scalar*, Scalar const*, Scalar const*, Function,                            \
                                           ElementwiseBinaryPlan32 const&, cudaStream_t, int);                         \
  template void enqueue_elementwise_binary(Scalar*, Scalar const*, Scalar const*, Function,                            \
                                           ElementwiseBinaryPlan64 const&, cudaStream_t, int)

#define UNI20_INSTANTIATE_BINARY_SCALARS(Function)                                                                     \
  UNI20_INSTANTIATE_BINARY(Function, float);                                                                           \
  UNI20_INSTANTIATE_BINARY(Function, double);                                                                          \
  UNI20_INSTANTIATE_BINARY(Function, uni20::cfloat);                                                                   \
  UNI20_INSTANTIATE_BINARY(Function, uni20::cdouble)

UNI20_INSTANTIATE_BINARY_SCALARS(uni20::linalg::add);
UNI20_INSTANTIATE_BINARY_SCALARS(uni20::linalg::subtract);
UNI20_INSTANTIATE_BINARY_SCALARS(uni20::linalg::multiply);
UNI20_INSTANTIATE_BINARY_SCALARS(uni20::linalg::divide);

#undef UNI20_INSTANTIATE_BINARY_SCALARS
#undef UNI20_INSTANTIATE_BINARY

#define UNI20_INSTANTIATE_INPLACE_BINARY(Function, Scalar)                                                             \
  template void enqueue_elementwise_inplace_binary(Scalar*, Scalar const*, Function,                                   \
                                                   ElementwiseInplaceBinaryPlan32 const&, cudaStream_t, int);          \
  template void enqueue_elementwise_inplace_binary(Scalar*, Scalar const*, Function,                                   \
                                                   ElementwiseInplaceBinaryPlan64 const&, cudaStream_t, int)

#define UNI20_INSTANTIATE_INPLACE_BINARY_SCALARS(Function)                                                             \
  UNI20_INSTANTIATE_INPLACE_BINARY(Function, float);                                                                   \
  UNI20_INSTANTIATE_INPLACE_BINARY(Function, double);                                                                  \
  UNI20_INSTANTIATE_INPLACE_BINARY(Function, uni20::cfloat);                                                           \
  UNI20_INSTANTIATE_INPLACE_BINARY(Function, uni20::cdouble)

UNI20_INSTANTIATE_INPLACE_BINARY_SCALARS(uni20::linalg::add);
UNI20_INSTANTIATE_INPLACE_BINARY_SCALARS(uni20::linalg::subtract);
UNI20_INSTANTIATE_INPLACE_BINARY_SCALARS(uni20::linalg::multiply);
UNI20_INSTANTIATE_INPLACE_BINARY_SCALARS(uni20::linalg::divide);

#undef UNI20_INSTANTIATE_INPLACE_BINARY_SCALARS
#undef UNI20_INSTANTIATE_INPLACE_BINARY

} // namespace uni20::linalg::detail::cuda_reference
