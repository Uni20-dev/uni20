#include <uni20/async/debug_cuda_scheduler.hpp>
#include <uni20/backend/cuda/runtime.hpp>
#include <uni20/linalg/async/copy.hpp>
#include <uni20/linalg/async/reductions.hpp>
#include <uni20/linalg/backends/cuda/reductions.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/copy_into.hpp>
#include <uni20/tensor/mdspec_tensor_view.hpp>
#include <uni20/tensor/reductions.hpp>
#include <uni20/tensor/tensor.hpp>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <cmath>
#include <complex>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <thread>
#include <tuple>
#include <utility>
#include <vector>

namespace
{

using uni20::index_type;
namespace reference = uni20::linalg::detail::cuda_reference;

// A strided view of a CUDA tensor's storage, usable wherever a tensor is.
template <std::size_t Rank, class Tensor>
auto cuda_strided_view(Tensor& tensor, std::array<index_type, Rank> const& extents,
                       std::array<index_type, Rank> const& strides, std::size_t offset = 0)
{
  auto mutable_base = tensor.mdspec();
  auto const_base = std::as_const(tensor).mdspec();
  using mutable_base_type = decltype(mutable_base);
  using const_base_type = decltype(const_base);
  using extents_type = stdex::dextents<index_type, Rank>;
  using mapping_type = uni20::layout_stride::mapping<extents_type>;
  using mutable_type =
      uni20::mdspec<typename mutable_base_type::element_type, extents_type, uni20::layout_stride,
                    typename mutable_base_type::accessor_type, typename mutable_base_type::data_descriptor_type>;
  using const_type =
      uni20::mdspec<typename const_base_type::element_type, extents_type, uni20::layout_stride,
                    typename const_base_type::accessor_type, typename const_base_type::data_descriptor_type>;
  mapping_type const mapping(extents_type(extents), strides);
  return uni20::MdspecTensorView<mutable_type, const_type, uni20::CudaStorage>{
      mutable_type{mutable_base.data_descriptor().offset_by(offset), mapping, mutable_base.accessor()},
      const_type{const_base.data_descriptor().offset_by(offset), mapping, const_base.accessor()}};
}

// The same strided view of a host tensor's storage.
template <std::size_t Rank, class Tensor>
auto host_strided_view(Tensor& tensor, std::array<index_type, Rank> const& extents,
                       std::array<index_type, Rank> const& strides, std::size_t offset = 0)
{
  using element_type = typename Tensor::value_type;
  using extents_type = stdex::dextents<index_type, Rank>;
  using mapping_type = uni20::layout_stride::mapping<extents_type>;
  using mutable_type = uni20::mdspan<element_type, extents_type, uni20::layout_stride>;
  using const_type = uni20::mdspan<element_type const, extents_type, uni20::layout_stride>;
  mapping_type const mapping(extents_type(extents), strides);
  element_type* data = tensor.mdspan().data_handle() + offset;
  return uni20::MdspecTensorView<mutable_type, const_type, uni20::HostStorage>{mutable_type{data, mapping},
                                                                               const_type{data, mapping}};
}

// Deterministic values in [-1, 1), with distinct real and imaginary parts.
template <class Scalar> Scalar test_value(std::uint64_t index)
{
  auto component = [](std::uint64_t seed) {
    seed = seed * 6364136223846793005ULL + 1442695040888963407ULL;
    return static_cast<double>(seed >> 11) * 0x1p-52 - 1.0;
  };
  if constexpr (uni20::is_complex_v<Scalar>)
  {
    using real_type = uni20::make_real_t<Scalar>;
    return Scalar(static_cast<real_type>(component(2 * index)), static_cast<real_type>(component(2 * index + 1)));
  }
  else
  {
    return static_cast<Scalar>(component(index));
  }
}

template <class Tensor> void fill_test_values(Tensor& tensor, std::uint64_t seed)
{
  using scalar_type = typename Tensor::value_type;
  auto span = tensor.mdspan();
  scalar_type* data = span.data_handle();
  auto const size = static_cast<std::size_t>(span.mapping().required_span_size());
  for (std::size_t index = 0; index < size; ++index)
    data[index] = test_value<scalar_type>(seed + index);
}

template <class Scalar> double tolerance() { return std::same_as<uni20::make_real_t<Scalar>, float> ? 1e-5 : 1e-12; }

template <class Scalar> void expect_close(Scalar actual, Scalar expected)
{
  double const scale = std::max(1.0, static_cast<double>(std::abs(expected)));
  EXPECT_LE(static_cast<double>(std::abs(actual - expected)), tolerance<Scalar>() * scale)
      << "actual " << actual << ", expected " << expected;
}

template <class Value> auto bits(Value value)
{
  if constexpr (sizeof(Value) == sizeof(std::uint32_t))
    return std::bit_cast<std::uint32_t>(value);
  else
    return std::bit_cast<std::uint64_t>(value);
}

double norm_of(std::initializer_list<double> values)
{
  reference::NormPartial partial{};
  for (double value : values)
    reference::accumulate_norm(partial, std::fabs(value));
  return reference::finish_norm(partial);
}

class CudaReductionTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
      cudaError_t const status = cudaGetDeviceCount(&device_count_);
      if (status != cudaSuccess) GTEST_SKIP() << "CUDA device discovery failed: " << cudaGetErrorString(status);
      if (device_count_ == 0) GTEST_SKIP() << "no CUDA devices are available";
    }

    int device_count_ = 0;
};

// Two logical elements at offsets 1 and 2 from the descriptor's origin.
struct ShiftedLayout
{
    template <class Extents> struct mapping : uni20::layout_right::mapping<Extents>
    {
        static_assert(Extents::rank() == 1 && Extents::static_extent(0) == 2);
        using layout_type = ShiftedLayout;
        static constexpr bool is_always_exhaustive() noexcept { return false; }
        static constexpr bool is_exhaustive() noexcept { return false; }
        constexpr index_type operator()(index_type i) const noexcept { return 1 + i; }
        constexpr index_type required_span_size() const noexcept { return 3; }
        constexpr uni20::offset_range reachable_range() const noexcept { return {.minimum = 1, .maximum = 2}; }
    };
};

TEST_F(CudaReductionTest, AddsMappingOriginAndDescriptorOffsetOnce)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  uni20::Tensor<double, 1> host(4);
  host[0] = 200;
  host[1] = 100;
  host[2] = 2;
  host[3] = 3;
  auto const device = uni20::to_device(host, 0);
  auto const base = device.mdspec();
  using shape = stdex::extents<index_type, 2>;
  using spec = uni20::mdspec<double const, shape, ShiftedLayout, typename decltype(base)::accessor_type,
                             typename decltype(base)::data_descriptor_type>;
  spec descriptor{base.data_descriptor().offset_by(1), ShiftedLayout::mapping<shape>{}, base.accessor()};
  uni20::MdspecTensorView<spec, spec, uni20::CudaStorage> view{descriptor, descriptor};
  EXPECT_EQ(uni20::sum_host(view), 5);
  EXPECT_EQ(uni20::maximum_host(view), 3);
}

TEST_F(CudaReductionTest, CanonicalScalarSupportsTransfersAndHostReductions)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  uni20::ScalarTensor<double> host;
  host[] = -7;
  auto const device = uni20::to_device(host, 0);
  static_assert(uni20::StridedMdspecLike<decltype(device.mdspec())>);
  EXPECT_EQ(uni20::to_host(device)[], -7);
  EXPECT_EQ(uni20::sum_host(device), -7);
  EXPECT_EQ(uni20::maximum_host(device), -7);
  EXPECT_EQ(uni20::norm_host(device), 7);
  EXPECT_EQ(uni20::inner_product_host(device, device), 49);

  uni20::CudaTensor<double, 0> copy(runtime.device_resources(0), uni20::extents<index_type>{});
  uni20::copy(copy, device);
  EXPECT_EQ(uni20::to_host(copy)[], -7);
}

TEST_F(CudaReductionTest, AsyncSumsSuspendUntilOneStreamIsAvailable)
{
  for (std::size_t capacity : {1U, 4U})
  {
    SCOPED_TRACE(capacity);
    auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = capacity});
    uni20::async::DebugCudaScheduler scheduler(uni20::cuda::Device::get(0));
    uni20::async::ScopedScheduler scoped(&scheduler);
    uni20::Tensor<double, 1> host(2);
    host[0] = 2;
    host[1] = 3;
    uni20::async::Async<uni20::CudaTensor<double, 1>> input = uni20::to_device(host, 0);

    std::vector<uni20::cuda::Stream> occupied;
    for (std::size_t i = 0; i < capacity; ++i)
      occupied.push_back(runtime.device_resources(0).streams().acquire());

    auto first = uni20::sum_host(input);
    auto second = uni20::sum_host(input);
    auto first_reader = first.read();
    auto second_reader = second.read();
    scheduler.run_all(uni20::async::ExternalWaitPolicy::allow);
    EXPECT_FALSE(first_reader.await_ready());
    EXPECT_FALSE(second_reader.await_ready());
    EXPECT_EQ(scheduler.external_waits(), 2U);

    // One available lease must suffice for both sums, including workspace allocation.
    occupied.pop_back();
    EXPECT_EQ(first_reader.get_wait(scheduler), 5);
    EXPECT_EQ(second_reader.get_wait(scheduler), 5);
  }
}

TEST_F(CudaReductionTest, FreshCopyAndHostReductionsSuspendOnOneStream)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  uni20::async::DebugCudaScheduler scheduler;
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::Tensor<double, 1> host(2);
  host[0] = 3;
  host[1] = 4;
  uni20::async::Async<uni20::CudaTensor<double, 1>> input = uni20::to_device(host, 0);
  uni20::async::Async<uni20::CudaTensor<double, 1>> output;
  host[0] = 2;
  host[1] = -1;
  uni20::async::Async<uni20::CudaTensor<double, 1>> rhs = uni20::to_device(host, 0);
  auto& pool = runtime.device_resources(0).streams();
  auto held = pool.acquire();
  auto sum = uni20::sum_host(input);
  uni20::copy(output, input);
#if UNI20_BACKEND_CUBLAS
  auto norm = uni20::norm_host(uni20::linalg::CublasBackend{}, input);
  auto dot = uni20::inner_product_host(uni20::linalg::CublasBackend{}, input, rhs);
#else
  auto norm = uni20::norm_host(input);
  auto dot = uni20::inner_product_host(input, rhs);
#endif
  auto reference_norm = uni20::norm_host(uni20::linalg::CudaReferenceBackend{}, input);
  auto reference_dot = uni20::inner_product_host(uni20::linalg::CudaReferenceBackend{}, input, rhs);

  scheduler.run_all(uni20::async::ExternalWaitPolicy::allow);
  EXPECT_EQ(scheduler.external_waits(), 6U);
  held = {};
  EXPECT_EQ(sum.get_wait(scheduler), 7);
  EXPECT_EQ(norm.get_wait(scheduler), 5);
  EXPECT_EQ(reference_norm.get_wait(scheduler), 5);
  EXPECT_EQ(dot.get_wait(scheduler), 2);
  EXPECT_EQ(reference_dot.get_wait(scheduler), 2);
  auto copied = uni20::to_host(output.get_wait(scheduler));
  EXPECT_EQ(copied[0], 3);
  EXPECT_EQ(copied[1], 4);
}

TEST_F(CudaReductionTest, SignedMaximumReadsOnlyStridedElements)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  uni20::Tensor<float, 2> host(2, 1030);
  for (index_type i = 0; i < 1030; ++i)
  {
    host[0, i] = -8.0F;
    host[1, i] = 100.0F;
  }
  host[0, 1029] = -2.0F;

  auto device = uni20::to_device(host, 0);
  auto first_row = cuda_strided_view(device, std::array<index_type, 1>{1030}, std::array<index_type, 1>{2});
  EXPECT_EQ(uni20::maximum_host(first_row), -2.0F);

  host[0, 775] = std::numeric_limits<float>::quiet_NaN();
  uni20::copy(device, host);
  EXPECT_TRUE(std::isnan(uni20::maximum_host(first_row)));
}

template <class Scalar> void expect_compensated_sum()
{
  using real_type = uni20::make_real_t<Scalar>;
  constexpr index_type block = reference::reduction_threads;
  constexpr index_type grid = block * reference::reduction_maximum_blocks;
  uni20::Tensor<Scalar, 1> host(2 * grid + 4);
  Scalar large{0x1p60};
  Scalar unit{1};
  if constexpr (uni20::is_complex_v<Scalar>)
  {
    large.imag(0x1p61);
    unit.imag(-3);
  }

  // Three grid-stride loads by one thread.
  host[3] = large;
  host[grid + 3] = unit;
  host[2 * grid + 3] = -large;
  // Three lanes in one block. The first tree step pairs the large and small terms.
  host[block] = large;
  host[block + block / 2] = real_type{2} * unit;
  host[block + block / 4] = -large;
  // Three separate blocks, paired the same way in the final reduction.
  host[2 * block] = large;
  host[(2 + block / 2) * block] = real_type{4} * unit;
  host[(2 + block / 4) * block] = -large;

  auto const expected = real_type{7} * unit;
  EXPECT_EQ(uni20::sum_host(host), expected);
  auto device = uni20::to_device(host, 0);
  EXPECT_EQ(uni20::sum_host(device), expected);
}

TEST_F(CudaReductionTest, SumRecoversCancellationWithinThreadsAndAcrossBlocks)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  expect_compensated_sum<float>();
  expect_compensated_sum<double>();
  expect_compensated_sum<uni20::cfloat>();
  expect_compensated_sum<uni20::cdouble>();
}

TEST_F(CudaReductionTest, SumPreservesNonfiniteComponents)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  double const inf = std::numeric_limits<double>::infinity();
  uni20::Tensor<uni20::cdouble, 1> host(3);
  host[0] = {inf, -inf};
  host[1] = {1, 2};
  EXPECT_EQ(uni20::sum_host(uni20::to_device(host, 0)), (uni20::cdouble{inf, -inf}));

  host[2] = {-inf, 3};
  auto result = uni20::sum_host(uni20::to_device(host, 0));
  EXPECT_TRUE(std::isnan(result.real()));
  EXPECT_EQ(result.imag(), -inf);

  host[0] = {1, std::numeric_limits<double>::quiet_NaN()};
  host[2] = {4, 5};
  result = uni20::sum_host(uni20::to_device(host, 0));
  EXPECT_EQ(result.real(), 6);
  EXPECT_TRUE(std::isnan(result.imag()));
}

TEST_F(CudaReductionTest, PairedDiagonalSumTransfersOnlyTheScalar)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  using scalar_type = uni20::cdouble;
  uni20::RowMajorTensor<scalar_type, 4> host(4, 3, 5, 2);
  // Large off-diagonal values reveal a mistaken stride or pair order.
  std::fill_n(host.storage().data(), host.size(), scalar_type{1000.0, -1000.0});
  for (index_type i = 0; i < 2; ++i)
    for (index_type j = 0; j < 3; ++j)
      host[i, j, j, i] = {static_cast<double>(3 * i + j + 1), static_cast<double>(i - j)};

  auto device = uni20::to_device(host, 0);
  auto diagonal = uni20::diagonal_view<0, 3, 1, 2>(std::as_const(device));
  static_assert(!uni20::ImmediateTensorView<decltype(diagonal)>);
  auto const expected = scalar_type{21.0, -3.0};
  EXPECT_EQ(uni20::sum_host(diagonal), expected);
  EXPECT_EQ(uni20::sum_host(uni20::diagonal_view<0, 3, 1, 2>(host)), expected);

  uni20::CudaTensor<scalar_type, 4> empty(runtime.device_resources(0), 2, 0, 3, 2);
  EXPECT_EQ(uni20::sum_host(uni20::diagonal_view<0, 3, 1, 2>(empty)), scalar_type{});
}

TEST(CudaReductionPartialTest, NormCombinesScaledRangesLikeReferenceBlas)
{
  EXPECT_EQ(norm_of({}), 0.0);
  EXPECT_DOUBLE_EQ(norm_of({3.0, 4.0}), 5.0);
  // Squaring these directly would overflow or underflow.
  EXPECT_DOUBLE_EQ(norm_of({3e200, 4e200}), 5e200);
  EXPECT_DOUBLE_EQ(norm_of({3e-200, 4e-200}), 5e-200);
  EXPECT_DOUBLE_EQ(norm_of({-3e-300, 4e-300}), 5e-300);
  // Mixed ranges.
  EXPECT_DOUBLE_EQ(norm_of({1e300, 1.0, 1e-300}), 1e300);
  EXPECT_DOUBLE_EQ(norm_of({1.0, 1e-160}), 1.0);
  EXPECT_DOUBLE_EQ(norm_of({3e-160, 4e-160, 1e-200}), 5e-160);
  EXPECT_TRUE(std::isinf(norm_of({1.0, std::numeric_limits<double>::infinity()})));
  EXPECT_TRUE(std::isnan(norm_of({1.0, std::numeric_limits<double>::quiet_NaN()})));
  EXPECT_TRUE(std::isnan(norm_of({1e300, std::numeric_limits<double>::quiet_NaN()})));

  reference::NormPartial left{};
  reference::NormPartial right{};
  reference::accumulate_norm(left, 3e200);
  reference::accumulate_norm(right, 4e200);
  reference::accumulate_norm(right, 1e-300);
  EXPECT_DOUBLE_EQ(reference::finish_norm(reference::combine(left, right)), 5e200);
}

template <class Scalar> void expect_strided_reductions_match_host()
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto& resources = runtime.device_resources(0);
  uni20::Tensor<Scalar, 2> host_lhs(5, 6);
  uni20::Tensor<Scalar, 2> host_rhs(5, 6);
  fill_test_values(host_lhs, 1);
  fill_test_values(host_rhs, 101);
  auto const lhs = uni20::to_device(host_lhs, 0);
  uni20::CudaMatrix<Scalar, uni20::RowMajor> row_major_rhs(resources, 5, 6);
  uni20::copy(row_major_rhs, host_rhs);

  // Operands in different orders.
  expect_close(uni20::inner_product_host(lhs, row_major_rhs), uni20::inner_product_host(host_lhs, host_rhs));
  expect_close(uni20::norm_host(row_major_rhs), uni20::norm_host(host_rhs));

  // Every other column, and columns of one buffer at different offsets.
  auto device_lhs = uni20::to_device(host_lhs, 0);
  auto device_slice = cuda_strided_view<2>(device_lhs, {5, 3}, {1, 10});
  auto host_slice = host_strided_view<2>(host_lhs, {5, 3}, {1, 10});
  expect_close(uni20::norm_host(device_slice), uni20::norm_host(host_slice));
  auto device_shifted = cuda_strided_view<2>(device_lhs, {5, 3}, {1, 10}, 5);
  auto host_shifted = host_strided_view<2>(host_lhs, {5, 3}, {1, 10}, 5);
  expect_close(uni20::inner_product_host(device_slice, device_shifted),
               uni20::inner_product_host(host_slice, host_shifted));

  // A transposed view against a row-major tensor of the same logical shape.
  auto device_transposed = cuda_strided_view<2>(device_lhs, {6, 5}, {5, 1});
  auto host_transposed = host_strided_view<2>(host_lhs, {6, 5}, {5, 1});
  uni20::CudaMatrix<Scalar, uni20::RowMajor> device_other(resources, 6, 5);
  uni20::Tensor<Scalar, 2> host_other(6, 5);
  fill_test_values(host_other, 501);
  uni20::copy(device_other, host_other);
  expect_close(uni20::inner_product_host(device_transposed, device_other),
               uni20::inner_product_host(host_transposed, host_other));
}

TEST_F(CudaReductionTest, StridedAndDifferentlyOrderedOperandsMatchTheHost)
{
  expect_strided_reductions_match_host<float>();
  expect_strided_reductions_match_host<double>();
  expect_strided_reductions_match_host<uni20::cfloat>();
  expect_strided_reductions_match_host<uni20::cdouble>();
}

TEST_F(CudaReductionTest, NormIsSafeFromOverflowAndUnderflow)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::Tensor<double, 1> host(4);
  host[0] = 3e200;
  host[1] = 0.0;
  host[2] = 4e200;
  host[3] = 0.0;
  auto device = uni20::to_device(host, 0);
  auto every_other = cuda_strided_view<1>(device, {2}, {2});
  EXPECT_DOUBLE_EQ(uni20::norm_host(every_other), 5e200);

  host[0] = 3e-200;
  host[2] = -4e-200;
  uni20::copy(device, host);
  EXPECT_DOUBLE_EQ(uni20::norm_host(every_other), 5e-200);

  uni20::Tensor<uni20::cdouble, 1> complex_host(2);
  complex_host[0] = uni20::cdouble(3e200, 0.0);
  complex_host[1] = uni20::cdouble(0.0, 4e200);
  auto complex_device = uni20::to_device(complex_host, 0);
  auto complex_view = cuda_strided_view<1>(complex_device, {2}, {1});
  double result = 0.0;
  auto input = uni20::mdspec_of(std::as_const(complex_view));
  ASSERT_TRUE(uni20::linalg::try_dispatch_kernel(uni20::linalg::CudaReferenceBackend{}, uni20::linalg::norm_op{},
                                                 result, input));
  EXPECT_DOUBLE_EQ(result, 5e200);
}

TEST_F(CudaReductionTest, ResultsAreBitwiseReproducible)
{
  constexpr index_type count = 1'000'003;
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::Tensor<double, 1> host_lhs(2 * count);
  uni20::Tensor<uni20::cdouble, 1> host_rhs(2 * count);
  fill_test_values(host_lhs, 7);
  fill_test_values(host_rhs, 9);
  auto device_lhs = uni20::to_device(host_lhs, 0);
  auto device_rhs = uni20::to_device(host_rhs, 0);
  auto lhs = cuda_strided_view<1>(device_lhs, {count}, {2});
  auto complex_lhs = cuda_strided_view<1>(device_rhs, {count}, {2});
  auto complex_rhs = cuda_strided_view<1>(device_rhs, {count}, {2}, 1);

  double const norm = uni20::norm_host(lhs);
  uni20::cdouble const product = uni20::inner_product_host(complex_lhs, complex_rhs);
  for (int repeat = 0; repeat < 3; ++repeat)
  {
    EXPECT_EQ(bits(uni20::norm_host(lhs)), bits(norm));
    uni20::cdouble const again = uni20::inner_product_host(complex_lhs, complex_rhs);
    EXPECT_EQ(bits(again.real()), bits(product.real()));
    EXPECT_EQ(bits(again.imag()), bits(product.imag()));
  }
  expect_close(norm, uni20::norm_host(host_strided_view<1>(host_lhs, {count}, {2})));
}

TEST_F(CudaReductionTest, ResultsMatchBitwiseAcrossDevicesOfOneKind)
{
  if (device_count_ < 2) GTEST_SKIP() << "requires two CUDA devices";
  cudaDeviceProp first{};
  cudaDeviceProp second{};
  ASSERT_EQ(cudaGetDeviceProperties(&first, 0), cudaSuccess);
  ASSERT_EQ(cudaGetDeviceProperties(&second, 1), cudaSuccess);
  if (first.major != second.major || first.minor != second.minor)
    GTEST_SKIP() << "the two devices have different compute capabilities";

  constexpr index_type count = 300'007;
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0, 1}, .streams_per_device = 2});
  uni20::Tensor<uni20::cfloat, 1> host(3 * count);
  fill_test_values(host, 11);
  auto device0 = uni20::to_device(host, 0);
  auto device1 = uni20::to_device(host, 1);
  auto view0 = cuda_strided_view<1>(device0, {count}, {3});
  auto view1 = cuda_strided_view<1>(device1, {count}, {3});
  EXPECT_EQ(bits(uni20::norm_host(view0)), bits(uni20::norm_host(view1)));
  auto const product0 = uni20::inner_product_host(view0, cuda_strided_view<1>(device0, {count}, {3}, 1));
  auto const product1 = uni20::inner_product_host(view1, cuda_strided_view<1>(device1, {count}, {3}, 1));
  EXPECT_EQ(bits(product0.real()), bits(product1.real()));
  EXPECT_EQ(bits(product0.imag()), bits(product1.imag()));
}

TEST_F(CudaReductionTest, RanksBeyondTheKernelRankReduceSliceBySlice)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  // Strides 3^k with extent 2 cannot merge, so the compact rank stays at nine.
  constexpr std::size_t rank = 9;
  std::array<index_type, rank> extents{};
  std::array<index_type, rank> strides{};
  index_type stride = 1;
  for (std::size_t axis = 0; axis < rank; ++axis)
  {
    extents[axis] = 2;
    strides[axis] = stride;
    stride *= 3;
  }
  uni20::Tensor<double, 1> host(stride + 1);
  fill_test_values(host, 21);
  auto device = uni20::to_device(host, 0);
  auto device_view = cuda_strided_view<rank>(device, extents, strides);
  auto device_shifted = cuda_strided_view<rank>(device, extents, strides, 1);

  reference::LoweredStridedElementwisePlan<1, reference::reduction_maximum_rank> plan;
  ASSERT_TRUE(reference::try_lower_strided_elementwise_plan<reference::reduction_maximum_rank>(
      uni20::make_multi_iteration_plan(std::tuple{uni20::mdspec_of(std::as_const(device_view)).mapping()}), plan));
  ASSERT_EQ(plan.host_loop_dimensions.size(), 1U);

  expect_close(uni20::norm_host(device_view), uni20::norm_host(host_strided_view<rank>(host, extents, strides)));
  expect_close(uni20::inner_product_host(device_view, device_shifted),
               uni20::inner_product_host(host_strided_view<rank>(host, extents, strides),
                                         host_strided_view<rank>(host, extents, strides, 1)));
}

TEST_F(CudaReductionTest, SixtyFourBitPlansGiveTheSameBitsAsThirtyTwoBitPlans)
{
  constexpr index_type count = 100'003;
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::Tensor<uni20::cdouble, 1> host(3 * count);
  fill_test_values(host, 31);
  auto device = uni20::to_device(host, 0);
  auto lhs_view = cuda_strided_view<1>(device, {count}, {3});
  auto rhs_view = cuda_strided_view<1>(device, {count}, {3}, 2);
  auto lhs = uni20::mdspec_of(std::as_const(lhs_view));
  auto rhs = uni20::mdspec_of(std::as_const(rhs_view));

  auto const norm_source = uni20::make_multi_iteration_plan(std::tuple{lhs.mapping()});
  reference::LoweredStridedElementwisePlan<1, reference::reduction_maximum_rank> norm_32;
  ASSERT_TRUE(reference::try_lower_strided_elementwise_plan<reference::reduction_maximum_rank>(norm_source, norm_32));
  ASSERT_EQ(norm_32.index_kind, reference::ElementwiseIndexKind::index_32);
  reference::LoweredStridedElementwisePlan<1, reference::reduction_maximum_rank> norm_64;
  norm_64.index_kind = reference::ElementwiseIndexKind::index_64;
  ASSERT_TRUE(reference::elementwise_plan_detail::try_lower(norm_source, norm_64.plan_64, 0));
  auto stream = runtime.device_resources(0).streams().acquire();
  EXPECT_EQ(bits(reference::reference_norm_of_plan<uni20::cdouble>(norm_64, lhs, stream)),
            bits(reference::reference_norm_of_plan<uni20::cdouble>(norm_32, lhs, stream)));

  auto const product_source = uni20::make_multi_iteration_plan(std::tuple{lhs.mapping(), rhs.mapping()});
  reference::LoweredStridedElementwisePlan<2, reference::reduction_maximum_rank> product_32;
  ASSERT_TRUE(
      reference::try_lower_strided_elementwise_plan<reference::reduction_maximum_rank>(product_source, product_32));
  reference::LoweredStridedElementwisePlan<2, reference::reduction_maximum_rank> product_64;
  product_64.index_kind = reference::ElementwiseIndexKind::index_64;
  ASSERT_TRUE(reference::elementwise_plan_detail::try_lower(product_source, product_64.plan_64, 0));
  auto const narrow = reference::reference_inner_product_of_plan<uni20::cdouble>(product_32, lhs, rhs, stream);
  auto const wide = reference::reference_inner_product_of_plan<uni20::cdouble>(product_64, lhs, rhs, stream);
  EXPECT_EQ(bits(wide.real()), bits(narrow.real()));
  EXPECT_EQ(bits(wide.imag()), bits(narrow.imag()));
}

// The workspace is allocated before the execution stream is leased: allocation
// leases a stream from the same pool, so the other order waits for itself.
TEST_F(CudaReductionTest, ReductionsCompleteWithOneStreamPerDevice)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  uni20::Tensor<double, 2> host(5, 6);
  fill_test_values(host, 41);
  auto device = uni20::to_device(host, 0);
  auto slice = cuda_strided_view<2>(device, {5, 3}, {1, 10});
  auto shifted = cuda_strided_view<2>(device, {5, 3}, {1, 10}, 5);
  expect_close(uni20::norm_host(slice), uni20::norm_host(host_strided_view<2>(host, {5, 3}, {1, 10})));
  expect_close(uni20::inner_product_host(slice, shifted),
               uni20::inner_product_host(host_strided_view<2>(host, {5, 3}, {1, 10}),
                                         host_strided_view<2>(host, {5, 3}, {1, 10}, 5)));
}

TEST_F(CudaReductionTest, ConcurrentReductionsShareTwoStreams)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::Tensor<double, 1> host(2 * 4096);
  fill_test_values(host, 51);
  auto device = uni20::to_device(host, 0);
  auto view = cuda_strided_view<1>(device, {4096}, {2});
  auto const expected = bits(uni20::norm_host(view));
  std::atomic<int> mismatches{0};
  std::vector<std::thread> threads;
  for (int thread = 0; thread < 8; ++thread)
  {
    threads.emplace_back([&] {
      for (int call = 0; call < 50; ++call)
      {
        if (bits(uni20::norm_host(view)) != expected) ++mismatches;
      }
    });
  }
  for (auto& thread : threads)
    thread.join();
  EXPECT_EQ(mismatches.load(), 0);
}

TEST_F(CudaReductionTest, EmptyOperandsReduceToZero)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::CudaMatrix<double> device(runtime.device_resources(0), 4, 4);
  auto empty = cuda_strided_view<2>(device, {4, 0}, {1, 8});
  EXPECT_EQ(uni20::norm_host(empty), 0.0);
  EXPECT_EQ(uni20::inner_product_host(empty, empty), 0.0);
}

} // namespace
