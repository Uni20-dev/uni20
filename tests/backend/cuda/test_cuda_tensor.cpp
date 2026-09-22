#include <uni20/async/cuda_task.hpp>
#include <uni20/async/debug_cuda_scheduler.hpp>
#include <uni20/backend/cuda/task_awaiters.hpp>
#include <uni20/storage/cuda_storage.hpp>
#include <uni20/tensor/tensor.hpp>

#include <gtest/gtest.h>

#include <algorithm>
#include <array>
#include <bit>
#include <concepts>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

namespace
{

using tensor_type = uni20::CudaTensor<double, 2>;
using direct_tensor_type = uni20::Tensor<double, 2, uni20::CudaStorage>;
using mutable_span_type = typename tensor_type::mdspan_type;
using const_span_type = typename tensor_type::const_mdspan_type;
using mutable_mdspec_type = decltype(std::declval<tensor_type&>().mdspec());
using const_mdspec_type = decltype(std::declval<tensor_type const&>().mdspec());
using host_tensor_type = uni20::Tensor<double, 2>;
using read_lease_type = decltype(uni20::acquire_cuda_read_access_sync(std::declval<tensor_type const&>()));
using write_lease_type = decltype(uni20::acquire_cuda_write_access_sync(std::declval<tensor_type&>()));
using read_access_type = decltype(uni20::acquire_cuda_read_access_async(std::declval<tensor_type const&>(),
                                                                        std::declval<uni20::cuda::Stream const&>()));
using write_access_type = decltype(uni20::acquire_cuda_write_access_async(std::declval<tensor_type&>(),
                                                                          std::declval<uni20::cuda::Stream const&>()));
using descriptor_read_lease_type =
    decltype(uni20::acquire_cuda_read_access_sync(std::declval<const_mdspec_type const&>()));
using descriptor_write_lease_type =
    decltype(uni20::acquire_cuda_write_access_sync(std::declval<mutable_mdspec_type&>()));
using descriptor_read_access_type = decltype(uni20::acquire_cuda_read_access_async(
    std::declval<const_mdspec_type const&>(), std::declval<uni20::cuda::Stream const&>()));
using descriptor_write_access_type = decltype(uni20::acquire_cuda_write_access_async(
    std::declval<mutable_mdspec_type&>(), std::declval<uni20::cuda::Stream const&>()));

template <class Tensor>
concept HasMutableMdspan = requires(Tensor& tensor) { tensor.mdspan(); };

template <class Tensor>
concept HasConstMdspan = requires(Tensor const& tensor) { tensor.mdspan(); };

template <class Value>
concept HasStorageObserver = requires(Value& value) { value.storage(); };

template <class Tensor>
concept CanSynchronizedCudaReadRvalue = requires { uni20::acquire_cuda_read_access_sync(std::declval<Tensor&&>()); };

template <class Tensor>
concept CanStreamReadRvalue = requires(uni20::cuda::Stream const& stream) {
  uni20::acquire_cuda_read_access_async(std::declval<Tensor&&>(), stream);
};

template <class Tensor>
concept CanAcquireHostReadSync = requires(Tensor const& tensor) { uni20::acquire_host_read_access_sync(tensor); };

template <class Tensor>
concept CanAcquireCudaReadSync = requires(Tensor const& tensor) { uni20::acquire_cuda_read_access_sync(tensor); };

template <class Tensor>
concept CanAcquireCudaReadAsync = requires(Tensor const& tensor, uni20::cuda::Stream const& stream) {
  uni20::acquire_cuda_read_access_async(tensor, stream);
};

template <class Tensor>
concept CanReshapeInplace = requires(Tensor& tensor) { uni20::reshape_inplace(tensor, 3, 2); };

template <class Tensor>
concept CanTransferReshape = requires(Tensor&& tensor) { uni20::reshape(std::move(tensor), 3, 2); };

struct MdspecCallCounts
{
    int mutable_calls = 0;
    mutable int const_calls = 0;
};

struct DescriptorSelectedStoragePolicy
{
    [[nodiscard]] static constexpr auto backend_selector() noexcept { return uni20::CudaStorage::backend_selector(); }
};

class CountingCudaTensorView {
  public:
    using tensor_type = uni20::CudaTensor<double, 2>;
    using storage_policy = DescriptorSelectedStoragePolicy;
    using extents_type = typename tensor_type::extents_type;

    CountingCudaTensorView(tensor_type& tensor, MdspecCallCounts& calls) : tensor_(&tensor), calls_(&calls) {}

    [[nodiscard]] auto mdspec()
    {
      ++calls_->mutable_calls;
      return tensor_->mdspec();
    }

    [[nodiscard]] auto mdspec() const
    {
      ++calls_->const_calls;
      return std::as_const(*tensor_).mdspec();
    }

    [[nodiscard]] auto backend_selector() const { return tensor_->backend_selector(); }

    [[nodiscard]] auto extents() const -> extents_type const& { return tensor_->extents(); }

    [[nodiscard]] auto extent(std::size_t axis) const { return tensor_->extent(axis); }

  private:
    tensor_type* tensor_;
    MdspecCallCounts* calls_;
};

using owning_read_lease_type = decltype(uni20::acquire_cuda_read_access_sync(std::declval<tensor_type&&>()));
using owning_read_access_type = decltype(uni20::acquire_cuda_read_access_async(
    std::declval<tensor_type&&>(), std::declval<uni20::cuda::Stream const&>()));
using counting_read_lease_type =
    decltype(uni20::acquire_cuda_read_access_sync(std::declval<CountingCudaTensorView const&>()));

static_assert(std::same_as<typename tensor_type::storage_type, uni20::cuda::CudaBuffer<double>>);
static_assert(std::same_as<tensor_type, direct_tensor_type>);
static_assert(std::same_as<typename mutable_span_type::data_handle_type, double*>);
static_assert(std::same_as<typename const_span_type::data_handle_type, double const*>);
static_assert(std::same_as<typename mutable_span_type::reference, double&>);
static_assert(std::same_as<typename const_span_type::reference, double const&>);
static_assert(std::same_as<typename mutable_mdspec_type::data_handle_type, double*>);
static_assert(std::same_as<typename const_mdspec_type::data_handle_type, double const*>);
static_assert(!tensor_type::immediately_readable);
static_assert(!tensor_type::immediately_writable);
static_assert(tensor_type::deferred_readable);
static_assert(tensor_type::deferred_writable);
static_assert(uni20::MdspecLike<mutable_mdspec_type>);
static_assert(uni20::MutableMdspecLike<mutable_mdspec_type>);
static_assert(uni20::MdspecLike<const_mdspec_type>);
static_assert(uni20::CudaAccessibleMdspec<mutable_mdspec_type>);
static_assert(uni20::CudaAccessibleMdspec<const_mdspec_type>);
static_assert(!uni20::HostAccessibleMdspec<mutable_mdspec_type>);
static_assert(!uni20::HostAccessibleMdspec<const_mdspec_type>);
static_assert(!uni20::MdspanLike<mutable_mdspec_type>);
static_assert(!uni20::MdspanLike<const_mdspec_type>);
static_assert(!uni20::ImmediateTensorView<tensor_type>);
static_assert(!uni20::MutableImmediateTensorView<tensor_type>);
static_assert(!HasMutableMdspan<tensor_type>);
static_assert(!HasConstMdspan<tensor_type>);
static_assert(uni20::TensorView<tensor_type>);
static_assert(uni20::MutableTensorView<tensor_type>);
static_assert(uni20::RankedStridedTensorView<tensor_type, 2>);
static_assert(uni20::MutableRankedStridedTensorView<tensor_type, 2>);
static_assert(uni20::OwningTensor<tensor_type>);
static_assert(uni20::ReadTensorLease<read_lease_type>);
static_assert(uni20::WriteTensorLease<write_lease_type>);
static_assert(uni20::CudaReadTensorLease<read_lease_type>);
static_assert(uni20::CudaWriteTensorLease<write_lease_type>);
static_assert(uni20::CudaReadMdspanLease<descriptor_read_lease_type>);
static_assert(uni20::CudaWriteMdspanLease<descriptor_write_lease_type>);
static_assert(!uni20::ImmediateTensorView<descriptor_read_lease_type>);
static_assert(!uni20::ImmediateTensorView<descriptor_write_lease_type>);
static_assert(!HasStorageObserver<descriptor_read_lease_type>);
static_assert(!HasStorageObserver<descriptor_write_lease_type>);
static_assert(!uni20::HostReadTensorLease<read_lease_type>);
static_assert(!uni20::HostWriteTensorLease<write_lease_type>);
static_assert(!CanAcquireHostReadSync<tensor_type>);
static_assert(!CanAcquireCudaReadSync<host_tensor_type>);
static_assert(!CanAcquireCudaReadAsync<host_tensor_type>);
static_assert(std::same_as<typename read_lease_type::storage_policy, uni20::CudaStorage>);
static_assert(std::same_as<typename write_lease_type::storage_policy, uni20::CudaStorage>);
static_assert(std::same_as<typename read_lease_type::mdspan_type::data_handle_type, double const*>);
static_assert(std::same_as<typename write_lease_type::mdspan_type::data_handle_type, double*>);
static_assert(std::same_as<typename read_lease_type::mdspan_type::reference, double const&>);
static_assert(std::same_as<typename write_lease_type::mdspan_type::reference, double&>);
static_assert(
    std::same_as<decltype(std::declval<read_lease_type const&>().storage()), tensor_type::storage_type const&>);
static_assert(!HasStorageObserver<write_lease_type>);
static_assert(uni20::async::TaskAwaitable<read_access_type>);
static_assert(uni20::async::TaskAwaitable<write_access_type>);
static_assert(uni20::async::TaskAwaitable<descriptor_read_access_type>);
static_assert(uni20::async::TaskAwaitable<descriptor_write_access_type>);
static_assert(uni20::ReadTensorLease<owning_read_lease_type>);
static_assert(uni20::async::TaskAwaitable<owning_read_access_type>);
static_assert(CanSynchronizedCudaReadRvalue<tensor_type>);
static_assert(CanStreamReadRvalue<tensor_type>);
static_assert(uni20::TensorView<CountingCudaTensorView>);
static_assert(uni20::MutableTensorView<CountingCudaTensorView>);
static_assert(!std::same_as<typename CountingCudaTensorView::storage_policy, uni20::CudaStorage>);
static_assert(std::same_as<typename counting_read_lease_type::storage_policy, DescriptorSelectedStoragePolicy>);
static_assert(!uni20::OwningTensor<CountingCudaTensorView>);
static_assert(!CanSynchronizedCudaReadRvalue<CountingCudaTensorView>);
static_assert(!CanStreamReadRvalue<CountingCudaTensorView>);
static_assert(!std::copy_constructible<tensor_type>);
static_assert(std::move_constructible<tensor_type>);
static_assert(CanReshapeInplace<tensor_type>);
static_assert(CanTransferReshape<tensor_type>);
using reshaped_tensor_type = decltype(uni20::reshape(std::declval<tensor_type&&>(), 3, 2));
static_assert(uni20::TensorView<reshaped_tensor_type>);
static_assert(!uni20::ImmediateTensorView<reshaped_tensor_type>);
static_assert(std::convertible_to<typename mutable_span_type::reference, double>);
static_assert(std::assignable_from<typename mutable_span_type::reference, double>);

using complex_tensor_type = uni20::CudaTensor<uni20::complex<double>, 2>;
using conjugated_tensor_type = decltype(uni20::conj(std::declval<complex_tensor_type&>()));
using conjugated_mdspec_type = decltype(std::declval<conjugated_tensor_type const&>().mdspec());

static_assert(uni20::TensorView<conjugated_tensor_type>);
static_assert(!uni20::ImmediateTensorView<conjugated_tensor_type>);
static_assert(!HasConstMdspan<conjugated_tensor_type>);
static_assert(uni20::mdspan_needs_conjugation_v<conjugated_mdspec_type>);
static_assert(
    std::same_as<typename conjugated_mdspec_type::data_descriptor_type,
                 typename decltype(std::declval<complex_tensor_type const&>().mdspec())::data_descriptor_type>);

TEST(CudaPointerAccessorTest, AccessReturnsMappedElementReference)
{
  std::array<double, 3> values{1.0, 2.0, 3.0};
  uni20::cuda::CudaPointerAccessor<double> accessor;

  auto& reference = accessor.access(values.data(), 1);

  EXPECT_EQ(&reference, &values[1]);
  reference = 4.0;
  EXPECT_DOUBLE_EQ(values[1], 4.0);
  EXPECT_EQ(accessor.offset(values.data(), 2), values.data() + 2);
}

class CudaTensorTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
      int device_count = 0;
      cudaError_t const status = cudaGetDeviceCount(&device_count);
      if (status != cudaSuccess)
      {
        GTEST_SKIP() << "CUDA device discovery failed: " << cudaGetErrorString(status);
      }
      if (device_count == 0)
      {
        GTEST_SKIP() << "no CUDA devices are available";
      }
    }
};

template <class Scalar> void check_cuda_initialization(bool uninitialized)
{
  using real_type = uni20::make_real_t<Scalar>;
  using word_type = std::conditional_t<sizeof(real_type) == 4, std::uint32_t, std::uint64_t>;
  constexpr std::size_t components = uni20::Complex<Scalar> ? 2 : 1;
  constexpr std::size_t word_count = 6 * components;
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 2});
  // Allocate the pinned destination before submitting initialization so host
  // allocation cannot add an implicit synchronization between the two streams.
  void* host_memory = nullptr;
  ASSERT_EQ(cudaMallocHost(&host_memory, word_count * sizeof(word_type)), cudaSuccess);
  std::unique_ptr<word_type, decltype(&cudaFreeHost)> output(static_cast<word_type*>(host_memory), cudaFreeHost);
  std::fill_n(output.get(), word_count, ~word_type{0});
  // Keep this stream checked out so allocation and initialization necessarily
  // use another stream. Reading relies only on the buffer's published writer.
  auto consumer = resources.streams().acquire();
  using tensor = uni20::CudaTensor<Scalar, 2>;
  tensor value = uninitialized ? tensor(uni20::uninitialized, resources, 2, 3) : tensor(resources, 2, 3);
  {
    auto read = value.storage().read_synchronized_with(consumer);
    uni20::cuda::ScopedDevice guard(resources.device().ordinal());
    uni20::cuda::check(cudaMemcpyAsync(output.get(), read.data(), word_count * sizeof(word_type),
                                       cudaMemcpyDeviceToHost, consumer.native_handle()),
                       "read initialized CUDA tensor", resources.device().ordinal());
    consumer.synchronize();
    read.release_after_synchronization();
  }
  word_type const expected =
      uninitialized ? std::bit_cast<word_type>(uni20::numeric_limits<real_type>::signaling_NaN()) : word_type{0};
  for (std::size_t i = 0; i < word_count; ++i)
    EXPECT_EQ(output.get()[i], expected);
}

TEST_F(CudaTensorTest, ShapeConstructionZerosRealAndComplexOnAnotherStream)
{
  check_cuda_initialization<float>(false);
  check_cuda_initialization<double>(false);
  check_cuda_initialization<uni20::complex<float>>(false);
  check_cuda_initialization<uni20::complex<double>>(false);
}

TEST_F(CudaTensorTest, ExplicitUninitializedPreservesSignalingNanBitsOnAnotherStream)
{
#if UNI20_FILL_UNINITIALIZED_SNAN
  check_cuda_initialization<float>(true);
  check_cuda_initialization<double>(true);
  check_cuda_initialization<uni20::complex<float>>(true);
  check_cuda_initialization<uni20::complex<double>>(true);
#else
  GTEST_SKIP() << "diagnostic signaling-NaN filling is disabled";
#endif
}

TEST_F(CudaTensorTest, ShapeResetInitializesNewStorageToZero)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  tensor_type value(uni20::uninitialized, resources, 2, 3);
  value.reset_shape(tensor_type::extents_type{3, 2});
  std::array<double, 6> output;
  {
    auto lease = uni20::acquire_cuda_read_access_sync(value);
    uni20::cuda::ScopedDevice guard(resources.device().ordinal());
    uni20::cuda::check(cudaMemcpy(output.data(), lease.mdspan().data_handle(), sizeof(output), cudaMemcpyDeviceToHost),
                       "read reset CUDA tensor", resources.device().ordinal());
  }
  for (double element : output)
    EXPECT_EQ(element, 0.0);
}

TEST_F(CudaTensorTest, ExplicitUninitializedStorageCanBeCompletelyOverwritten)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  tensor_type value(uni20::uninitialized, resources, 2, 3);
  {
    auto stream = resources.streams().acquire();
    auto write = value.mdspec().data_descriptor().buffer().write_synchronized_with(stream);
    uni20::cuda::ScopedDevice guard(resources.device().ordinal());
    uni20::cuda::check(cudaMemsetAsync(write.data(), 0, write.size_bytes(), stream.native_handle()),
                       "overwrite uninitialized CUDA tensor", stream.device());
  }
  std::array<double, 6> output;
  {
    auto lease = uni20::acquire_cuda_read_access_sync(value);
    uni20::cuda::ScopedDevice guard(resources.device().ordinal());
    uni20::cuda::check(cudaMemcpy(output.data(), lease.mdspan().data_handle(), sizeof(output), cudaMemcpyDeviceToHost),
                       "read overwritten CUDA tensor", resources.device().ordinal());
  }
  for (double element : output)
    EXPECT_EQ(element, 0.0);
}

TEST_F(CudaTensorTest, EmptyAndRankZeroInitialization)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  tensor_type empty(resources, 0, 3);
  tensor_type empty_uninitialized(uni20::uninitialized, resources, 0, 3);
  EXPECT_EQ(empty.storage().size(), 0U);
  EXPECT_EQ(empty_uninitialized.storage().size(), 0U);

  using scalar_tensor = uni20::CudaTensor<double, 0>;
  scalar_tensor scalar(resources, scalar_tensor::extents_type{});
  double output = 1.0;
  {
    auto lease = uni20::acquire_cuda_read_access_sync(scalar);
    uni20::cuda::ScopedDevice guard(resources.device().ordinal());
    uni20::cuda::check(cudaMemcpy(&output, lease.mdspan().data_handle(), sizeof(output), cudaMemcpyDeviceToHost),
                       "read rank-zero CUDA tensor", resources.device().ordinal());
  }
  EXPECT_EQ(output, 0.0);
}

TEST_F(CudaTensorTest, NonnumericElementsRequireExplicitUninitializedConstruction)
{
  struct Element
  {
      int value = 7;
  };
  static_assert(std::is_trivially_copyable_v<Element>);
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  using tensor = uni20::CudaTensor<Element, 1>;
  EXPECT_THROW((tensor(resources, 2)), std::invalid_argument);

  tensor value(uni20::uninitialized, resources, 2);
  EXPECT_EQ(value.storage().size(), 2U);
  uni20::cuda::CudaBuffer<Element> raw(resources, 2);
  EXPECT_EQ(raw.size(), 2U);
}

TEST_F(CudaTensorTest, PackedInitializationPublishesToChildrenAndPreservesZeroPadding)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 2});
  std::array<std::size_t, 3> const offsets{0, 8, 16};
  auto storage = uni20::CudaStorage::make_packed_storage<double>(resources, offsets.back(), offsets,
                                                                 uni20::StorageInitialization::Zero);
  {
    auto read = storage.buffer(1).blocking_read_access();
    std::array<std::uint64_t, 8> output;
    uni20::cuda::ScopedDevice guard(resources.device().ordinal());
    uni20::cuda::check(cudaMemcpy(output.data(), read.data(), sizeof(output), cudaMemcpyDeviceToHost),
                       "read zero-initialized CUDA child buffer", resources.device().ordinal());
    for (auto bits : output)
      EXPECT_EQ(bits, 0U);
  }

#if UNI20_FILL_UNINITIALIZED_SNAN
  std::array<std::size_t, 2> const block_ends{6, 12};
  auto undefined = uni20::CudaStorage::make_packed_storage<double>(resources, offsets.back(), offsets,
                                                                   uni20::StorageInitialization::Uninitialized);
  uni20::CudaStorage::initialize_packed_padding(undefined, offsets, block_ends);
  auto const expected_nan = std::bit_cast<std::uint64_t>(uni20::numeric_limits<double>::signaling_NaN());
  for (std::size_t ordinal = 0; ordinal < block_ends.size(); ++ordinal)
  {
    auto read = undefined.buffer(ordinal).blocking_read_access();
    std::array<std::uint64_t, 8> output;
    uni20::cuda::ScopedDevice guard(resources.device().ordinal());
    uni20::cuda::check(cudaMemcpy(output.data(), read.data(), sizeof(output), cudaMemcpyDeviceToHost),
                       "read diagnostic CUDA child buffer", resources.device().ordinal());
    for (std::size_t i = 0; i < output.size(); ++i)
      EXPECT_EQ(output[i], i < block_ends[ordinal] - offsets[ordinal] ? expected_nan : 0U);
  }
#endif
}

TEST_F(CudaTensorTest, ExplicitResourcesConstructionOwnsDeviceBufferAndDescriptor)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  tensor_type tensor(resources, 2, 3);

  EXPECT_EQ(tensor.rows(), 2);
  EXPECT_EQ(tensor.cols(), 3);
  EXPECT_EQ(tensor.size(), 6);
  EXPECT_EQ(tensor.storage().size(), 6U);
  EXPECT_EQ(tensor.storage().device(), resources.device());

  auto span = tensor.mdspec();
  EXPECT_EQ(&span.data_descriptor().buffer(), &tensor.storage());
  EXPECT_EQ(span.data_descriptor().element_offset(), 0U);
  EXPECT_EQ(span.mapping()(1, 2), 5U);

  tensor_type const& const_tensor = tensor;
  auto const_span = const_tensor.mdspec();
  EXPECT_EQ(&const_span.data_descriptor().buffer(), &tensor.storage());
  EXPECT_EQ(const_span.mapping()(1, 2), 5U);
  EXPECT_EQ(tensor.backend_selector(), uni20::CudaStorage::backend_selector());
}

TEST_F(CudaTensorTest, ShapeResetKeepsTheOriginalDeviceResources)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  tensor_type tensor(resources, 2, 3);
  auto* const original_resources = &tensor.storage().resources();

  tensor.reset_shape(tensor_type::extents_type{4, 5});

  EXPECT_EQ(&tensor.storage().resources(), original_resources);
  EXPECT_EQ(tensor.rows(), 4);
  EXPECT_EQ(tensor.cols(), 5);
  EXPECT_EQ(tensor.storage().size(), 20U);
}

TEST_F(CudaTensorTest, OwningReshapePreservesDeferredStorageResources)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  tensor_type tensor(resources, 2, 3);
  auto* const original_resources = &tensor.storage().resources();

  uni20::reshape_inplace(tensor, 3, 2);

  EXPECT_EQ(tensor.rows(), 3);
  EXPECT_EQ(tensor.cols(), 2);
  EXPECT_EQ(tensor.storage().size(), 6U);
  EXPECT_EQ(&tensor.storage().resources(), original_resources);

  auto reshaped = uni20::reshape(std::move(tensor), 1, 6);

  EXPECT_EQ(reshaped.rows(), 1);
  EXPECT_EQ(reshaped.cols(), 6);
  EXPECT_EQ(reshaped.storage().size(), 6U);
  EXPECT_EQ(&reshaped.storage().resources(), original_resources);
  static_assert(uni20::TensorView<decltype(reshaped)>);
  static_assert(!uni20::ImmediateTensorView<decltype(reshaped)>);
}

TEST_F(CudaTensorTest, SynchronizedCudaAccessResolvesPointerMdspans)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  tensor_type tensor(resources, 2, 3);
  std::array<double, 6> const input{1, 2, 3, 4, 5, 6};
  std::array<double, 6> output{};

  {
    auto lease = uni20::acquire_cuda_write_access_sync(tensor);
    static_assert(uni20::MutableImmediateTensorView<decltype(lease)>);
    EXPECT_EQ(lease.backend_selector(), tensor.backend_selector());
    EXPECT_NE(lease.mdspan().data_handle(), nullptr);

    uni20::cuda::ScopedDevice guard(resources.device().ordinal());
    uni20::cuda::check(cudaMemcpy(lease.mdspan().data_handle(), input.data(), sizeof(input), cudaMemcpyHostToDevice),
                       "cudaMemcpy test host-to-device", resources.device().ordinal());
    uni20::cuda::check(cudaDeviceSynchronize(), "cudaDeviceSynchronize test upload", resources.device().ordinal());
  }

  {
    auto lease = uni20::acquire_cuda_read_access_sync(std::as_const(tensor));
    static_assert(uni20::ImmediateTensorView<decltype(lease)>);
    static_assert(!uni20::MutableImmediateTensorView<decltype(lease)>);
    EXPECT_EQ(&lease.storage(), &tensor.storage());

    uni20::cuda::ScopedDevice guard(resources.device().ordinal());
    uni20::cuda::check(cudaMemcpy(output.data(), lease.mdspan().data_handle(), sizeof(output), cudaMemcpyDeviceToHost),
                       "cudaMemcpy test device-to-host", resources.device().ordinal());
  }

  EXPECT_EQ(output, input);
}

TEST_F(CudaTensorTest, DescriptorAcquisitionResolvesPolicyFreePointerMdspans)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  tensor_type tensor(resources, 2, 3);
  std::array<double, 6> const input{1, 2, 3, 4, 5, 6};
  std::array<double, 6> output{};

  auto write_descriptor = tensor.mdspec();
  {
    auto lease = uni20::acquire_cuda_write_access_sync(write_descriptor);
    static_assert(uni20::CudaWriteMdspanLease<decltype(lease)>);
    uni20::cuda::ScopedDevice guard(resources.device().ordinal());
    uni20::cuda::check(cudaMemcpy(lease.mdspan().data_handle(), input.data(), sizeof(input), cudaMemcpyHostToDevice),
                       "cudaMemcpy descriptor upload", resources.device().ordinal());
  }

  auto read_descriptor = std::as_const(tensor).mdspec();
  {
    auto lease = uni20::acquire_cuda_read_access_sync(read_descriptor);
    static_assert(uni20::CudaReadMdspanLease<decltype(lease)>);
    uni20::cuda::ScopedDevice guard(resources.device().ordinal());
    uni20::cuda::check(cudaMemcpy(output.data(), lease.mdspan().data_handle(), sizeof(output), cudaMemcpyDeviceToHost),
                       "cudaMemcpy descriptor download", resources.device().ordinal());
  }

  EXPECT_EQ(output, input);
}

TEST_F(CudaTensorTest, StreamOrderedAccessCanBeCoAwaitedAsTensorView)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  uni20::async::DebugCudaScheduler scheduler(resources.device());
  tensor_type tensor(resources, 2, 3);
  bool submitted = false;

  auto task = [](tensor_type& value, bool& observed) static -> uni20::async::CudaTask {
    auto stream = co_await uni20::cuda::acquire_stream(value.storage().resources().streams());
    auto lease = co_await uni20::acquire_cuda_write_access_async(value, stream);
    static_assert(uni20::MutableImmediateTensorView<decltype(lease)>);
    auto const size_bytes =
        static_cast<std::size_t>(lease.mdspan().mapping().required_span_size()) * sizeof(tensor_type::element_type);
    uni20::cuda::check(cudaMemsetAsync(lease.mdspan().data_handle(), 0, size_bytes, stream.native_handle()),
                       "cudaMemsetAsync tensor lease", stream.device());
    observed = true;
    co_return;
  }(tensor, submitted);

  scheduler.schedule(std::move(task), resources.device().ordinal());
  scheduler.run_all();
  EXPECT_TRUE(submitted);
  tensor.storage().synchronize();

  std::array<double, 6> output{};
  output.fill(1.0);
  auto lease = uni20::acquire_cuda_read_access_sync(std::as_const(tensor));
  {
    uni20::cuda::ScopedDevice guard(resources.device().ordinal());
    uni20::cuda::check(cudaMemcpy(output.data(), lease.mdspan().data_handle(), sizeof(output), cudaMemcpyDeviceToHost),
                       "cudaMemcpy tensor lease verification", resources.device().ordinal());
  }
  EXPECT_EQ(output, (std::array<double, 6>{}));
}

TEST_F(CudaTensorTest, StreamOrderedDescriptorAccessCanBeCoAwaitedAsMdspanLease)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  uni20::async::DebugCudaScheduler scheduler(resources.device());
  tensor_type tensor(resources, 2, 3);
  bool submitted = false;

  auto task = [](tensor_type& value, bool& observed) static -> uni20::async::CudaTask {
    auto stream = co_await uni20::cuda::acquire_stream(value.storage().resources().streams());
    auto descriptor = value.mdspec();
    auto lease = co_await uni20::acquire_cuda_write_access_async(descriptor, stream);
    static_assert(uni20::CudaWriteMdspanLease<decltype(lease)>);
    auto const size_bytes =
        static_cast<std::size_t>(lease.mdspan().mapping().required_span_size()) * sizeof(tensor_type::element_type);
    uni20::cuda::check(cudaMemsetAsync(lease.mdspan().data_handle(), 0, size_bytes, stream.native_handle()),
                       "cudaMemsetAsync descriptor lease", stream.device());
    observed = true;
    co_return;
  }(tensor, submitted);

  scheduler.schedule(std::move(task), resources.device().ordinal());
  scheduler.run_all();
  EXPECT_TRUE(submitted);
  tensor.storage().synchronize();
}

TEST_F(CudaTensorTest, DeferredAcquisitionCapturesEachMdspecOnce)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  tensor_type tensor(resources, 2, 3);
  MdspecCallCounts calls;
  CountingCudaTensorView view(tensor, calls);

  {
    auto lease = uni20::acquire_cuda_read_access_sync(std::as_const(view));
    EXPECT_NE(lease.mdspan().data_handle(), nullptr);
  }
  EXPECT_EQ(calls.const_calls, 1);
  EXPECT_EQ(calls.mutable_calls, 0);

  calls = {};
  {
    auto lease = uni20::acquire_cuda_write_access_sync(view);
    EXPECT_NE(lease.mdspan().data_handle(), nullptr);
  }
  EXPECT_EQ(calls.const_calls, 1);
  EXPECT_EQ(calls.mutable_calls, 1);

  auto stream = resources.streams().acquire();
  calls = {};
  {
    auto acquisition = uni20::acquire_cuda_read_access_async(std::as_const(view), stream);
    auto lease = acquisition.await_resume();
    EXPECT_NE(lease.mdspan().data_handle(), nullptr);
  }
  EXPECT_EQ(calls.const_calls, 1);
  EXPECT_EQ(calls.mutable_calls, 0);

  calls = {};
  {
    auto acquisition = uni20::acquire_cuda_write_access_async(view, stream);
    auto lease = acquisition.await_resume();
    EXPECT_NE(lease.mdspan().data_handle(), nullptr);
  }
  EXPECT_EQ(calls.const_calls, 1);
  EXPECT_EQ(calls.mutable_calls, 1);
}

TEST_F(CudaTensorTest, OwningRvalueReadLeaseMovesTheBuffer)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  tensor_type tensor(resources, 2, 3);
  std::array<double, 6> const input{1, 2, 3, 4, 5, 6};
  std::array<double, 6> output{};

  {
    auto lease = uni20::acquire_cuda_write_access_sync(tensor);
    uni20::cuda::ScopedDevice guard(resources.device().ordinal());
    uni20::cuda::check(cudaMemcpy(lease.mdspan().data_handle(), input.data(), sizeof(input), cudaMemcpyHostToDevice),
                       "cudaMemcpy owning read test upload", resources.device().ordinal());
  }

  auto lease = uni20::acquire_cuda_read_access_sync(std::move(tensor));
  auto moved_lease = std::move(lease);
  {
    uni20::cuda::ScopedDevice guard(resources.device().ordinal());
    uni20::cuda::check(
        cudaMemcpy(output.data(), moved_lease.mdspan().data_handle(), sizeof(output), cudaMemcpyDeviceToHost),
        "cudaMemcpy owning read test download", resources.device().ordinal());
  }

  EXPECT_EQ(output, input);
}

TEST_F(CudaTensorTest, OwningRvalueStreamReadPublishesCompletion)
{
  uni20::cuda::DeviceResources resources({.device = uni20::cuda::Device::get(0), .stream_count = 1});
  tensor_type tensor(resources, 2, 3);
  std::array<double, 6> const input{1, 2, 3, 4, 5, 6};
  std::array<double, 6> output{};

  {
    auto lease = uni20::acquire_cuda_write_access_sync(tensor);
    uni20::cuda::ScopedDevice guard(resources.device().ordinal());
    uni20::cuda::check(cudaMemcpy(lease.mdspan().data_handle(), input.data(), sizeof(input), cudaMemcpyHostToDevice),
                       "cudaMemcpy owning stream read test upload", resources.device().ordinal());
  }

  auto stream = resources.streams().acquire();
  {
    auto acquisition = uni20::acquire_cuda_read_access_async(std::move(tensor), stream);
    auto lease = acquisition.await_resume();
    uni20::cuda::check(cudaMemcpyAsync(output.data(), lease.mdspan().data_handle(), sizeof(output),
                                       cudaMemcpyDeviceToHost, stream.native_handle()),
                       "cudaMemcpyAsync owning stream read test download", resources.device().ordinal());
  }

  EXPECT_EQ(output, input);
}

TEST_F(CudaTensorTest, DefaultConstructionUsesInstalledRuntimeResources)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::CudaTensor<double, 2> tensor(3, 4);

  EXPECT_EQ(tensor.rows(), 3);
  EXPECT_EQ(tensor.cols(), 4);
  EXPECT_EQ(&tensor.storage().resources(), &runtime.default_device_resources());
}

} // namespace
