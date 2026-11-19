#include <uni20/async/debug_cuda_scheduler.hpp>
#include <uni20/backend/cuda/runtime.hpp>
#include <uni20/common/trace.hpp>
#include <uni20/linalg/async.hpp>
#include <uni20/mdspan/generated_layout.hpp>
#include <uni20/tensor/async.hpp>
#include <uni20/tensor/copy.hpp>
#include <uni20/tensor/mdspec_tensor_view.hpp>
#include <uni20/tensor/tensor.hpp>

#include "../backend/cuda/test_cuda_stream_gate.hpp"

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <array>
#include <atomic>
#include <chrono>
#include <concepts>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <string_view>
#include <thread>
#include <tuple>
#include <type_traits>
#include <utility>

namespace uni20_cuda_copy_test
{
// Trivially copyable without a default constructor.
struct Packet
{
    std::uint32_t value;
    Packet() = delete;
    constexpr explicit Packet(std::uint32_t x) : value(x) {}
};

// Trivially copyable and wider than the fixed-width copy kernels.
struct WideElement
{
    double values[5];
};
} // namespace uni20_cuda_copy_test

// The documented customization that lets host storage leave Packet elements uninitialized.
template <> struct uni20::is_integer_t<uni20_cuda_copy_test::Packet> : std::true_type
{};

namespace
{

using uni20_cuda_copy_test::Packet;
using uni20_cuda_copy_test::WideElement;
using host_matrix_type = uni20::Tensor<double, 2>;
using cuda_matrix_type = uni20::CudaTensor<double, 2>;
using async_cuda_matrix_type = uni20::async::Async<cuda_matrix_type>;
using row_major_host_matrix_type = uni20::RowMajorTensor<double, 2>;
using complex_type = uni20::complex<double>;
using complex_host_matrix_type = uni20::Tensor<complex_type, 2>;
using complex_cuda_matrix_type = uni20::CudaTensor<complex_type, 2>;
using cuda_vector_type = uni20::CudaTensor<double, 1>;
using complex_cuda_conjugated_view = uni20::ConjugatedTensorView<complex_cuda_matrix_type>;
using cuda_extents_type = stdex::dextents<uni20::index_type, 2>;
using resolved_cuda_output_mdspan =
    stdex::mdspan<double, cuda_extents_type, stdex::layout_left, uni20::cuda::CudaPointerAccessor<double>>;
using resolved_cuda_input_mdspan =
    stdex::mdspan<double const, cuda_extents_type, stdex::layout_left, uni20::cuda::CudaPointerAccessor<double const>>;
using resolved_complex_cuda_output_mdspan =
    stdex::mdspan<complex_type, cuda_extents_type, stdex::layout_left, uni20::cuda::CudaPointerAccessor<complex_type>>;
using resolved_complex_cuda_input_mdspan = stdex::mdspan<complex_type const, cuda_extents_type, stdex::layout_left,
                                                         uni20::cuda::CudaPointerAccessor<complex_type const>>;
using conjugated_cuda_descriptor = decltype(std::declval<complex_cuda_conjugated_view const&>().mdspec());
using conjugated_cuda_lease =
    decltype(uni20::acquire_cuda_read_access_sync(std::declval<conjugated_cuda_descriptor const&>()));
using resolved_conjugated_cuda_mdspan = std::remove_cvref_t<decltype(std::declval<conjugated_cuda_lease&>().mdspan())>;

using namespace std::chrono_literals;

template <class Layout, class Mdspec>
auto remap_mdspec(Mdspec const& source, typename Layout::template mapping<typename Mdspec::extents_type> mapping)
{
  using source_type = std::remove_cvref_t<Mdspec>;
  using result_type = uni20::mdspec<typename source_type::element_type, typename source_type::extents_type, Layout,
                                    typename source_type::accessor_type, typename source_type::data_descriptor_type>;
  return result_type{source.data_descriptor(), std::move(mapping), source.accessor()};
}

template <class Tensor>
auto make_strided_matrix_mdspec(Tensor& tensor, std::size_t offset, std::array<uni20::index_type, 2> const& strides)
{
  auto base = tensor.mdspec();
  using base_type = decltype(base);
  using layout_type = stdex::layout_stride;
  using mapping_type = layout_type::mapping<cuda_extents_type>;
  using result_type = uni20::mdspec<typename base_type::element_type, cuda_extents_type, layout_type,
                                    typename base_type::accessor_type, typename base_type::data_descriptor_type>;
  return result_type{base.data_descriptor().offset_by(offset), mapping_type{cuda_extents_type{2, 3}, strides},
                     base.accessor()};
}

template <class Tensor> auto make_vector_mdspec(Tensor& tensor, std::size_t offset, uni20::index_type extent)
{
  auto base = tensor.mdspec();
  using base_type = decltype(base);
  using mapping_type = typename base_type::mapping_type;
  return base_type{base.data_descriptor().offset_by(offset), mapping_type{typename base_type::extents_type{extent}},
                   base.accessor()};
}

struct DescriptorSelectedStoragePolicy
{
    [[nodiscard]] static constexpr auto backend_selector() noexcept
    {
      return uni20::linalg::backend_list<uni20::linalg::CpuReferenceBackend>{uni20::linalg::CpuReferenceBackend{}};
    }
};

template <std::size_t Rank> struct StrideProbe
{
    [[nodiscard]] static constexpr std::size_t rank() noexcept { return Rank; }
    [[nodiscard]] constexpr uni20::index_type extent(std::size_t axis) const noexcept { return extents_[axis]; }
    [[nodiscard]] constexpr uni20::index_type stride(std::size_t axis) const noexcept { return strides_[axis]; }

    std::array<uni20::index_type, Rank> extents_{};
    std::array<uni20::index_type, Rank> strides_{};
};

class CudaDescriptorMatrixView {
  public:
    using storage_policy = DescriptorSelectedStoragePolicy;
    using extents_type = typename cuda_matrix_type::extents_type;

    explicit CudaDescriptorMatrixView(cuda_matrix_type const& tensor) : tensor_(&tensor) {}

    [[nodiscard]] auto mdspec() const { return tensor_->mdspec(); }

    [[nodiscard]] static constexpr auto backend_selector() noexcept { return storage_policy::backend_selector(); }

    [[nodiscard]] auto extents() const noexcept -> extents_type const& { return tensor_->extents(); }

    [[nodiscard]] auto extent(std::size_t axis) const noexcept { return tensor_->extent(axis); }

  private:
    cuda_matrix_type const* tensor_;
};

#if UNI20_BACKEND_CUSOLVER
// Singular values stay on the device. Host-only decompositions such as svd or
// eigh reject CUDA operands with a static_assert instead of copying to the host.
template <class AsyncTensor>
concept CanComputeAsyncSingularValuesOnDevice = requires(AsyncTensor const& matrix, AsyncTensor&& consumed) {
  { uni20::linalg::singular_values(matrix) } -> std::same_as<uni20::async::Async<cuda_vector_type>>;
  { uni20::linalg::singular_values(std::move(consumed)) } -> std::same_as<uni20::async::Async<cuda_vector_type>>;
};
static_assert(CanComputeAsyncSingularValuesOnDevice<async_cuda_matrix_type>);
#endif

template <class AsyncTensor>
concept CanFormDeferredAsyncReshape = requires(AsyncTensor& tensor) { uni20::async::reshape_view(tensor, 4); };

static_assert(uni20::TensorView<CudaDescriptorMatrixView>);
static_assert(CanFormDeferredAsyncReshape<async_cuda_matrix_type>);

// Host and CUDA tensors never convert into each other; to_host, to_device and
// copy(destination, source) are the explicit transfers.
static_assert(!std::constructible_from<host_matrix_type, cuda_matrix_type const&>);
static_assert(!std::constructible_from<cuda_matrix_type, host_matrix_type const&>);
static_assert(!std::constructible_from<complex_host_matrix_type, complex_cuda_conjugated_view const&>);
static_assert(std::constructible_from<complex_cuda_matrix_type, complex_cuda_conjugated_view const&>);
// Materialization keeps the source's storage domain.
static_assert(std::same_as<decltype(uni20::make_tensor(std::declval<cuda_matrix_type const&>())), cuda_matrix_type>);
static_assert(std::same_as<decltype(uni20::make_tensor(std::declval<complex_cuda_conjugated_view const&>())),
                           complex_cuda_matrix_type>);
static_assert(std::same_as<decltype(uni20::Tensor(std::declval<complex_cuda_conjugated_view const&>())),
                           complex_cuda_matrix_type>);
static_assert(
    std::same_as<decltype(uni20::make_tensor(std::declval<async_cuda_matrix_type const&>())), async_cuda_matrix_type>);
static_assert(std::same_as<decltype(uni20::to_host(std::declval<async_cuda_matrix_type const&>())),
                           uni20::async::Async<host_matrix_type>>);
static_assert(!std::same_as<typename CudaDescriptorMatrixView::storage_policy, uni20::CudaStorage>);
static_assert(uni20::cuda::BufferMdspec<uni20::tensor_mdspec_t<CudaDescriptorMatrixView>>);
static_assert(uni20::CudaAccessibleAccessor<uni20::cuda::CudaPointerAccessor<double>>);
static_assert(!uni20::HostAccessibleAccessor<uni20::cuda::CudaPointerAccessor<double>>);
static_assert(uni20::CudaAccessibleMdspan<resolved_cuda_output_mdspan>);
static_assert(uni20::CudaAccessibleMdspan<resolved_cuda_input_mdspan>);
static_assert(uni20::MutableMdspanLike<resolved_complex_cuda_output_mdspan>);
static_assert(uni20::CudaAccessibleMdspan<resolved_complex_cuda_input_mdspan>);
static_assert(std::same_as<uni20::remove_proxy_reference_t<typename resolved_complex_cuda_output_mdspan::reference>,
                           complex_type>);
static_assert(
    std::same_as<uni20::logical_value_t<typename resolved_complex_cuda_input_mdspan::reference>, complex_type>);
static_assert(std::same_as<typename resolved_conjugated_cuda_mdspan::accessor_type,
                           uni20::cuda::CudaConjugatingPointerAccessor<double>>);
static_assert(!uni20::cuda::BufferMdspec<resolved_cuda_output_mdspan>);
static_assert(!uni20::linalg::detail::cuda_reference::SupportedCopyMdspans<resolved_cuda_output_mdspan,
                                                                           resolved_cuda_input_mdspan>);
using generated_cuda_descriptor =
    uni20::mdspec<double const, cuda_extents_type, uni20::GeneratedLayout,
                  uni20::cuda::CudaPointerAccessor<double const>, uni20::cuda::CudaBufferView<double const>>;
static_assert(uni20::CudaAccessibleMdspec<generated_cuda_descriptor>);
static_assert(!uni20::linalg::detail::cuda_reference::SupportedCopyMdspans<resolved_cuda_output_mdspan,
                                                                           generated_cuda_descriptor>);

void CUDART_CB set_copy_flag(void* raw_flag)
{
  static_cast<std::atomic<bool>*>(raw_flag)->store(true, std::memory_order_release);
}

class ErrorModeGuard {
  public:
    ErrorModeGuard() : previous_(trace::get_formatting_options().errors_abort())
    {
      trace::get_formatting_options().set_errors_abort(false);
    }

    ~ErrorModeGuard() { trace::get_formatting_options().set_errors_abort(previous_); }

  private:
    bool previous_;
};

// Transpose `host` on the device through a row-major copy and compare every
// element's bytes, so payloads that are not ordinary values (signaling NaNs)
// must survive as well.
template <class Scalar> void expect_transposing_device_copy_preserves_bytes(uni20::Tensor<Scalar, 2> const& host)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto source = uni20::to_device(host, 0);
  uni20::CudaMatrix<Scalar, uni20::RowMajor> transposed(runtime.device_resources(0), host.extent(0), host.extent(1));
  uni20::copy(transposed, source);
  auto const result = uni20::to_host(transposed);
  for (uni20::index_type row = 0; row < host.extent(0); ++row)
  {
    for (uni20::index_type column = 0; column < host.extent(1); ++column)
    {
      Scalar const expected = host[row, column];
      Scalar const actual = result[row, column];
      EXPECT_EQ(std::memcmp(&actual, &expected, sizeof(Scalar)), 0) << "element " << row << ", " << column;
    }
  }
}

template <class Scalar> uni20::Tensor<Scalar, 2> make_counting_matrix()
{
  uni20::Tensor<Scalar, 2> matrix(3, 4);
  for (uni20::index_type row = 0; row < 3; ++row)
  {
    for (uni20::index_type column = 0; column < 4; ++column)
      matrix[row, column] = static_cast<Scalar>(1 + row + 3 * column);
  }
  return matrix;
}

// A strided view of a CUDA tensor's storage, usable wherever a tensor is.
template <std::size_t Rank, class Tensor>
auto cuda_strided_view(Tensor& tensor, std::array<uni20::index_type, Rank> const& extents,
                       std::array<uni20::index_type, Rank> const& strides, std::size_t offset = 0)
{
  auto mutable_base = tensor.mdspec();
  auto const_base = std::as_const(tensor).mdspec();
  using mutable_base_type = decltype(mutable_base);
  using const_base_type = decltype(const_base);
  using extents_type = stdex::dextents<uni20::index_type, Rank>;
  using mapping_type = stdex::layout_stride::mapping<extents_type>;
  using mutable_type =
      uni20::mdspec<typename mutable_base_type::element_type, extents_type, stdex::layout_stride,
                    typename mutable_base_type::accessor_type, typename mutable_base_type::data_descriptor_type>;
  using const_type =
      uni20::mdspec<typename const_base_type::element_type, extents_type, stdex::layout_stride,
                    typename const_base_type::accessor_type, typename const_base_type::data_descriptor_type>;
  mapping_type const mapping(extents_type(extents), strides);
  return uni20::MdspecTensorView<mutable_type, const_type, uni20::CudaStorage>{
      mutable_type{mutable_base.data_descriptor().offset_by(offset), mapping, mutable_base.accessor()},
      const_type{const_base.data_descriptor().offset_by(offset), mapping, const_base.accessor()}};
}

// A strided view of a host tensor's storage.
template <std::size_t Rank, class Tensor>
auto host_strided_view(Tensor& tensor, std::array<uni20::index_type, Rank> const& extents,
                       std::array<uni20::index_type, Rank> const& strides)
{
  using element_type = typename Tensor::value_type;
  using extents_type = stdex::dextents<uni20::index_type, Rank>;
  using mapping_type = stdex::layout_stride::mapping<extents_type>;
  using mutable_type = stdex::mdspan<element_type, extents_type, stdex::layout_stride>;
  using const_type = stdex::mdspan<element_type const, extents_type, stdex::layout_stride>;
  mapping_type const mapping(extents_type(extents), strides);
  element_type* data = tensor.mdspan().data_handle();
  return uni20::MdspecTensorView<mutable_type, const_type, uni20::HostStorage>{mutable_type{data, mapping},
                                                                               const_type{data, mapping}};
}

// A copy backend that declines every instance. Nested in a staging backend, it
// checks that a step's decline leaves the output untouched.
struct DecliningCopyBackend
{
    static constexpr std::string_view name = "declining_copy";
};

template <class Output, class Input>
consteval auto kernel_accepts_types(DecliningCopyBackend const&, uni20::linalg::copy_op const&, Output&, Input&)
{
  return uni20::linalg::kernel_types_maybe;
}

template <class Output, class Input>
uni20::linalg::KernelAttempt try_kernel(DecliningCopyBackend, uni20::linalg::copy_op const&, Output&, Input&)
{
  return uni20::linalg::KernelAttempt::unsupported_instance;
}

// The CUDA copy selector is the storage selector followed by peer staging,
// which retains the storage selector for its own steps.
using cuda_copy_selector =
    decltype(uni20::linalg::select_backend_for_storage<uni20::CudaStorage>(uni20::linalg::copy_op{}));
using cuda_copy_entries = decltype(std::declval<cuda_copy_selector>().entries);
static_assert(std::same_as<std::tuple_element_t<std::tuple_size_v<cuda_copy_entries> - 1, cuda_copy_entries>,
                           uni20::linalg::CudaPeerStagingBackend<uni20::CudaStorage::backend_selector_type>>);
static_assert(std::tuple_size_v<cuda_copy_entries> ==
              std::tuple_size_v<decltype(uni20::CudaStorage::backend_selector().entries)> + 1);

// CUDA storage represents only trivially copyable elements, so other element
// types fail at compile time rather than reaching a copy kernel.
template <class Element>
concept CudaStorable = requires { typename uni20::CudaStorage::storage_t<Element>; };
static_assert(CudaStorable<Packet>);
static_assert(CudaStorable<WideElement>);
static_assert(!CudaStorable<std::string>);
static_assert(!std::is_trivially_copyable_v<std::string>);

// Every trivially copyable element has a strided device copy; wide ones loop over words.
static_assert(uni20::linalg::detail::cuda_reference::supports_elementwise_copy<Packet>);
static_assert(uni20::linalg::detail::cuda_reference::supports_elementwise_copy<WideElement>);
static_assert(!uni20::linalg::detail::cuda_reference::ElementwiseCopyWords<WideElement>::fixed_width);

// A rank-zero copy is one element, which the reference backend always moves.
using rank_zero_host = uni20::Tensor<double, 0, uni20::HostStorage, stdex::layout_stride>;
using rank_zero_cuda = uni20::CudaTensor<double, 0, stdex::layout_stride>;
static_assert(uni20::linalg::detail::cuda_staging::rank_v<uni20::tensor_mdspec_t<rank_zero_cuda>> == 0);
static_assert(!uni20::linalg::detail::cuda_staging::host_staging_types_compatible<
              decltype(uni20::linalg::select_backend_for_storage<uni20::CudaStorage>(uni20::linalg::copy_op{})),
              decltype(uni20::linalg::select_backend_for_storage<uni20::HostStorage>(uni20::linalg::copy_op{})),
              uni20::linalg::CudaReferenceBackend, uni20::tensor_mdspec_t<rank_zero_cuda>,
              uni20::tensor_mdspec_t<rank_zero_host const>>());
static_assert(!uni20::linalg::detail::cuda_staging::peer_staging_types_compatible<
              uni20::linalg::CudaReferenceBackend, uni20::tensor_mdspec_t<rank_zero_cuda>,
              uni20::tensor_mdspec_t<rank_zero_cuda const>>());

template <class Matrix> void fill_matrix(Matrix& matrix, double value)
{
  for (uni20::index_type row = 0; row < matrix.extent(0); ++row)
  {
    for (uni20::index_type column = 0; column < matrix.extent(1); ++column)
      matrix[row, column] = value;
  }
}

class CudaCopyTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
      cudaError_t const status = cudaGetDeviceCount(&device_count_);
      if (status != cudaSuccess)
      {
        GTEST_SKIP() << "CUDA device discovery failed: " << cudaGetErrorString(status);
      }
      if (device_count_ == 0) GTEST_SKIP() << "no CUDA devices are available";
    }

    static host_matrix_type make_matrix()
    {
      host_matrix_type matrix(2, 3);
      matrix[0, 0] = 1.0;
      matrix[1, 0] = 2.0;
      matrix[0, 1] = 3.0;
      matrix[1, 1] = 4.0;
      matrix[0, 2] = 5.0;
      matrix[1, 2] = 6.0;
      return matrix;
    }

    static void expect_matrix(host_matrix_type const& matrix)
    {
      ASSERT_EQ(matrix.rows(), 2);
      ASSERT_EQ(matrix.cols(), 3);
      EXPECT_DOUBLE_EQ((matrix[0, 0]), 1.0);
      EXPECT_DOUBLE_EQ((matrix[1, 0]), 2.0);
      EXPECT_DOUBLE_EQ((matrix[0, 1]), 3.0);
      EXPECT_DOUBLE_EQ((matrix[1, 1]), 4.0);
      EXPECT_DOUBLE_EQ((matrix[0, 2]), 5.0);
      EXPECT_DOUBLE_EQ((matrix[1, 2]), 6.0);
    }

    int device_count_ = 0;
};

TEST(CudaCopyPlanningTest, ElementwisePlanCompactsAndDecodesIndependentPaddedStrides)
{
  using mapping_type = stdex::layout_stride::mapping<cuda_extents_type>;
  cuda_extents_type const extents{2, 3};
  std::array<uni20::index_type, 2> const output_strides{1, 3};
  std::array<uni20::index_type, 2> const input_strides{1, 2};
  double output_storage[8]{};
  double input_storage[6]{};
  stdex::mdspan output{output_storage, mapping_type{extents, output_strides}};
  stdex::mdspan input{input_storage, mapping_type{extents, input_strides}};
  uni20::linalg::detail::cuda_reference::LoweredElementwiseCopyPlan plan;

  ASSERT_TRUE(uni20::linalg::detail::cuda_reference::try_make_elementwise_plan(output, input, plan));
  ASSERT_EQ(plan.index_kind, uni20::linalg::detail::cuda_reference::ElementwiseIndexKind::index_32);
  auto const& layout = plan.plan_32;
  EXPECT_EQ(layout.compact_rank, 2);
  EXPECT_EQ(layout.element_count, 6);
  EXPECT_EQ(layout.offsets(0), (std::array<std::int32_t, 2>{0, 0}));
  EXPECT_EQ(layout.offsets(1), (std::array<std::int32_t, 2>{1, 1}));
  EXPECT_EQ(layout.offsets(2), (std::array<std::int32_t, 2>{3, 2}));
  EXPECT_EQ(layout.offsets(3), (std::array<std::int32_t, 2>{4, 3}));
  EXPECT_EQ(layout.offsets(5), (std::array<std::int32_t, 2>{7, 5}));
}

TEST(CudaCopyPlanningTest, ElementwisePlanUses64BitOffsetsWhenAReachableOffsetExceeds32Bits)
{
  using extents_type = stdex::dextents<uni20::index_type, 1>;
  using mapping_type = stdex::layout_stride::mapping<extents_type>;
  auto const large_stride = static_cast<uni20::index_type>(std::numeric_limits<std::int32_t>::max()) + 1;
  mapping_type output_mapping{extents_type{2}, std::array<uni20::index_type, 1>{large_stride}};
  mapping_type input_mapping{extents_type{2}, std::array<uni20::index_type, 1>{1}};
  double value = 0;
  stdex::mdspan output{&value, output_mapping};
  stdex::mdspan input{&value, input_mapping};
  uni20::linalg::detail::cuda_reference::LoweredElementwiseCopyPlan plan;

  ASSERT_TRUE(uni20::linalg::detail::cuda_reference::try_make_elementwise_plan(output, input, plan));
  ASSERT_EQ(plan.index_kind, uni20::linalg::detail::cuda_reference::ElementwiseIndexKind::index_64);
  EXPECT_EQ(plan.plan_64.offsets(1), (std::array<std::int64_t, 2>{static_cast<std::int64_t>(large_stride), 1}));
}

TEST(CudaCopyPlanningTest, LoweredPlanVisitsOnlyTheSelectedPayload)
{
  uni20::linalg::detail::cuda_reference::LoweredElementwiseCopyPlan plan;
  plan.plan_32.element_count = 7;
  plan.plan_64.element_count = 11;

  std::size_t visited = 0;
  auto record = [&](auto const& selected) { visited = static_cast<std::size_t>(selected.element_count); };

  plan.index_kind = uni20::linalg::detail::cuda_reference::ElementwiseIndexKind::index_32;
  plan.visit(record);
  EXPECT_EQ(visited, 7);

  plan.index_kind = uni20::linalg::detail::cuda_reference::ElementwiseIndexKind::index_64;
  plan.visit(record);
  EXPECT_EQ(visited, 11);
}

TEST(CudaCopyPlanningTest, EmptyIterationLowersToAZeroRank32BitPayload)
{
  using extents_type = stdex::dextents<uni20::index_type, 1>;
  using mapping_type = stdex::layout_stride::mapping<extents_type>;
  mapping_type mapping{extents_type{0}, std::array<uni20::index_type, 1>{1}};
  double value = 0;
  stdex::mdspan output{&value, mapping};
  stdex::mdspan input{&value, mapping};
  uni20::linalg::detail::cuda_reference::LoweredElementwiseCopyPlan plan;

  ASSERT_TRUE(uni20::linalg::detail::cuda_reference::try_make_elementwise_plan(output, input, plan));
  ASSERT_EQ(plan.index_kind, uni20::linalg::detail::cuda_reference::ElementwiseIndexKind::index_32);
  EXPECT_EQ(plan.plan_32.element_count, 0);
  EXPECT_EQ(plan.plan_32.compact_rank, 0);
}

TEST(CudaCopyPlanningTest, ElementwisePlanDeclinesNegativeStridesUntilCudaRebasingIsDefined)
{
  using extents_type = stdex::dextents<uni20::index_type, 1>;
  using mapping_type = stdex::layout_stride::mapping<extents_type>;
  mapping_type output_mapping{extents_type{2}, std::array<uni20::index_type, 1>{-1}};
  mapping_type input_mapping{extents_type{2}, std::array<uni20::index_type, 1>{1}};
  double value = 0;
  stdex::mdspan output{&value, output_mapping};
  stdex::mdspan input{&value, input_mapping};
  uni20::linalg::detail::cuda_reference::LoweredElementwiseCopyPlan plan;

  EXPECT_FALSE(uni20::linalg::detail::cuda_reference::try_make_elementwise_plan(output, input, plan));
}

TEST(CudaCopyPlanningTest, NonpositiveActiveStridesDecline)
{
  using uni20::linalg::detail::cuda_reference::active_strides_are_positive;

  EXPECT_TRUE(active_strides_are_positive(StrideProbe<1>{.extents_ = {3}, .strides_ = {1}}));
  EXPECT_FALSE(active_strides_are_positive(StrideProbe<1>{.extents_ = {3}, .strides_ = {0}}));
  EXPECT_FALSE(active_strides_are_positive(StrideProbe<1>{.extents_ = {3}, .strides_ = {-1}}));
  EXPECT_TRUE(active_strides_are_positive(StrideProbe<1>{.extents_ = {1}, .strides_ = {-1}}));
  EXPECT_TRUE(active_strides_are_positive(StrideProbe<1>{.extents_ = {0}, .strides_ = {-1}}));
  EXPECT_TRUE(active_strides_are_positive(StrideProbe<2>{.extents_ = {0, 3}, .strides_ = {1, -1}}));
}

TEST_F(CudaCopyTest, PageableHostRoundTripUsesExplicitTransferFunctions)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto host = make_matrix();

  auto device = uni20::to_device(host, 0);
  auto result = uni20::to_host(device);

  EXPECT_EQ(device.storage().device().ordinal(), 0);
  expect_matrix(result);
}

TEST_F(CudaCopyTest, MaterializationKeepsCudaSourcesOnTheirDevice)
{
  if (device_count_ < 2) GTEST_SKIP() << "requires two CUDA devices";
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0, 1}, .streams_per_device = 2});
  complex_host_matrix_type source(2, 2);
  source[0, 0] = complex_type{1.0, 2.0};
  source[1, 0] = complex_type{-3.0, 4.0};
  source[0, 1] = complex_type{5.0, -6.0};
  source[1, 1] = complex_type{-7.0, -8.0};
  auto device = uni20::to_device(source, 1);

  auto copied = uni20::make_tensor(device);
  auto conjugated = uni20::make_tensor(uni20::conj(device));
  uni20::Tensor deduced(uni20::conj(device));

  EXPECT_EQ(copied.storage().device().ordinal(), 1);
  EXPECT_EQ(conjugated.storage().device().ordinal(), 1);
  EXPECT_EQ(deduced.storage().device().ordinal(), 1);
  auto const copied_host = uni20::to_host(copied);
  auto const conjugated_host = uni20::to_host(conjugated);
  auto const deduced_host = uni20::to_host(deduced);
  for (uni20::index_type row = 0; row < 2; ++row)
  {
    for (uni20::index_type col = 0; col < 2; ++col)
    {
      EXPECT_EQ((copied_host[row, col]), (source[row, col]));
      EXPECT_EQ((conjugated_host[row, col]), uni20::conj(source[row, col]));
      EXPECT_EQ((deduced_host[row, col]), uni20::conj(source[row, col]));
    }
  }
}

TEST_F(CudaCopyTest, AsyncMaterializationStaysOnDeviceAndToHostTransfers)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::async::DebugCudaScheduler scheduler(uni20::cuda::Device::get(0));
  uni20::async::ScopedScheduler scoped(&scheduler);
  async_cuda_matrix_type device = uni20::to_device(make_matrix(), 0);

  auto materialized = uni20::make_tensor(device);
  auto host = uni20::to_host(device);

  EXPECT_EQ(materialized.get_wait(scheduler).storage().device().ordinal(), 0);
  expect_matrix(uni20::to_host(materialized.get_wait(scheduler)));
  expect_matrix(host.get_wait(scheduler));
}

TEST_F(CudaCopyTest, FixedOutputPageableTransfersResizeAndRoundTrip)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto source = make_matrix();
  cuda_matrix_type device(runtime.device_resources(0), 1, 1);
  host_matrix_type result(1, 1);

  uni20::copy(device, source);
  uni20::copy(result, device);

  EXPECT_EQ(device.rows(), source.rows());
  EXPECT_EQ(device.cols(), source.cols());
  expect_matrix(result);
}

TEST_F(CudaCopyTest, ExplicitCudaBackendUsesDescriptorRatherThanStoragePolicy)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto device = uni20::to_device(make_matrix(), 0);
  CudaDescriptorMatrixView input(device);
  host_matrix_type result;

  uni20::copy(uni20::linalg::CudaReferenceBackend{}, result, input);

  expect_matrix(result);
}

TEST_F(CudaCopyTest, DeviceToHostCopyObservesConjugatingAccessor)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  complex_host_matrix_type source(2, 2);
  source[0, 0] = complex_type{1.0, 2.0};
  source[1, 0] = complex_type{-3.0, 4.0};
  source[0, 1] = complex_type{5.0, -6.0};
  source[1, 1] = complex_type{-7.0, -8.0};

  auto device = uni20::to_device(source, 0);
  auto result = uni20::to_host(uni20::conj(device));

  EXPECT_EQ((result[0, 0]), (complex_type{1.0, -2.0}));
  EXPECT_EQ((result[1, 0]), (complex_type{-3.0, -4.0}));
  EXPECT_EQ((result[0, 1]), (complex_type{5.0, 6.0}));
  EXPECT_EQ((result[1, 1]), (complex_type{-7.0, 8.0}));
}

TEST_F(CudaCopyTest, PageableCopyConvertsBetweenHostAndDeviceLayouts)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto const source = make_matrix();
  row_major_host_matrix_type row_major_source(2, 3);
  uni20::copy(row_major_source, source);

  cuda_matrix_type device(runtime.device_resources(0), 2, 3);
  uni20::copy(device, row_major_source);
  expect_matrix(uni20::to_host(device));

  row_major_host_matrix_type row_major_result(2, 3);
  uni20::copy(row_major_result, device);
  host_matrix_type result(2, 3);
  uni20::copy(result, row_major_result);
  expect_matrix(result);
}

TEST_F(CudaCopyTest, PageableCopyStagesNonContiguousHostAndDeviceViews)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto const source = make_matrix();

  // Columns 0, 2 and 4 of a 2 x 6 allocation on each side.
  host_matrix_type host_storage(2, 6);
  auto host_columns = host_strided_view<2>(host_storage, {2, 3}, {1, 4});
  uni20::copy(host_columns, source);
  cuda_matrix_type device_storage(runtime.device_resources(0), 2, 6);
  auto device_columns = cuda_strided_view<2>(device_storage, {2, 3}, {1, 4});

  uni20::copy(device_columns, host_columns);
  host_matrix_type contiguous(2, 3);
  uni20::copy(contiguous, device_columns);
  expect_matrix(contiguous);

  cuda_matrix_type device(runtime.device_resources(0), 2, 3);
  uni20::copy(device, host_columns);
  expect_matrix(uni20::to_host(device));

  host_matrix_type other_storage(2, 6);
  auto other_columns = host_strided_view<2>(other_storage, {2, 3}, {1, 4});
  uni20::copy(other_columns, device);
  uni20::copy(contiguous, other_columns);
  expect_matrix(contiguous);

  // Neither side contiguous: device columns into a transposed host view.
  host_matrix_type transposed_storage(3, 2);
  auto transposed = host_strided_view<2>(transposed_storage, {2, 3}, {3, 1});
  uni20::copy(transposed, device_columns);
  for (uni20::index_type row = 0; row < 2; ++row)
  {
    for (uni20::index_type column = 0; column < 3; ++column)
      EXPECT_DOUBLE_EQ((transposed_storage[column, row]), (source[row, column]));
  }
}

TEST_F(CudaCopyTest, PageableStagedCopiesObserveConjugatingAccessors)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  complex_host_matrix_type source(2, 2);
  source[0, 0] = complex_type{1.0, 2.0};
  source[1, 0] = complex_type{-3.0, 4.0};
  source[0, 1] = complex_type{5.0, -6.0};
  source[1, 1] = complex_type{-7.0, -8.0};
  auto device = uni20::to_device(source, 0);

  uni20::RowMajorTensor<complex_type, 2> row_major(2, 2);
  uni20::copy(row_major, uni20::conj(device));
  uni20::CudaTensor<complex_type, 2, uni20::RowMajor> device_row_major(runtime.device_resources(0), 2, 2);
  uni20::copy(device_row_major, uni20::conj(source));
  auto const roundtrip = uni20::to_host(device_row_major);
  for (uni20::index_type row = 0; row < 2; ++row)
  {
    for (uni20::index_type column = 0; column < 2; ++column)
    {
      EXPECT_EQ((row_major[row, column]), uni20::conj(source[row, column]));
      EXPECT_EQ((roundtrip[row, column]), uni20::conj(source[row, column]));
    }
  }
}

TEST_F(CudaCopyTest, DeviceToDeviceCopyExecutesConjugatingAccessorKernel)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  complex_host_matrix_type source(2, 2);
  source[0, 0] = complex_type{1.0, 2.0};
  source[1, 0] = complex_type{-3.0, 4.0};
  source[0, 1] = complex_type{5.0, -6.0};
  source[1, 1] = complex_type{-7.0, -8.0};

  auto device_source = uni20::to_device(source, 0);
  complex_cuda_matrix_type device_output(runtime.device_resources(0), 2, 2);
  auto conjugated = uni20::conj(device_source);
  auto output_span = device_output.mdspec();
  auto input_span = conjugated.mdspec();
  auto preparation = uni20::linalg::detail::cuda_reference::prepare_copy(output_span, input_span);
  ASSERT_EQ(preparation.attempt, uni20::linalg::KernelAttempt::success);
  EXPECT_EQ(preparation.execution, uni20::linalg::detail::cuda_reference::CopyExecution::elementwise_kernel);

  uni20::copy(device_output, conjugated);

  auto result = uni20::to_host(device_output);
  EXPECT_EQ((result[0, 0]), (complex_type{1.0, -2.0}));
  EXPECT_EQ((result[1, 0]), (complex_type{-3.0, -4.0}));
  EXPECT_EQ((result[0, 1]), (complex_type{5.0, 6.0}));
  EXPECT_EQ((result[1, 1]), (complex_type{-7.0, 8.0}));
}

TEST_F(CudaCopyTest, SameBufferConjugatingCopyDeclinesWithoutMutation)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  complex_host_matrix_type source(2, 2);
  source[0, 0] = complex_type{1.0, 2.0};
  source[1, 0] = complex_type{-3.0, 4.0};
  source[0, 1] = complex_type{5.0, -6.0};
  source[1, 1] = complex_type{-7.0, -8.0};

  auto device = uni20::to_device(source, 0);
  auto conjugated = uni20::conj(device);
  auto output_span = device.mdspec();
  auto input_span = conjugated.mdspec();

  EXPECT_EQ(uni20::linalg::detail::cuda_reference::copy(output_span, input_span),
            uni20::linalg::KernelAttempt::unsupported_instance);
  EXPECT_FALSE(uni20::linalg::try_dispatch_kernel(uni20::linalg::CudaReferenceBackend{}, uni20::linalg::copy_op{},
                                                  output_span, input_span));
  auto same_input_span = uni20::mdspec_of(std::as_const(device));
  EXPECT_TRUE(uni20::linalg::try_dispatch_kernel(uni20::linalg::CudaReferenceBackend{}, uni20::linalg::copy_op{},
                                                 output_span, same_input_span));

  auto result = uni20::to_host(device);
  EXPECT_EQ((result[0, 0]), (complex_type{1.0, 2.0}));
  EXPECT_EQ((result[1, 0]), (complex_type{-3.0, 4.0}));
  EXPECT_EQ((result[0, 1]), (complex_type{5.0, -6.0}));
  EXPECT_EQ((result[1, 1]), (complex_type{-7.0, -8.0}));
}

TEST_F(CudaCopyTest, EmptySameBufferConjugatingCopySucceeds)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  complex_cuda_matrix_type device(runtime.device_resources(0), 1, 1);
  auto conjugated = uni20::conj(device);
  auto output_base = device.mdspec();
  auto input_base = conjugated.mdspec();
  using output_type = decltype(output_base);
  using input_type = decltype(input_base);
  typename output_type::mapping_type output_mapping{typename output_type::extents_type{0, 1}};
  typename input_type::mapping_type input_mapping{typename input_type::extents_type{0, 1}};
  output_type output{output_base.data_descriptor(), output_mapping, output_base.accessor()};
  input_type input{input_base.data_descriptor(), input_mapping, input_base.accessor()};

  auto preparation = uni20::linalg::detail::cuda_reference::prepare_copy(output, input);

  EXPECT_EQ(preparation.attempt, uni20::linalg::KernelAttempt::success);
  EXPECT_FALSE(preparation.has_work);
  EXPECT_EQ(uni20::linalg::detail::cuda_reference::copy(output, input), uni20::linalg::KernelAttempt::success);
}

TEST_F(CudaCopyTest, EmptySameBufferDifferentOffsetCopySucceeds)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  cuda_matrix_type device(runtime.device_resources(0), 1, 2);
  auto output_base = device.mdspec();
  auto input_base = std::as_const(device).mdspec();
  using output_type = decltype(output_base);
  using input_type = decltype(input_base);
  typename output_type::mapping_type output_mapping{typename output_type::extents_type{0, 1}};
  typename input_type::mapping_type input_mapping{typename input_type::extents_type{0, 1}};
  output_type output{output_base.data_descriptor(), output_mapping, output_base.accessor()};
  input_type input{input_base.data_descriptor().offset_by(1), input_mapping, input_base.accessor()};

  auto preparation = uni20::linalg::detail::cuda_reference::prepare_copy(output, input);

  EXPECT_EQ(preparation.attempt, uni20::linalg::KernelAttempt::success);
  EXPECT_FALSE(preparation.has_work);
  EXPECT_EQ(uni20::linalg::detail::cuda_reference::copy(output, input), uni20::linalg::KernelAttempt::success);
}

TEST_F(CudaCopyTest, DisjointSameBufferCopyUsesOneAccessState)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  uni20::Tensor<double, 1> host(6);
  host[0] = 1.0;
  host[1] = 2.0;
  host[2] = 3.0;
  host[3] = -1.0;
  host[4] = -1.0;
  host[5] = -1.0;
  auto device = uni20::to_device(host, 0);
  auto output = make_vector_mdspec(device, 3, 3);
  auto input = make_vector_mdspec(std::as_const(device), 0, 3);

  auto preparation = uni20::linalg::detail::cuda_reference::prepare_copy(output, input);
  ASSERT_EQ(preparation.attempt, uni20::linalg::KernelAttempt::success);
  ASSERT_TRUE(preparation.has_work);
  ASSERT_EQ(preparation.output_buffer, preparation.input_buffer);
  ASSERT_EQ(uni20::linalg::detail::cuda_reference::copy(output, input), uni20::linalg::KernelAttempt::success);

  auto result = uni20::to_host(device);
  EXPECT_DOUBLE_EQ(result[0], 1.0);
  EXPECT_DOUBLE_EQ(result[1], 2.0);
  EXPECT_DOUBLE_EQ(result[2], 3.0);
  EXPECT_DOUBLE_EQ(result[3], 1.0);
  EXPECT_DOUBLE_EQ(result[4], 2.0);
  EXPECT_DOUBLE_EQ(result[5], 3.0);
}

TEST_F(CudaCopyTest, HostToDeviceCopyObservesConjugatingAccessor)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  complex_host_matrix_type source(2, 2);
  source[0, 0] = complex_type{1.0, 2.0};
  source[1, 0] = complex_type{-3.0, 4.0};
  source[0, 1] = complex_type{5.0, -6.0};
  source[1, 1] = complex_type{-7.0, -8.0};

  auto device = uni20::to_device(uni20::conj(source), 0);
  auto result = uni20::to_host(device);

  EXPECT_EQ((result[0, 0]), (complex_type{1.0, -2.0}));
  EXPECT_EQ((result[1, 0]), (complex_type{-3.0, -4.0}));
  EXPECT_EQ((result[0, 1]), (complex_type{5.0, 6.0}));
  EXPECT_EQ((result[1, 1]), (complex_type{-7.0, 8.0}));
}

TEST_F(CudaCopyTest, SameDeviceCopyUsesCudaReferenceFallback)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto source = uni20::to_device(make_matrix(), 0);
  cuda_matrix_type destination(runtime.device_resources(0), 2, 3);

  uni20::copy(destination, source);

  expect_matrix(uni20::to_host(destination));
}

TEST_F(CudaCopyTest, SameDeviceCopyConvertsBetweenColumnMajorAndRowMajorMappings)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto source = uni20::to_device(make_matrix(), 0);
  cuda_matrix_type row_major_storage(runtime.device_resources(0), 2, 3);
  cuda_matrix_type roundtrip(runtime.device_resources(0), 2, 3);

  auto source_span = std::as_const(source).mdspec();
  auto row_major_base = row_major_storage.mdspec();
  using row_major_mapping = stdex::layout_right::mapping<cuda_extents_type>;
  auto row_major_output = remap_mdspec<stdex::layout_right>(row_major_base, row_major_mapping{cuda_extents_type{2, 3}});

  auto preparation = uni20::linalg::detail::cuda_reference::prepare_copy(row_major_output, source_span);
  ASSERT_EQ(preparation.attempt, uni20::linalg::KernelAttempt::success);
  EXPECT_EQ(preparation.execution, uni20::linalg::detail::cuda_reference::CopyExecution::elementwise_kernel);
  ASSERT_EQ(preparation.elementwise_plan.index_kind,
            uni20::linalg::detail::cuda_reference::ElementwiseIndexKind::index_32);
  auto const& elementwise_plan = preparation.elementwise_plan.plan_32;
  ASSERT_EQ(elementwise_plan.compact_rank, 2);
  EXPECT_EQ(elementwise_plan.strides[0], (std::array<std::int32_t, 2>{1, 2}));
  EXPECT_EQ(elementwise_plan.strides[1], (std::array<std::int32_t, 2>{3, 1}));
  ASSERT_EQ(uni20::linalg::detail::cuda_reference::copy(row_major_output, source_span),
            uni20::linalg::KernelAttempt::success);

  auto row_major_input_base = std::as_const(row_major_storage).mdspec();
  auto row_major_input =
      remap_mdspec<stdex::layout_right>(row_major_input_base, row_major_mapping{cuda_extents_type{2, 3}});
  auto roundtrip_output = roundtrip.mdspec();
  ASSERT_EQ(uni20::linalg::detail::cuda_reference::copy(roundtrip_output, row_major_input),
            uni20::linalg::KernelAttempt::success);

  expect_matrix(uni20::to_host(roundtrip));
}

TEST_F(CudaCopyTest, StridedDeviceCopyMovesIntegerElementsOfEverySize)
{
  expect_transposing_device_copy_preserves_bytes(make_counting_matrix<std::int8_t>());
  expect_transposing_device_copy_preserves_bytes(make_counting_matrix<std::int16_t>());
  expect_transposing_device_copy_preserves_bytes(make_counting_matrix<std::int32_t>());
  expect_transposing_device_copy_preserves_bytes(make_counting_matrix<std::int64_t>());
}

TEST_F(CudaCopyTest, StridedDeviceCopyPreservesSignalingNanPayloads)
{
  auto matrix = make_counting_matrix<float>();
  matrix[1, 2] = std::numeric_limits<float>::signaling_NaN();
  expect_transposing_device_copy_preserves_bytes(matrix);
}

TEST_F(CudaCopyTest, SameDeviceCopyHandlesPaddedStridesAndBufferOffsets)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto source = uni20::to_device(make_matrix(), 0);
  cuda_vector_type padded_storage(runtime.device_resources(0), 10);
  cuda_matrix_type roundtrip(runtime.device_resources(0), 2, 3);
  std::array<uni20::index_type, 2> const padded_strides{1, 3};

  auto source_span = std::as_const(source).mdspec();
  auto padded_output = make_strided_matrix_mdspec(padded_storage, 1, padded_strides);
  auto preparation = uni20::linalg::detail::cuda_reference::prepare_copy(padded_output, source_span);
  ASSERT_EQ(preparation.attempt, uni20::linalg::KernelAttempt::success);
  EXPECT_EQ(preparation.execution, uni20::linalg::detail::cuda_reference::CopyExecution::elementwise_kernel);
  EXPECT_EQ(preparation.output_offset, 1);
  EXPECT_EQ(preparation.output_span_size, 8);
  ASSERT_EQ(uni20::linalg::detail::cuda_reference::copy(padded_output, source_span),
            uni20::linalg::KernelAttempt::success);

  auto padded_input = make_strided_matrix_mdspec(std::as_const(padded_storage), 1, padded_strides);
  auto roundtrip_output = roundtrip.mdspec();
  ASSERT_EQ(uni20::linalg::detail::cuda_reference::copy(roundtrip_output, padded_input),
            uni20::linalg::KernelAttempt::success);

  expect_matrix(uni20::to_host(roundtrip));
}

TEST_F(CudaCopyTest, SameDeviceCopyOrdersPendingSourceAndDestinationReuse)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 3});
  auto source = uni20::to_device(make_matrix(), 0);
  cuda_matrix_type destination(runtime.device_resources(0), 2, 3);
  uni20::test::CudaStreamGate gate;

  {
    auto stream = runtime.device_resources(0).streams().acquire();
    auto source_predecessor = source.storage().write_synchronized_with(stream);
    uni20::cuda::ScopedDevice device(0);
    gate.close(stream.native_handle(), 0);
  }

  uni20::copy(destination, source);

  std::atomic<bool> destination_reader_completed = false;
  {
    auto stream = runtime.device_resources(0).streams().acquire();
    auto destination_reader = destination.storage().read_synchronized_with(stream);
    uni20::cuda::ScopedDevice device(0);
    uni20::cuda::check(cudaLaunchHostFunc(stream.native_handle(), set_copy_flag, &destination_reader_completed),
                       "cudaLaunchHostFunc same-device destination reader", 0);
  }

  bool const predecessor_entered = gate.wait_entered();
  if (!predecessor_entered) gate.open();
  ASSERT_TRUE(predecessor_entered);
  EXPECT_FALSE(destination_reader_completed.load(std::memory_order_acquire));

  gate.open();
  runtime.device_resources(0).streams().synchronize();
  EXPECT_TRUE(destination_reader_completed.load(std::memory_order_acquire));
  expect_matrix(uni20::to_host(destination));
}

TEST_F(CudaCopyTest, RowMajorRoundTripPreservesLogicalOrder)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  row_major_host_matrix_type host(2, 3);
  host[0, 0] = 1.0;
  host[0, 1] = 2.0;
  host[0, 2] = 3.0;
  host[1, 0] = 4.0;
  host[1, 1] = 5.0;
  host[1, 2] = 6.0;

  auto result = uni20::to_host(uni20::to_device(host, 0));

  static_assert(std::same_as<typename decltype(result)::layout_type, stdex::layout_right>);
  EXPECT_DOUBLE_EQ((result[0, 0]), 1.0);
  EXPECT_DOUBLE_EQ((result[0, 1]), 2.0);
  EXPECT_DOUBLE_EQ((result[0, 2]), 3.0);
  EXPECT_DOUBLE_EQ((result[1, 0]), 4.0);
  EXPECT_DOUBLE_EQ((result[1, 1]), 5.0);
  EXPECT_DOUBLE_EQ((result[1, 2]), 6.0);
}

TEST_F(CudaCopyTest, ZeroExtentRoundTripIsAnEmptySuccessfulTransfer)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 1});
  host_matrix_type host(3, 0);

  auto device = uni20::to_device(host, 0);
  auto result = uni20::to_host(device);

  EXPECT_EQ(device.rows(), 3);
  EXPECT_EQ(device.cols(), 0);
  EXPECT_EQ(result.rows(), 3);
  EXPECT_EQ(result.cols(), 0);
}

TEST_F(CudaCopyTest, PeerCopyPreservesValues)
{
  if (device_count_ < 2) GTEST_SKIP() << "peer-copy test requires two CUDA devices";
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0, 1}, .streams_per_device = 2});
  auto source = uni20::to_device(make_matrix(), 0);

  auto destination = uni20::to_device(source, 1);

  EXPECT_EQ(destination.storage().device().ordinal(), 1);
  expect_matrix(uni20::to_host(destination));
}

TEST_F(CudaCopyTest, PeerCopyPreservesSourceAndDestinationLedgers)
{
  if (device_count_ < 2) GTEST_SKIP() << "peer-copy test requires two CUDA devices";
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0, 1}, .streams_per_device = 3});
  auto source = uni20::to_device(make_matrix(), 0);
  cuda_matrix_type destination(runtime.device_resources(1), 2, 3);
  uni20::test::CudaStreamGate gate;

  {
    auto stream = runtime.device_resources(1).streams().acquire();
    auto destination_predecessor = destination.storage().write_synchronized_with(stream);
    uni20::cuda::ScopedDevice device(1);
    gate.close(stream.native_handle(), 1);
  }

  uni20::copy(destination, source);

  std::atomic<bool> source_writer_completed = false;
  {
    auto stream = runtime.device_resources(0).streams().acquire();
    auto source_writer = source.storage().write_synchronized_with(stream);
    uni20::cuda::ScopedDevice device(0);
    uni20::cuda::check(cudaLaunchHostFunc(stream.native_handle(), set_copy_flag, &source_writer_completed),
                       "cudaLaunchHostFunc peer source writer", 0);
  }

  std::atomic<bool> destination_reader_completed = false;
  {
    auto stream = runtime.device_resources(1).streams().acquire();
    auto destination_reader = destination.storage().read_synchronized_with(stream);
    uni20::cuda::ScopedDevice device(1);
    uni20::cuda::check(cudaLaunchHostFunc(stream.native_handle(), set_copy_flag, &destination_reader_completed),
                       "cudaLaunchHostFunc peer destination reader", 1);
  }

  bool const predecessor_entered = gate.wait_entered();
  if (!predecessor_entered) gate.open();
  ASSERT_TRUE(predecessor_entered);
  EXPECT_FALSE(source_writer_completed.load(std::memory_order_acquire));
  EXPECT_FALSE(destination_reader_completed.load(std::memory_order_acquire));

  gate.open();
  runtime.device_resources(0).streams().synchronize();
  runtime.device_resources(1).streams().synchronize();
  EXPECT_TRUE(source_writer_completed.load(std::memory_order_acquire));
  EXPECT_TRUE(destination_reader_completed.load(std::memory_order_acquire));
  expect_matrix(uni20::to_host(destination));
}

TEST_F(CudaCopyTest, AsyncCopyConstructsOutputAndCarriesDeviceCompletion)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::async::DebugCudaScheduler scheduler(uni20::cuda::Device::get(0));
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<cuda_matrix_type> input = uni20::to_device(make_matrix(), 0);
  uni20::async::Async<cuda_matrix_type> output;

  uni20::copy(output, input);

  auto const& device_result = output.get_wait(scheduler);
  expect_matrix(uni20::to_host(device_result));
}

TEST_F(CudaCopyTest, AsyncConjugatingCopyRunsElementwiseKernel)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::async::DebugCudaScheduler scheduler(uni20::cuda::Device::get(0));
  uni20::async::ScopedScheduler scoped(&scheduler);
  complex_host_matrix_type source(2, 2);
  source[0, 0] = complex_type{1.0, 2.0};
  source[1, 0] = complex_type{-3.0, 4.0};
  source[0, 1] = complex_type{5.0, -6.0};
  source[1, 1] = complex_type{-7.0, -8.0};
  uni20::async::Async<complex_cuda_matrix_type> input = uni20::to_device(source, 0);
  auto conjugated = uni20::async::conj(input);
  uni20::async::Async<complex_cuda_matrix_type> output;

  uni20::copy(output, conjugated);

  auto result = uni20::to_host(output.get_wait(scheduler));
  EXPECT_EQ((result[0, 0]), (complex_type{1.0, -2.0}));
  EXPECT_EQ((result[1, 0]), (complex_type{-3.0, -4.0}));
  EXPECT_EQ((result[0, 1]), (complex_type{5.0, 6.0}));
  EXPECT_EQ((result[1, 1]), (complex_type{-7.0, 8.0}));
}

TEST_F(CudaCopyTest, AsyncPageableCopiesAndMaterializationUseBlockingCudaBoundary)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::async::DebugCudaScheduler scheduler(uni20::cuda::Device::get(0));
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<host_matrix_type> host_input = make_matrix();

  auto device_output = uni20::to_device(host_input, runtime.device_resources(0));
  auto host_output = uni20::to_host(device_output);

  expect_matrix(host_output.get_wait(scheduler));
}

TEST_F(CudaCopyTest, AsyncPeerCopyRunsOnDestinationDevice)
{
  if (device_count_ < 2) GTEST_SKIP() << "async peer-copy test requires two CUDA devices";
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0, 1}, .streams_per_device = 2});
  uni20::async::DebugCudaScheduler scheduler(uni20::cuda::Device::get(0));
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<cuda_matrix_type> input = uni20::to_device(make_matrix(), 0);
  uni20::async::Async<cuda_matrix_type> output = cuda_matrix_type(runtime.device_resources(1), 2, 3);

  uni20::copy(output, input);

  auto const& device_result = output.get_wait(scheduler);
  EXPECT_EQ(device_result.storage().device().ordinal(), 1);
  expect_matrix(uni20::to_host(device_result));
}

TEST_F(CudaCopyTest, PeerCopyStagesLayoutChangesAndConjugation)
{
  if (device_count_ < 2) GTEST_SKIP() << "peer-copy test requires two CUDA devices";
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0, 1}, .streams_per_device = 2});
  auto const source = make_matrix();
  auto device0 = uni20::to_device(source, 0);

  uni20::CudaTensor<double, 2, uni20::RowMajor> row_major(runtime.device_resources(1), 2, 3);
  uni20::copy(row_major, device0);
  host_matrix_type result(2, 3);
  uni20::copy(result, row_major);
  expect_matrix(result);

  // A padded source: columns 0, 2 and 4 of a 2 x 6 allocation on device 0.
  cuda_matrix_type padded_storage(runtime.device_resources(0), 2, 6);
  auto padded = cuda_strided_view<2>(padded_storage, {2, 3}, {1, 4});
  uni20::copy(padded, device0);
  cuda_matrix_type device1(runtime.device_resources(1), 2, 3);
  uni20::copy(device1, padded);
  expect_matrix(uni20::to_host(device1));

  complex_host_matrix_type complex_source(2, 2);
  complex_source[0, 0] = complex_type{1.0, 2.0};
  complex_source[1, 0] = complex_type{-3.0, 4.0};
  complex_source[0, 1] = complex_type{5.0, -6.0};
  complex_source[1, 1] = complex_type{-7.0, -8.0};
  auto complex_device0 = uni20::to_device(complex_source, 0);
  complex_cuda_matrix_type conjugated(runtime.device_resources(1), 2, 2);
  uni20::copy(conjugated, uni20::conj(complex_device0));
  auto const conjugated_result = uni20::to_host(conjugated);
  for (uni20::index_type row = 0; row < 2; ++row)
  {
    for (uni20::index_type column = 0; column < 2; ++column)
      EXPECT_EQ((conjugated_result[row, column]), uni20::conj(complex_source[row, column]));
  }
}

TEST_F(CudaCopyTest, AsyncPeerCopyStagesLayoutChanges)
{
  if (device_count_ < 2) GTEST_SKIP() << "async peer-copy test requires two CUDA devices";
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0, 1}, .streams_per_device = 2});
  uni20::async::DebugCudaScheduler scheduler(uni20::cuda::Device::get(0));
  uni20::async::ScopedScheduler scoped(&scheduler);
  using row_major_cuda_matrix = uni20::CudaTensor<double, 2, uni20::RowMajor>;
  uni20::async::Async<cuda_matrix_type> input = uni20::to_device(make_matrix(), 0);
  uni20::async::Async<row_major_cuda_matrix> output = row_major_cuda_matrix(runtime.device_resources(1), 2, 3);

  uni20::copy(output, input);

  auto const& device_result = output.get_wait(scheduler);
  EXPECT_EQ(device_result.storage().device().ordinal(), 1);
  host_matrix_type result(2, 3);
  uni20::copy(result, device_result);
  expect_matrix(result);
}

TEST_F(CudaCopyTest, ReferenceKernelLeavesLayoutChangingTransfersToHostStaging)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  row_major_host_matrix_type row_major_source(2, 3);
  uni20::copy(row_major_source, make_matrix());
  host_matrix_type sentinel(2, 3);
  fill_matrix(sentinel, -1.0);
  auto device = uni20::to_device(sentinel, 0);
  auto output = uni20::mdspec_of(device);
  auto input = uni20::mdspec_of(std::as_const(row_major_source));

  EXPECT_FALSE(uni20::linalg::try_dispatch_kernel(uni20::linalg::CudaReferenceBackend{}, uni20::linalg::copy_op{},
                                                  output, input));
  auto const untouched = uni20::to_host(device);
  for (uni20::index_type row = 0; row < 2; ++row)
  {
    for (uni20::index_type column = 0; column < 3; ++column)
      EXPECT_DOUBLE_EQ((untouched[row, column]), -1.0);
  }

  EXPECT_TRUE(uni20::linalg::try_dispatch_kernel(uni20::linalg::cuda_host_transfer_selector(), uni20::linalg::copy_op{},
                                                 output, input));
  expect_matrix(uni20::to_host(device));
}

TEST_F(CudaCopyTest, HostStagingBackendCopiesAloneAndDeclinesWithoutWritingTheOutput)
{
  using uni20::linalg::copy_op;
  using uni20::linalg::try_dispatch_kernel;
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto const source = make_matrix();
  row_major_host_matrix_type row_major_source(2, 3);
  uni20::copy(row_major_source, source);
  auto const host_copies = uni20::linalg::select_backend_for_storage<uni20::HostStorage>(copy_op{});
  uni20::linalg::CudaHostStagingBackend staging{
      uni20::linalg::select_backend_for_storage<uni20::CudaStorage>(copy_op{}), host_copies,
      uni20::linalg::CudaReferenceBackend{}};

  // Alone, it handles a layout change and a pair that needs no staging.
  cuda_matrix_type device(runtime.device_resources(0), 2, 3);
  auto device_output = uni20::mdspec_of(device);
  auto row_major_input = uni20::mdspec_of(std::as_const(row_major_source));
  EXPECT_TRUE(try_dispatch_kernel(staging, copy_op{}, device_output, row_major_input));
  expect_matrix(uni20::to_host(device));

  cuda_matrix_type same_order(runtime.device_resources(0), 2, 3);
  auto same_order_output = uni20::mdspec_of(same_order);
  auto column_major_input = uni20::mdspec_of(source);
  EXPECT_TRUE(try_dispatch_kernel(staging, copy_op{}, same_order_output, column_major_input));
  expect_matrix(uni20::to_host(same_order));

  row_major_host_matrix_type row_major_result(2, 3);
  auto host_output = uni20::mdspec_of(row_major_result);
  auto device_input = uni20::mdspec_of(std::as_const(device));
  EXPECT_TRUE(try_dispatch_kernel(staging, copy_op{}, host_output, device_input));
  host_matrix_type result(2, 3);
  uni20::copy(result, row_major_result);
  expect_matrix(result);

  // The device layout step declines after the upload has filled the staging
  // buffer, and before the download has started; neither output changes.
  uni20::linalg::CudaHostStagingBackend declining{DecliningCopyBackend{}, host_copies,
                                                  uni20::linalg::CudaReferenceBackend{}};
  host_matrix_type sentinel(2, 3);
  fill_matrix(sentinel, -1.0);
  auto sentinel_device = uni20::to_device(sentinel, 0);
  auto sentinel_device_output = uni20::mdspec_of(sentinel_device);
  EXPECT_FALSE(try_dispatch_kernel(declining, copy_op{}, sentinel_device_output, row_major_input));
  row_major_host_matrix_type sentinel_host(2, 3);
  fill_matrix(sentinel_host, -1.0);
  auto sentinel_host_output = uni20::mdspec_of(sentinel_host);
  EXPECT_FALSE(try_dispatch_kernel(declining, copy_op{}, sentinel_host_output, device_input));

  auto const device_after = uni20::to_host(sentinel_device);
  for (uni20::index_type row = 0; row < 2; ++row)
  {
    for (uni20::index_type column = 0; column < 3; ++column)
    {
      EXPECT_DOUBLE_EQ((device_after[row, column]), -1.0);
      EXPECT_DOUBLE_EQ((sentinel_host[row, column]), -1.0);
    }
  }
}

TEST_F(CudaCopyTest, PeerStagingBackendCopiesAloneAndDeclinesWithoutWritingTheOutput)
{
  if (device_count_ < 2) GTEST_SKIP() << "peer-copy test requires two CUDA devices";
  using uni20::linalg::copy_op;
  using uni20::linalg::try_dispatch_kernel;
  using row_major_cuda_matrix = uni20::CudaTensor<double, 2, uni20::RowMajor>;
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0, 1}, .streams_per_device = 2});
  auto const device0 = uni20::to_device(make_matrix(), 0);
  auto input = uni20::mdspec_of(device0);

  row_major_cuda_matrix row_major(runtime.device_resources(1), 2, 3);
  auto output = uni20::mdspec_of(row_major);
  EXPECT_FALSE(try_dispatch_kernel(uni20::linalg::CudaReferenceBackend{}, copy_op{}, output, input));

  uni20::linalg::CudaPeerStagingBackend staging{uni20::linalg::CudaReferenceBackend{}};
  EXPECT_TRUE(try_dispatch_kernel(staging, copy_op{}, output, input));
  host_matrix_type result(2, 3);
  uni20::copy(result, row_major);
  expect_matrix(result);

  // Operands on one device are not a peer-staging instance.
  row_major_cuda_matrix same_device(runtime.device_resources(0), 2, 3);
  auto same_device_output = uni20::mdspec_of(same_device);
  EXPECT_FALSE(try_dispatch_kernel(staging, copy_op{}, same_device_output, input));

  uni20::linalg::CudaPeerStagingBackend declining{DecliningCopyBackend{}};
  host_matrix_type sentinel(2, 3);
  fill_matrix(sentinel, -1.0);
  row_major_cuda_matrix sentinel_device(runtime.device_resources(1), 2, 3);
  uni20::copy(sentinel_device, sentinel);
  auto sentinel_output = uni20::mdspec_of(sentinel_device);
  EXPECT_FALSE(try_dispatch_kernel(declining, copy_op{}, sentinel_output, input));
  host_matrix_type after(2, 3);
  uni20::copy(after, sentinel_device);
  for (uni20::index_type row = 0; row < 2; ++row)
  {
    for (uni20::index_type column = 0; column < 3; ++column)
      EXPECT_DOUBLE_EQ((after[row, column]), -1.0);
  }
}

TEST_F(CudaCopyTest, ElementsWithoutDefaultConstructorsTransferInAnyLayout)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto& resources = runtime.device_resources(0);
  uni20::Tensor<Packet, 2> host(uni20::uninitialized, 2, 3);
  for (uni20::index_type row = 0; row < 2; ++row)
  {
    for (uni20::index_type column = 0; column < 3; ++column)
      host[row, column] = Packet{static_cast<std::uint32_t>(10 * row + column + 1)};
  }
  auto expect_packets = [&](auto const& result) {
    for (uni20::index_type row = 0; row < 2; ++row)
    {
      for (uni20::index_type column = 0; column < 3; ++column)
        EXPECT_EQ((result[row, column].value), (host[row, column].value));
    }
  };

  // Contiguous, through the reference backend alone.
  expect_packets(uni20::to_host(uni20::to_device(host, resources)));

  // Staged both ways, then a strided device copy.
  uni20::CudaMatrix<Packet, uni20::RowMajor> row_major(uni20::uninitialized, resources, 2, 3);
  uni20::copy(row_major, host);
  uni20::RowMajorTensor<Packet, 2> row_major_host(uni20::uninitialized, 2, 3);
  uni20::copy(row_major_host, row_major);
  expect_packets(row_major_host);
  uni20::CudaMatrix<Packet> column_major(uni20::uninitialized, resources, 2, 3);
  uni20::copy(column_major, row_major);
  expect_packets(uni20::to_host(column_major));
}

TEST_F(CudaCopyTest, WideElementsCopyInAnyLayout)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  auto& resources = runtime.device_resources(0);
  uni20::Tensor<WideElement, 2> host(3, 4);
  for (uni20::index_type row = 0; row < 3; ++row)
  {
    for (uni20::index_type column = 0; column < 4; ++column)
    {
      for (int component = 0; component < 5; ++component)
        host[row, column].values[component] = 100.0 * row + 10.0 * column + component;
    }
  }
  auto expect_elements = [&](auto const& result) {
    for (uni20::index_type row = 0; row < 3; ++row)
    {
      for (uni20::index_type column = 0; column < 4; ++column)
      {
        for (int component = 0; component < 5; ++component)
          EXPECT_EQ((result[row, column].values[component]), (host[row, column].values[component]));
      }
    }
  };

  // A strided device copy uses the word loop; the host-device copies stage.
  auto device = uni20::to_device(host, resources);
  uni20::CudaMatrix<WideElement, uni20::RowMajor> row_major(uni20::uninitialized, resources, 3, 4);
  uni20::copy(row_major, device);
  expect_elements(uni20::to_host(row_major));
  uni20::RowMajorTensor<WideElement, 2> row_major_host(3, 4);
  uni20::copy(row_major_host, device);
  expect_elements(row_major_host);
  uni20::CudaMatrix<WideElement> column_major(uni20::uninitialized, resources, 3, 4);
  uni20::copy(column_major, row_major_host);
  expect_elements(uni20::to_host(column_major));
}

TEST_F(CudaCopyTest, RankZeroStridedTransfersRoundTrip)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  rank_zero_host host;
  host[] = 7.0;
  rank_zero_cuda device(uni20::uninitialized, runtime.device_resources(0), rank_zero_cuda::extents_type{});
  uni20::copy(device, host);
  rank_zero_host back;
  uni20::copy(back, device);
  EXPECT_EQ(back[], 7.0);
}

TEST_F(CudaCopyTest, AsyncToDeviceOfAConjugatedViewStagesFromAnotherDevice)
{
  if (device_count_ < 2) GTEST_SKIP() << "async peer-copy test requires two CUDA devices";
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0, 1}, .streams_per_device = 2});
  uni20::async::DebugCudaScheduler scheduler(uni20::cuda::Device::get(0));
  uni20::async::ScopedScheduler scoped(&scheduler);
  complex_host_matrix_type host(1, 2);
  host[0, 0] = complex_type{1.0, 2.0};
  host[0, 1] = complex_type{-3.0, 4.0};
  uni20::async::Async<complex_cuda_matrix_type> device = uni20::to_device(host, 0);

  auto moved = uni20::to_device(uni20::async::conj(device), runtime.device_resources(1));

  auto const& result = moved.get_wait(scheduler);
  EXPECT_EQ(result.storage().device().ordinal(), 1);
  auto const values = uni20::to_host(result);
  EXPECT_EQ((values[0, 0]), uni20::conj(host[0, 0]));
  EXPECT_EQ((values[0, 1]), uni20::conj(host[0, 1]));
}

TEST_F(CudaCopyTest, AsyncPeerMaterializationCarriesCompletionIntoOwnedResult)
{
  if (device_count_ < 2) GTEST_SKIP() << "async peer-copy test requires two CUDA devices";
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0, 1}, .streams_per_device = 2});
  uni20::async::DebugCudaScheduler scheduler(uni20::cuda::Device::get(0));
  uni20::async::ScopedScheduler scoped(&scheduler);
  uni20::async::Async<cuda_matrix_type> input = uni20::to_device(make_matrix(), 0);

  auto output = uni20::to_device(input, runtime.device_resources(1));

  auto const& device_result = output.get_wait(scheduler);
  EXPECT_EQ(device_result.storage().device().ordinal(), 1);
  expect_matrix(uni20::to_host(device_result));
}

TEST_F(CudaCopyTest, AsyncCopyRejectsOutputInputEpochAliasing)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::async::Async<cuda_matrix_type> tensor = uni20::to_device(make_matrix(), 0);
  ErrorModeGuard const error_mode;

  EXPECT_THROW(uni20::copy(tensor, tensor), std::runtime_error);
}

} // namespace
