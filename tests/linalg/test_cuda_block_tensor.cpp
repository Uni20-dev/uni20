#include <uni20/async/debug_scheduler.hpp>
#include <uni20/async/tbb_scheduler.hpp>
#include <uni20/backend/cuda/runtime.hpp>
#include <uni20/storage/cuda_storage.hpp>
#include <uni20/symmetry/block_tensor.hpp>
#include <uni20/symmetry/block_tensor_concepts.hpp>
#include <uni20/symmetry/block_tensor_linear.hpp>
#include <uni20/symmetry/block_tensor_permute.hpp>
#include <uni20/symmetry/block_tensor_svd.hpp>
#include <uni20/tensor/copy_into.hpp>

#include <cuda_runtime_api.h>
#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <limits>
#include <ranges>
#include <vector>

namespace
{

using host_storage = uni20::PackedCompleteBlockStorage<>;
using cuda_storage = uni20::PackedCompleteBlockStorage<uni20::CudaStorage>;
using parallel_cuda_storage = uni20::ParallelPackedCompleteBlockStorage<uni20::CudaStorage>;
using aligned_cuda_storage = uni20::PackedCompleteBlockStorage<uni20::CudaStorage, 64>;
using domain_type = uni20::Domain<uni20::BlockSpace>;
using codomain_type = uni20::Codomain<uni20::BlockSpace>;
using host_tensor = uni20::BlockTensor<double, domain_type, codomain_type, host_storage>;
using cuda_tensor = uni20::BlockTensor<double, domain_type, codomain_type, cuda_storage>;
using parallel_cuda_tensor = uni20::BlockTensor<double, domain_type, codomain_type, parallel_cuda_storage>;
using aligned_cuda_tensor = uni20::BlockTensor<double, domain_type, codomain_type, aligned_cuda_storage>;

static_assert(uni20::BlockTensorStorageFor<cuda_storage, double, 2, 2>);
static_assert(uni20::BlockTensorStorageFor<parallel_cuda_storage, double, 2, 2>);
static_assert(uni20::LocalBlockStorageFor<cuda_storage, double, 2, 2>);
static_assert(uni20::LocalBlockStorageFor<parallel_cuda_storage, double, 2, 2>);
static_assert(!uni20::ImmediateLocalBlockStorageFor<cuda_storage, double, 2, 2>);
static_assert(uni20::BlockTensorView<cuda_tensor>);
static_assert(uni20::MutableBlockTensorView<cuda_tensor>);
static_assert(!uni20::ImmediateBlockTensorView<cuda_tensor>);
static_assert(uni20::cuda::BufferMdspec<uni20::tensor_mdspec_t<uni20::block_tensor_const_block_t<cuda_tensor>>>);

class CudaBlockTensorTest : public ::testing::Test {
  protected:
    void SetUp() override
    {
      cudaError_t const status = cudaGetDeviceCount(&device_count_);
      if (status != cudaSuccess) GTEST_SKIP() << "CUDA device discovery failed: " << cudaGetErrorString(status);
      if (device_count_ == 0) GTEST_SKIP() << "no CUDA devices are available";
    }

    int device_count_ = 0;
};

#if UNI20_BACKEND_CUBLAS
TEST(CudaPartitionedBlockTensorTest, AllocationWideReductionEligibilityMatchesCublasIntegerRange)
{
  auto const maximum = static_cast<std::size_t>(std::numeric_limits<int>::max());
  EXPECT_TRUE(uni20::linalg::cuda_partitioned_reduction_size_supported(maximum));
  EXPECT_FALSE(uni20::linalg::cuda_partitioned_reduction_size_supported(maximum + 1));
}
#endif

TEST_F(CudaBlockTensorTest, PackedBlocksRoundTripAndAllocateLikePreservesPlacement)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::cuda::ScopedDevice scoped_device(0);

  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", 1}});
  uni20::BlockSpace const rows(symmetry, {{q0, 2}, {q1, 3}}, "rows");
  uni20::BlockSpace const columns(symmetry, {{q0, 4}, {q1, 5}}, "columns");

  host_tensor host(symmetry, domain_type{rows}, codomain_type{columns});
  double value = 1.0;
  for (std::size_t ordinal = 0; ordinal < host.stored_block_count(); ++ordinal)
  {
    auto block = host.block_by_ordinal(ordinal);
    for (uni20::index_type column = 0; column < block.extent(1); ++column)
      for (uni20::index_type row = 0; row < block.extent(0); ++row)
        block[row, column] = value++;
  }

  cuda_tensor device(symmetry, domain_type{rows}, codomain_type{columns});
  ASSERT_EQ(device.stored_block_count(), host.stored_block_count());
  EXPECT_EQ(device.storage().buffer().device().ordinal(), 0);
  ASSERT_EQ(device.storage().buffer().buffer_count(), device.stored_block_count());
  ASSERT_EQ(device.stored_block_count(), 2U);
  EXPECT_TRUE(device.storage().buffer().buffer(0).shares_allocation_with(device.storage().buffer().buffer(1)));
  EXPECT_EQ(device.storage().buffer().buffer(0).allocation_offset(), device.storage().offsets()[0]);
  EXPECT_EQ(device.storage().buffer().buffer(1).allocation_offset(), device.storage().offsets()[1]);
  for (std::size_t ordinal = 0; ordinal < device.stored_block_count(); ++ordinal)
  {
    auto output = device.block_by_ordinal(ordinal);
    auto input = host.block_by_ordinal(ordinal);
    uni20::copy(output, input);
  }

  auto allocated = device.allocate_like();
  EXPECT_EQ(allocated.storage().buffer().device(), device.storage().buffer().device());
  EXPECT_TRUE(std::ranges::equal(allocated.storage().offsets(), device.storage().offsets()));
  ASSERT_EQ(allocated.storage().buffer().buffer_count(), allocated.stored_block_count());
  EXPECT_TRUE(allocated.storage().buffer().buffer(0).shares_allocation_with(allocated.storage().buffer().buffer(1)));
  EXPECT_FALSE(allocated.storage().buffer().buffer(0).shares_allocation_with(device.storage().buffer().buffer(0)));

  host_tensor result(symmetry, domain_type{rows}, codomain_type{columns});
  for (std::size_t ordinal = 0; ordinal < device.stored_block_count(); ++ordinal)
  {
    auto output = result.block_by_ordinal(ordinal);
    auto input = device.block_by_ordinal(ordinal);
    uni20::copy(output, input);
  }

  for (std::size_t ordinal = 0; ordinal < result.stored_block_count(); ++ordinal)
  {
    auto expected = host.block_by_ordinal(ordinal);
    auto actual = result.block_by_ordinal(ordinal);
    for (uni20::index_type column = 0; column < actual.extent(1); ++column)
      for (uni20::index_type row = 0; row < actual.extent(0); ++row)
        EXPECT_DOUBLE_EQ((actual[row, column]), (expected[row, column]));
  }
}

TEST_F(CudaBlockTensorTest, FixedLinearOperationsRemainCudaResident)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::cuda::ScopedDevice scoped_device(0);

  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", 1}});
  uni20::BlockSpace const rows(symmetry, {{q0, 2}, {q1, 3}}, "rows");
  uni20::BlockSpace const columns(symmetry, {{q0, 4}, {q1, 5}}, "columns");

  host_tensor host_input(symmetry, domain_type{rows}, codomain_type{columns});
  double value = 1.0;
  for (std::size_t ordinal = 0; ordinal < host_input.stored_block_count(); ++ordinal)
  {
    auto block = host_input.block_by_ordinal(ordinal);
    for (uni20::index_type column = 0; column < block.extent(1); ++column)
      for (uni20::index_type row = 0; row < block.extent(0); ++row)
        block[row, column] = value++;
  }

  cuda_tensor input(symmetry, domain_type{rows}, codomain_type{columns});
  cuda_tensor output(symmetry, domain_type{rows}, codomain_type{columns});
  for (std::size_t ordinal = 0; ordinal < input.stored_block_count(); ++ordinal)
  {
    auto device_block = input.block_by_ordinal(ordinal);
    auto host_block = host_input.block_by_ordinal(ordinal);
    uni20::copy(device_block, host_block);
  }

  uni20::set_zero(output);
  uni20::copy(output, input);
  uni20::scale(output, 2.0);
  uni20::axpy(output, -0.5, input);

  host_tensor actual(symmetry, domain_type{rows}, codomain_type{columns});
  for (std::size_t ordinal = 0; ordinal < actual.stored_block_count(); ++ordinal)
  {
    auto host_block = actual.block_by_ordinal(ordinal);
    auto device_block = output.block_by_ordinal(ordinal);
    uni20::copy(host_block, device_block);
  }

  for (std::size_t ordinal = 0; ordinal < actual.stored_block_count(); ++ordinal)
  {
    auto expected = host_input.block_by_ordinal(ordinal);
    auto result = actual.block_by_ordinal(ordinal);
    for (uni20::index_type column = 0; column < result.extent(1); ++column)
      for (uni20::index_type row = 0; row < result.extent(0); ++row)
        EXPECT_DOUBLE_EQ((result[row, column]), (1.5 * expected[row, column]));
  }
}

TEST_F(CudaBlockTensorTest, AlignedPackedLinearOperationsKeepPaddingZero)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::cuda::ScopedDevice scoped_device(0);

  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", 1}});
  uni20::BlockSpace const rows(symmetry, {{q0, 3}, {q1, 2}}, "rows");
  uni20::BlockSpace const columns(symmetry, {{q0, 3}, {q1, 2}}, "columns");

  host_tensor host(symmetry, domain_type{rows}, codomain_type{columns});
  double value = 1.0;
  for (std::size_t ordinal = 0; ordinal < host.stored_block_count(); ++ordinal)
  {
    auto block = host.block_by_ordinal(ordinal);
    for (uni20::index_type column = 0; column < block.extent(1); ++column)
      for (uni20::index_type row = 0; row < block.extent(0); ++row)
        block[row, column] = value++;
  }

  aligned_cuda_tensor device(symmetry, domain_type{rows}, codomain_type{columns});
  ASSERT_TRUE(device.storage().has_padding());
  EXPECT_TRUE(std::ranges::equal(device.storage().offsets(), std::vector<std::size_t>{0, 16, 20}));
  EXPECT_TRUE(std::ranges::equal(device.storage().block_ends(), std::vector<std::size_t>{9, 20}));
  for (std::size_t ordinal = 0; ordinal < device.stored_block_count(); ++ordinal)
    uni20::copy(device.block_by_ordinal(ordinal), host.block_by_ordinal(ordinal));

#if UNI20_BACKEND_CUBLAS
  EXPECT_DOUBLE_EQ(uni20::inner_product_host(device, device), 819.0);
  EXPECT_DOUBLE_EQ(uni20::norm_host(device), std::sqrt(819.0));
#endif

  auto copy_physical = [](aligned_cuda_tensor const& tensor) {
    auto stream = tensor.storage().buffer().resources().streams().acquire();
    auto access = tensor.storage().buffer().read_synchronized_with(stream);
    std::vector<double> physical(tensor.storage().buffer().size());
    uni20::cuda::check(cudaMemcpyAsync(physical.data(), access.data(), physical.size() * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream.native_handle()),
                       "copy aligned packed CUDA storage", 0);
    stream.synchronize();
    access.release_after_synchronization();
    return physical;
  };

  auto physical = copy_physical(device);
  for (std::size_t offset = 9; offset < 16; ++offset)
    EXPECT_DOUBLE_EQ(physical[offset], 0.0);

  uni20::scale(device, 2.0);
#if UNI20_BACKEND_CUBLAS
  EXPECT_DOUBLE_EQ(uni20::norm_host(device), 2.0 * std::sqrt(819.0));
#endif
  physical = copy_physical(device);
  for (std::size_t offset = 9; offset < 16; ++offset)
    EXPECT_DOUBLE_EQ(physical[offset], 0.0);

  uni20::set_zero(device);
#if UNI20_BACKEND_CUBLAS
  EXPECT_DOUBLE_EQ(uni20::norm_host(device), 0.0);
#endif
  physical = copy_physical(device);
  for (double const element : physical)
    EXPECT_DOUBLE_EQ(element, 0.0);
}

TEST_F(CudaBlockTensorTest, NonfiniteScaledPayloadsKeepAlignedPaddingZero)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::cuda::ScopedDevice scoped_device(0);

  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", 1}});
  uni20::BlockSpace const space(symmetry, {{q0, 3}, {q1, 2}}, "space");
  host_tensor host(symmetry, domain_type{space}, codomain_type{space});
  for (std::size_t ordinal = 0; ordinal < host.stored_block_count(); ++ordinal)
  {
    auto block = host.block_by_ordinal(ordinal);
    for (uni20::index_type column = 0; column < block.extent(1); ++column)
      for (uni20::index_type row = 0; row < block.extent(0); ++row)
        block[row, column] = 1.0;
  }

  auto copy_payload_to_device = [&](aligned_cuda_tensor& tensor) {
    for (std::size_t ordinal = 0; ordinal < tensor.stored_block_count(); ++ordinal)
      uni20::copy(tensor.block_by_ordinal(ordinal), host.block_by_ordinal(ordinal));
  };
  auto expect_zero_padding = [](aligned_cuda_tensor const& tensor) {
    auto stream = tensor.storage().buffer().resources().streams().acquire();
    auto access = tensor.storage().buffer().read_synchronized_with(stream);
    std::vector<double> physical(tensor.storage().buffer().size());
    uni20::cuda::check(cudaMemcpyAsync(physical.data(), access.data(), physical.size() * sizeof(double),
                                       cudaMemcpyDeviceToHost, stream.native_handle()),
                       "copy aligned nonfinite CUDA storage", 0);
    stream.synchronize();
    access.release_after_synchronization();
    for (std::size_t offset = 9; offset < 16; ++offset)
      EXPECT_DOUBLE_EQ(physical[offset], 0.0);
  };
  auto expect_infinite_payload = [](aligned_cuda_tensor const& tensor) {
    uni20::ColumnMajorTensor<double, 2> host_block(3, 3);
    uni20::copy(host_block, tensor.block_by_ordinal(0));
    EXPECT_TRUE(std::isinf(host_block[0, 0]));
  };

  double const infinity = std::numeric_limits<double>::infinity();
  aligned_cuda_tensor input(symmetry, domain_type{space}, codomain_type{space});
  aligned_cuda_tensor output(symmetry, domain_type{space}, codomain_type{space});
  copy_payload_to_device(input);
  copy_payload_to_device(output);

  uni20::scale(output, infinity);
  expect_zero_padding(output);
  expect_infinite_payload(output);

  uni20::assign_scale(output, infinity, input);
  expect_zero_padding(output);
  expect_infinite_payload(output);

  uni20::set_zero(output);
  uni20::axpy(output, infinity, input);
  expect_zero_padding(output);
  expect_infinite_payload(output);
}

TEST_F(CudaBlockTensorTest, ParallelPackedLinearOperationsDispatchFromSchedulerBatch)
{
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 4});
  uni20::cuda::ScopedDevice scoped_device(0);
  uni20::async::TbbScheduler scheduler{4};
  uni20::async::ScopedScheduler scoped_scheduler(&scheduler);

  uni20::Symmetry const symmetry{"N:U(1)"};
  auto const q0 = uni20::QNum::identity(symmetry);
  auto const q1 = uni20::make_qnum(symmetry, {{"N", 1}});
  uni20::BlockSpace const rows(symmetry, {{q0, 2}, {q1, 3}}, "rows");
  uni20::BlockSpace const columns(symmetry, {{q0, 4}, {q1, 5}}, "columns");

  host_tensor host_input(symmetry, domain_type{rows}, codomain_type{columns});
  double value = 1.0;
  for (std::size_t ordinal = 0; ordinal < host_input.stored_block_count(); ++ordinal)
  {
    auto block = host_input.block_by_ordinal(ordinal);
    for (uni20::index_type column = 0; column < block.extent(1); ++column)
      for (uni20::index_type row = 0; row < block.extent(0); ++row)
        block[row, column] = value++;
  }

  parallel_cuda_tensor input(symmetry, domain_type{rows}, codomain_type{columns});
  parallel_cuda_tensor output(symmetry, domain_type{rows}, codomain_type{columns});
  for (std::size_t ordinal = 0; ordinal < input.stored_block_count(); ++ordinal)
    uni20::copy(input.block_by_ordinal(ordinal), host_input.block_by_ordinal(ordinal));

  uni20::set_zero(output);
  uni20::copy(output, input);
  uni20::scale(output, 2.0);
  uni20::axpy(output, -0.5, input);

  host_tensor actual(symmetry, domain_type{rows}, codomain_type{columns});
  for (std::size_t ordinal = 0; ordinal < actual.stored_block_count(); ++ordinal)
    uni20::copy(actual.block_by_ordinal(ordinal), output.block_by_ordinal(ordinal));

  for (std::size_t ordinal = 0; ordinal < actual.stored_block_count(); ++ordinal)
  {
    auto expected = host_input.block_by_ordinal(ordinal);
    auto result = actual.block_by_ordinal(ordinal);
    for (uni20::index_type column = 0; column < result.extent(1); ++column)
      for (uni20::index_type row = 0; row < result.extent(0); ++row)
        EXPECT_DOUBLE_EQ((result[row, column]), (1.5 * expected[row, column]));
  }
}

TEST_F(CudaBlockTensorTest, SixLegPermutationPreservesDenseAxesAndChargeSectors)
{
  using namespace uni20;
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::cuda::ScopedDevice scoped_device(0);
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  auto const q1 = make_qnum(sym, {{"N", 1}});
  BlockSpace const charged(sym, {{q0, 2}, {q1, 2}}, "charged");
  BlockSpace const neutral(sym, {{q0, 2}}, "neutral");
  using DomainType = Domain<BlockSpace, BlockSpace, BlockSpace, BlockSpace>;
  using CodomainType = Codomain<BlockSpace, BlockSpace>;
  using Host = BlockTensor<double, DomainType, CodomainType, host_storage>;
  using Device = BlockTensor<double, DomainType, CodomainType, cuda_storage>;
  Host host(sym, DomainType{charged, neutral, neutral, neutral}, CodomainType{charged, neutral});
  for (std::size_t sector = 0; sector < 2; ++sector)
  {
    auto block = host.block(Host::key_type{{sector, 0, 0, 0, sector, 0}});
    for (index_type a = 0; a < 2; ++a)
      for (index_type b = 0; b < 2; ++b)
        for (index_type c = 0; c < 2; ++c)
          for (index_type d = 0; d < 2; ++d)
            for (index_type e = 0; e < 2; ++e)
              for (index_type f = 0; f < 2; ++f)
                block[a, b, c, d, e, f] = static_cast<double>(1 + a + 2 * b + 3 * c + 5 * d + 7 * e + 11 * f +
                                                              (a + c) * (e + f) + 13 * sector);
  }
  Device device(sym, host.domain(), host.codomain(), runtime.device_resources(0));
  for (std::size_t ordinal = 0; ordinal < host.stored_block_count(); ++ordinal)
    uni20::copy(device.block_by_ordinal(ordinal), host.block_by_ordinal(ordinal));

  // Materialize a noncontiguous six-axis view on the device before reading it back.
  auto view = permute<0, 3, 2, 1, 4, 5>(device);
  Device reordered(sym, view.domain(), view.codomain(), runtime.device_resources(0));
  uni20::copy(reordered, view);
  Host roundtrip(sym, view.domain(), view.codomain());
  for (std::size_t ordinal = 0; ordinal < roundtrip.stored_block_count(); ++ordinal)
    uni20::copy(roundtrip.block_by_ordinal(ordinal), reordered.block_by_ordinal(ordinal));
  ASSERT_TRUE(std::ranges::equal(roundtrip.stored_keys(), host.stored_keys()));
  for (std::size_t sector = 0; sector < 2; ++sector)
  {
    auto original = host.block(Host::key_type{{sector, 0, 0, 0, sector, 0}});
    auto actual = roundtrip.block(Host::key_type{{sector, 0, 0, 0, sector, 0}});
    for (index_type a = 0; a < 2; ++a)
      for (index_type b = 0; b < 2; ++b)
        for (index_type c = 0; c < 2; ++c)
          for (index_type d = 0; d < 2; ++d)
            for (index_type e = 0; e < 2; ++e)
              for (index_type f = 0; f < 2; ++f)
                EXPECT_DOUBLE_EQ((actual[a, d, c, b, e, f]), (original[a, b, c, d, e, f]));
  }
}

#if UNI20_BACKEND_CUSOLVER
TEST_F(CudaBlockTensorTest, SixLegMatrixBlockSvdMaterializesFiveLegCudaFactors)
{
  using namespace uni20;
  auto runtime = uni20::cuda::initialize({.device_ordinals = {0}, .streams_per_device = 2});
  uni20::cuda::ScopedDevice scoped_device(0);
  Symmetry const sym{"N:U(1)"};
  auto const q0 = QNum::identity(sym);
  auto const q1 = make_qnum(sym, {{"N", 1}});
  BlockSpace const rows(sym, {{q0, 2}, {q1, 2}}, "rows");
  BlockSpace const columns(sym, {{q0, 3}, {q1, 3}}, "columns");
  LocalSpace const local(sym, {q0, q0}, "repeated-local-charge");
  QNumSpace const fixed(q1, "fixed");
  QNumSpace const identity(q0, "identity");
  using DomainType = Domain<BlockSpace, QNumSpace, Dual<QNumSpace>, LocalSpace>;
  using CodomainType = Codomain<BlockSpace, QNumSpace>;
  using Host = BlockTensor<double, DomainType, CodomainType, host_storage>;
  using Device = BlockTensor<double, DomainType, CodomainType, cuda_storage>;
  Host host(sym, DomainType{rows, fixed, dual(fixed), local}, CodomainType{columns, identity});
  ASSERT_EQ(host.stored_block_count(), 4);
  for (std::size_t ordinal = 0; ordinal < host.stored_block_count(); ++ordinal)
  {
    auto block = host.block_by_ordinal(ordinal);
    for (index_type i = 0; i < 2; ++i)
      for (index_type j = 0; j < 3; ++j)
        block[i, j] = static_cast<double>(1 + i + 2 * j + 3 * i * j + 7 * ordinal);
  }
  Device device(sym, host.domain(), host.codomain(), runtime.device_resources(0));
  static_assert(Device::order() == 6 && Device::dense_block_order() == 2);
  for (std::size_t ordinal = 0; ordinal < host.stored_block_count(); ++ordinal)
    uni20::copy(device.block_by_ordinal(ordinal), host.block_by_ordinal(ordinal));
  auto decomposition = block_svd(device);
  ASSERT_EQ(decomposition.sectors().size(), 2);
  auto factors = materialize_svd(decomposition, select_svd_states(decomposition.spectrum()));
  using Left = decltype(factors.left_singular_vectors);
  using Right = decltype(factors.right_singular_vectors_adjoint);
  static_assert(Left::order() == 3 && Right::order() == 5);
  static_assert(std::same_as<typename Right::storage_policy, PackedSparseBlockStorage<CudaStorage>>);
  EXPECT_EQ(&factors.right_singular_vectors_adjoint.allocation_context(), &runtime.device_resources(0));
  EXPECT_EQ(factors.right_singular_vectors_adjoint.storage().buffer().device().ordinal(), 0);
  auto const& bond = factors.singular_values.domain().template space<0>();
  ASSERT_EQ(bond.size(), 2);
  for (std::size_t sector = 0; sector < 2; ++sector)
  {
    auto found = std::ranges::find_if(bond, [&](auto const& entry) { return entry.q == rows[sector].q; });
    ASSERT_NE(found, bond.end());
    auto const s = static_cast<std::size_t>(found - bond.begin());
    auto values = factors.singular_values.diagonal_values(typename decltype(factors.singular_values)::key_type{{s, s}});
    Tensor<double, 2> lhs(static_cast<index_type>(values.size()), 3);
    uni20::copy(lhs, factors.left_singular_vectors.block(Left::key_type{{s, sector}}));
    for (std::size_t state = 0; state < 2; ++state)
    {
      Tensor<double, 2> rhs(2, static_cast<index_type>(values.size()));
      uni20::copy(rhs, factors.right_singular_vectors_adjoint.block(Right::key_type{{sector, state, s}}));
      auto original = host.block(Host::key_type{{sector, state, sector}});
      for (index_type i = 0; i < 2; ++i)
        for (index_type j = 0; j < 3; ++j)
        {
          double reconstructed = 0.0;
          for (index_type k = 0; k < lhs.extent(0); ++k)
            reconstructed += lhs[k, j] * values[k] * rhs[i, k];
          EXPECT_NEAR(reconstructed, (original[i, j]), 1.0e-11);
        }
    }
  }
}
#endif

} // namespace
